import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


class Client:
    def __init__(self, client_id, config, device, labeled_loader, unlabeled_loader, valloader):
        self.client_id = client_id
        self.config = config
        self.device = device
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.valloader = valloader

        # Iterators for step-based mini-batch sampling
        self.iter_labeled = None
        self.iter_unlabeled = None

        # Data Augmentation Module:
        # RandAugment with num_ops=1 and magnitude=10 as specified in Section 4.1 (Eq. 5)
        self.rand_augment = transforms.RandAugment(num_ops=1, magnitude=10)

    def apply_strong_augment(self, x):
        """
        Applies strong augmentation (RandAugment(1, 10)) to a batch of images.
        Safely handles both [0, 1] tensors and pre-normalized tensors without uint8 wrap-around.
        """
        if x.dim() != 4:
            return x

        augmented = []
        for img in x.detach().cpu():
            img_min = img.min()
            img_max = img.max()
            # If tensor is outside [0, 1] (e.g. standardized with mean/std), rescale to [0, 255] uint8
            if img_min < 0.0 or img_max > 1.0:
                scale = (img_max - img_min).clamp(min=1e-6)
                img_uint8 = ((img - img_min) / scale * 255.0).round().clamp(0, 255).to(torch.uint8)
                aug_uint8 = self.rand_augment(img_uint8)
                aug_img = (aug_uint8.to(img.dtype) / 255.0) * scale + img_min
            else:
                img_uint8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
                aug_uint8 = self.rand_augment(img_uint8)
                aug_img = aug_uint8.to(img.dtype) / 255.0
            augmented.append(aug_img)

        return torch.stack(augmented, dim=0).to(self.device)

    def compute_confidence(self, logits, metric=None):
        r"""
        Compute a non-negative confidence score h(\cdot) >= 0 for the given logits.
        Metrics supported:
          - 'variance': Variance of logits across classes (default in FedLabel paper, Eq. 3).
          - 'entropy': Normalized certainty (log(N) - H(p)) >= 0 so higher is more confident
                       and ratios h(s^{-*}) / h(s^*) in Eq. (7) remain in [0, 1].
          - 'max_prob': Maximum softmax probability in [0, 1].
        """
        if metric is None:
            metric = getattr(self.config, "confidence_metric", "variance")

        if metric == "variance":
            return torch.var(logits, dim=-1)
        elif metric == "entropy":
            num_classes = logits.size(-1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = math.log(max(num_classes, 2))
            return torch.clamp(max_entropy - entropy, min=0.0)
        elif metric == "max_prob":
            return torch.max(torch.softmax(logits, dim=-1), dim=-1)[0]
        else:
            raise ValueError(f"Unknown confidence metric: {metric}")

    def _get_next_batch(self, loader, current_iter):
        """
        Retrieves the next mini-batch from a dataloader cycling infinitely.
        """
        if current_iter is None:
            current_iter = iter(loader)
        try:
            batch = next(current_iter)
        except StopIteration:
            current_iter = iter(loader)
            batch = next(current_iter)
        return batch, current_iter

    @staticmethod
    def _set_bn_eval(model):
        """Sets BatchNorm layers to eval mode when batch size is 1 to prevent crash."""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    @staticmethod
    def _set_bn_train(model):
        """Restores BatchNorm layers to train mode."""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()

    def update(self, global_model):
        """
        Executes the three-stage FedLabel update (Algorithm 1):
          Stage 1: Supervised Local Update (tau labeled steps, Eq. 1-2)
          Stage 2: Semi-Supervised Update (tau' unlabeled steps with consistency loss, Eq. 3-8, 10)
          Stage 3: Merging Updates: Delta w_k = Delta w_{L,k} + Delta w_{U,k}
        Returns:
          delta_w: Dict of weight updates tensor-by-tensor.
          r_k: Total number of samples used (labeled + threshold-passing unlabeled).
        """
        global_model = global_model.to(self.device)
        global_model.eval()
        global_state = global_model.state_dict()
        param_keys = {k for k, _ in global_model.named_parameters()}

        # -------------------------------------------------------------
        # Stage 1: Supervised Local Update (Labeled Training, Eq. 1-2)
        # -------------------------------------------------------------
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer_l = optim.SGD(local_model.parameters(), lr=self.config.local_lr)
        criterion = nn.CrossEntropyLoss()

        use_epochs = getattr(self.config, "use_epochs", False)
        total_labeled_used = 0

        if use_epochs:
            for _ in range(self.config.tau):
                for x, y in self.labeled_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if x.size(0) == 0:
                        continue
                    if x.size(0) == 1:
                        self._set_bn_eval(local_model)
                    else:
                        self._set_bn_train(local_model)
                    optimizer_l.zero_grad()
                    loss = criterion(local_model(x), y)
                    loss.backward()
                    optimizer_l.step()
                    total_labeled_used += x.size(0)
        else:
            # Paper default: tau local mini-batch SGD iterations
            for _ in range(self.config.tau):
                (x, y), self.iter_labeled = self._get_next_batch(self.labeled_loader, self.iter_labeled)
                x, y = x.to(self.device), y.to(self.device)
                if x.size(0) == 0:
                    continue
                if x.size(0) == 1:
                    self._set_bn_eval(local_model)
                else:
                    self._set_bn_train(local_model)
                optimizer_l.zero_grad()
                loss = criterion(local_model(x), y)
                loss.backward()
                optimizer_l.step()
                total_labeled_used += x.size(0)

        local_state = local_model.state_dict()

        # Calculate labeled update difference: Delta w_{L,k} = w_{L,k} - w_global
        delta_w_L = {
            k: (local_state[k].cpu() - global_state[k].cpu())
            if torch.is_floating_point(local_state[k])
            else torch.zeros_like(local_state[k].cpu())
            for k in global_state.keys()
        }

        # -------------------------------------------------------------
        # Stage 2: Semi-Supervised Update (Unlabeled Training, Eq. 3-8, 10)
        # -------------------------------------------------------------
        unlabeled_model = copy.deepcopy(global_model).to(self.device)
        unlabeled_model.train()

        optimizer_u = optim.SGD(unlabeled_model.parameters(), lr=self.config.local_lr)
        criterion_u_sum = nn.CrossEntropyLoss(reduction="sum")

        total_valid_unlabeled = 0
        unlabeled_updated = False

        if self.unlabeled_loader is not None and len(self.unlabeled_loader) > 0:
            unlabeled_steps = []
            if use_epochs:
                for _ in range(self.config.tau_prime):
                    for x_u, _ in self.unlabeled_loader:
                        unlabeled_steps.append(x_u)
            else:
                for _ in range(self.config.tau_prime):
                    (x_u, _), self.iter_unlabeled = self._get_next_batch(
                        self.unlabeled_loader, self.iter_unlabeled
                    )
                    unlabeled_steps.append(x_u)

            local_model.eval()
            for x_u in unlabeled_steps:
                batch_size = x_u.size(0)
                if batch_size == 0:
                    continue
                x_u = x_u.to(self.device)

                with torch.no_grad():
                    # 1) Logits from global model and local model on unlabeled data
                    logits_g = global_model(x_u)
                    logits_l = local_model(x_u)

                    # 2) Confidence calculation: h(s)
                    conf_g = self.compute_confidence(logits_g)
                    conf_l = self.compute_confidence(logits_l)

                    # 3) Model Selection (Eq. 3): Identify s* and s^{-*}
                    mask_g_better = conf_g > conf_l

                    selected_logits = torch.where(mask_g_better.unsqueeze(1), logits_g, logits_l)
                    discarded_logits = torch.where(mask_g_better.unsqueeze(1), logits_l, logits_g)

                    selected_conf = torch.where(mask_g_better, conf_g, conf_l)
                    discarded_conf = torch.where(mask_g_better, conf_l, conf_g)

                    # 4) Thresholding (Eq. 4): pseudo-label hat{y}
                    probs_s = torch.softmax(selected_logits, dim=-1)
                    max_probs, pseudo_labels = torch.max(probs_s, dim=-1)

                    mask_threshold = max_probs > self.config.beta

                if not mask_threshold.any():
                    continue

                x_u_filtered = x_u[mask_threshold]
                pseudo_labels_filtered = pseudo_labels[mask_threshold]
                discarded_logits_filtered = discarded_logits[mask_threshold]
                selected_conf_filtered = selected_conf[mask_threshold]
                discarded_conf_filtered = discarded_conf[mask_threshold]

                num_passed = x_u_filtered.size(0)
                total_valid_unlabeled += num_passed

                # 5) Apply strong augmentation: psi(xi) (Eq. 5)
                x_u_aug = self.apply_strong_augment(x_u_filtered)

                optimizer_u.zero_grad()
                unlabeled_model.train()
                if num_passed == 1:
                    self._set_bn_eval(unlabeled_model)
                else:
                    self._set_bn_train(unlabeled_model)

                logits_u = unlabeled_model(x_u_aug)

                # Cross-Entropy Loss normalized by mini-batch size b as in Eq. (10)
                loss_ce = criterion_u_sum(logits_u, pseudo_labels_filtered) / float(batch_size)

                # Global-Local Consistency Loss (Eq. 6-7):
                # Check if discarded model agrees with pseudo-label: argmax s^{-*}(xi) == hat{y}_xi
                discarded_preds = torch.argmax(discarded_logits_filtered, dim=-1)
                consistency_mask = discarded_preds == pseudo_labels_filtered

                loss_kl = 0.0
                if consistency_mask.any():
                    log_probs_u = F.log_softmax(logits_u[consistency_mask], dim=-1)
                    probs_discarded = F.softmax(discarded_logits_filtered[consistency_mask], dim=-1)

                    # Adaptive weight (Eq. 7): lambda_k = lambda_0 * (h(s^{-*}) / h(s^*)) <= lambda_0
                    ratio = discarded_conf_filtered[consistency_mask] / (
                        selected_conf_filtered[consistency_mask] + 1e-8
                    )
                    lambda_k = self.config.lambda_0 * torch.clamp(ratio, min=0.0, max=1.0)

                    # KL Divergence: KL( s(w_U), s^{-*} )
                    kl_loss_per_sample = F.kl_div(
                        log_probs_u, probs_discarded, reduction="none"
                    ).sum(dim=-1)

                    # Normalized by mini-batch size b to match Eq. (8) & Eq. (10)
                    loss_kl = (lambda_k * kl_loss_per_sample).sum() / float(batch_size)

                loss_total = loss_ce + loss_kl
                loss_total.backward()
                optimizer_u.step()
                unlabeled_updated = True

        unlabeled_state = unlabeled_model.state_dict()

        # Calculate unlabeled update difference: Delta w_{U,k} = w_{U,k} - w_global
        delta_w_U = {
            k: (unlabeled_state[k].cpu() - global_state[k].cpu())
            if torch.is_floating_point(unlabeled_state[k])
            else torch.zeros_like(unlabeled_state[k].cpu())
            for k in global_state.keys()
        }

        # -------------------------------------------------------------
        # Stage 3: Merging Updates (Algorithm 1, Line 8)
        # -------------------------------------------------------------
        # Sum parameter updates: Delta w_k = Delta w_{L,k} + Delta w_{U,k}
        # For non-parameter floating-point buffers (e.g., BatchNorm running_mean/running_var),
        # average the buffer diffs so global_state + delta_w never drives running_var negative.
        delta_w = {}
        for k in global_state.keys():
            if k in param_keys:
                delta_w[k] = delta_w_L[k] + delta_w_U[k]
            elif torch.is_floating_point(global_state[k]):
                if unlabeled_updated:
                    delta_w[k] = 0.5 * (delta_w_L[k] + delta_w_U[k])
                else:
                    delta_w[k] = delta_w_L[k]
            else:
                delta_w[k] = torch.zeros_like(global_state[k].cpu())

        # Total number of samples used in local updates: r_k^{(t,0)}
        r_k = total_labeled_used + total_valid_unlabeled
        if r_k <= 0:
            r_k = len(self.labeled_loader.dataset) if len(self.labeled_loader.dataset) > 0 else 1

        return delta_w, r_k