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

        self.iter_labeled = None
        self.iter_unlabeled = None

        # RandAugment(1, 10) as specified in ICCV 2023 paper
        self.rand_augment = transforms.RandAugment(num_ops=1, magnitude=10)

    def apply_strong_augment(self, x):
        """
        Applies strong augmentation to a batch of images (B, C, H, W)
        while preserving the input tensor's exact numerical range.
        """
        if x.dim() != 4:
            return x

        _, c, h, w = x.shape

        if c == 1:
            pad = max(1, h // 8)
            padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            augmented = []
            for img in padded:
                top = torch.randint(0, 2 * pad + 1, (1,)).item()
                left = torch.randint(0, 2 * pad + 1, (1,)).item()
                augmented.append(img[:, top : top + h, left : left + w])
            return torch.stack(augmented, dim=0).to(self.device)

        augmented = []
        for img in x.detach().cpu():
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img_min = img.min()
            img_max = img.max()
            scale = (img_max - img_min).clamp(min=1e-6)
            img_uint8 = ((img - img_min) / scale * 255.0).round().clamp(0, 255).to(torch.uint8)
            aug_uint8 = self.rand_augment(img_uint8)
            aug_img = (aug_uint8.to(img.dtype) / 255.0) * scale + img_min
            augmented.append(aug_img)

        return torch.stack(augmented, dim=0).to(self.device)

    def compute_confidence(self, logits, metric=None):
        r"""
        Compute a non-negative confidence score h(\cdot) >= 0 for the given logits.
        """
        if metric is None:
            metric = getattr(self.config, "confidence_metric", "variance")

        safe_logits = torch.clamp(logits, min=-40.0, max=40.0)

        if metric == "variance":
            return torch.var(safe_logits, dim=-1, unbiased=False)
        elif metric == "entropy":
            num_classes = safe_logits.size(-1)
            probs = torch.softmax(safe_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-8)), dim=-1)
            max_entropy = math.log(max(num_classes, 2))
            return torch.clamp(max_entropy - entropy, min=0.0)
        elif metric == "max_prob":
            return torch.max(torch.softmax(safe_logits, dim=-1), dim=-1)[0]
        else:
            raise ValueError(f"Unknown confidence metric: {metric}")

    def _get_next_batch(self, loader, current_iter):
        if current_iter is None:
            current_iter = iter(loader)
        try:
            batch = next(current_iter)
        except StopIteration:
            current_iter = iter(loader)
            batch = next(current_iter)
        return batch, current_iter

    def update(self, global_model):
        global_model = global_model.to(self.device)
        global_model.eval()
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        param_keys = {k for k, _ in global_model.named_parameters()}

        # -------------------------------------------------------------
        # Stage 1: Supervised Local Update (Labeled Training)
        # -------------------------------------------------------------
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer_l = optim.SGD(
            local_model.parameters(),
            lr=self.config.local_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()

        use_epochs = getattr(self.config, "use_epochs", False)

        if use_epochs:
            for _ in range(self.config.tau):
                for x, y in self.labeled_loader:
                    x, y = x.to(self.device), y.to(self.device).long().view(-1)
                    if x.size(0) <= 1:
                        continue
                    optimizer_l.zero_grad()
                    loss = criterion(local_model(x), y)
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
                        optimizer_l.step()
        else:
            for _ in range(self.config.tau):
                (x, y), self.iter_labeled = self._get_next_batch(self.labeled_loader, self.iter_labeled)
                x, y = x.to(self.device), y.to(self.device).long().view(-1)
                if x.size(0) <= 1:
                    continue
                optimizer_l.zero_grad()
                loss = criterion(local_model(x), y)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
                    optimizer_l.step()

        local_state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}

        delta_w_L = {
            k: (local_state[k] - global_state[k])
            if torch.is_floating_point(local_state[k])
            else torch.zeros_like(local_state[k])
            for k in global_state.keys()
        }

        # -------------------------------------------------------------
        # Stage 2: Semi-Supervised Update (Unlabeled Training)
        # -------------------------------------------------------------
        unlabeled_model = copy.deepcopy(global_model).to(self.device)
        unlabeled_model.train()

        optimizer_u = optim.SGD(
            unlabeled_model.parameters(),
            lr=self.config.local_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion_u = nn.CrossEntropyLoss()

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
                if x_u.size(0) <= 1:
                    continue
                x_u = x_u.to(self.device)

                with torch.no_grad():
                    logits_g = global_model(x_u)
                    logits_l = local_model(x_u)

                    if not (torch.isfinite(logits_g).all() and torch.isfinite(logits_l).all()):
                        continue

                    conf_g = self.compute_confidence(logits_g)
                    conf_l = self.compute_confidence(logits_l)

                    mask_g_better = conf_g > conf_l

                    selected_logits = torch.where(mask_g_better.unsqueeze(1), logits_g, logits_l)
                    discarded_logits = torch.where(mask_g_better.unsqueeze(1), logits_l, logits_g)

                    selected_conf = torch.where(mask_g_better, conf_g, conf_l)
                    discarded_conf = torch.where(mask_g_better, conf_l, conf_g)

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
                if num_passed <= 1:
                    continue

                total_valid_unlabeled += num_passed

                x_u_aug = self.apply_strong_augment(x_u_filtered)

                optimizer_u.zero_grad()
                logits_u = unlabeled_model(x_u_aug)

                loss_ce = criterion_u(logits_u, pseudo_labels_filtered)

                discarded_preds = torch.argmax(discarded_logits_filtered, dim=-1)
                consistency_mask = discarded_preds == pseudo_labels_filtered

                loss_kl = 0.0
                if consistency_mask.any():
                    log_probs_u = F.log_softmax(logits_u[consistency_mask], dim=-1)
                    log_probs_discarded = F.log_softmax(
                        discarded_logits_filtered[consistency_mask], dim=-1
                    )

                    ratio = discarded_conf_filtered[consistency_mask] / (
                        selected_conf_filtered[consistency_mask].clamp(min=1e-6)
                    )
                    lambda_k = self.config.lambda_0 * torch.clamp(ratio, min=0.0, max=1.0)

                    kl_loss_per_sample = F.kl_div(
                        log_probs_u, log_probs_discarded, reduction="none", log_target=True
                    ).sum(dim=-1)

                    loss_kl = (lambda_k * kl_loss_per_sample).mean()

                loss_total = loss_ce + loss_kl
                if torch.isfinite(loss_total):
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(unlabeled_model.parameters(), max_norm=5.0)
                    optimizer_u.step()
                    unlabeled_updated = True

        unlabeled_state = {k: v.detach().cpu() for k, v in unlabeled_model.state_dict().items()}

        delta_w_U = {
            k: (unlabeled_state[k] - global_state[k])
            if torch.is_floating_point(unlabeled_state[k])
            else torch.zeros_like(unlabeled_state[k])
            for k in global_state.keys()
        }

        # -------------------------------------------------------------
        # Stage 3: Merging Updates
        # -------------------------------------------------------------
        delta_w = {}
        for k in global_state.keys():
            if k in param_keys:
                combined = delta_w_L[k] + delta_w_U[k]
                delta_w[k] = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            elif torch.is_floating_point(global_state[k]):
                if unlabeled_updated:
                    # Average the BatchNorm running buffers so running_var stays mathematically positive
                    buf_avg = 0.5 * (delta_w_L[k] + delta_w_U[k])
                    delta_w[k] = torch.nan_to_num(buf_avg, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    delta_w[k] = torch.nan_to_num(delta_w_L[k], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                delta_w[k] = torch.zeros_like(global_state[k])

        r_k = len(self.labeled_loader.dataset) + total_valid_unlabeled
        if r_k <= 0:
            r_k = 1

        return delta_w, r_k