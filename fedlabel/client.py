import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
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
        # RandAugment with num_ops=1 and magnitude=10 as specified in the ICCV paper
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandAugment(num_ops=1, magnitude=10),
            transforms.ToTensor()
        ])

    def apply_strong_augment(self, x):
        """
        Applies strong augmentation (RandAugment) to a batch of images.
        Works across 1-channel and 3-channel image formats.
        """
        if x.dim() == 4:
            # Batch of 2D images: (B, C, H, W)
            augmented = [self.augment(img.cpu()) for img in x]
            return torch.stack(augmented).to(self.device)
        return x

    def compute_confidence(self, logits, metric=None):
        r"""
        Compute a flexible confidence score h(\cdot) for the given logits.
        Metrics supported:
          - 'variance': Variance of logits across classes (default in FedLabel paper).
          - 'entropy': Negative predictive entropy (higher means more confident).
          - 'max_prob': Maximum softmax probability.
        """
        if metric is None:
            metric = getattr(self.config, "confidence_metric", "variance")
            
        if metric == "variance":
            return torch.var(logits, dim=-1)
        elif metric == "entropy":
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return -entropy  # Negated so that higher is more confident
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

    def update(self, global_model):
        """
        Executes the three-stage FedLabel update:
          Stage 1: Supervised Local Update (tau labeled steps)
          Stage 2: Semi-Supervised Update (tau' unlabeled steps with consistency loss)
          Stage 3: Merging Updates: Delta w_k = Delta w_{L,k} + Delta w_{U,k}
        Returns:
          delta_w: Dict of weight updates tensor-by-tensor.
          r_k: Total number of samples used (labeled + threshold-passing unlabeled).
        """
        # Ensure the global model is evaluated on the client device
        global_model = global_model.to(self.device)
        global_model.eval()
        global_state = global_model.state_dict()
        
        # -------------------------------------------------------------
        # Stage 1: Supervised Local Update (Labeled Training)
        # -------------------------------------------------------------
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        
        optimizer_l = optim.SGD(local_model.parameters(), lr=self.config.local_lr)
        criterion = nn.CrossEntropyLoss()
        
        use_epochs = getattr(self.config, "use_epochs", False)
        
        if use_epochs:
            for _ in range(self.config.tau):
                for x, y in self.labeled_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if x.size(0) <= 1:
                        continue
                    optimizer_l.zero_grad()
                    loss = criterion(local_model(x), y)
                    loss.backward()
                    optimizer_l.step()
        else:
            # Paper default: tau local mini-batch SGD iterations
            for _ in range(self.config.tau):
                (x, y), self.iter_labeled = self._get_next_batch(self.labeled_loader, self.iter_labeled)
                x, y = x.to(self.device), y.to(self.device)
                if x.size(0) <= 1:
                    continue
                optimizer_l.zero_grad()
                loss = criterion(local_model(x), y)
                loss.backward()
                optimizer_l.step()
                
        local_state = local_model.state_dict()
        
        # Calculate labeled update difference tensor-by-tensor: Delta w_{L,k} = w_{L,k} - w_global
        delta_w_L = {
            k: (local_state[k].cpu() - global_state[k].cpu())
            if torch.is_floating_point(local_state[k])
            else torch.zeros_like(local_state[k].cpu())
            for k in global_state.keys()
        }
        
        # -------------------------------------------------------------
        # Stage 2: Semi-Supervised Update (Unlabeled Training)
        # -------------------------------------------------------------
        unlabeled_model = copy.deepcopy(global_model).to(self.device)
        unlabeled_model.train()
        
        optimizer_u = optim.SGD(unlabeled_model.parameters(), lr=self.config.local_lr)
        
        total_valid_unlabeled = 0
        
        if self.unlabeled_loader is not None and len(self.unlabeled_loader) > 0:
            if use_epochs:
                unlabeled_steps = []
                for _ in range(self.config.tau_prime):
                    for x_u, _ in self.unlabeled_loader:
                        unlabeled_steps.append(x_u)
            else:
                unlabeled_steps = []
                for _ in range(self.config.tau_prime):
                    (x_u, _), self.iter_unlabeled = self._get_next_batch(self.unlabeled_loader, self.iter_unlabeled)
                    unlabeled_steps.append(x_u)
                    
            for x_u in unlabeled_steps:
                x_u = x_u.to(self.device)
                if x_u.size(0) <= 1:
                    continue
                    
                with torch.no_grad():
                    local_model.eval()
                    # 1) Logits from global model and local model on unlabeled data
                    logits_g = global_model(x_u)
                    logits_l = local_model(x_u)
                    
                    # 2) Confidence calculation: h(s)
                    conf_g = self.compute_confidence(logits_g)
                    conf_l = self.compute_confidence(logits_l)
                    
                    # 3) Model Selection: Identify s* and s^{-*}
                    mask_g_better = conf_g > conf_l
                    
                    selected_logits = torch.where(mask_g_better.unsqueeze(1), logits_g, logits_l)
                    discarded_logits = torch.where(mask_g_better.unsqueeze(1), logits_l, logits_g)
                    
                    selected_conf = torch.where(mask_g_better, conf_g, conf_l)
                    discarded_conf = torch.where(mask_g_better, conf_l, conf_g)
                    
                    # 4) Thresholding: pseudo-label hat{y}
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
                
                # 5) Apply strong augmentation: phi(xi)
                x_u_aug = self.apply_strong_augment(x_u_filtered)
                
                optimizer_u.zero_grad()
                unlabeled_model.train()
                logits_u = unlabeled_model(x_u_aug)
                
                # Cross-Entropy Loss on strongly augmented data
                loss_ce = criterion(logits_u, pseudo_labels_filtered)
                
                # Global-Local Consistency Loss:
                # Check if discarded model agrees with pseudo-label: argmax s^{-*}(xi) == hat{y}_xi
                discarded_preds = torch.argmax(discarded_logits_filtered, dim=-1)
                consistency_mask = (discarded_preds == pseudo_labels_filtered)
                
                loss_kl = 0.0
                if consistency_mask.any():
                    log_probs_u = F.log_softmax(logits_u[consistency_mask], dim=-1)
                    probs_discarded = F.softmax(discarded_logits_filtered[consistency_mask], dim=-1)
                    
                    # Adaptive weight: lambda_k = lambda_0 * (h(s^{-*}) / h(s^*)) <= lambda_0
                    ratio = discarded_conf_filtered[consistency_mask] / (selected_conf_filtered[consistency_mask] + 1e-8)
                    lambda_k = self.config.lambda_0 * torch.clamp(ratio, max=1.0)
                    
                    # KL Divergence: KL( s(w_U), s^{-*} )
                    kl_loss_per_sample = F.kl_div(log_probs_u, probs_discarded, reduction='none').sum(dim=-1)
                    
                    # Normalized across all valid threshold-passing samples in the mini-batch as in Eq. (8)
                    loss_kl = (lambda_k * kl_loss_per_sample).sum() / float(num_passed)
                    
                loss_total = loss_ce + loss_kl
                loss_total.backward()
                optimizer_u.step()
                
        unlabeled_state = unlabeled_model.state_dict()
        
        # Calculate unlabeled update difference tensor-by-tensor: Delta w_{U,k} = w_{U,k} - w_global
        delta_w_U = {
            k: (unlabeled_state[k].cpu() - global_state[k].cpu())
            if torch.is_floating_point(unlabeled_state[k])
            else torch.zeros_like(unlabeled_state[k].cpu())
            for k in global_state.keys()
        }
        
        # -------------------------------------------------------------
        # Stage 3: Merging Updates
        # -------------------------------------------------------------
        # Delta w_k = Delta w_{L,k} + Delta w_{U,k}
        delta_w = {k: delta_w_L[k] + delta_w_U[k] for k in global_state.keys()}
        
        # Total number of samples used: labeled dataset + unlabeled samples passing threshold
        r_k = len(self.labeled_loader.dataset) + total_valid_unlabeled
        if r_k <= 0:
            r_k = 1
            
        return delta_w, r_k
