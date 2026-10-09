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
        
        # Data Augmentation Module
        # Uses RandAugment with num_ops=1 and magnitude=10 as specified in the paper
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandAugment(num_ops=1, magnitude=10),
            transforms.ToTensor()
        ])

    def compute_confidence(self, logits):
        """
        Compute a confidence score h(\cdot) for the given logits.
        The paper typically uses the variance of the logits as the uncertainty/confidence score.
        """
        return torch.var(logits, dim=-1)

    def update(self, global_model):
        """
        Performs local client update (labeled and unlabeled training) based on FedLabel algorithm.
        Returns the delta weight updates and the total number of samples used (r_k).
        """
        # Ensure the global model evaluates on the correct device for the client
        global_model = global_model.to(self.device)
        global_model.eval()
        
        # -------------------------------------------------------------
        # Stage 1: Supervised Local Update (Labeled Training)
        # -------------------------------------------------------------
        # Initialize local_model weights with global_model weights using deepcopy
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        
        optimizer_l = optim.SGD(local_model.parameters(), lr=self.config.local_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train for tau epochs on the labeled data
        for epoch in range(self.config.tau):
            for x, y in self.labeled_loader:
                x, y = x.to(self.device), y.to(self.device)
                if x.size(0) <= 1:
                    continue
                optimizer_l.zero_grad()
                logits = local_model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer_l.step()
                
        global_state = global_model.state_dict()
        local_state = local_model.state_dict()
        
        # Calculate labeled update difference (Delta w_L,k)
        delta_w_L = {k: local_state[k].cpu() - global_state[k].cpu() for k in global_state.keys()}
        
        # -------------------------------------------------------------
        # Stage 2: Semi-Supervised Update (Unlabeled Training)
        # -------------------------------------------------------------
        # Initialize unlabeled_model weights with global_model weights using deepcopy
        unlabeled_model = copy.deepcopy(global_model).to(self.device)
        unlabeled_model.train()
        
        optimizer_u = optim.SGD(unlabeled_model.parameters(), lr=self.config.local_lr)
        
        valid_unlabeled_samples = 0
        
        if self.unlabeled_loader is not None:
            # Train for tau' epochs on the unlabeled data
            for epoch in range(self.config.tau_prime):
                epoch_valid_samples = 0
                for x_u, _ in self.unlabeled_loader:
                    x_u = x_u.to(self.device)
                    if x_u.size(0) <= 1:
                        continue
                        
                    with torch.no_grad():
                        local_model.eval()
                        # Compute logits using both global and local models
                        logits_g = global_model(x_u)
                        logits_l = local_model(x_u)
                        
                        conf_g = self.compute_confidence(logits_g)
                        conf_l = self.compute_confidence(logits_l)
                        
                        # Model Selection: Identify the model with the higher confidence score
                        mask_g_better = conf_g > conf_l
                        
                        selected_logits = torch.where(mask_g_better.unsqueeze(1), logits_g, logits_l)
                        discarded_logits = torch.where(mask_g_better.unsqueeze(1), logits_l, logits_g)
                        
                        selected_conf = torch.where(mask_g_better, conf_g, conf_l)
                        discarded_conf = torch.where(mask_g_better, conf_l, conf_g)
                        
                        probs_s = torch.softmax(selected_logits, dim=-1)
                        max_probs, pseudo_labels = torch.max(probs_s, dim=-1)
                        
                        # Thresholding
                        mask_threshold = max_probs > self.config.beta
                        
                    if not mask_threshold.any():
                        continue
                        
                    x_u_filtered = x_u[mask_threshold]
                    pseudo_labels_filtered = pseudo_labels[mask_threshold]
                    discarded_logits_filtered = discarded_logits[mask_threshold]
                    selected_conf_filtered = selected_conf[mask_threshold]
                    discarded_conf_filtered = discarded_conf[mask_threshold]
                    
                    epoch_valid_samples += x_u_filtered.size(0)
                    
                    # Apply RandAugment to the filtered unlabeled data
                    x_u_aug = torch.stack([self.augment(img.cpu()) for img in x_u_filtered]).to(self.device)
                    
                    optimizer_u.zero_grad()
                    unlabeled_model.train()
                    logits_u = unlabeled_model(x_u_aug)
                    
                    # CrossEntropy Loss against the pseudo-label
                    loss_ce = criterion(logits_u, pseudo_labels_filtered)
                    
                    # Global-Local Consistency Loss
                    discarded_preds = torch.argmax(discarded_logits_filtered, dim=-1)
                    consistency_mask = (discarded_preds == pseudo_labels_filtered)
                    
                    loss_kl = 0.0
                    if consistency_mask.any():
                        log_probs_u = F.log_softmax(logits_u[consistency_mask], dim=-1)
                        probs_discarded = F.softmax(discarded_logits_filtered[consistency_mask], dim=-1)
                        
                        # Adaptive weight
                        lambda_k = self.config.lambda_0 * (discarded_conf_filtered[consistency_mask] / (selected_conf_filtered[consistency_mask] + 1e-8))
                        
                        kl_loss_per_sample = F.kl_div(log_probs_u, probs_discarded, reduction='none').sum(dim=-1)
                        loss_kl = (lambda_k * kl_loss_per_sample).mean()
                        
                    loss_total = loss_ce + loss_kl
                    loss_total.backward()
                    optimizer_u.step()
                    
                if epoch == self.config.tau_prime - 1:
                    valid_unlabeled_samples = epoch_valid_samples
                    
        unlabeled_state = unlabeled_model.state_dict()
        # Calculate unlabeled update difference (Delta w_U,k)
        delta_w_U = {k: unlabeled_state[k].cpu() - global_state[k].cpu() for k in global_state.keys()}
        
        # -------------------------------------------------------------
        # Stage 3: Merging Updates
        # -------------------------------------------------------------
        # Sum both differences
        delta_w = {k: delta_w_L[k] + delta_w_U[k] for k in global_state.keys()}
        
        # Total number of samples used
        r_k = len(self.labeled_loader.dataset) + valid_unlabeled_samples
        
        return delta_w, r_k
