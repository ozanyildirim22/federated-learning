import torch
import random


class Server:
    def __init__(self, global_model, global_lr=1.0):
        self.global_model = global_model
        self.global_lr = global_lr

    def select_clients(self, clients, m):
        """
        Randomly selects m clients from the total pool of clients C^(t,0).
        """
        m = min(m, len(clients))
        return random.sample(clients, m)

    def aggregate(self, client_updates):
        """
        Aggregates the Delta w_k (weight updates) received from clients by computing a weighted average
        based on the number of samples used by each client (r_k) (Eq. 9):
        w^(t+1) = w^(t) + global_lr * sum ( (r_k / sum(r_k')) * Delta w_k )
        """
        total_samples = sum([r_k for _, r_k in client_updates])
        if total_samples <= 0:
            return

        global_state = self.global_model.state_dict()
        param_keys = {k for k, _ in self.global_model.named_parameters()}

        aggregated_update = {
            key: torch.zeros_like(val)
            for key, val in global_state.items()
            if torch.is_floating_point(val)
        }

        for delta_w, r_k in client_updates:
            weight = float(r_k) / float(total_samples)
            for key in aggregated_update.keys():
                if key in delta_w and torch.is_floating_point(delta_w[key]):
                    aggregated_update[key] += delta_w[key].to(aggregated_update[key].device) * weight

        for key in aggregated_update.keys():
            if key in param_keys:
                global_state[key] += self.global_lr * aggregated_update[key]
            else:
                global_state[key] += aggregated_update[key]

        self.global_model.load_state_dict(global_state)