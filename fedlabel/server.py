import torch
import random


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def select_clients(self, clients, m):
        """
        Randomly selects m clients from the total pool of clients C^(t,0).
        """
        m = min(m, len(clients))
        return random.sample(clients, m)

    def aggregate(self, client_updates):
        """
        Aggregates the Delta w_k (weight updates) received from clients by computing a weighted average
        based on the number of samples used by each client (r_k):
        w^(t+1) = w^(t) + sum ( (r_k / sum(r_k')) * Delta w_k )
        """
        total_samples = sum([r_k for _, r_k in client_updates])
        if total_samples <= 0:
            return

        global_state = self.global_model.state_dict()

        aggregated_update = {
            key: torch.zeros_like(val)
            for key, val in global_state.items()
            if torch.is_floating_point(val)
        }

        for delta_w, r_k in client_updates:
            weight = float(r_k) / float(total_samples)
            for key in aggregated_update.keys():
                if key in delta_w and torch.is_floating_point(delta_w[key]):
                    upd = delta_w[key].to(aggregated_update[key].device)
                    if torch.isfinite(upd).all():
                        aggregated_update[key] += upd * weight

        for key in aggregated_update.keys():
            global_state[key] += aggregated_update[key]
            # Guarantee BatchNorm running_var is strictly positive so sqrt(running_var) never NaNs
            if "running_var" in key:
                global_state[key].clamp_(min=1e-5)

        self.global_model.load_state_dict(global_state)