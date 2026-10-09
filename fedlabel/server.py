import torch
import random

class Server:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def select_clients(self, clients, m):
        """
        Randomly selects m clients from the total pool.
        """
        return random.sample(clients, m)
        
    def aggregate(self, client_updates):
        """
        Aggregates the Delta w_k (weight updates) received from clients by computing a weighted average
        based on the number of samples used by each client (r_k).
        """
        total_samples = sum([r_k for _, r_k in client_updates])
        if total_samples == 0:
            return
            
        # Initialize an empty state_dict for aggregated updates
        # Using the same device as the global model for the zeros
        aggregated_update = {
            key: torch.zeros_like(val) 
            for key, val in self.global_model.state_dict().items()
        }
        
        # Weighted average based on r_k / sum(r_k')
        for delta_w, r_k in client_updates:
            weight = r_k / total_samples
            for key in aggregated_update.keys():
                # delta_w tensors are stored on CPU to save GPU memory; move them to the correct device here
                aggregated_update[key] += delta_w[key].to(aggregated_update[key].device) * weight
                
        # Apply the update to the global model
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] += aggregated_update[key]
            
        self.global_model.load_state_dict(global_state)
