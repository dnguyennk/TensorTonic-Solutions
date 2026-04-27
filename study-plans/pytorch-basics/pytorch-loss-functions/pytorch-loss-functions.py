import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred_float = torch.tensor(pred, dtype = torch.float32)
    target_float = torch.tensor(target, dtype = torch.float32)
    target_long = torch.tensor(target, dtype = torch.long)
    if method == 'mse':
        return ((pred_float - target_float)**2).mean().item()
    elif method == 'cross_entropy':
        # pred shape : (N, C) logits ///////// target shape : (N, ) class indices
        N = pred_float.shape[0]
        # Step 1: log-sum-exp trick
        z_max = pred_float.max(dim = 1, keepdim = True).values # (N, 1)
        z_shifted = pred_float - z_max # (N, C) max = 0
        log_sum_exp = z_max.squeeze(1) + torch.log(torch.exp(z_shifted).sum(dim=1)) #(N,)
        #Step 2: get logit @ target
        target_logits = pred_float.gather(1, target_long.unsqueeze(1)).squeeze(1) # = pred_float[torch.arange(N), target_long]
        # Step 3: per-sample NLL = log_sum_exp − z_target 
        per_sample_loss = log_sum_exp - target_logits
        return per_sample_loss.mean().item()
    elif method == 'huber':
        diff = pred_float - target_float  # residual a = pred - target
        abs_diff = diff.abs() # |a|
        quadratic = 0.5 * diff**2 # ½a² 
        linear = delta*(abs_diff - 0.5*delta) # δ(|a| − δ/2)  
        # Select which branch based on |a| ≤ δ
        loss = torch.where(abs_diff <= delta, quadratic, linear)
        return loss.mean().item()
        
        