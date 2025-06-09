import torch
import torch.nn.functional as F
from ..utils import generate_guassian_depth_target

def get_klv_depth_loss(pred, target):
    # params
    downsample = 8
    cam_depth_range = [2.0, 58.0, 0.5] # depth bound
    constant_std = 0.5
    ds = torch.arange(*cam_depth_range, dtype=torch.float).view(-1, 1, 1)
    D, _, _ = ds.shape

    depth_labels = target['depth'].unsqueeze(1) #(bs, N, H, W)
    depth_preds = pred['depth'] #(bs, D, H//8, W//8)

    depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
        downsample, cam_depth_range, constant_std=constant_std)
    
    depth_values = depth_values.view(-1)
    fg_mask = (depth_values >= cam_depth_range[0]) & (depth_values <= (cam_depth_range[1] - cam_depth_range[2]))        
    
    depth_gaussian_labels = depth_gaussian_labels.view(-1, D)[fg_mask]
    depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, D)[fg_mask]
    
    depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
    
    return depth_loss * 0.01 # depth weight