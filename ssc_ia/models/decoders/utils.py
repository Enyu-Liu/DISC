import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean
from ..utils import vox2pix

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


def _down_sample2D(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

def _up_sample2D(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


def generate_grid_bev(L, W, interval):
    l_indices = torch.arange(0, L, step=interval)
    w_indices = torch.arange(0, W, step=interval)

    grid_l, grid_w = torch.meshgrid(l_indices, w_indices, indexing='ij')  
    sampled_coords = torch.stack([grid_l.flatten(), grid_w.flatten()], dim=-1)
    return sampled_coords


def select_ins_points(mask, height_prob, params, scene_shape, top_k=500, kernel_size=2, num_points=3, thres=0.25):
    """Select reference points for potential instance in bev plane
    NOTE: Only support bs==1 for now.
    Args:
        mask: mask in (L, W) plane in shape of (bs, L, W)
        height_prob: height prob of each bev points in shape of (bs, L, W, H)
        num_points: num of selected points in z axis
    Return:
        ref3ds: (N*num_points, 3)
        ref2ds: (M, 2) in image plane Only keep points can be project into front image
        mask2ds: (N*num_points)
        hs_w : (N*4, 1) height probs for selected points
    """
    L, W, H = height_prob.shape[1:]
    num_points = num_points
    height_prob = height_prob.squeeze(0) #(L, W, H)
    mask = mask.squeeze(0) #(L, W)
    h_weights, h_indices = torch.topk(height_prob, k=num_points, dim=2)  #(L, W, num_points)

    # Devide bev plane into multiple grids
    x_blocks = L // kernel_size
    y_blocks = W // kernel_size
    bev_grids = mask.view(x_blocks, kernel_size, y_blocks, kernel_size)
    bev_grids = bev_grids.permute(0, 2, 1, 3).contiguous().view(-1, kernel_size**2) #(num_grid, K**2)

    # Get query coors using local grids
    max_vals, max_idx = torch.max(bev_grids, dim=-1)
    block_coords = torch.arange(0, len(max_vals)).to(max_idx)
    x_block = block_coords // y_blocks
    y_block = block_coords % y_blocks
    x_offsets = max_idx // kernel_size
    y_offsets = max_idx % kernel_size
    selected_coor = torch.stack([
        x_block * kernel_size + x_offsets, 
        y_block * kernel_size + y_offsets,
    ], dim=-1)

    top_vals, top_idx = torch.topk(max_vals, top_k)
    point_mask = top_vals > thres
    point_mask = point_mask if point_mask.sum() > 100 else top_vals > 0.
    top_idx = top_idx[point_mask]
    top_vals = top_vals[point_mask]
    ref3ds = selected_coor[top_idx] #(N, 2)

    hs = h_indices[ref3ds[:, 0], ref3ds[:, 1], :].unsqueeze(-1).reshape(-1, 1) #(N*4, 1)
    hs_w = h_weights[ref3ds[:, 0], ref3ds[:, 1], :].unsqueeze(-1).reshape(-1, 1) #(N*4, 1)
    ref3ds = ref3ds.unsqueeze(1).repeat(1, num_points, 1).reshape(-1, 2) #(N*4, 2)
    ref3ds = torch.cat([ref3ds, hs], dim=-1) #(N*4, 3)

    # NOTE : Add random distrub
    disturb = True
    if disturb:
        disturbance = torch.rand(size=(ref3ds.shape)) - 0.5  # Voxel coordinate (-0.5, 0.5)
        ref3ds_dis = ref3ds +  disturbance.to(ref3ds.device)

    ref3ds_norm = (ref3ds_dis / (torch.tensor(scene_shape) - 1).to(ref3ds)).unsqueeze(0) #(bs, N*4, 3)
    ref2ds = vox2pix(ref3ds_norm, *params).squeeze(0) #(N*4, 2)  # NOTE: To be focused on [h, w]
    mask2ds = ((ref2ds[:, 0] >= 0) & (ref2ds[:, 0] < 1) & 
                (ref2ds[:, 1] >= 0) & (ref2ds[:, 1] < 1))   # (N*4)

    ref2ds = ref2ds[mask2ds] #(M, 2)        
    return ref3ds, ref2ds, mask2ds, hs_w, top_vals

def ins_mask_projected(seg_preds, vol_pts, scene_shape):
    """Project pixel to bev plane and get bev mask for instance 
    Args:
        * seg_preds: (bs, 1, h, w) with sigmoid
        * vol_pts: (bs, h*w, 3) each pixel's coor in 3D scene plane
        * scene_shape: [L, W, H]
    Return:
        * bev_ins_mask in shape of (bs, L, W)
    """
    bs = seg_preds.shape[0]
    L, W, _ = scene_shape
    assert bs == 1 
    seg_preds = seg_preds.squeeze().flatten() #(h*w)
    bev_coor = vol_pts.squeeze(0)[:, :2].long() #(h*w, 2)
    inside_mask = ((bev_coor[:, 0] >= 0) & (bev_coor[:, 0] < L) & 
                   (bev_coor[:, 1] >= 0) & (bev_coor[:, 1] < W))
    bev_coor = bev_coor[inside_mask] #(N, 2)
    seg_preds = seg_preds[inside_mask] #(N, )

    bev_coor, indices = torch.unique(bev_coor, dim=0, return_inverse=True) #(M, 2)
    # seg_preds_tmp, _= scatter_max(seg_preds, indices, dim=0) #(M, )
    seg_preds_tmp = scatter_mean(seg_preds, indices, dim=0)
    seg_preds = seg_preds_tmp

    bev_ins_mask_tmp = torch.zeros((L, W)).to(seg_preds)
    bev_ins_mask_tmp[bev_coor[:, 0], bev_coor[:, 1]] = seg_preds
    bev_ins_mask = bev_ins_mask_tmp

    return bev_ins_mask.unsqueeze(0) # (bs, L, W)