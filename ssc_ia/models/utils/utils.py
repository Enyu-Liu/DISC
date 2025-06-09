from functools import reduce

import torch
import torch.nn.functional as F


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)

def index_batch_element_with_mask(x, mask):
    """ Index batch data with relative mask

    Args:
        x : (bs, n, ...)
        mask : (bs, n) torch.bool
    Return:
        x' : (bs, n', ...) where n' is equal to the num of 'True' in mask
    """
    assert x.shape[1] == mask.shape[1]
    indices = mask.nonzero()
    pass

def flatten_fov_from_voxels(x3d, fov_mask):
    # assert x3d.shape[0] == 1
    # if fov_mask.dim() == 2:
    #     assert fov_mask.shape[0] == 1
    #     fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask[0]].transpose(1, 2)

def flatten_non_empty_from_voxels(x3d, nonempty_mask):
    bs = x3d.size(0)
    indices = torch.stack([torch.nonzero(mask) for mask in nonempty_mask], dim=0).squeeze()
    indices = indices.unsqueeze(0) if bs==1 else indices
    batch_indices = torch.arange(bs, dtype=torch.long, device=nonempty_mask.device).\
        view(-1, 1).expand(-1, indices.size(1))
    return x3d.flatten(2).transpose(1, 2)[batch_indices, indices, :]

def index_fov_back_to_voxels(x3d, fov, fov_mask):
    # assert x3d.shape[0] == fov.shape[0] == 1
    if fov_mask.dim() == 2:
        # assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
        fov_mask = fov_mask[0] # TODO: only deal with bs=1
    fov_concat = torch.zeros_like(x3d).flatten(2)
    fov_concat[..., fov_mask] = fov.transpose(1, 2)
    return torch.where(fov_mask, fov_concat, x3d.flatten(2)).reshape(*x3d.shape)

def index_nonempty_back_to_voxels(x3d, nonempty, nonempty_mask):
    feature_list = []
    for bs in range(x3d.size(0)):
        feature_list.append(index_fov_back_to_voxels(x3d[bs].unsqueeze(0), 
                                                     nonempty[bs].unsqueeze(0),
                                                     nonempty_mask[bs]))
    return torch.cat(feature_list, dim=0)


def flatten_index_back_to_original_space(indices, scene_shape):
    """
    NOTE: Flattned index computed by :
        index = x(w * h) + y * h + z 
    """
    l, w, h = scene_shape
    x = indices // (w * h)
    y = (indices % (w * h)) // h
    z = indices % h 
    return torch.stack((x, y, z), dim=1) # (num_indice, 3) voxel_pts

def interpolate_flatten(x, src_shape, dst_shape, mode='nearest'):
    """Inputs & returns shape as [bs, n, (c)]
    """
    if len(x.shape) == 3:
        bs, n, c = x.shape
        x = x.transpose(1, 2)
    elif len(x.shape) == 2:
        bs, n, c = *x.shape, 1
    assert cumprod(src_shape) == n
    x = F.interpolate(
        x.reshape(bs, c, *src_shape).float(), dst_shape, mode=mode,
        align_corners=False).flatten(2).transpose(1, 2).to(x.dtype)
    if c == 1:
        x = x.squeeze(2)
    return x


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    # h * w
    areas = shapes[:, 0] * shapes[:, 1]
    
    # Add zero at the list beginnig
    start_indices = torch.cat((shapes.new_zeros((1,)), areas.cumsum(0)[:-1]))
    
    return start_indices


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    assert L == cumprod(shape), 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def pix2cam(p_pix, depth, K, post_rot, post_tran):
    """
    Args:
        p_pix: (bs, 2, H, W)
        post_rot : (bs, 3, 3)
        post_tran : (bs, 3)
    """
    H, W = p_pix.shape[-2:]
    p_pix = p_pix.flatten(2).transpose(1, 2) - post_tran[:, :2].unsqueeze(1) #(b, N, 2)
    p_pix = post_rot[:, :2, :2].inverse().unsqueeze(1) @ p_pix.unsqueeze(-1) #(b, 1, 2, 2) @ (b, N, 2, 1)
    p_pix = p_pix.squeeze(-1).transpose(1, 2).reshape(-1, 2, H, W) #(b, 2, H, W)

    p_pix = torch.cat([p_pix * depth, depth], dim=1)  # bs, 3, h, w
    return K.inverse() @ p_pix.flatten(2)


def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(1)) / vox_size - offset
    return p_vox


def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, post_rot, post_tran, offset=0.5, downsample_z=1):
    p_cam = pix2cam(p_pix, depth, K, post_rot, post_tran)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox


def cam2pix(p_cam, K, image_shape, post_rots, post_trans):
    """
    Args:
        p_cam : (bs, N, 3, 1)
    Return:
        p_pix: (bs, N, 2)
    """
    p_pix = (K @ p_cam).squeeze(-1)  # .clamp(min=1e-3) (bs, N, 3)
    p_pix = p_pix / p_pix[..., 2].unsqueeze(-1) #(bs, N, 3)
    p_pix = (post_rots[:, :2, :2] @ p_pix[..., :2].unsqueeze(-1)).squeeze(-1) + post_trans[..., :2]
    p_pix = p_pix / (torch.tensor(image_shape[::-1]).to(p_pix) - 1) #(bs, N, 2)
    return p_pix


def vox2pix(p_vox, K, E, vox_origin, vox_size, image_shape, scene_shape, post_rots, post_trans, offset=0.5): 
    """
    Args:
        p_vox : 3D normalized coors (bs, N, 3)
    """
    p_vox = p_vox * torch.tensor(scene_shape).to(p_vox) * vox_size + vox_origin + offset*vox_size # (bs, N, 3)
    ones = torch.ones(p_vox.shape[0], p_vox.shape[1], 1, 1).to(p_vox) #(bs, N, 1, 1)
    p_cam = (E @ torch.cat([p_vox.unsqueeze(-1), ones], dim=2)) #(bs, N, 4, 1)
    return cam2pix(p_cam[..., :-1, :], K, image_shape, post_rots, post_trans).clamp(0, 1)


def volume_rendering(
        volume,
        image_grid,
        K,
        E,
        vox_origin,
        vox_size,
        image_shape,
        depth_args=(2, 50, 1),
):
    depth = torch.arange(*depth_args).to(image_grid)  # (D,)
    p_pix = F.pad(image_grid, (0, 0, 0, 0, 0, 1), value=1)  # (B, 3, H, W)
    p_pix = p_pix.unsqueeze(-1) * depth.reshape(1, 1, 1, 1, -1)

    p_cam = K.inverse() @ p_pix.flatten(2)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size)
    p_vox = p_vox.reshape(1, *image_shape, depth.size(0), -1)  # (B, H, W, D, 3)
    p_vox = p_vox / (torch.tensor(volume.shape[-3:]) - 1).to(p_vox)

    return F.grid_sample(volume, torch.flip(p_vox, dims=[-1]) * 2 - 1, padding_mode='zeros'), depth


def render_depth(volume, image_grid, K, E, vox_origin, vox_size, image_shape, depth_args):
    sigmas, z = volume_rendering(volume, image_grid, K, E, vox_origin, vox_size, image_shape,
                                 depth_args)
    beta = z[1] - z[0]
    T = torch.exp(-torch.cumsum(F.pad(sigmas[..., :-1], (1, 0)) * beta, dim=-1))
    alpha = 1 - torch.exp(-sigmas * beta)
    depth_map = torch.sum(T * alpha * z, dim=-1).reshape(1, *image_shape)
    depth_map = depth_map  # + d[..., 0]
    return depth_map


def inverse_warp(img, image_grid, depth, pose, K, padding_mode='zeros'):
    """
    img: (B, 3, H, W)
    image_grid: (B, 2, H, W)
    depth: (B, H, W)
    pose: (B, 3, 4)
    """
    p_cam = pix2cam(image_grid, depth.unsqueeze(1), K)
    p_cam = (pose @ F.pad(p_cam, (0, 0, 0, 1), value=1))[:, :3]
    p_pix = cam2pix(p_cam, K, img.shape[2:])
    p_pix = p_pix.reshape(*depth.shape, 2) * 2 - 1
    projected_img = F.grid_sample(img, p_pix, padding_mode=padding_mode)
    valid_mask = p_pix.abs().max(dim=-1)[0] <= 1
    return projected_img, valid_mask


def project_3d_points2_2d(points, view):
    """Project 3D points to 2D plane with specific view
    Args:
        points : 3d scene points in shape (bs, num, 1, 3)
        view : str included in ['top', 'side', 'front']
    Return:
        points_2d in shape (bs, num, 1, 2)
    """
    assert points.shape[-1]==3
    assert view in ['top', 'side', 'front']
    if view == 'top':
        points_2d = points[..., :-1]
    elif view == 'side':
        x_coors = points[..., 0]
        z_coors = points[..., -1]
        points_2d = torch.stack((x_coors, z_coors), dim=-1)
    else:
        points_2d = points[..., 1:]
    return points_2d