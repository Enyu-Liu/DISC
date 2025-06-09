import os

import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import label

from ..models.utils.utils import pix2vox, generate_grid

COLORS = np.array([
    [255, 255, 255, 255], #custom add for empty pixel
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255], #person
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255], # road
    [255, 150, 255, 255], # parking
    [75, 0, 75, 255], # sidewalk
    [175, 0, 75, 255], # other-ground
    [255, 200, 0, 255], #building
    [255, 120, 50, 255],
    [0, 175, 0, 255], #vegetation
    [135, 60, 0, 255],
    [150, 240, 80, 255], #terrain
    [255, 240, 150, 255],
    [255, 0, 0, 255],
    [100, 100, 50, 255], # custom add for out-of-scene pixels
]).astype(np.uint8)



def project_voxels_to_image(data_input, target, save=False):
    """Project voxel gt to front image in order to get semantic seg gt in front image
    
    Args:
        data_input: Original input data
        target: Original target data
        save: Save visiable seg image
    Return:
        semantic_target: torch.tensor in shape of (b, *image_shape)
    """
    bk_indices = target['bk_cls_info']['indices'][1:]
    scale = data_input['scale_3ds'][0]
    projected_pix_coor = data_input[f'projected_pix_{scale}'].squeeze() #(n_voxels, 2)
    pix_z_coor = data_input[f'pix_z_{scale}'].squeeze() #(n_voxels)
    fov_mask = data_input[f'fov_mask_{scale}'].squeeze() #(n_voxels)
    image_shape = data_input['img'].shape[-2:] # (img_h, img_w)
    voxel_data = target['target'].squeeze().reshape(-1) #(vl, vw, vh)

    # init semantic image
    semantic_image = np.zeros(image_shape, dtype=np.int32)
    depth_image = np.zeros(image_shape, dtype=np.float32)

    # filter voxels outside image
    pixel_x = projected_pix_coor[fov_mask][:, 0] # n_voxels_fov
    pixel_y = projected_pix_coor[fov_mask][:, 1] # n_voxels_fov
    depth_values = pix_z_coor[fov_mask] # n_voxels_fov
    indices = torch.nonzero(fov_mask, as_tuple=False).squeeze()
    
    # Iterate over each voxel
    for i in range(len(pixel_x)):
        x, y = pixel_x[i], pixel_y[i]
        vol_indic = indices[i]
        if voxel_data[vol_indic] != 0 and voxel_data[vol_indic] != 255:  # non-empty
            if semantic_image[y, x] == 0 or depth_values[i] < depth_image[y, x]:  # use depth to choose ultimate pixel label
                semantic_image[y, x] = voxel_data[vol_indic]
                depth_image[y, x] = depth_values[i]
    
    if save:
        save_visable_results(semantic_image, data_input)
    # get bk semantic map
    semantic_image = torch.from_numpy(semantic_image)
    return torch.cat([(semantic_image==i).to(torch.int32).unsqueeze(0) for i in bk_indices], dim=0)

def project_image_to_voxels(data_input, target, image_scale=1, mode='ins', save=False):
    """Project image to voxels to get finer-grained segmentation gt
    
    Args:
        data_input: Original input data
        target: Original target data
        mode: 'ins' for instance seg map, 'bk' for bksetg map.
        save: Save visiable seg image
    Return:
        semantic_target: torch.tensor in shape of (b, *image_shape)
    """
    ins_indices = target['ins_cls_info']['indices'][1:]
    bk_indices = target['bk_cls_info']['indices'][1:]
    num_cls = len(ins_indices) + len(bk_indices) + 1
    # get useful data information from dataloader
    image_shape = data_input['img'].shape[-2:] # (img_h, img_w)
    depth = data_input['depth'].unsqueeze(0)# (bs, img_h, img_w)
    cam_K = data_input['cam_K'].unsqueeze(0) #(bs, 3, 3)
    cam_E = data_input['cam_pose'].unsqueeze(0) # (bs, 4, 4)
    voxel_origin = data_input['voxel_origin'].unsqueeze(0) #(bs, 3)
    voxel_data = target['target'].squeeze() #(vl, vw, vh)
    voxel_scale = data_input['scale_3ds'][0]

    # Generate intermediate variables
    image_grid = generate_grid(image_shape)
    image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # (1, 2, img_h, img_w)      

    #NOTE: About 20% of the pixels are outside the 3D scene range
    vol_pts = pix2vox(
        image_grid*image_scale,
        depth.unsqueeze(1),
        cam_K,
        cam_E,
        voxel_origin,
        post_rot=data_input['post_rot'].unsqueeze(0), # (1, 3, 3)
        post_tran=data_input['post_tran'].unsqueeze(0), #(1, 3)
        vox_size=0.2*voxel_scale,  # voxel size
        downsample_z=1).squeeze().long()     # (370*1220, 3)

    # Flatten voxel_data for easier indexing
    extended_voxel_data = torch.cat((voxel_data.flatten(), torch.tensor([num_cls])), 0)
    dim1, dim2, dim3 = voxel_data.shape
    # Convert 3D indices to 1D linear indices
    linear_indices = vol_pts[..., 0] * dim2 * dim3 + vol_pts[..., 1] * dim3 + vol_pts[..., 2]
    linear_indices = torch.where(
        (vol_pts[..., 0] >= dim1) | (vol_pts[..., 1] >= dim2) | (vol_pts[..., 2] >= dim3),
        torch.tensor(extended_voxel_data.size(0) - 1), 
        linear_indices
    )
    # Map linear indices to class labels using extended voxel data
    semantic_map_flat = extended_voxel_data[linear_indices]

    # Reshape to get the semantic segmentation labels map
    semantic_image = semantic_map_flat.view(image_shape[0], image_shape[1]).long().numpy()
    semantic_image[semantic_image == 255] = num_cls

    # generate seg gt of specific clses
    # semantic_image = torch.from_numpy(semantic_image)
    indices = np.concatenate((ins_indices, [num_cls])) if mode=='ins' else np.concatenate((bk_indices, [num_cls]))
    mask = np.isin(semantic_image, indices)
    semantic_image_renew = np.where(mask, semantic_image, 0)
    
    if save:
        save_visable_results(semantic_image, data_input, specfic_id='complete')
        save_visable_results(semantic_image_renew, data_input, specfic_id='partial')
    semantic_image_renew[semantic_image == num_cls] = 255  # NOTE: KEEP 255
    return semantic_image_renew
    # return torch.cat([(semantic_image==i).to(torch.int32).unsqueeze(0) for i in bk_indices], dim=0)


def instance_bev_seg_map(data_input, target, save=False):
    """Generate instance seg map for assign loss
    """
    ins_indices = target['ins_cls_info']['indices'][1:]
    bk_indices = target['bk_cls_info']['indices'][1:]

    # bev_remap = project_voxels_to_single_bev(data_input, target, save=save, mode='ins')
    seg_maps = project_voxels_to_bev(data_input, target, save=save, mode='ins').numpy() #(H, W)
    instance_list = []
    
    # Iterate over all classes except the background (0)
    for class_id, class_mask in zip(ins_indices, seg_maps):
        map_id = ins_indices.index(class_id) + 1 
        # # Create binary mask for the current class
        # class_mask = (seg_map == class_id)
        
        # If there are no pixels for this class, continue to the next class
        if np.sum(class_mask) == 0:
            continue
        
        # Label connected components in the binary mask
        labeled_mask, num_instances = label(class_mask)
        
        # Create instance masks and store them in the output list
        for instance_id in range(1, num_instances + 1):
            instance_mask = (labeled_mask == instance_id)
            instance_list.append({
                'label': map_id,
                'mask': instance_mask
            })
    
    if save:
        for i, ins in enumerate(instance_list):
            ins_label = ins_indices[ins['label'] - 1]
            mask = ins['mask'] * ins_label
            specfic_id = f'ins_{i}_{ins_label}'
            save_visable_results(mask, data_input, specfic_id=specfic_id)

    return instance_list


def project_voxels_to_bev(data_input, target, save=False, mode='bk'):
    """
    Generate BEV semantic segmentation masks for each class from the 3D semantic segmentation target
    NOTE: This function does not consider the occlusion caused by the viewing angle from top to bottom.
    """
    ins_indices = target['ins_cls_info']['indices'][1:]
    bk_indices = target['bk_cls_info']['indices'][1:]
    ins_names = target['ins_cls_info']['names'][1:]
    bk_names = target['bk_cls_info']['names'][1:]

    indices = bk_indices if mode=='bk' else ins_indices
    names = bk_names if mode=='bk' else ins_names
    voxel_data = target['target'] #(vl, vw, vh)
    masks = [torch.max(voxel_data==i, axis=2).values.to(torch.int32).unsqueeze(0) for i in indices]
    if save:
        for i, mask in enumerate(masks):
            mask = torch.where(mask.squeeze().bool(), indices[i], 0).numpy()
            if mask.sum() != 0:
                save_visable_results(mask, data_input, specfic_id=names[i])
    return torch.cat(masks, dim=0)


def project_voxels_to_single_bev(data_input, target, save=False, mode='bk'):
    ins_indices = target['ins_cls_info']['indices'][1:]
    bk_indices = target['bk_cls_info']['indices'][1:]
    voxel_data = target['target'] # (256, 256, 32)
    L, W, H = voxel_data.shape
    indices = bk_indices if mode=='bk' else ins_indices
    # Initialize the BEV map
    bev_map = np.zeros((L, W), dtype=np.uint8)
    for h in range(0, H-1):
        # Extract the current layer
        layer = voxel_data[:, :, h]
        mask = np.isin(layer, indices)
        
        # Update the BEV map where the mask is true (considering occlusion)
        bev_map[mask] = layer[mask]
    
    if save:
        save_visable_results(bev_map, data_input, specfic_id='bev')
    
    # Remap gt labels
    bk_indices_ = np.concatenate(([0], indices))   # bug here
    indice_remap = {y:x for (x, y) in enumerate(bk_indices_)}
    bev_remap = np.copy(bev_map)
    for old_label, new_label in indice_remap.items():
        bev_remap[bev_map == old_label] = new_label
    return bev_remap

def save_visable_results(semantic_image, data_input, data_root='./vis/seg_image/', specfic_id=None):
    """Visuliz segmentation gt image
    Args:
        semantic_image : semantic image label in numpy format
    """
    frame_id = data_input['frame_id']
    sequence = data_input['sequence'] 
    segimg_name = frame_id + '.png' if specfic_id==None else frame_id + '_' + str(specfic_id) + '.png'
    img_path = os.path.join(data_root, *[sequence, segimg_name])
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    color_image = np.zeros((semantic_image.shape[0], semantic_image.shape[1], 4), dtype=np.uint8)  # RGBA format
    color_image = COLORS[semantic_image]
    # convert array to PIL image
    # color_image = (resize(color_image, (47, 153, 4), anti_aliasing=True)* 255).astype(np.uint8) # NOTE: Temp
    color_image_pil = Image.fromarray(color_image)
    color_image_pil.save(img_path)