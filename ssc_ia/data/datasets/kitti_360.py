import glob
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms as T

from ...utils.helper import vox2pix
from ...utils.projection import project_voxels_to_single_bev, project_image_to_voxels


SPLITS = {
    'train':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync',
     '2013_05_28_drive_0007_sync'),
    'val': ('2013_05_28_drive_0006_sync', ),
    'test': ('2013_05_28_drive_0009_sync', ),
}

KITTI_360_CLASS_FREQ = torch.tensor([
    2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103, 105540713, 16292249, 45297267,
    14454132, 110397082, 6766219, 295883213, 50037503, 1561069, 406330, 30516166, 1950115
])

bk_cls_info = dict(indices=np.array([0, 7, 8, 9, 10, 11, 12, 13, 14, 17]), 
                   names=['empty', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain', 'other-structure'])
ins_cls_info = dict(indices=np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 18]),
                    names=['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'pole', 'traffic-sign', 'other-object'])

class KITTI360(Dataset):

    META_INFO = {
        'class_weights':
        1 / torch.log(KITTI_360_CLASS_FREQ + 1e-6),
        'class_names':
        ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object')
    }

    def __init__(
        self,
        split,
        data_root,
        label_root,
        depth_root,
        data_config,
        project_scale=2,
        frustum_size=4,
        context_prior=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.data_config = data_config
        self.sequences = SPLITS[split]
        self.split = split

        self.depth_root = depth_root
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.context_prior = context_prior
        self.flip = self.data_config['flip']
        self.num_classes = 19

        self.voxel_origin = np.array((0, -25.6, -2))
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_shape = self.data_config['input_size']

        self.scans = []
        calib = self.read_calib()
        for sequence in self.sequences:
            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            proj_matrix = P @ T_velo_2_cam

            glob_path = osp.join(self.data_root, 'sequences', sequence, 'voxels', '*.bin')
            for voxel_path in glob.glob(glob_path):
                self.scans.append({
                    'sequence': sequence,
                    'P': P,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix': proj_matrix,
                    'voxel_path': voxel_path,
                })

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx]
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = osp.basename(scan['voxel_path'])
        frame_id = osp.splitext(filename)[0]
        data = {
            'frame_id': frame_id,
            'sequence': sequence,
            'P': P,
            'cam_pose': T_velo_2_cam,
            'proj_matrix': proj_matrix,
            'voxel_origin': self.voxel_origin
        }
        label = {}
        label['class_fres'] = KITTI_360_CLASS_FREQ
        label['bk_cls_info'] = bk_cls_info
        label['ins_cls_info'] = ins_cls_info

        scale_3ds = (self.output_scale, self.project_scale)
        data['scale_3ds'] = scale_3ds
        cam_K = P[:3, :3]
        data['cam_K'] = cam_K
        
        flip = random.random() > 0.5 if self.flip and self.split == 'train' else False
        target_1_path = osp.join(self.label_root, sequence, frame_id + f'_1_{scale_3ds[0]}.npy')
        with_target = self.split != 'test' and osp.exists(target_1_path)
        if with_target:
            target = np.load(target_1_path)
            if flip:
                target = np.flip(target, axis=1).copy()
            label['target'] = target

        # NOTE: Load and process img and corresponding depth map
        assert self.depth_root is not None
        depth_path = osp.join(self.depth_root, 'sequences', sequence, frame_id + '.npy')
        depth = np.load(depth_path)
        depth = Image.fromarray(depth) 
        
        img_path = osp.join(self.data_root, 'sequences', sequence, 'image_00/data_rect',
                            frame_id + '.png')
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at path: {img_path}")
            # print(f"Exception: {e}")
            raise e

        # Apply the same augment on both depth map and img
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip)
        resize, resize_dims, crop, flip, rotate = img_augs
        img, depth, post_rot2, post_tran2 = self.img_transform(
            img, depth, post_rot, post_tran, resize=resize, 
            resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
        )
        # for convenience, make augmentation matrices 3x3
        post_tran = np.zeros(3)
        post_rot = np.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2        

        depth = np.asarray(depth, dtype=np.float32)
        img = np.asarray(img, dtype=np.float32) / 255.0
        data['img'] = self.transforms(img)  # (3, H, W)  # TO Tensor
        data['depth'] = depth
        data['post_tran'] = post_tran
        data['post_rot'] = post_rot
        label['depth'] = depth

        for scale_3d in scale_3ds:
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(T_velo_2_cam, cam_K, self.voxel_origin,
                                                     self.voxel_size * scale_3d, self.img_shape,
                                                     self.scene_size, post_rot, post_tran)
            data[f'projected_pix_{scale_3d}'] = projected_pix
            data[f'pix_z_{scale_3d}'] = pix_z
            data[f'fov_mask_{scale_3d}'] = fov_mask

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)

        # Generate augmentation gt
        if self.split != 'test':
            bev_segmap = project_voxels_to_single_bev(data, label)
            bev_occ_map = project_voxels_to_single_bev(data, label, mode='ins')
            seg_img = project_image_to_voxels(data, label, mode='ins', save=False)
            label['bev_seg_bk'] = torch.from_numpy(bev_segmap)
            label['bev_seg_ins'] = torch.from_numpy(bev_occ_map) # num_cls
            label['seg_img'] = torch.from_numpy(seg_img)

        return data, label

    @staticmethod
    def read_calib():
        P = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out['P2'] = P
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out
    
    def parse_poses(self, filename):
        """Returns T_cam_2_cam actually, different from the original implementation in SemanticKITTI API
        """
        poses = []
        with open(filename) as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                pose = np.zeros((4, 4))
                pose[:3] = np.array(values).reshape((3, 4))
                pose[3, 3] = 1.0
                poses.append(pose)
        return poses

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])    
    
    def sample_augmentation(self, H, W, flip=None, scale=None):
        """Get aug params for depth net
        Args :  
            H, W : Original imag shape
            data_config : include aug dict and custom input size
        """
        fH, fW = self.data_config['input_size'] # resized shape
        
        if self.split=='train':
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate
    
    def img_transform(self, img, depth, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        depth = self.img_transform_core(depth, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, depth, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img    