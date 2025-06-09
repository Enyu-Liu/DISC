point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

empty_idx = 0

semantic_kitti_class_frequencies = [
        5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
        8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
        4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
        1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
    ]


data_config={
    'input_size': (384, 1280),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# model params #
numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)


model = dict(
    depth_net=dict(
        type='GeometryDepthNet',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=27,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
    ),
    img_view_transformer=dict(
        type='ViewTransformerLSS',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    )
)