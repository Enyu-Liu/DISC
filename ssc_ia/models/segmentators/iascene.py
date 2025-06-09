import importlib

import numpy as np
import torch.nn as nn
import torch
from mmengine.config import Config
from omegaconf import OmegaConf

from ..bevpipelines import LssVolume
from ... import build_from_configs
from .. import encoders, bevpipelines, decoders
from ..losses import (ce_ssc_loss, geo_scal_loss, sem_scal_loss, semantic_seg_loss, height_loss, get_klv_depth_loss)

class IAScene(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        view_scales,
        embed_dims,
        bevpipeline,
        volume_scale,
        image_shape,
        num_classes,
        class_frequencies=None,
        criterions=None,
        voxel_size=0.2, 
        scene_shape=[256, 256, 32],
        **kwargs,
    ):
        super().__init__()
        self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        self.criterions = criterions
        self.image_shape = image_shape
        self.project_scale = volume_scale
        self.voxel_size = voxel_size*volume_scale
        self.scene_shape = [l//self.project_scale for l in scene_shape]
        # build model structure
        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        
        # lss transformer
        self.lss_trans = LssVolume(image_shape, embed_dims)

        # bev pipeline
        self.generate_bev = build_from_configs(bevpipelines, bevpipeline, embed_dims=embed_dims, num_cls=num_classes, \
                                               scene_shape=self.scene_shape, image_shape=self.image_shape, voxel_size=self.voxel_size)

        # Decoder 
        self.ins_decoder = build_from_configs(decoders, decoder, num_cls=num_classes, \
                                              scene_shape=self.scene_shape, project_scale=self.project_scale)

    
    def forward(self, inputs):
        # Symphonies encoder
        feats = self.encoder(inputs['img']) 

        depth, K, E, voxel_origin, post_rot, post_tran, projected_pix = list(map(lambda k: inputs[k],
            ('depth', 'cam_K', 'cam_pose', 'voxel_origin', 'post_rot', 'post_tran', 'projected_pix_2')))

        # Lss View Transformer
        # TODO : Return volume, mono depth pred
        volume, depth_feat, mono_depth_pred = self.lss_trans(feats, inputs)
        # Generate BEV
        volume, bev_feat, seg_logits = self.generate_bev(volume, feats, depth, K, E, voxel_origin, \
                                                         post_rot, post_tran, projected_pix) # seg_logits for all cls

        # Decoder Layer (Instance Feature Enhance/BK Feature Enhance)
        params = [K, E, voxel_origin, self.voxel_size, self.image_shape, self.scene_shape, post_rot, post_tran]
        out_dict = self.ins_decoder(feats, volume, bev_feat, depth, params)
        out_dict['depth'] = [mono_depth_pred]

        return out_dict
    
    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'bev_seg': semantic_seg_loss,
            'geo_h': height_loss,
            'depth': get_klv_depth_loss,
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'][0])
        losses = {}
        for i in range(len(preds['ssc_logits'])):
            scale = 1 if i == len(preds['ssc_logits']) - 1 else 0.5
            preds_single = {key:value[i] for key, value in preds.items()}
            for loss in self.criterions:
                l = loss_map[loss](preds_single, target)
                if isinstance(l, dict):
                    for key, value in l.items():
                        losses['loss_' + loss + '_' + key + '_' + str(i)] = value * scale
                else:
                    losses['loss_' + loss + '_' + str(i)] = l * scale
        return losses


