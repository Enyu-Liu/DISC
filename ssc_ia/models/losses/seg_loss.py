import torch
import torch.nn.functional as F
from mmseg.registry import MODELS


def _mask_seg_loss(pred, target):
    # target = torch.where(((target > 0) & (target != 255)), 1, target).long()
    pred = torch.sigmoid(pred)
    target = torch.where(((target > 0) & (target != 255)), 1, target).long()
    target = torch.where((target==255), 0, target)
    img_size = target.shape[-2:] 
    pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False) #(bs, 1, H, W)
    # weight matrix
    pos_weight = torch.ones_like(target) * 10.0  # 5.0 for positive
    neg_weight = torch.ones_like(target)
    weight = torch.where(target == 1, pos_weight, neg_weight).unsqueeze(1).to(pred) #(bs, 1, H, W)
    loss_cse = F.binary_cross_entropy(pred.float(), target.unsqueeze(1).float(), weight=weight, reduction='mean')
    loss_dice = MODELS.build({'type':'DiceLoss', 'use_sigmoid':False})
    loss_dice = loss_dice(pred, target.unsqueeze(1))
    return loss_dice + loss_cse

# TODO: To be sorted out
def _seg_loss(pred, target, cls_fres, indices):
    cls_fres = torch.cat([cls_fres[:, indices].sum().unsqueeze(0),\
                            cls_fres[:, indices][:, 1:].squeeze(0)], dim=0) # 11  #torch.cat(bk, ins)
    cls_weight = 1 / torch.log(cls_fres + 1e-6),
    cls_weight = cls_weight[0] /  cls_weight[0].sum() # [11]

    img_size = target.shape[-2:] 
    pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False) #(bs, num_k, H, W)
    loss_cse = F.cross_entropy(pred.float(), target.long(), \
                            cls_weight.float(), ignore_index=255, reduction='mean')

    # Compute Dice Loss (target needs to be one-hot encoded)
    target_one_hot = F.one_hot(target.long(), num_classes=pred.size(1)).permute(0, 3, 1, 2).float() #(bs, num_k, H, W)
    loss_dice = MODELS.build({'type':'DiceLoss', 'use_sigmoid':True})
    loss_dice = loss_dice(pred, target_one_hot)

    return loss_dice + loss_cse
    
def semantic_seg_loss(pred, target):
    loss = {}
    ins_indices = target['ins_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()
    bk_indices = target['bk_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()

    loss_ins = 0.
    for i in range(len(pred['bev_seg_ins'])):
        w = 1. if i == len(pred['bev_seg_ins'])-1 else 0.5
        loss_ins += _seg_loss(pred['bev_seg_ins'][i], target['bev_seg_ins'], target['class_fres'], indices=ins_indices) * w
    loss['ins'] = loss_ins

    loss_bk = 0.
    for i in range(len(pred['bev_seg_bk'])):
        w = 1. if i == len(pred['bev_seg_bk'])-1 else 0.5
        loss_bk += _seg_loss(pred['bev_seg_bk'][i], target['bev_seg_bk'], target['class_fres'], indices=bk_indices) * w
    loss['bk'] = loss_bk
    
    loss['img'] = _mask_seg_loss(pred['seg_img'], target['seg_img'])   # Ignor 255
    return loss


def _h_loss(pred, target, opposite_indices):
    indices = opposite_indices  # Opposite to the mode
    non_cls = indices + [255]
    target_mask = torch.ones_like(target)
    for ignored_class in non_cls:
        target_mask[target == ignored_class] = 0
    pred = F.interpolate(pred, size=target.shape[1:], mode='trilinear') #(bs, 1, L, W, H)
    # Focal loss
    focal_loss = MODELS.build({'type':'FocalLoss', 'use_sigmoid':True, 'loss_weight':5.0})
    return focal_loss(pred, target_mask) * 5.

def _h_point_loss(pred, target, opposite_indices, scale=2):
    points = pred['coor']
    h_logits = pred['h_logits']
    indices = opposite_indices  # Opposite to the mode
    non_cls = indices + [255]
    target_mask = torch.ones_like(target)
    for ignored_class in non_cls:
        target_mask[target == ignored_class] = 0
    target = F.interpolate(target_mask.to(h_logits).unsqueeze(1), scale_factor=1/scale, mode='trilinear') #(bs, 1, L, W, H)
    target = target[..., points[:, 0], points[:, 1], :] #(bs, 1, N, H)
    focal_loss = MODELS.build({'type':'FocalLoss', 'use_sigmoid':True, 'loss_weight':5.0})
    return focal_loss(h_logits[None, None, :, :], target)


def height_loss(pred, target):
    loss = {}
    ins_indices = target['ins_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()
    bk_indices = target['bk_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()
    target = target['target']
    
    loss_ins = 0.
    for i in range(len(pred['h_ins'])):
        w = 1. if i == len(pred['h_ins'])-1 else 0.5
        loss_ins += _h_loss(pred['h_ins'][i].unsqueeze(1), target.long(),opposite_indices=bk_indices) * w
    loss['ins'] = loss_ins

    loss_bk = 0.
    for i in range(len(pred['h_bk'])):
        w = 1. if i == len(pred['h_bk'])-1 else 0.5
        loss_bk += _h_loss(pred['h_bk'][i].unsqueeze(1), target.long(), opposite_indices=ins_indices) * w
    loss['bk'] = loss_bk
    # loss['h_logits'] = _h_point_loss(pred['query_dict'], target.long())
    return loss



    
