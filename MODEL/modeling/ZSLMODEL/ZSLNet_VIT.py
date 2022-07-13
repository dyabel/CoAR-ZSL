import torch
import torch.nn as nn
import torch.nn.functional as F

from MODEL.modeling.backbone import resnet101_features,ViT
import MODEL.modeling.utils as utils

from os.path import join
import pickle
import numpy as np
import time
from MODEL.modeling.lossModule import SupConLoss_clear
from torch import distributed as dist
from .nets import *
from .class_name import *
from copy import deepcopy
import random
# from sklearn.metrics import pairwise_distances

import os
base_architecture_to_features = {
    'resnet101': resnet101_features,
}

class ZSLNet_VIT(nn.Module):
    def __init__(self, backbone, img_size, c, w, h,
                 attribute_num, cls_num, ucls_num, attr_group, w2v,dataset_name,
                 scale=20.0, device=None,cfg=None):

        super(ZSLNet_VIT, self).__init__()
        self.device = device

        self.img_size = img_size

        # self.attritube_num = len(CUB_select)
        self.attribute_num = attribute_num

        print('attribute_num',self.attribute_num)

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h
        print("TEMP",cfg.MODEL.LOSS.TEMP)

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.cls_num = cls_num
        self.attr_group = attr_group
        self.att_assign = {}
        self.backbone = backbone
        for key in attr_group:
            for att in attr_group[key]:
                self.att_assign[att] = key - 1

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        self.w2v_att = F.normalize(self.w2v_att,dim=-1)

        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.attr_proto_size = self.feat_channel
        self.part_num = self.attribute_num
        self.fc_proto = nn.Identity()
        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()
        self.Hinge_loss = nn.HingeEmbeddingLoss(margin=cfg.MODEL.LOSS.MARGIN)
        self.iters = -1
        self.rank = dist.get_rank()
        self.dataset_name = dataset_name
        if dataset_name == 'CUB':
            self.class_names = CUB_CLASS
            self.attr_names = CUB_ATTRIBUTE
            hid_size = 1024
        if dataset_name == 'AwA2':
            self.class_names = AwA2_CLASS
            self.attr_names = AwA2_ATTRIBUTE
            hid_size = 1024
        if dataset_name == 'SUN':
            self.class_names = SUN_CLASS
            hid_size = 1024
        if cfg.MODEL.ORTH:
            print('#'*100)
            print('orth')
            self.attribute_vector = nn.Parameter(nn.init.orthogonal_(torch.empty(self.part_num,self.part_num)),requires_grad=True)
        else:
            self.attribute_vector = nn.Parameter(torch.eye(self.part_num),requires_grad=False)
      
        self.memory_max_size = 1024
        out_channel = self.part_num
        #layers
        self.fc_attention_channel = nn.Linear(self.attr_proto_size,self.part_num)
        nn.init.xavier_uniform_(self.fc_attention_channel.weight)
        nn.init.constant_(self.fc_attention_channel.bias,0.)
        self.contrastive_embedding = nn.Linear(self.attr_proto_size,cfg.MODEL.HID)
     
        self.proto_model = ProtoModel(self.part_num,hid_size,self.attr_proto_size,with_cn=True)
        self.atten_thr = cfg.MODEL.ATTEN_THR
        self.feat_memory = torch.empty(0,self.feat_channel).to(self.device)
        self.label_memory = torch.empty(0).to(self.device)
        self.contrast_loss = SupConLoss_clear(cfg.MODEL.LOSS.TEMP)
        self.alpha = cfg.MODEL.LOSS.ALPHA
        self.beta = cfg.MODEL.LOSS.BETA


    def vit_features(self, x):
        x,patch_feature = self.backbone(x)
        return x,patch_feature

    def attr_decorrelation(self, query):
    
       loss_sum = 0

       for key in self.attr_group:
           group = self.attr_group[key]
           if query.ndim == 3:
               proto_each_group = query[:,group,:]  # g1 * v
               channel_l2_norm = torch.norm(proto_each_group, p=2, dim=1)
           else:
               proto_each_group = query[group, :]  # g1 * v
               channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
           loss_sum += channel_l2_norm.mean()

       loss_sum = loss_sum.float()/len(self.attr_group)

       return loss_sum

    def attentionModule(self, global_feat, patch_feat, seen_att):

        N,S,C = patch_feat.shape
        W = int(np.sqrt(S))
        H = W
        seen_att_normalized = F.normalize(seen_att,dim=-1)
        parts_map = self.fc_attention_channel(patch_feat)
        parts_map = parts_map.transpose(1,2).reshape(N,-1,W,H)
        # parts_map = self.extract_4(x4)
        # parts_map = self.extract_1(x1)
      
        global_semantic_feat = F.avg_pool2d(parts_map, kernel_size=(W, H)).squeeze()
        self.global_semantic_feat = F.normalize(global_semantic_feat,dim=-1)
        semantic_score = self.global_semantic_feat @ seen_att_normalized.T * 25.

        global_feat = self.fc_proto(global_feat)
        global_feat = F.normalize(global_feat,dim=-1)
        cls_proto = self.proto_model(seen_att_normalized*np.sqrt(self.part_num),True)
        cls_proto = torch.nn.functional.normalize(cls_proto,p=2,dim=1)
        visual_score = torch.einsum('bd,nd->bn', global_feat, cls_proto) * self.scale

        att_weight = F.max_pool2d(parts_map,kernel_size=(W,H)).squeeze().detach()
        self.att_weight = att_weight.gt(self.atten_thr)
        parts_map_flatten = parts_map.reshape(N,-1,W*H).softmax(dim=1)
        part_feats = torch.einsum('blr,brv->blv',parts_map_flatten, patch_feat.detach())
        # atten_channel_weight = torch.sigmoid(self.fc_attention_channel2(F.relu(self.fc_attention_channel1(parts_map.reshape(N,self.part_num,-1)))))
        # part_feats = atten_channel_weight * global_feat[:,None].expand(N,self.part_num,C)
        # part_feats = F.relu(part_feats)
        self.visual_score_reverse = None
        self.topk = None
        self.cls_proto_reverse = None

        return part_feats, visual_score, global_feat, semantic_score, cls_proto, parts_map
        
    def forward(self, x, att=None, label=None, seen_att=None,att_unseen=None):
        if att is not None:
            # att = att[:,CUB_select]
            att[att < 0] = 0.
            if self.part_num > self.attribute_num:
                att = torch.cat((att,att.new_ones(len(att)).unsqueeze(1)*0.0),1)
            att = F.normalize(att,dim=-1)
            att_binary = att.clone()
            att_binary[att_binary > 0] = 1.
        if seen_att is not None:
            # seen_att = seen_att[:,CUB_select]
            if self.part_num > self.attribute_num:
                seen_att = torch.cat((seen_att,seen_att.new_ones(len(seen_att)).unsqueeze(1)*0.0),1)
        self.iters += 1
        feat,patch_feature= self.vit_features(x)
        part_feats, visual_score, global_feat, semantic_score, cls_proto, atten_map = self.attentionModule(feat,patch_feature,seen_att)
      


        L_proto = torch.tensor(0).float().to(self.device)
        Lcls = torch.tensor(0).float().to(self.device)
        Lcls_att = torch.tensor(0).float().to(self.device)
        L_proto_align = torch.tensor(0).float().to(self.device)
        Lcpt = torch.tensor(0).float().to(self.device)
        Lreg = torch.tensor(0).float().to(self.device)
        Lad = torch.tensor(0).float().to(self.device)

        part_feats = F.normalize(part_feats, dim=-1)
        att_proto = self.proto_model(F.normalize(self.attribute_vector,-1) * np.sqrt(self.part_num), False)
        att_proto = F.normalize(att_proto,dim=-1)

      

        if not self.training:
            return visual_score

        part_filter = self.att_weight&att_binary.bool()
        if part_filter.sum()>0:
            attr_proto_dist = torch.cat([cosine_distance(part_feat,att_proto).unsqueeze(0) for part_feat in part_feats],dim=0)
            index_pos = torch.arange(len(att_proto)).view(-1,1).expand(attr_proto_dist.size(0),-1,1).to(self.device)
            tmp = torch.arange(len(att_proto))
            index_neg = torch.cat([tmp[tmp!=i].unsqueeze(0) for i in range(len(att_proto))],dim=0).expand(attr_proto_dist.size(0),-1,-1).to(self.device)
            pos_dists = attr_proto_dist.gather(2,index_pos).squeeze()
            neg_dists = attr_proto_dist.gather(2,index_neg).squeeze()
            L_proto += F.relu(pos_dists - self.alpha * neg_dists.min(dim=-1)[0]  + self.beta)[part_filter].mean()
        if self.part_num > self.attribute_num:
            att[:,-1] = 0.1


        # """
        Lcls += self.CLS_loss(visual_score, label)
        Lcls_att += self.CLS_loss(semantic_score, label)

        part_feats = part_feats[part_filter]
        part_label = (torch.arange(self.part_num)[None, ...].repeat(len(att), 1).to(self.device))[part_filter].reshape(-1)

        if len(part_label)>0 and len(torch.unique(part_label)) != len(part_label):
            contrastive_embeddings = F.normalize(self.contrastive_embedding(part_feats),dim=-1)
            L_proto_align += self.contrast_loss(contrastive_embeddings,part_label)

        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'AD_loss': Lad,
            'CPT_loss': Lcpt,
            'ATTCLS_loss': Lcls_att,
            'Proto_loss': L_proto,
            'Proto_align_loss': L_proto_align,
            'scale': scale,
        }

        return loss_dict

    def CPT(self, atten_map):
       N, L, W, H = atten_map.shape
       xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
       yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

       xp = xp.repeat(1, H)
       yp = yp.repeat(W, 1)

       atten_map_t = atten_map.view(N, L, -1)
       value, idx = atten_map_t.max(dim=-1)

       tx = idx // H
       ty = idx - H * tx

       xp = xp.unsqueeze(0).unsqueeze(0)
       yp = yp.unsqueeze(0).unsqueeze(0)
       tx = tx.unsqueeze(-1).unsqueeze(-1)
       ty = ty.unsqueeze(-1).unsqueeze(-1)

       pos = (xp - tx) ** 2 + (yp - ty) ** 2

       loss = atten_map * pos

       loss = loss.reshape(N, -1).mean(-1)
       loss = loss.mean()

       return loss


def build_ZSLNet_VIT(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attribute_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = utils.get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE


    c,w,h = 1024, img_size//32, img_size//32

    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    model_dir = cfg.PRETRAINED_MODELS

    vit = ViT(model_name='vit_large_patch16_224_in21k',pretrained=pretrained)

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)


    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return ZSLNet_VIT(backbone=vit, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attribute_num=attribute_num,
                  attr_group=attr_group, w2v=w2v,dataset_name=dataset_name,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device,cfg=cfg)