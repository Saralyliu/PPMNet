import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange

# add
import clip
import math
from model.get_cam import get_img_cam
from pytorch_grad_cam import GradCAM
from clip.clip_text import new_class_names, new_class_names_coco
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import copy
from util.vis1 import Visualizer
import util.utils as utils
Visualizer.initialize(True)
from einops import rearrange

def zeroshot_classifier(classnames, templates, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
def get_maskT(map,T):
    map[map>=T]=1
    map[map<T]=0
    return map

def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity

def get_gram_matrixsq(s,q): #N*c
    fea_T = q.permute(0, 2, 1)    # C*N
    fea_norm = s.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(s, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram
def get_gram_matrixss(fea): #N*c
    fea_T = fea.permute(0, 2, 1)    # C*N
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram
def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.root = args.data_root
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        #
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        channel = 516
        # if self.shot==1:
        #     channel = 516
        # else:
        #     channel = 524
        self.query_merge = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512+2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls_supp = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 2, kernel_size=1))

        self.transformer = Transformer(shot=self.shot)

        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        # if args.shot > 1:
        #     self.kshot_trans_dim = args.kshot_trans_dim
        #     if self.kshot_trans_dim == 0:
        #         self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
        #         self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
        #     else:
        #         self.kshot_rw = nn.Sequential(
        #             nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))
        self.map_mode = 'Cosine'
        # add
        self.annotation_root = args.annotation_root
        self.clip_model, _ = clip.load(args.clip_path)
    def forward(self, x, q_cam,s_cam, class_name, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):

        # mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        # mask = (mask == 1).float()
        h, w = x.shape[-2:]
        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")

        # extract the cnn features
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)
        supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x)

        supp_feat_cnn = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat_cnn = self.down_supp(supp_feat_cnn)
        query_feat_cnn = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_cnn = self.down_query(query_feat_cnn)

        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list_ori = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        q_camT = get_maskT(q_cam.unsqueeze(1).clone(), 0.6)
        q_camT = F.interpolate(q_camT, size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]),
                               mode='bilinear',
                               align_corners=True)
        q_camT = (q_camT == 1).float()
        s_camT = get_maskT(s_cam.clone(), 0.6)
        s_camT = F.interpolate(s_camT, size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]),
                               mode='bilinear',
                               align_corners=True)
        s_camT = rearrange(s_camT, "b n h w -> (b n) 1 h w")
        s_camT = (s_camT == 1).float()

        supp_feat_itemfore = eval('supp_feat_' + self.low_fea_id)
        supp_feat_itemfore = supp_feat_itemfore * F.interpolate(s_camT, size=(
            supp_feat_itemfore.size(2), supp_feat_itemfore.size(3)),
                                                                mode='bilinear', align_corners=True)
        supp_feat_itemfore = rearrange(supp_feat_itemfore, "(b n) c h w -> b n c h w", n=self.shot)
        sq_feat_list_orifore = [supp_feat_itemfore[:, i, ...] for i in range(self.shot)]
        query_feat_item = eval('query_feat_' + self.low_fea_id)
        query_feat_itemfore = query_feat_item * F.interpolate(q_camT, size=(
            query_feat_item.size(2), query_feat_item.size(3)), mode='bilinear', align_corners=True)

        sq_feat_list_orifore.append(query_feat_itemfore)

        #clip图像特征提取
        tmp_supp_clip_fts, supp_attn_maps = self.clip_model.encode_image(s_x, h, w, extract=True)[:]
        tmp_que_clip_fts, que_attn_maps = self.clip_model.encode_image(x, h, w, extract=True)[:]

        supp_clip_fts = [ss[1:, :, :] for ss in tmp_supp_clip_fts]
        que_clip_fts = [ss[1:, :, :] for ss in tmp_que_clip_fts]

        tmp_supp_clip_feat_all = [ss.permute(1, 2, 0) for ss in supp_clip_fts]
        supp_clip_feat_all = [aw.reshape(
            tmp_supp_clip_feat_all[0].shape[0], tmp_supp_clip_feat_all[0].shape[1], int(math.sqrt(tmp_supp_clip_feat_all[0].shape[2])),
            int(math.sqrt(tmp_supp_clip_feat_all[0].shape[2]))).float()
            for aw in tmp_supp_clip_feat_all]

        tmp_que_clip_feat_all = [qq.permute(1, 2, 0) for qq in que_clip_fts]
        que_clip_feat_all = [aw.reshape(
            tmp_que_clip_feat_all[0].shape[0], tmp_que_clip_feat_all[0].shape[1], int(math.sqrt(tmp_que_clip_feat_all[0].shape[2])),
            int(math.sqrt(tmp_que_clip_feat_all[0].shape[2]))).float()
            for aw in tmp_que_clip_feat_all]

        # get the vtp from memory
        q_cam = F.interpolate(q_cam.unsqueeze(1), size=(query_feat_cnn.shape[2], query_feat_cnn.shape[3]),
                          mode='bilinear',
                          align_corners=True)
        img_cam = q_cam.repeat(1,2,1,1)
  

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))

        est_val_list = []
        est_val_listq = []
        est_val_lists_sf = []
        est_val_lists_qf = []
        for i, supp_item in enumerate(sq_feat_list_orifore):

            supp_gramf = get_gram_matrix(supp_item)
            gram_diffq = que_gram - supp_gramf
            gram_diffqnorm = gram_diffq.norm(dim=(1, 2)) / norm_max

            if i < self.shot:
                est_val_list.append(gram_diffqnorm.reshape(bs, 1, 1, 1))  # q-sf
                # =====================s_sf s_qf
                supp_gram = get_gram_matrix(supp_feat_list_ori[i])
                norm_maxs = torch.ones_like(supp_gram).norm(dim=(1, 2))
                que_gramf = get_gram_matrix(sq_feat_list_orifore[-1])
                norm_maxqf = torch.ones_like(que_gramf).norm(dim=(1, 2))

                gram_diffs_sf = supp_gram - supp_gramf
                gram_diffs_sfnorm = gram_diffs_sf.norm(dim=(1, 2)) / norm_maxs

                gram_diffs_qf = supp_gram - que_gramf
                gram_diffs_qfnorm = gram_diffs_qf.norm(dim=(1, 2)) / norm_maxqf

                est_val_lists_sf.append(gram_diffs_sfnorm.reshape(bs, 1, 1, 1))  # s-sf
                est_val_lists_qf.append(gram_diffs_qfnorm.reshape(bs, 1, 1, 1))  # s-qf
            else:
                est_val_listq.append(gram_diffqnorm.reshape(bs, 1, 1, 1))  # q-qf

        est_val_total = torch.cat(est_val_list, 1)
        # =======================================
        s_val_mean = est_val_total.mean(1, True)
        est_val_listq.append(s_val_mean)  # qs
        est_val_totalsq = torch.cat(est_val_listq, 1)
        sumsq = est_val_totalsq.sum(1, True)
        # 计算调整后的权重 η_L^i
        eta_Lsq = self.calculate_eta(est_val_totalsq, sumsq)
        # 对调整后的权重进行归一化
        weight_softqs = torch.softmax(eta_Lsq, 1)
        # =================================ssq
        weight_softsqlist = []
        for i, s_sf in enumerate(est_val_lists_sf):
            est_val_totals_sq = torch.cat([s_sf, est_val_lists_qf[i]], 1)  # s_sf_qf
            sums_sq = est_val_totals_sq.sum(1, True)
            # 计算调整后的权重 η_L^i
            eta_Ls_sq = self.calculate_eta(est_val_totals_sq, sums_sq)
            # 对调整后的权重进行归一化
            weight_softs_sq = torch.softmax(eta_Ls_sq, 1)
            weight_softsqlist.append(weight_softs_sq)

        if self.shot > 1:
            sum = est_val_total.mean(1, True)
            # 计算调整后的权重 η_L^i
            eta_L = self.calculate_eta(est_val_total, sum)
            # 对调整后的权重进行归一化
            weight_soft = torch.softmax(eta_L, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # get the vvp
        if self.shot == 1:
 
            similarity2 = get_similarity(que_clip_feat_all[10], supp_clip_feat_all[10], s_camT)
            similarity1 = get_similarity(que_clip_feat_all[11], supp_clip_feat_all[11], s_camT)

        else:
            s_camT = rearrange(s_camT, "(b n) c h w -> b n c h w", n=self.shot)
            supp_clip_feat_all = [rearrange(ss, "(b n) c h w -> b n c h w", n=self.shot) for ss in supp_clip_feat_all]
            clip_similarity_1 = [
                get_similarity(que_clip_feat_all[11], supp_clip_feat_all[11][:, i, ...], s_camT[:, i, ...]) for i in
                range(self.shot)]
            clip_similarity_2 = [
                get_similarity(que_clip_feat_all[10], supp_clip_feat_all[10][:, i, ...], s_camT[:, i, ...]) for i in
                range(self.shot)]
            s_camT = rearrange(s_camT, "b n c h w -> (b n) c h w")
            similarity1 = torch.cat(clip_similarity_1, dim=1)
            similarity2 = torch.cat(clip_similarity_2, dim=1)
            similarity1 = (weight_soft * similarity1).sum(1, True)
            similarity2 = (weight_soft * similarity2).sum(1, True)
        clip_similarity = torch.cat([similarity1, similarity2], dim=1).cuda()
        clip_similarity = F.interpolate(clip_similarity, size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]),
                                        mode='bilinear', align_corners=True)

        #=================================================
        supp_pro = Weighted_GAP(supp_feat_cnn, \
                                F.interpolate(s_camT, size=(supp_feat_cnn.size(2), supp_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        supp_pro_bin = supp_pro.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])
        quer_pro = Weighted_GAP(query_feat_cnn, \
                                F.interpolate(q_camT, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        quer_pro_bin = quer_pro.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])


        supp_pro_bin = rearrange(supp_pro_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_pro_bin = (supp_pro_bin * (weight_soft.unsqueeze(2))).sum(1, True)

        merge_pro_bin = torch.cat([quer_pro_bin.unsqueeze(1), supp_pro_bin], 1) #先q后s
        merge_pro_bin = (merge_pro_bin * (weight_softqs.unsqueeze(2))).sum(1)
        # merge_feat_bin = 0.5*quer_pro_bin+0.5*supp_feat_bin
        query_feat = self.query_merge(torch.cat([query_feat_cnn, merge_pro_bin, img_cam * 10, clip_similarity * 10], dim=1))

        meta_out, weights = self.transformer(query_feat, supp_feat, s_camT, img_cam, clip_similarity)
        base_out = self.base_learnear(query_feat_5)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM )
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1).cuda()
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        map_h, map_w = meta_map_bg.shape[-2], meta_map_bg.shape[-1]
        base_map = F.interpolate(base_map, size=(map_h, map_w), mode='bilinear', align_corners=True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        if self.training != True:
            for i in range(1):
                q_camTnew = final_out.max(1)[1].unsqueeze(0).float()
                query_feat_itemfore1 = eval('query_feat_' + self.low_fea_id)
                query_feat_itemfore1 = query_feat_itemfore1 * F.interpolate(q_camTnew, size=(
                    query_feat_itemfore1.size(2), query_feat_itemfore1.size(3)), mode='bilinear', align_corners=True)
                sq_feat_list_orifore[-1] = query_feat_itemfore1

                est_val_list = []
                est_val_listq = []
                for i, supp_item in enumerate(sq_feat_list_orifore):
                    supp_gram = get_gram_matrix(supp_item)
                    gram_diff = que_gram - supp_gram
                    gram_diffnorm = gram_diff.norm(dim=(1, 2)) / norm_max
                    if i < self.shot:
                        est_val_list.append(gram_diffnorm.reshape(bs, 1, 1, 1))
                    else:
                        est_val_listq.append(gram_diffnorm.reshape(bs, 1, 1, 1))
                est_val_total = torch.cat(est_val_list, 1)
                # =======================================
                s_val_mean = est_val_total.mean(1, True)
                est_val_listq.append(s_val_mean)
                est_val_totalsq = torch.cat(est_val_listq, 1)
                sumsq = est_val_totalsq.sum(1, True)
                # 计算调整后的权重 η_L^i
                eta_Lsq = self.calculate_eta(est_val_totalsq, sumsq)
                # 对调整后的权重进行归一化
                weight_softqs = torch.softmax(eta_Lsq, 1)
                if self.shot > 1:
                    sum = est_val_total.mean(1, True)
                    # 计算调整后的权重 η_L^i
                    eta_L = self.calculate_eta(est_val_total, sum)
                    # 对调整后的权重进行归一化
                    weight_soft = torch.softmax(eta_L, 1)
                else:
                    weight_soft = torch.ones_like(est_val_total)
                est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

                quer_pro = Weighted_GAP(query_feat_cnn, \
                                        F.interpolate(q_camTnew, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
                                                      mode='bilinear', align_corners=True))
                quer_pro_bin = quer_pro.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])

                supp_pro_bin = (supp_pro_bin * (weight_soft.unsqueeze(2))).sum(1, True)

                merge_pro_bin = torch.cat([quer_pro_bin.unsqueeze(1), supp_pro_bin], 1)
                merge_pro_bin = (merge_pro_bin * (weight_softqs.unsqueeze(2))).sum(1)
                # merge_feat_bin = 0.5*quer_pro_bin+0.5*supp_feat_bin
                query_feat = self.query_merge(
                    torch.cat([query_feat_cnn, merge_pro_bin, img_cam * 10, clip_similarity * 10], dim=1))

                meta_out, weights = self.transformer(query_feat, supp_feat, s_camT, img_cam, clip_similarity)
                meta_out_soft = meta_out.softmax(1)
                meta_map_bg = meta_out_soft[:, 0:1, :, :]
                meta_map_fg = meta_out_soft[:, 1:, :, :]
                est_map = est_val.expand_as(meta_map_fg)
                meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
                meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

                merge_map = torch.cat([meta_map_bg, base_map], 1)
                merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

                final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return final_out.max(1)[1], main_loss + aux_loss1, distil_loss / 3, aux_loss2
        else:
            return final_out, meta_out, base_out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.transformer.mix_transformer.parameters()},
                {'params': model.supp_merge.parameters(), "lr": LR * 10},
                {'params': model.cls_supp.parameters(), "lr": LR * 10},
                {'params': model.query_merge.parameters(), "lr": LR * 10},
                {'params': model.cls_merge.parameters(), "lr": LR * 10},
                {'params': model.down_supp.parameters(), "lr": LR * 10},
                {'params': model.down_query.parameters(), "lr": LR * 10},
                {'params': model.gram_merge.parameters(), "lr": LR * 10},
            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False
        for param in model.clip_model.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
    def calculate_eta(self,omega_L, omega_L_mean, epsilon=1e-7):
        return 1 / (omega_L / omega_L_mean + epsilon)

    # 归一化调整后的权重
    def normalize_eta(self, eta_L):
        eta_L_sum = np.sum(eta_L)
        return eta_L / eta_L_sum

    def query_region_activate(self, query_fea, prototypes, mode):
        """
        Input:  query_fea:      [b, c, h, w]
                prototypes:     [b, n, c, 1, 1]
                mode:           Cosine/Conv/Learnable
        Oualphaut: activation_map: [b, n, h, w]
        """
        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Conv':
            map_temp = torch.bmm(prototypes.squeeze(-1).squeeze(-1), que_temp)  # [b, n, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map