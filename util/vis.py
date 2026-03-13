import random

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

###########################################################showout

path = './vis/'
if not os.path.exists(path): os.makedirs(path)

def showout(map, index,name,count):
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    pred = map.squeeze()
    pred = pred.detach().cpu().numpy()
    plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(pred, interpolation='bicubic',cmap='gray')
    plt.axis('off')
    save_name = feature_save_path + '/' +str(count)+ '_' + str(index) + '.png'
    plt.savefig(save_name)
    plt.show()

def show_image(map, name, count):
    # path = '/home/liuyu/code/BAM/savefig/'
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(map)
    plt.axis('off')
    save_name = feature_save_path + '/' +str(count)+ '_' + 'image' + '.jpg'
    plt.savefig(save_name)

def show(img,mask,overlay_color):
    # blue 51 102 255
    # green 204 255 204
    # yellow 255,230,153
    overlay_color = overlay_color
    transparency = 0.5
    plt.ion()
    img = img
    mask = mask
    mask = mask // np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (
            overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (
            overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (
            overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])

    return im_over.astype(np.uint8)

def save_result(support,s_label,query,q_label,query_pred,supp_pred,i_iter,group,q_heat,s_heat):
    result_dir = os.path.join(path, 'result',str(group),'pred')
    result_dir1 = os.path.join(path, 'result', str(group),'heatmap')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    support = support.squeeze().detach().cpu().numpy()
    s_label = s_label.squeeze().detach().cpu().numpy()
    query = query.squeeze().detach().cpu().numpy()
    q_label = q_label.squeeze().detach().cpu().numpy()
    cv2.imwrite('%s/%s_Sb.png' % (result_dir, i_iter), support)
    cv2.imwrite('%s/%s_qb.png' % (result_dir, i_iter), query)
    overlay_color = [0, 255, 0]
    overlay_color1 = [255,0, 0]
    support_save = show(support,s_label,overlay_color)
    cv2.imwrite('%s/%s_Sa.png' % (result_dir, i_iter), support_save)
    query_save =show(query,q_label,overlay_color)
    cv2.imwrite('%s/%s_qa.png' % (result_dir, i_iter), query_save)
    quer_presave =show(query,query_pred,overlay_color1)
    cv2.imwrite('%s/%s_qpred.png' % (result_dir, i_iter), quer_presave)
    supp_presave = show(support, supp_pred, overlay_color1)
    cv2.imwrite('%s/%s_spred.png' % (result_dir, i_iter), supp_presave)
    cv2.imwrite('%s/%s_sheat.png' % (result_dir1, i_iter), s_heat)
    cv2.imwrite('%s/%s_qheat.png' % (result_dir1, i_iter), q_heat)

def vis_heatmap(img,feature,name,count):
    result_dir1 = os.path.join(path, str(name))
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    #img = img.squeeze().detach().cpu().numpy()
    img_size = img.shape[1]
    # feature = torch.softmax(feature,1)
    feature = (feature ** 2).sum(1)
    # feature = feature.squeeze(0)[0]

    for j in range(feature.size(0)):
        fm = feature[j].detach().cpu().numpy()
        fm = 1-fm
        #  activate map
        fm = cv2.resize(fm, (img_size, img_size))
        fm = 255 * (fm - np.min(fm)) / (
                np.max(fm) - np.min(fm) + 1e-12
        )

        # bbox = localize_from_map(fm, threshold_ratio=1.0)
        fm = np.uint8(np.floor(fm))
        fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)


        overlapped = img * 0.3 + fm * 0.7
        overlapped[overlapped > 255] = 255
        # overlapped = draw_bbox(overlapped, [bbox])
        overlapped = overlapped.astype(np.uint8)

        grid_img = 255 * np.ones(
            (img_size, 3 * img_size + 2 * 10, 3), dtype=np.uint8
        )
        grid_img[:, :img_size, :] = img[:, :, ::-1]
        grid_img[:, img_size + 10:2 * img_size + 10, :] = fm
        grid_img[:, 2 * img_size + 2 * 10:, :] = overlapped
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("", grid_img)
        # cv2.waitKey(0)
        cv2.imwrite('%s/%s_heat.png' % (result_dir1, count), grid_img)


def generate_heat_map(img, density_map,name,count):
    result_dir = os.path.join(path, str(name))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap = density_map.squeeze().detach().cpu().numpy()
    # heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap[:, :, 0:2] = heatmap[:, :, 0:2] * 0.9
    merge_img = img.copy()
    heatmap_img = heatmap.copy()
    overlay = img.copy()
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, merge_img, 1 - alpha, 0, merge_img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap_img, alpha, merge_img, 1 - alpha, 0, merge_img) # 将热度图覆盖到原图
    # merge_img = cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(merge_img)
    plt.axis('off')
    save_name = result_dir + '/' + str(count) + '_' + str(name) + '.jpg'
    plt.savefig(save_name)
    # cv2.imshow("", merge_img)
    # cv2.waitKey(0)
    # return merge_img
def show(img,mask,overlay_color):
    # blue 51 102 255
    # green 204 255 204
    # yellow 255,230,153
    overlay_color = overlay_color
    transparency = 0.5
    plt.ion()
    img = img
    mask = mask
    # mask = mask // np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (
            overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (
            overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (
            overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])

    return im_over.astype(np.uint8)

def show_result(image,label,name,count,overlay_color):
    result_dir = os.path.join(path,str(name))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    label = label.squeeze().detach().cpu().numpy()
    color_save = show(image,label,overlay_color)
    plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(color_save)
    plt.axis('off')
    save_name = result_dir + '/' + str(count) + '_' + str(name) + '.jpg'
    plt.savefig(save_name)
    # cv2.imwrite('%s/%s_out.png' % (result_dir, count), color_save)
def showcos(map,name,count):
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    pred = map.squeeze()
    pred = pred.detach().cpu().numpy()
    plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(pred, interpolation='bicubic',cmap='gray')
    plt.axis('off')
    save_name = feature_save_path + '/' +str(count)+ '_'+ str(name)+ '.png'
    plt.savefig(save_name)
    plt.show()
def show_image(image, name, count):
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    plt.figure(figsize=(473, 473), dpi=1)
    plt.imshow(image)
    plt.axis('off')
    save_name = feature_save_path + '/' + str(count) + '_' + str(name) + '.jpg'
    plt.savefig(save_name)

def feature_heatmap(img,feature,name,i,count):
    result_dir = os.path.join(path, str(name),str(count),str(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[1]
    feature = feature.squeeze().detach().cpu().numpy()
    feature_map_num = feature.shape[0]
    for j in range(feature_map_num):
        fm = feature[j]
        fm = cv2.resize(fm, (img_size, img_size))
        fm = 255*(fm - np.min(fm)) / (
            np.max(fm) - np.min(fm) + 1e-12)
        fm = np.uint8(np.floor(fm))
        fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)
        merge_img = img.copy()
        heatmap_img = fm.copy()
        overlay = img.copy()
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, merge_img, 1 - alpha, 0, merge_img)  # 将背景热度图覆盖到原图
        cv2.addWeighted(heatmap_img, alpha, merge_img, 1 - alpha, 0, merge_img)  # 将热度图覆盖到原图
        merge_img = cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB)
        # grid_img = cv2.addWeighted(img, 0.6, fm, 1.0, 0)
        # grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
        plt.imshow(merge_img)
        plt.axis('off')
        save_name = result_dir + '/' + str(count) + '_' + str(name)+ str(i) + '_' +str(j) + '.jpg'
        plt.savefig(save_name)


r""" Visualize model predictions """
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# from . import utils
from matplotlib import pyplot as plt
import os

class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        # cls.vis_path = './vis/'
        # if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b,count):
        spt_img_b = to_cpu(spt_img_b)
        spt_mask_b = to_cpu(spt_mask_b)
        qry_img_b = to_cpu(qry_img_b)
        qry_mask_b = to_cpu(qry_mask_b)
        pred_mask_b = to_cpu(pred_mask_b)
        # cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b)):
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, count, sample_idx)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, iou=None):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]


        qry_img = cls.to_numpy(qry_img, 'img') #3,473,473

        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask') #473,473
        pred_mask = cls.to_numpy(pred_mask, 'mask')

        for spt_img in spt_imgs:
            plt.imshow(spt_img)
            plt.axis('off')
            save_name = cls.vis_path + '%d_%d_class-%d' % (batch_idx, sample_idx, cls_id) + 's_image' + '.jpg'
            plt.savefig(save_name)

        plt.imshow(qry_img)
        plt.axis('off')
        save_name = cls.vis_path + '%d_%d_class-%d' % (batch_idx, sample_idx, cls_id) + 'q_image' + '.jpg'
        plt.savefig(save_name)

        # plt.imshow(qry_mask, interpolation='bicubic', cmap='gray')
        # # plt.axis('off')
        # save_name = cls.vis_path + '%d_%d_class-%d' % (batch_idx, sample_idx, cls_id) + 'q_pre' + '.jpg'
        # plt.savefig(save_name)
        #
        # plt.imshow(pred_mask, interpolation='bicubic', cmap='gray')
        # # plt.axis('off')
        # save_name = cls.vis_path + '%d_%d_class-%d' % (batch_idx, sample_idx, cls_id)+'q_m' + '.jpg'
        # plt.savefig(save_name)
        # plt.show()

        # (Image.fromarray(pred_mask.astype(np.uint8))).save(cls.vis_path + '%d_%d_class-%d' % (batch_idx, sample_idx, cls_id)+'q_pre' + '.jpg')

        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        merged_pil = cls.merge_image_pair(spt_masked_pils + [pred_masked_pil, qry_masked_pil])

        iou = iou.item() if iou else 0.0

        # [spt_masked_pil.save(
        #     cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + 's' + '.jpg') for spt_masked_pil in spt_masked_pils]

        pred_masked_pil.save(
            cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + 'pre' + '.jpg')
        qry_masked_pil.save(
            cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + 'q' + '.jpg')

        # merged_pil.save(cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img

