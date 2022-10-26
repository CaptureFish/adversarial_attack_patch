import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble_tool.utils import denorm
from pytorchYOLOv4.tool.utils import load_class_names
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import time
import sys
import numpy as np

from ipdb import set_trace as st

# 这里也有很多关于method_num的选项，为什么选用不同的生成器需要变化这么多？
# latent code 意义不大
# 集成所有的代码需要的工具 
class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

def eval_rowPtach(generator, batch_size, device
                , latent_shift, alpah_latent
                , input_imgs, label, patch_scale, cls_id_attacked
                , denormalisation
                , model_name, detector
                , patch_transformer, patch_applier
                , by_rectangle
                , enable_rotation
                , enable_randomLocation
                , enable_crease
                , enable_projection
                , enable_rectOccluding
                , enable_blurred
                , enable_with_bbox
                , enable_show_plt
                , enable_clear_output
                , cls_conf_threshold
                , patch_mode = 0
                , multi_score = False
                , enable_no_random=False
                , fake_images_default=None):

                    
    

    fake_images = fake_images_default

    # enable_empty_patch == False: no patch
    enable_empty_patch =False
    if(patch_mode == 1):
        enable_empty_patch = True

    min_loss_det = 999
    for i_fimage in range(1):
        if not(enable_clear_output):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(  adv_patch=fake_images[i_fimage], 
                                                                        lab_batch=label, 
                                                                        img_size=416,
                                                                        patch_mask=[],
                                                                        by_rectangle=by_rectangle,  
                                                                        do_rotate=enable_rotation, 
                                                                        rand_loc=enable_randomLocation, 
                                                                        with_black_trans=False, 
                                                                        scale_rate = patch_scale, 
                                                                        with_crease=enable_crease, 
                                                                        with_projection=enable_projection,
                                                                        with_rectOccluding=enable_rectOccluding,
                                                                        enable_empty_patch=enable_empty_patch,
                                                                        enable_no_random=enable_no_random,
                                                                        enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs[0], adv_batch_t)
        else:
            p_img_batch = input_imgs

        # # test
        # print("label       size : "+str(label.size()))
        # print("label            : "+str(label))
        # print("fake_images size : "+str(fake_images.size()))
        # print("p_img_batch size : "+str(p_img_batch.size()))
        # # test
        # import numpy as np
        # def show(img):
        #     npimg = img.numpy()
        #     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
        #                                     interpolation='nearest')
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        # plt.ion()
        # plt.figure()
        # show(make_grid(fake_images.detach().cpu()[:16,:,:,:]))
        # plt.figure()
        # show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
        # plt.ioff()
        # plt.show()
        # sys.exit()

        # loss.
        if(model_name == "yolov4"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            # default is multi_score. Here, "max_prob_cls" is "cls*obj"
            loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        if(model_name == "yolov3"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        if(model_name == "yolov2"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        if(model_name == "fasterrcnn"):
            max_prob, bboxes = detector.detect(tensor_image_inputs=p_img_batch, cls_id_attacked=cls_id_attacked, threshold=0.5)
            # loss_det
            loss_det = torch.mean(max_prob)
            # no loss_overlap
            loss_overlap = torch.tensor(0.0).to(device)
        
        if(min_loss_det > loss_det):
            min_loss_det = loss_det
    
    loss_det = min_loss_det

    # darw bbox
    if enable_with_bbox and len(bboxes)>0:
        trans_2pilimage = transforms.ToPILImage()
        batch = p_img_batch.size()[0]
        for b in range(batch):
            img_pil = trans_2pilimage(p_img_batch[b].cpu())
            img_width = img_pil.size[0]
            img_height = img_pil.size[1]
            namesfile = 'pytorchYOLOv4/data/coco.names'
            class_names = load_class_names(namesfile)
            # sample first image
            # print("bbox : "+str(bbox))
            bbox = bboxes[b]
            for box in bbox:
                # print("box size : "+str(box.size()))
                cls_id = box[6].int()
                cls_name = class_names[cls_id]
                cls_conf = box[5]
                if(cls_id == cls_id_attacked):
                    if(cls_conf > cls_conf_threshold):
                        if(model_name == "yolov2"):
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            left        = (x_center.item() - width.item() / 2) * img_width
                            right       = (x_center.item() + width.item() / 2) * img_width
                            top         = (y_center.item() - height.item() / 2) * img_height
                            bottom      = (y_center.item() + height.item() / 2) * img_height
                        if(model_name == "yolov4") or (model_name == "yolov3") or (model_name == "fasterrcnn"):
                            left        = int(box[0] * img_width)
                            right       = int(box[2] * img_width)
                            top         = int(box[1] * img_height)
                            bottom      = int(box[3] * img_height)
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline ="red")
                        # text
                        color = [255,0,0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height)/18))
                        sentence = str(cls_name)+" ("+str(round(float(cls_conf), 2))+")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
            if enable_show_plt:
                # show, debug
                plt.imshow(img_pil)
                plt.show()
            trans_2tensor = transforms.ToTensor()
            p_img_batch[b] = trans_2tensor(img_pil)
    
    return p_img_batch, fake_images, bboxes










# # test
# import numpy as np
# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
#                                     interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
# plt.ion()
# plt.figure()
# show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
# plt.ioff()
# plt.show()
# sys.exit()
