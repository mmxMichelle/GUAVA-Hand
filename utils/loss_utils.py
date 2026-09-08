

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from fused_ssim import fused_ssim
import lightning as L
from submodules.lpipsPyTorch import LPIPS

C1 = 0.01 ** 2
C2 = 0.03 ** 2
def cal_l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def cal_l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def cal_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# [NEW - Contribution 2]
# Convert SSIM similarity into a minimization loss.
def cal_ssim_loss(img1, img2):
    return 1.0 - cal_ssim(img1, img2)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def fast_ssim(img1, img2):
    ssim_value  = fused_ssim(img1, img2)
    return ssim_value

def cal_mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def cal_psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class Optimization_Loss(L.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg.OPTIMIZE
        self.bg_color=0.0

        self.perpetual_loss_f=LPIPS('alex', '0.1')
        self.perpetual_loss_f.eval()

        self.l1_loss_f=F.l1_loss
        # SSIM similarity, retained for evaluation/debugging if needed.
        self.ssim_f=cal_ssim

        # [NEW - Contribution 2]
        # SSIM loss used for optimisation.
        self.ssim_loss_f = cal_ssim_loss


    def  init_perpetual_loss(self,perpetual_loss_model):
        self.perpetual_loss_f=perpetual_loss_model

    def forward(self,render_results,batch,extra_results,iter_idx):

        batch_size=batch['image'].shape[0]
        render_images=render_results['renders']
        gt_images=batch['image']
        gt_masks=batch['mask']
        lambda_perpetual=self.cfg.lambda_perpetual
        if iter_idx > self.cfg.perpetual_increase_iter:
            lambda_perpetual=self.cfg.lambda_perpetual_high
        gt_images=gt_images*(gt_masks)+(1-gt_masks)*self.bg_color
        loss_dict={}

        if iter_idx < 1000:
            render_images=render_images*(gt_masks)+(1-gt_masks)*self.bg_color
        loss_dict['image_loss']=self.l1_loss_f(render_images,gt_images)*self.cfg.lambda_l1
        loss_dict['perpetual_loss']=self.perpetual_loss_f(render_images,gt_images)*lambda_perpetual

        if self.cfg.lambda_head_crop>0:
            head_lambdas=[self.cfg.lambda_l1*self.cfg.lambda_head_crop,lambda_perpetual*self.cfg.lambda_head_crop]
            loss_dict['head_loss']=self.cal_box_loss(render_images,gt_images,batch['head_box'],
                                                     [self.l1_loss_f,self.perpetual_loss_f],head_lambdas)
        if self.cfg.lambda_hand_crop>0:
            # [NEW - Contribution 2]
            hand_lambdas=[
                self.cfg.lambda_l1*self.cfg.lambda_hand_crop,
                lambda_perpetual*self.cfg.lambda_hand_crop,
                self.cfg.lambda_hand_ssim * self.cfg.lambda_hand_crop,
                ]
            hand_loss_funs = [
                self.l1_loss_f,
                self.perpetual_loss_f,
                self.ssim_loss_f,]

            loss_dict['hand_loss']=self.cal_box_loss(render_images,gt_images,batch['left_hand_box'],
                                                     hand_loss_funs,hand_lambdas)
            loss_dict['hand_loss']+=self.cal_box_loss(render_images,gt_images,batch['right_hand_box'],
                                                     hand_loss_funs,hand_lambdas)
        if 'raw_renders' in render_results:
            raw_images=render_results['raw_renders']
            loss_dict['image_loss']=loss_dict['image_loss']+self.l1_loss_f(raw_images,gt_images)*self.cfg.lambda_l1
            loss_dict['perpetual_loss']=loss_dict['perpetual_loss']+self.perpetual_loss_f(raw_images,gt_images)*lambda_perpetual
            if self.cfg.lambda_head_crop>0:
                loss_dict['head_loss']+=self.cal_box_loss(raw_images,gt_images,batch['head_box'],
                                                     [self.l1_loss_f,self.perpetual_loss_f],head_lambdas)
            if self.cfg.lambda_hand_crop>0:
                loss_dict['hand_loss']+=self.cal_box_loss(raw_images,gt_images,batch['left_hand_box'],
                                                     hand_loss_funs,hand_lambdas)
                loss_dict['hand_loss']+=self.cal_box_loss(raw_images,gt_images,batch['right_hand_box'],
                                                     hand_loss_funs,hand_lambdas)

        loss_local_xyz=F.relu((extra_results['uv_point_xyz']).norm(dim=-1) - self.cfg.threshold_local_xyz).mean() * self.cfg.lambda_local_xyz
        loss_local_scale=F.relu(extra_results['uv_point_scale'] - self.cfg.threshold_scale).norm(dim=-1).mean() * self.cfg.lambda_local_scale

        loss_dict['local_xyz_loss']=loss_local_xyz
        loss_dict['local_scale_loss']=loss_local_scale
        show_loss={}
        for key in loss_dict.keys():
            show_loss[key]=loss_dict[key].item()

        return loss_dict,show_loss

    def cal_box_loss(self,render_images,gt_images,box,loss_funs,loss_lambdas):
        #box:left,right,top,bottom
        batch_size = render_images.size(0)
        gt_crops,render_crops=[],[]
        loss=0.0
        for i in range(batch_size):
            gt_crop=gt_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            render_crop=render_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            if gt_crop.shape[1]<1 or gt_crop.shape[2]<1:
                continue
            gt_crop=F.interpolate(gt_crop[None],(256,256),mode='bilinear')
            render_crop=F.interpolate(render_crop[None],(256,256),mode='bilinear')
            gt_crops.append(gt_crop)
            render_crops.append(render_crop)
        render_crops=torch.cat(render_crops,dim=0)
        gt_crops=torch.cat(gt_crops,dim=0)
        for ii in range(len(loss_funs)):
            loss=loss+loss_funs[ii](render_crops,gt_crops)*loss_lambdas[ii]

        return loss
