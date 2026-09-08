
import os,shutil
import lightning
from lightning.fabric.strategies import DDPStrategy
import torch,torchvision
import numpy as np
from tqdm import tqdm
from utils.general_utils import  rtqdm, biuld_logger

from models.UbodyAvatar import Ubody_Gaussian_inferer,Ubody_Gaussian,GaussianRenderer
from utils.loss_utils import cal_psnr,cal_ssim,Optimization_Loss

class Trainer:
    def __init__(
            self, meta_cfg, infer_model:Ubody_Gaussian_inferer,render_model:GaussianRenderer, optimizer, scheduler,
            train_dataloader, val_dataloader, devices, debug=False,timestamp='202412',
            use_distributed_sampler=True
        ):

        self._debug = debug
        self._meta_cfg, self._best_metric = meta_cfg, None
        self._dump_dir = os.path.join('outputs','debugs') if debug else \
                         os.path.join('outputs', meta_cfg.TRAIN.exp_str, timestamp,)

        self.use_distributed_sampler = use_distributed_sampler
        self._save_codes(timestamp)
        if not debug:
            os.makedirs(os.path.join(self._dump_dir, 'visuals_training','valid_render'), exist_ok=True)
            os.makedirs(os.path.join(self._dump_dir, 'visuals_training',"train_render"), exist_ok=True)
            os.makedirs(os.path.join(self._dump_dir, 'checkpoints'), exist_ok=True)
            self.logger = biuld_logger(os.path.join(self._dump_dir, 'train_log.txt'), name=f'train_{timestamp}')
            self.logger.debug(meta_cfg._raw_string)
        else:
            if os.path.exists(os.path.join(self._dump_dir, 'debug_visuals_training')):
                shutil.rmtree(os.path.join(self._dump_dir, 'debug_visuals_training'))
            os.makedirs(os.path.join(self._dump_dir, 'debug_visuals_training',"train_render"), exist_ok=True)
            os.makedirs(os.path.join(self._dump_dir, 'debug_visuals_training','valid_render'), exist_ok=True)
            self.logger = biuld_logger(os.path.join(self._dump_dir, 'debug.txt'), name=f'train_{timestamp}')
        # build trainer
        self.lightning_fabric = lightning.Fabric(
            accelerator='cuda', strategy= DDPStrategy(find_unused_parameters=True), devices=devices,
        )
        self.lightning_fabric.launch()
        # loop config
        self._log_interval = 100
        self._total_iters = meta_cfg.TRAIN.train_iter
        self._check_interval = meta_cfg.TRAIN.check_interval if not debug else 200
        self._visual_train_interval = 1000 if not debug else 100

        # training materials
        self.scheduler = scheduler
        self.infer_model, self.optimizer = self.lightning_fabric.setup(infer_model, optimizer,)
        self.loss_model=Optimization_Loss(meta_cfg,).to(self.lightning_fabric.device)

        render_model = self.lightning_fabric.setup(render_model)
        self.render_model=render_model
        self.train_dataloader = self.lightning_fabric.setup_dataloaders(train_dataloader,use_distributed_sampler=self.use_distributed_sampler)
        self.val_dataloader = self.lightning_fabric.setup_dataloaders(val_dataloader,use_distributed_sampler=self.use_distributed_sampler)#use_distributed_sampler=self.use_distributed_sampler

    def run_fit(self, init_iter=1):

        train_render_path=os.path.join(self._dump_dir, 'visuals_training',"train_render")
        if self._debug:
            train_render_path=os.path.join(self._dump_dir, 'debug_visuals_training',"train_render")

        # build bar
        fit_bar = tqdm(range(init_iter, self._total_iters+1)) if self._debug else \
                  rtqdm(range(init_iter, self._total_iters+1))
        train_iter = iter(self.train_dataloader)
        self._set_state(train=True)
        bg=0.0
        for iter_idx in fit_bar:
            # get data and prepare
            try:

                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch_data = next(train_iter)

            # forward
            vertex_gs_dict,uv_point_gs_dict,extra_dict = self.infer_model(batch_data['source'], )

            ubody_gaussians=Ubody_Gaussian(self._meta_cfg.MODEL,vertex_gs_dict,uv_point_gs_dict,pruning=False)
            ubody_gaussians.init_ehm(self.infer_model.ehm)
            deform_gaussian_assets=ubody_gaussians(batch_data['target'])
            #rendering gaussian
            render_results=self.render_model(deform_gaussian_assets,batch_data['target']['render_cam_params'],bg=bg)
            extra_results={'uv_point_xyz':uv_point_gs_dict['local_pos'],'uv_point_scale':uv_point_gs_dict['scales'],
                           'vertices':self.infer_model.smplx_deform_res['vertices'],'uv_point_opacity':uv_point_gs_dict['opacities'],
                           'vertex_opacity':vertex_gs_dict['opacities'],'vertex_scale':vertex_gs_dict['scales'],}

            loss_metrics,show_metric=self.loss_model(render_results,batch_data['target'],extra_results,iter_idx)
            loss = sum(loss_metrics.values())
            self.lightning_fabric.backward(loss)
            for param in self.infer_model.parameters():
                if param.grad is not None: param.grad.nan_to_num_()

            # backward and step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            with torch.no_grad():
                # logger
                self._logger(iter_idx, fit_bar, loss_metrics, show_metric)

                #visual training results
                if iter_idx%self._visual_train_interval==0 :
                    gt_rgb = batch_data['target']['image'].clamp(0, 1).cpu()
                    gt_mask=batch_data['target']['mask'].clamp(0, 1).cpu()
                    pred_rgb = render_results['renders'].clamp(0, 1).cpu()
                    gt_rgb=gt_rgb*gt_mask+(1-gt_mask)*bg

                    if 'raw_renders' in render_results:
                        pred_raw_rgb = render_results['raw_renders'].clamp(0, 1).cpu()
                        pred_rgb=torch.cat([pred_rgb,pred_raw_rgb])
                    source_rgb=batch_data['source']['image'].clamp(0, 1).cpu()
                    gt_rgb[:, :, -150:, -150:] = self._resize(source_rgb, (150, 150))
                    cat_imgs=torch.cat([gt_rgb, pred_rgb])
                    visulize_rgbs = torchvision.utils.make_grid(cat_imgs, nrow=self._meta_cfg.TRAIN.batch_size, padding=0)

                    torchvision.utils.save_image(visulize_rgbs, os.path.join(train_render_path, f'training_{iter_idx}_{self.lightning_fabric.global_rank}.jpg'))
                    del visulize_rgbs,source_rgb,gt_rgb,pred_rgb

                # checkpoints
                if iter_idx % self._check_interval == 0 or iter_idx == self._total_iters:
                    self._set_state(train=False)
                    self.run_val(iter_idx)

                    self._save_checkpoints('latest.pt',iter_idx)
                    self._set_state(train=True)
                if iter_idx==100000:
                    self._save_checkpoints('100000.pt',iter_idx)

                del batch_data,loss_metrics, show_metric,extra_results,render_results,ubody_gaussians,\
                    vertex_gs_dict,uv_point_gs_dict,deform_gaussian_assets
                if iter_idx%(50//self._meta_cfg.TRAIN.batch_size)==0 : torch.cuda.empty_cache() #20

    @torch.no_grad()
    def run_val(self, iter_idx, save_ckpt=True):
        val_iter = iter(self.val_dataloader)
        _valid_renderation_outputs = []
        bg=self._meta_cfg.MODEL.bg_color

        for idx, batch_data in enumerate(val_iter):
            vertex_gs_dict,up_point_gs_dict,_ = self.infer_model(batch_data['source'], )
            ubody_gaussians=Ubody_Gaussian(self._meta_cfg.MODEL,vertex_gs_dict,up_point_gs_dict)
            ubody_gaussians.init_ehm(self.infer_model.ehm)
            deform_gaussian_assets=ubody_gaussians(batch_data['target'])
            render_results=self.render_model(deform_gaussian_assets,batch_data['target']['render_cam_params'],bg=bg)

            gt_rgb = batch_data['target']['image'].clamp(0, 1).cpu()
            gt_mask=batch_data['target']['mask'].clamp(0, 1).cpu()
            pred_rgb = render_results['renders'].clamp(0, 1).cpu()
            gt_rgb=gt_rgb*gt_mask+(1-gt_mask)*bg

            psnr = float(cal_psnr(pred_rgb, gt_rgb).mean())
            ssim = float(cal_ssim(pred_rgb, gt_rgb).mean())
            source_rgb=batch_data['source']['image'].clamp(0, 1).cpu()
            # visulize
            gt_rgb[:, :, -150:, -150:] = self._resize(source_rgb, (150, 150))
            if 'raw_renders' in render_results:
                pred_raw_rgb = render_results['raw_renders'].clamp(0, 1).cpu()
                pred_rgb=torch.cat([pred_rgb,pred_raw_rgb])

            visulize_rgbs = torchvision.utils.make_grid(torch.cat([gt_rgb, pred_rgb]), nrow=3, padding=0)
            visulize_rgbs = self._resize(visulize_rgbs, 256)
            _valid_renderation_outputs.append({'PSNR': psnr, 'SSIM': ssim, 'Image': visulize_rgbs})#'LPIPS':lpips,
        merged_images = torchvision.utils.make_grid(
            torch.stack([r['Image'] for r in _valid_renderation_outputs[:15]]), nrow=3, padding=0
        )
        merged_psnr = np.mean([r['PSNR'] for r in _valid_renderation_outputs])
        merged_ssim = np.mean([r['SSIM'] for r in _valid_renderation_outputs])

        local_merged_ssims = torch.tensor(merged_ssim,device=self.lightning_fabric.device)
        local_merged_psnr = torch.tensor(merged_psnr,device=self.lightning_fabric.device)

        merged_ssim = self.lightning_fabric.all_reduce(local_merged_ssims,reduce_op="mean")
        merged_psnr = self.lightning_fabric.all_reduce(local_merged_psnr,reduce_op="mean")
        merged_ssim=merged_ssim.cpu().item()
        merged_psnr=merged_psnr.cpu().item()
        log_str = 'Step: {:05d} / {}, \tPSNR: {:.4f}, \tSSIM: {:.4f}. '.format(
            iter_idx, self._total_iters, merged_psnr, merged_ssim,
        )
        self.logger.debug(log_str)
        if save_ckpt:
            self._save_valid_renderation(iter_idx, merged_ssim, merged_images, log_str, larger_best=True)

        del _valid_renderation_outputs

    def _save_checkpoints(self, name='latest.pt',iter_idx=1, optimizer=False):
        if self._debug:
            return
        saving_path = os.path.join(self._dump_dir, 'checkpoints')
        # remove old best model
        try:
            if name.startswith('best'):
                models = os.listdir(saving_path)
                for m in models:
                    if m.startswith('best'):
                        os.remove(os.path.join(saving_path, m))
        except:
            pass
        state = {'model': self.infer_model, 'meta_cfg': self._meta_cfg._dump,'global_iter':iter_idx,
                 'render_model':self.render_model}
        if optimizer:
            state['optimizer'] = self.optimizer
        self.lightning_fabric.save(os.path.join(saving_path, name), state)
        self.logger.debug('Model saved at {}.'.format(os.path.join(saving_path, name)))

    def _save_valid_renderation(self, iter_idx, metric, images, log_string, larger_best=True):
        if self._debug:
            valid_renderation_path = os.path.join(self._dump_dir,'debug_visuals_training','valid_render', f'debug_{iter_idx}.jpg')
        else:
            valid_renderation_path = os.path.join(self._dump_dir, 'visuals_training','valid_render', f'{iter_idx}_{self.lightning_fabric.global_rank}.jpg')
        torchvision.utils.save_image(images, valid_renderation_path)
        best_path = 'best_{}_{:.3f}.pt'.format(iter_idx, metric)

        if self._best_metric is None:
            self._best_metric = metric
            self._save_checkpoints(best_path,iter_idx)
        else:
            if larger_best:
                if metric >= self._best_metric:
                    self._best_metric = metric

                    self._save_checkpoints(best_path,iter_idx)
            else:
                if metric <= self._best_metric:
                    self._best_metric = metric
                    self._save_checkpoints(best_path,iter_idx)


    def _logger(self, iter_idx, fit_bar, loss_metrics, show_metric):
        if not hasattr(self, 'log_stats'):
            self.log_stats, self.show_stats = [], []
        # build fit bar and file log
        learning_rate = self.optimizer.param_groups[0]['lr']
        loss_metrics = torch.utils._pytree.tree_map(lambda x: x.item(), loss_metrics)
        self.log_stats.append(loss_metrics); self.show_stats.append(show_metric)
        self.log_stats = self.log_stats[-100:]; self.show_stats = self.show_stats[-100:]
        show_metric = self._dict_mean(self.show_stats)
        show_loss = sum([float(loss_metrics[k]) for k in loss_metrics])
        fit_bar.set_postfix({'loss': "{:.4f}".format(show_loss), **show_metric})
        if iter_idx % self._log_interval == 0:
            log_metric = self._dict_mean(self.log_stats, "{:.4f}")
            log_loss = sum([float(log_metric[k]) for k in log_metric])

            log_string =  "{:05d} / {}: ".format(iter_idx, self._total_iters) + \
                            "lr={:.5f}, loss={:.4f} | ".format(learning_rate, log_loss) + \
                            ", ".join([f'{k}={v}' for k, v in log_metric.items()])
            if self._debug:
                self.logger.info(log_string)
            else:
                self.logger.debug(log_string)

    @staticmethod
    def _resize(frames, tgt_size=(256, 256)):
        if isinstance(tgt_size, torch.Tensor):
            tgt_size = (tgt_size.shape[-2], tgt_size.shape[-1])
        if frames.shape[-2:] == tgt_size:
            return frames
        else:
            frames = torchvision.transforms.functional.resize(
                frames, tgt_size, antialias=True
            )
            return frames

    @staticmethod
    def _dict_mean(dict_list, float_format='{:.2f}'):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = float_format.format(np.mean([d[key] for d in dict_list]))
        return mean_dict

    def _save_codes(self,timestamp):
        target_folder=os.path.join(self._dump_dir,"codes",f"{timestamp}_GubodyAvatar")
        source_folder='.'
        for root, _, files in os.walk(source_folder):
            if 'outputs' in root.split(os.sep) or 'assets' in root.split(os.sep):
                continue
            for file in files:
                if file.endswith(".py") or file.endswith(".yaml"):
                    source_file_path = os.path.join(root, file)

                    relative_path = os.path.relpath(root, source_folder)
                    target_subdir = os.path.join(target_folder, relative_path)

                    os.makedirs(target_subdir, exist_ok=True)

                    target_file_name = f"{file}"
                    target_file_path = os.path.join(target_subdir, target_file_name)

                    shutil.copy(source_file_path, target_file_path)
    def _set_state(self,train=True):
        if train:
            self.infer_model.train()
            self.render_model.train()
        else:
            self.infer_model.eval()
            self.render_model.eval()
