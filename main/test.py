import os
import torch
import argparse
import lightning
import numpy as np
import torchvision
from tqdm import tqdm
import time,json,imageio,copy
from dataset import TrackedData_infer
from models.UbodyAvatar import Ubody_Gaussian_inferer,Ubody_Gaussian,GaussianRenderer
from utils.general_utils import (
    ConfigDict, device_parser,
    calc_parameters, find_pt_file,add_extra_cfgs
)
import torchvision
from utils.general_utils import to8b
from omegaconf import OmegaConf
from utils.camera_utils import generate_novel_view_poses


def change_id_info(target_info,source_info):
    target_info=copy.deepcopy(target_info)
    target_info['smplx_coeffs']['shape']=source_info['smplx_coeffs']['shape']
    target_info['smplx_coeffs']['joints_offset']=source_info['smplx_coeffs']['joints_offset']
    target_info['smplx_coeffs']['head_scale']=source_info['smplx_coeffs']['head_scale']
    target_info['smplx_coeffs']['hand_scale']=source_info['smplx_coeffs']['hand_scale']
    target_info['flame_coeffs']['shape_params']=source_info['flame_coeffs']['shape_params']
    return target_info

def render_set(meta_cfg,infer_model:Ubody_Gaussian_inferer,render_model:GaussianRenderer,dataset:TrackedData_infer,dataset_name:str,root_path:str,):
    out_dir=os.path.join(root_path,dataset_name,)
    os.makedirs(out_dir,exist_ok=True)
    bg=0.0
    video_ids=list(dataset.videos_info.keys())

    for vidx,video_id in enumerate(video_ids):
        print(f'{video_id} [{vidx+1}/{len(video_ids)}]')
        out_videoid_dir=os.path.join(out_dir,video_id)
        out_render_path=os.path.join(out_videoid_dir,'render')
        out_gt_path=os.path.join(out_videoid_dir,'gt')
        os.makedirs(out_render_path,exist_ok=True)
        os.makedirs(out_gt_path,exist_ok=True)
        speed_info={}
        source_info=dataset._load_source_info(video_id)

        #warmming up
        vertex_gs_dict,up_point_gs_dict,_ = infer_model(source_info, )
        start_time = time.time()
        vertex_gs_dict,up_point_gs_dict,_ = infer_model(source_info, )
        infer_time = time.time() - start_time

        ubody_gaussians=Ubody_Gaussian(meta_cfg.MODEL,vertex_gs_dict,up_point_gs_dict,pruning=True)
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

        #saving ply
        ubody_gaussians.save_point_ply(out_videoid_dir)
        ubody_gaussians.save_gaussian_ply(out_videoid_dir)

        frames=dataset.videos_info[video_id]['frames_keys']
        all_render_time=0.0
        test_num=dataset.testing_split[video_id]
        rendering_imgs=[]

        #warmming up
        target_info=dataset._load_target_info(video_id,frames[0])
        deform_gaussian_assets=ubody_gaussians(target_info)
        render_results=render_model(deform_gaussian_assets,target_info['render_cam_params'],bg=bg)

        for idx,frame in tqdm(enumerate(frames[-test_num:])) :
            target_info=dataset._load_target_info(video_id,frame)
            start_time = time.time()
            deform_gaussian_assets=ubody_gaussians(target_info)
            render_results=render_model(deform_gaussian_assets,target_info['render_cam_params'],bg=bg)
            render_time=time.time() - start_time
            all_render_time+=render_time

            render_image=render_results['renders'][0]
            gt_mask=target_info['mask'][0]
            gt_image=target_info['image'][0]*(gt_mask)+(1-gt_mask)*bg
            torchvision.utils.save_image(gt_image, os.path.join(out_gt_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(render_image, os.path.join(out_render_path, '{0:05d}'.format(idx) + ".png"))

            cat_image=torch.cat([gt_image,render_image],dim=2)
            rendering_imgs.append(to8b(cat_image.detach().cpu().numpy()))

        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(out_videoid_dir, f'{video_id}_video.mp4'), rendering_imgs, fps=30, quality=8)

        render_speed=test_num/all_render_time #fps
        speed_info['infer_time (ms)']=infer_time*1000
        speed_info['render_speed (fps)']=render_speed
        with open(os.path.join(out_videoid_dir,'speed_info.json'), 'w') as f:
            json.dump(speed_info, f)

def render_cross_set(meta_cfg,infer_model:Ubody_Gaussian_inferer,render_model:GaussianRenderer,source_dataset:TrackedData_infer,target_dataset:TrackedData_infer,dataset_name:str,root_path:str,args):
    out_dir=os.path.join(root_path,dataset_name,)
    os.makedirs(out_dir,exist_ok=True)
    bg=0.0

    s_video_ids=list(source_dataset.videos_info.keys())
    t_video_ids=list(target_dataset.videos_info.keys())
    for s_vidx,s_video_id in enumerate(s_video_ids):
        source_info=source_dataset._load_source_info(s_video_id,)
        source_info['render_cam_params']=source_dataset._load_target_info(s_video_id,source_dataset.videos_info[s_video_id]['frames_keys'][0])
        # Warm up source inference, matching self-reenactment timing protocol.
        vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)

        start_time = time.time()
        vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
        infer_time = time.time() - start_time
        ubody_gaussians=Ubody_Gaussian(meta_cfg.MODEL,vertex_gs_dict,up_point_gs_dict,pruning=True)
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

        out_sub_dir=os.path.join(out_dir,s_video_id)
        os.makedirs(out_sub_dir,exist_ok=True)
        for t_vidx,t_video_id in enumerate(t_video_ids):
            print(f'{t_video_id} [{t_vidx+1}/{len(t_video_ids)}]')
            out_videoid_dir=os.path.join(out_sub_dir,f'{s_video_id}_{t_video_id}')
            out_render_path=os.path.join(out_videoid_dir,'render')
            os.makedirs(out_render_path,exist_ok=True)
            torchvision.utils.save_image(source_info['image'], os.path.join(out_videoid_dir,'source_image.png'))
            frames=target_dataset.videos_info[t_video_id]['frames_keys']

            test_num=target_dataset.testing_split[t_video_id]

            if test_num <= 0:
                raise RuntimeError(
                    f'Invalid testing split ({test_num}) for cross pair '
                    f'{s_video_id} -> {t_video_id}'
                )

            target_frames = list(frames[-test_num:])

            if len(target_frames) == 0:
                raise RuntimeError(
                    f'No target frames available for cross pair '
                    f'{s_video_id} -> {t_video_id}'
                )

            all_render_time = 0.0
            speed_info = {}

            metadata = {
                'source_video_id': str(s_video_id),
                'target_video_id': str(t_video_id),
                'source_data_path': str(args.source_data_path),
                'target_data_path': str(args.data_path),
                'num_target_frames': len(target_frames),
                'target_frame_keys': [str(frame) for frame in target_frames],
                'keep_source_cam': bool(args.keep_source_cam),
            }

            with open(os.path.join(out_videoid_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Warm up cross deformation + rendering, matching self-reenactment.
            warmup_info = target_dataset._load_target_info(
                t_video_id,
                target_frames[0]
            )
            warmup_info = change_id_info(warmup_info, source_info)

            if args.keep_source_cam:
                warmup_cam_params = source_info['render_cam_params']
            else:
                warmup_cam_params = warmup_info['render_cam_params']

            warmup_gaussian_assets = ubody_gaussians(warmup_info)
            _ = render_model(
                warmup_gaussian_assets,
                warmup_cam_params,
                bg=bg
            )
            rendering_imgs=[]
            for idx,frame in tqdm(enumerate(target_frames)) :
                target_info=target_dataset._load_target_info(t_video_id,frame)
                target_info=change_id_info(target_info,source_info)

                if args.keep_source_cam:
                    render_cam_parms=source_info['render_cam_params']
                else: render_cam_parms=target_info['render_cam_params']

                start_time = time.time()
                deform_gaussian_assets=ubody_gaussians(target_info)
                render_results=render_model(deform_gaussian_assets,render_cam_parms,bg=bg)

                render_time = time.time() - start_time
                all_render_time += render_time
                render_image=render_results['renders'][0]
                gt_mask=target_info['mask'][0]
                torchvision.utils.save_image(render_image, os.path.join(out_render_path, '{0:05d}'.format(idx) + ".png"))
                rendering_imgs.append(to8b(render_image.detach().cpu().numpy()))

            rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(out_videoid_dir, f'{s_video_id}_{t_video_id}_video.mp4'), rendering_imgs, fps=30, quality=8)
            render_speed = len(target_frames) / all_render_time

            speed_info['infer_time (ms)'] = infer_time * 1000
            speed_info['render_speed (fps)'] = render_speed

            with open(
                os.path.join(out_videoid_dir, 'speed_info.json'),
                'w'
            ) as f:
                json.dump(speed_info, f, indent=2)


def render_novel_views(meta_cfg,infer_model:Ubody_Gaussian_inferer,render_model:GaussianRenderer,dataset:TrackedData_infer,dataset_name:str,root_path:str,):
    #render norvel views for self-reenactment
    out_dir=os.path.join(root_path,dataset_name,)
    os.makedirs(out_dir,exist_ok=True)
    bg=0.0
    num_keyframes=120
    video_ids=list(dataset.videos_info.keys())

    for vidx,video_id in enumerate(video_ids):
        print(f'{video_id} [{vidx+1}/{len(video_ids)}]')
        out_videoid_dir=os.path.join(out_dir,f"{video_id}")
        out_render_path=os.path.join(out_videoid_dir,'render')
        os.makedirs(out_render_path,exist_ok=True)
        source_info=dataset._load_source_info(video_id)

        vertex_gs_dict,up_point_gs_dict,_ = infer_model(source_info, )
        novel_cam_params=generate_novel_view_poses(source_info,image_size=dataset.image_size,tanfov=dataset.tanfov,num_keyframes=num_keyframes)
        ubody_gaussians=Ubody_Gaussian(meta_cfg.MODEL,vertex_gs_dict,up_point_gs_dict,pruning=True)
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

        frames=dataset.videos_info[video_id]['frames_keys']
        rendering_imgs=[]
        for idx,frame in tqdm(enumerate(frames)) :

            target_info=dataset._load_target_info(video_id,frame)
            deform_gaussian_assets=ubody_gaussians(target_info)
            render_cam_param=novel_cam_params[idx%num_keyframes]
            render_results=render_model(deform_gaussian_assets,render_cam_param,bg=bg)

            render_image=render_results['renders'][0]
            gt_mask=target_info['mask'][0]
            torchvision.utils.save_image(render_image, os.path.join(out_render_path, '{0:05d}'.format(idx) + ".png"))
            rendering_imgs.append(to8b(render_image.detach().cpu().numpy()))

        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(out_videoid_dir, f'{video_id}_dynamic_novel_views_video.mp4'), rendering_imgs, fps=30, quality=8)

def render_static_novel_views(meta_cfg,infer_model:Ubody_Gaussian_inferer,render_model:GaussianRenderer,dataset:TrackedData_infer,dataset_name:str,root_path:str,args):
    #render norvel views in static mode
    out_dir=os.path.join(root_path,dataset_name,)
    os.makedirs(out_dir,exist_ok=True)
    bg=0.0
    num_keyframes=360
    video_ids=list(dataset.videos_info.keys())

    for vidx,video_id in enumerate(video_ids):
        print(f'{video_id} [{vidx+1}/{len(video_ids)}]')
        for ridx in tqdm(args.render_snovel_idx):

            frame_key=dataset.videos_info[video_id]['frames_keys'][ridx]
            out_videoid_dir=os.path.join(out_dir,frame_key,f"{video_id}")
            out_render_path=os.path.join(out_videoid_dir,'render')
            os.makedirs(out_render_path,exist_ok=True)
            source_info=dataset._load_source_info(video_id)

            vertex_gs_dict,up_point_gs_dict,_ = infer_model(source_info, )
            novel_cam_params=generate_novel_view_poses(source_info,image_size=dataset.image_size,tanfov=dataset.tanfov,num_keyframes=num_keyframes)
            ubody_gaussians=Ubody_Gaussian(meta_cfg.MODEL,vertex_gs_dict,up_point_gs_dict,pruning=True)
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()

            target_info=dataset._load_target_info(video_id,frame_key)
            deform_gaussian_assets=ubody_gaussians(target_info)

            rendering_imgs=[]
            for idx in tqdm(range(num_keyframes)) :
                render_cam_param=novel_cam_params[idx%num_keyframes]
                render_results=render_model(deform_gaussian_assets,render_cam_param,bg=bg)
                render_image=render_results['renders'][0]
                gt_mask=target_info['mask'][0]
                torchvision.utils.save_image(render_image, os.path.join(out_render_path, '{0:05d}'.format(idx) + ".png"))
                rendering_imgs.append(to8b(render_image.detach().cpu().numpy()))

            rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(out_videoid_dir, f'{video_id}_{ridx}_static_novel_views_video.mp4'), rendering_imgs, fps=30, quality=8)


def test(args,config_name, base_model, devices,data_path,model_path,save_path,out_name='test'):
    if config_name is None:
        model_config_path=os.path.join(f'{model_path}', f'config.yaml')
    else:
        model_config_path=os.path.join('configs/train', f'{config_name}.yaml')
    meta_cfg = ConfigDict(
        model_config_path=model_config_path
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)
    target_devices = device_parser(devices)
    device=f'cuda:{target_devices[0]}'
    meta_cfg=copy.deepcopy(meta_cfg)
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    infer_model.eval()
    render_model.eval()

    op_para_num, all_para_num = calc_parameters([infer_model,render_model])
    print('Number of parameters: {:.2f}M.'.format(all_para_num/1000000))

    if base_model is None:
        ckpt_ptah=os.path.join(model_path,'checkpoints')
        base_model=find_pt_file(ckpt_ptah,'best')
        if base_model is None:
            base_model=find_pt_file(ckpt_ptah,'latest')

    assert os.path.exists(base_model), f'Base model not found: {base_model}.'
    _state=torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    render_model.load_state_dict(_state['render_model'], strict=False)
    print('Load model from: {}.'.format(base_model))

    # load dataset
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path']=data_path
    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test',device=device,test_full=args.test_full)

    print(f'Test Dataset: {len(test_dataset)}.')
    with torch.no_grad():
        if not args.skip_self_act:
            print('Rendering self-reenactment')
            render_set(meta_cfg,infer_model,render_model,test_dataset,f'{out_name}_self_act',save_path)
        if args.render_dynamic_novel_views:
            print('Rendering dynamic novel views')
            render_novel_views(meta_cfg,infer_model,render_model,test_dataset,f'{out_name}_dyn_novel_views',save_path)
        if args.render_static_novel_views:
            print('Rendering static novel views')
            render_static_novel_views(meta_cfg,infer_model,render_model,test_dataset,f'{out_name}_sta_novel_views',save_path,args)

        if args.render_cross_act:
            print('Rendering cross-reenactment')
            meta_cfg['DATASET']['data_path']=args.source_data_path
            source_dataset=TrackedData_infer(cfg=meta_cfg, split='test',device=device,test_full=True)
            render_cross_set(meta_cfg,infer_model,render_model,source_dataset,test_dataset,f'{out_name}_cross_act',save_path,args)
            source_dataset._lmdb_engine.close()

    test_dataset._lmdb_engine.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', default=None, type=str)

    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--basemodel','-b', default=None, type=str)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--model_path','-m',type=str)
    parser.add_argument('--save_path','-s',type=str,default=None)
    parser.add_argument('--saving_name','-n',type=str,default='render')
    parser.add_argument('--non_test_full', action='store_true', default=False)

    parser.add_argument('--skip_self_act', action='store_true', default=False)
    parser.add_argument('--render_dynamic_novel_views', action='store_true', default=False)
    parser.add_argument('--render_static_novel_views', action='store_true', default=False)
    parser.add_argument('--render_snovel_idx', nargs='+', type=int, default=[0])

    parser.add_argument('--render_cross_act', action='store_true', default=False)
    parser.add_argument('--keep_source_cam', action='store_true', default=False)
    parser.add_argument('--source_data_path', type=str, default=None,help='source info for cross_reenactment')

    args = parser.parse_args()
    args.test_full = not args.non_test_full
    print("Command Line Args: {}".format(args))
    # launch
    torch.set_float32_matmul_precision('high')
    if args.render_cross_act:
        assert args.source_data_path is not None
    if args.save_path is None:
        args.save_path=args.model_path
    test(args,args.config_name, args.basemodel, args.devices, args.data_path,args.model_path,args.save_path,args.saving_name)