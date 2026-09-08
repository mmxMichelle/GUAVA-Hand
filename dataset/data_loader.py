import os
import json
import torch
import pickle
import random
import numpy as np
import torchvision
from copy import deepcopy
import time
import torch.utils.data._utils.collate as collate

from utils.lmdb import LMDBEngine
from utils.graphics_utils import get_full_proj_matrix

class TrackedData(torch.utils.data.Dataset):
    def __init__(self, cfg,split):
        super().__init__()

        self.cfg=cfg
        self.bg_color=0.0
        self.feature_img_size=cfg.MODEL.feature_img_size #img size for extrating feature map 14*xx
        self.feature_part_size=cfg.MODEL.feature_part_size
        self.image_size=cfg.MODEL.image_size
        self.tanfov,self.focal=1/cfg.MODEL.invtanfov,cfg.MODEL.invtanfov

        self.split = split
        assert self.split in ['train', 'valid', 'test'], f'Invalid split: {self.split}'
        self.data_path = cfg.DATASET.data_path
        self.traked_data=load_dict_pkl(os.path.join(self.data_path, 'optim_tracking_ehm.pkl'))

        self.traked_data_id_share=load_dict_pkl(os.path.join(self.data_path, 'id_share_params.pkl'))
        with open(os.path.join(self.data_path, 'videos_info.json'), 'r') as f:
            self.videos_info = json.load(f)

        if self.split in ['train', 'valid']:
            with open(os.path.join(self.data_path, 'dataset_frames.json'), 'r') as f:
                self.frames = json.load(f)[self.split]
        else:
            self.frames = []
            for video_id in self.videos_info.keys():
                self.frames += [f'{video_id}/{frame_id}' for frame_id in self.videos_info[video_id]['frames_keys']]


    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(os.path.join(self.data_path, 'img_lmdb'), write=False)

    def slice(self, slice):
        self.frames = self.frames[:slice]

    def shuffle_slice(self, slice_num):
        random.seed(time.time())
        random.shuffle(self._frames)
        self._frames = self._frames[:slice_num]

    def __getitem__(self, index):

        whole_frame_key = self.frames[index]
        return self._load_one_record(whole_frame_key)

    def __len__(self, ):
        return len(self.frames)

    def _choose_image(self, video_id,frame_key, number=1):
        assert number==1

        if self.split == 'train':
            candidate_key = [key for key in self.videos_info[video_id]['frames_keys'] if key != frame_key]
            source_key = random.sample(candidate_key, k=number)[0]
        else:
            source_key = self.videos_info[video_id]['frames_keys'][0]

        tracking_info,source_image,source_mask=self._load_one_info(video_id,source_key)

        return  tracking_info,source_image,source_mask

    def _load_one_record(self, whole_frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()

        # source image for extrating features
        video_id,frame_key = get_video_info(whole_frame_key)
        source_tracking_info,source_image,source_mask = self._choose_image(video_id,frame_key)
        source_image=source_image*source_mask
        source_image = torchvision.transforms.functional.resize(source_image, (self.feature_img_size, self.feature_img_size), antialias=True)

        # target image
        traget_tracking_info,target_image,target_mask=self._load_one_info(video_id,frame_key)
        target_image  = torchvision.transforms.functional.resize(target_image, (self.image_size, self.image_size), antialias=True)
        target_mask  = torchvision.transforms.functional.resize(target_mask, (self.image_size, self.image_size), antialias=True)
        #rendering cam_params
        view_matrix,full_proj_matrix=get_full_proj_matrix(traget_tracking_info['w2c_cam'],self.tanfov)
        traget_tracking_info['render_cam_params']={
            "world_view_transform":view_matrix,"full_proj_transform":full_proj_matrix,
            'tanfovx':self.tanfov,'tanfovy':self.tanfov,
            'image_height':self.image_size,'image_width':self.image_size,
            'camera_center':traget_tracking_info['c2w_cam'][:3,3]
        }

        one_record = {'source':{'image':source_image,},'target':{'image':target_image,'mask':target_mask}}
        one_record['source'].update(source_tracking_info)
        one_record['target'].update(traget_tracking_info)
        return one_record


    def _load_one_info(self,video_id,frame_key):
        _image = self._lmdb_engine[f'{video_id}/{frame_key}/body_image'].float() / 255.0
        _mask = self._lmdb_engine[f'{video_id}/{frame_key}/body_mask'].float() / 255.0

        tracking_info = deepcopy(self.traked_data[video_id][frame_key])

        tracking_info['smplx_coeffs'].update({"shape":self.traked_data_id_share[video_id]['smplx_shape'][0],
                                              "joints_offset":self.traked_data_id_share[video_id]['joints_offset'][0],
                                              "head_scale":self.traked_data_id_share[video_id]['head_scale'][0],
                                              "hand_scale":self.traked_data_id_share[video_id]['hand_scale'][0],
                                              })
        tracking_info['flame_coeffs'].update({"shape_params":self.traked_data_id_share[video_id]['flame_shape'][0]})

        tracking_info=data_to_tensor(tracking_info)
        tracking_info=squeeze_params(tracking_info)

        # Convert PyTorch 3D coordinate system to the COLMAP coordinate system.
        #(Since it is identical to the image coordinate, the same camera parameters
        # are employed for unprojection and Gaussian rendering.)
        RT=tracking_info['smplx_coeffs']['camera_RT_params']
        c2c_mat=torch.tensor([[-1, 0, 0, 0],
                              [ 0,-1, 0, 0],
                              [ 0, 0, 1, 0],
                              [ 0, 0, 0, 1],
                              ],dtype=torch.float32)
        RT_mat=torch.tensor([[  1, 0, 0, 0],
                              [ 0, 1, 0, 0],
                              [ 0, 0, 1, 0],
                              [ 0, 0, 0, 1],
                              ],dtype=torch.float32)
        RT_mat[:3,:4]=RT
        w2c_cam=torch.matmul(c2c_mat,RT_mat)
        c2w_cam=torch.linalg.inv(w2c_cam)
        tracking_info['w2c_cam'],tracking_info['c2w_cam']=w2c_cam,c2w_cam
        tracking_info['head_box'],tracking_info['left_hand_box'],tracking_info['right_hand_box']=self._load_box(tracking_info)

        return tracking_info,_image,_mask

    def _load_box(self,tracking_info):
        crop_info={'body_crop':tracking_info['body_crop'],'head_crop':tracking_info['head_crop'],
                   'left_hand_crop':tracking_info['left_hand_crop'],'right_hand_crop':tracking_info['right_hand_crop']}
        crop_info=data_to_tensor(crop_info)
        scale=self.cfg.MODEL.image_size/self.cfg.DATASET.origin_image_size
        image_size=self.cfg.MODEL.image_size
        head_crop_size=self.cfg.DATASET.head_crop_size
        hand_crop_size=self.cfg.DATASET.hand_crop_size

        head_box_o=torch.tensor([[0.0,0.0,1.0],[head_crop_size,0.0,1.0],[0.0,head_crop_size,1.0],[head_crop_size,head_crop_size,1.0]])#x,y
        hand_box_o=torch.tensor([[0.0,0.0,1.0],[hand_crop_size,0.0,1.0],[0.0,hand_crop_size,1.0],[hand_crop_size,hand_crop_size,1.0]])#x,y

        body_crop=crop_info['body_crop']
        head_crop=crop_info['head_crop']
        left_hand_crop=crop_info['left_hand_crop']
        right_hand_crop=crop_info['right_hand_crop']

        head_box=body_crop['M_o2c-hd']@head_crop['M_c2o']@head_box_o[:,:,None]
        left_hand_box=body_crop['M_o2c-hd']@left_hand_crop['M_c2o']@hand_box_o[:,:,None]
        right_hand_box=body_crop['M_o2c-hd']@right_hand_crop['M_c2o']@hand_box_o[:,:,None]
        head_box*=scale
        left_hand_box*=scale
        right_hand_box*=scale
        head_box = head_box.clamp(0, image_size - 1)
        left_hand_box = left_hand_box.clamp(0, image_size - 1)
        right_hand_box = right_hand_box.clamp(0, image_size - 1)


        head_left,head_right=int(head_box.min(dim=0)[0][0]),int(head_box.max(dim=0)[0][0])
        head_top,head_bottom=int(head_box.min(dim=0)[0][1]),int(head_box.max(dim=0)[0][1])

        left_hand_left,left_hand_right=int(left_hand_box.min(dim=0)[0][0]),int(left_hand_box.max(dim=0)[0][0])
        left_hand_top,left_hand_bottom=int(left_hand_box.min(dim=0)[0][1]),int(left_hand_box.max(dim=0)[0][1])
        right_hand_left,right_hand_right=int(right_hand_box.min(dim=0)[0][0]),int(right_hand_box.max(dim=0)[0][0])
        right_hand_top,right_hand_bottom=int(right_hand_box.min(dim=0)[0][1]),int(right_hand_box.max(dim=0)[0][1])
        #left right top bottom
        head_box=torch.tensor([head_left,head_right,head_top,head_bottom],dtype=torch.long)
        left_hand_box=torch.tensor([left_hand_left,left_hand_right,left_hand_top,left_hand_bottom],dtype=torch.long)
        right_hand_box=torch.tensor([right_hand_left,right_hand_right,right_hand_top,right_hand_bottom],dtype=torch.long)
        if head_box[0]==head_box[1] or head_box[2]==head_box[3]:
            head_box=torch.tensor([0,image_size-1,0,image_size-1],dtype=torch.long)

        return head_box,left_hand_box,right_hand_box

    def _crop_image_part(self,image,box):
        if image.shape[-1]!=self.image_size:
            image=torchvision.transforms.functional.resize(image, (self.image_size, self.image_size), antialias=True)
        img_part=image[ :, box[ 2]:box[ 3], box[ 0]:box[ 1]]
        center_x=(box[0]+box[1])/2
        center_y=(box[2]+box[3])/2
        x_scale=self.image_size/(box[1]-box[0])
        y_scale=self.image_size/(box[3]-box[2])
        x_offset=center_x*2/self.image_size-1 #(cx-w/2)/(w/2)=cx*2/w-1
        y_offset=center_y*2/self.image_size-1
        transform_info=torch.tensor([x_scale,y_scale,x_offset,y_offset],dtype=torch.float32)
        img_part=torchvision.transforms.functional.resize(img_part, (self.feature_part_size, self.feature_part_size), antialias=True)
        return img_part,transform_info


#for inference
class TrackedData_infer(TrackedData):
    def __init__(self, cfg,split,device,test_full=False):
        super().__init__(cfg,split)
        #number of frames for testing of each video
        testing_split_path=os.path.join(self.data_path, 'testing_split.json')
        if os .path.exists(testing_split_path) and not test_full:
            with open(testing_split_path, 'r') as f:
                self.testing_split = json.load(f)
        else:
            self.testing_split = {}
            for video_id in self.videos_info.keys():
                self.testing_split[video_id] = self.videos_info[video_id]['frames_num']
        self.device=device

    def _load_source_info(self,video_id,key_idx=0):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        collate_fun=collate.default_collate
        source_key = self.videos_info[video_id]['frames_keys'][key_idx]
        tracking_info,source_image,source_mask=self._load_one_info(video_id,source_key)
        source_image=source_image*source_mask#+(1-source_mask)*self.bg_color
        source_image = torchvision.transforms.functional.resize(source_image, (self.feature_img_size, self.feature_img_size), antialias=True)

        source_info={'image':source_image,}
        source_info.update(tracking_info)
        source_info=collate_fun([source_info])
        source_info=self._move_to_device(source_info, self.device)
        return source_info

    def _load_target_info(self,video_id,frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        collate_fun=collate.default_collate
        tracking_info,target_image,target_mask=self._load_one_info(video_id,frame_key)
        target_image  = torchvision.transforms.functional.resize(target_image, (self.image_size, self.image_size), antialias=True)
        target_mask  = torchvision.transforms.functional.resize(target_mask, (self.image_size, self.image_size), antialias=True)
        #rendering cam_params
        view_matrix,full_proj_matrix=get_full_proj_matrix(tracking_info['w2c_cam'],self.tanfov)
        tracking_info['render_cam_params']={
            "world_view_transform":view_matrix,"full_proj_transform":full_proj_matrix,
            'tanfovx':self.tanfov,'tanfovy':self.tanfov,
            'image_height':self.image_size,'image_width':self.image_size,
            'camera_center':tracking_info['c2w_cam'][:3,3]
        }
        target_info={'image':target_image,'mask':target_mask}
        target_info.update(tracking_info)
        target_info=collate_fun([target_info])
        target_info=self._move_to_device(target_info, self.device)
        return target_info

    def _load_one_info(self,video_id,frame_key):

        if video_id in self.traked_data_id_share.keys():
            return super()._load_one_info(video_id,frame_key)
        else:
            _image = self._lmdb_engine[f'{frame_key}/body_image'].float() / 255.0
            _mask = self._lmdb_engine[f'{frame_key}/body_mask'].float() / 255.0

            tracking_info = deepcopy(self.traked_data[frame_key])

            tracking_info['smplx_coeffs'].update({"shape":self.traked_data_id_share['smplx_shape'][0],
                                                "joints_offset":self.traked_data_id_share['joints_offset'][0],
                                                "head_scale":self.traked_data_id_share['head_scale'][0],
                                                "hand_scale":self.traked_data_id_share['hand_scale'][0],
                                                })
            tracking_info['flame_coeffs'].update({"shape_params":self.traked_data_id_share['flame_shape'][0]})

            tracking_info=data_to_tensor(tracking_info)
            tracking_info=squeeze_params(tracking_info)

            # Convert PyTorch 3D coordinate system to the COLMAP coordinate system.
            #(Since it is identical to the image coordinate, the same camera parameters
            # are employed for unprojection and Gaussian rendering.)
            RT=tracking_info['smplx_coeffs']['camera_RT_params']
            c2c_mat=torch.tensor([[-1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1],
                                ],dtype=torch.float32)
            RT_mat=torch.tensor([[  1, 0, 0, 0],
                                [ 0, 1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1],
                                ],dtype=torch.float32)
            RT_mat[:3,:4]=RT
            w2c_cam=torch.matmul(c2c_mat,RT_mat)
            c2w_cam=torch.linalg.inv(w2c_cam)
            tracking_info['w2c_cam'],tracking_info['c2w_cam']=w2c_cam,c2w_cam
            tracking_info['head_box'],tracking_info['left_hand_box'],tracking_info['right_hand_box']=self._load_box(tracking_info)

            return tracking_info,_image,_mask

    def _move_to_device(self, data, device):
        """Recursively move tensors to the specified device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(item, device) for item in data]
        else:
            return data

def get_video_info(frame_key):
    split_info=frame_key.split('/')
    video_id = split_info[0]
    frame_id = split_info[1]
    return video_id,frame_id

def data_to_tensor(data_dict, device='cpu'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray) or isinstance(data_dict[key], list):
            data_dict[key] = torch.as_tensor(data_dict[key], device=device,dtype=torch.float32)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_tensor(data_dict[key], device=device)
        else:
            continue
    return data_dict

def squeeze_params(tracking_info):
    tracking_info['smplx_coeffs']  = {kk: vv.squeeze() for kk, vv in tracking_info['smplx_coeffs'].items()}
    return tracking_info

def load_dict_pkl(path, encoding='') -> dict:
    assert os.path.exists(path)
    with open(path, 'rb') as fid:
        if encoding != '':
            ret = pickle.load(fid, encoding=encoding)
        else:
            ret = pickle.load(fid)
    return ret

def write_dict_pkl(path, a_dict:dict):
    a_dir = os.path.dirname(path)
    if len(a_dir) > 0: os.makedirs(a_dir, exist_ok=True)
    with open(path, 'wb') as fid:
        pickle.dump(a_dict, fid)

def print_tensor_shapes(d):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 2:
                print(f"{key}: {value.shape}")

        elif isinstance(value, dict):
            print_tensor_shapes(value)

def compare_dict_shapes(d1, d2):
    if d1.keys() != d2.keys():
        return False

    for key in d1:
        value1 = d1[key]
        value2 = d2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if value1.shape != value2.shape:
                print(f"不同的维度: {key}: {value1.shape} vs {value2.shape}")
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dict_shapes(value1, value2):
                return False
        else:
            if value2==value1:
                print(key)
                continue
            print(f"键 {key} 的值类型不匹配")
            return False

    return True

def load_canonical_render_prams():
    w2c_cam=torch.tensor([[ 1, 0, 0,    0],
                          [ 0, 1, 0, 0.6],
                          [ 0, 0, 1, 22],
                          [ 0, 0, 0, 1],
                          ],dtype=torch.float32,device='cuda')
    c2w_cam=torch.linalg.inv(w2c_cam)
    tanfov=torch.tensor([1.0/24],dtype=torch.float32,device='cuda')
    image_size=torch.tensor([512*2],dtype=torch.int,device='cuda')

    view_matrix,full_proj_matrix=get_full_proj_matrix(w2c_cam.cpu(),tanfov)
    render_cam_params={
            "world_view_transform":view_matrix.cuda(),"full_proj_transform":full_proj_matrix.cuda(),
            'tanfovx':tanfov,'tanfovy':tanfov,
            'image_height':image_size,'image_width':image_size,
            'camera_center':c2w_cam[:3,3]
        }
    return render_cam_params


