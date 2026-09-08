import torch
import numpy as np
from tqdm import tqdm
from utils.graphics_utils import get_full_proj_matrix

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """
    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position,FoVx,FoVy, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cuda:0'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, np.pi - 1e-5)

        theta = h
        v = v / np.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(np.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(np.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        c2w=create_cam2world_matrix(forward_vectors, camera_origins)

        w2c=torch.linalg.inv(c2w).squeeze(0)@torch.tensor([[1 ,0 ,0 ,0 ],#
                    [0 ,-1,0 ,0 ],
                    [0 ,0 ,-1,0 ],
                    [0 ,0 ,0 ,1 ]],dtype=torch.float32,device=device)
        R = torch.transpose(w2c[:3,:3],0,1).cpu().numpy()
        T= w2c[:3, 3].cpu().numpy()
        c2w=torch.linalg.inv(w2c)


        return w2c,c2w


def generate_novel_view_poses(tracking_info,image_size=512,tanfov=1/24.0,pitch_range = 0.3,yaw_range = 0.35,num_keyframes=120):
    camera_center=tracking_info['c2w_cam'][0,:3,3]
    device=tracking_info['c2w_cam'].device
    circle_cam_params=[]
    result_cam_params=[]
    FoVx=tanfov
    FoVy=tanfov
    radius=camera_center.square().sum().sqrt()

    lookat_position=[0.0,0.75,0.0]
    print("Generate multi-view poses for rendering")

    for frame_idx in tqdm(range(num_keyframes)):
        w2c_cam,c2w_cam=LookAtPoseSampler.sample(
            horizontal_mean=3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
            vertical_mean=3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
            lookat_position=torch.Tensor(lookat_position).to(device),FoVx=FoVx,FoVy=FoVy ,radius=radius, device=device)
        view_matrix,full_proj_matrix=get_full_proj_matrix(w2c_cam,tanfov)
        circle_cam_params.append({
                                "world_view_transform":view_matrix.unsqueeze(0),"full_proj_transform":full_proj_matrix.unsqueeze(0),
                                'tanfovx':torch.tensor([FoVx],device=device),'tanfovy':torch.tensor([FoVy],device=device),
                                'image_height':torch.tensor([image_size],device=device),'image_width':torch.tensor([image_size],device=device),
                                'camera_center':c2w_cam[:3,3].unsqueeze(0)
                                    })

    return circle_cam_params