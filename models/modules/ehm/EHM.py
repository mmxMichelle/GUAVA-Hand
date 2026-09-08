import torch
import numpy as np
import torch.nn as nn
from pytorch3d.structures import Meshes
import time


from ..smplx import SMPLX
from ..flame import FLAME
from utils import rotation_converter as converter
from ..flame.lbs import lbs,lbs_wobeta,lbs_get_transform,blend_shapes,vertices2joints
#from ..mano  import MANO

class EHM(nn.Module):
    def __init__(self, flame_assets_dir, smplx_assets_dir,
                       n_shape=300, n_exp=50, with_texture=False, uv_size=512,
                       check_pose=True,add_teeth=False):
        super().__init__()
        self.smplx = SMPLX(smplx_assets_dir, n_shape=n_shape, n_exp=n_exp, check_pose=check_pose, with_texture=with_texture,add_teeth=add_teeth,uv_size=uv_size)
        self.flame = FLAME(flame_assets_dir, n_shape=n_shape, n_exp=n_exp, with_texture=with_texture,add_teeth=add_teeth)
        #mano_assets_dir use_pca=True, num_pca_comps=6, flat_hand_mean=False
        #self.mano  = MANO(mano_assets_dir, use_pca=use_pca, num_pca_comps=num_pca_comps, flat_hand_mean=flat_hand_mean)

        v_template,v_head_template=self.smplx.v_template.clone(),self.flame.v_template.clone()
        tbody_joints = vertices2joints(self.smplx.J_regressor, v_template[None])
        flame_joints = vertices2joints(self.flame.J_regressor, v_head_template[None])
        v_template[self.smplx.smplx2flame_ind]=v_head_template - flame_joints[0, 3:5].mean(dim=0, keepdim=True) + tbody_joints[0, 23:25].mean(dim=0, keepdim=True)
        self.register_buffer('v_template', v_template)

        laplacian_matrix = Meshes(verts=[v_template], faces=[self.smplx.faces_tensor]).laplacian_packed().to_dense()
        self.register_buffer("laplacian_matrix", laplacian_matrix, persistent=False)
        D = torch.diag(laplacian_matrix)
        laplacian_matrix_negate_diag = laplacian_matrix - torch.diag(D) * 2
        self.register_buffer("laplacian_matrix_negate_diag", laplacian_matrix_negate_diag, persistent=False)

    def forward(self, body_param_dict:dict, flame_param_dict:dict=None, mano_param_dict:dict=None,
                static_offset=None, zero_expression=False, zero_jaw=False, zero_shape=False,pose_type='rotmat',):

        # for flame head model
        start_time=time.time()
        if flame_param_dict is not None:
            eye_pose_params    = flame_param_dict['eye_pose_params']# batch_size,6
            shape_params       = flame_param_dict['shape_params']# batch_size,300
            expression_params  = flame_param_dict['expression_params']# batch_size,50
            global_pose_params = flame_param_dict.get('pose_params', None)# batch_size,3
            jaw_params         = flame_param_dict.get('jaw_params', None)# batch_size,3
            eyelid_params      = flame_param_dict.get('eyelid_params', None) ## batch_size,2
            head_scale         = body_param_dict.get('head_scale', None) # batch_size

            batch_size = shape_params.shape[0]

            # Adjust shape params size if needed
            if shape_params.shape[1] < self.flame.n_shape:
                shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.flame.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
            if zero_expression: expression_params = torch.zeros_like(expression_params,device=shape_params.device)
            if zero_jaw: jaw_params = torch.zeros_like(jaw_params,device=shape_params.device)
            if zero_shape: shape_params = torch.zeros_like(shape_params,device=shape_params.device)

            neck_pose_params = self.flame.neck_pose.expand(batch_size, -1)
            global_pose_params = torch.zeros_like(global_pose_params,device=shape_params.device)
            neck_pose_params = torch.zeros_like(neck_pose_params,device=shape_params.device)
            betas = torch.cat([shape_params, expression_params], dim=1)
            full_pose = torch.cat([global_pose_params, neck_pose_params, jaw_params, eye_pose_params], dim=1)

            template_vertices = self.flame.v_template.clone().unsqueeze(0).expand(batch_size, -1, -1)
            if static_offset is not None: template_vertices = template_vertices + static_offset[:, self.smplx.smplx2flame_ind]
            head_vertices, head_joints = lbs(betas, full_pose, template_vertices,
                                             self.flame.shapedirs, self.flame.posedirs,
                                             self.flame.J_regressor, self.flame.parents,
                                             self.flame.lbs_weights, dtype=self.flame.dtype)

            if eyelid_params is not None:
                head_vertices = head_vertices + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None] #[:, :self.flame.n_ori_verts]
                head_vertices = head_vertices + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]#[:, :self.flame.n_ori_verts]
            head_vertices=head_vertices*head_scale[:,None]

        else:
            head_vertices = None

        # body paramerters
        shape_params      = body_param_dict.get('shape')
        expression_params = body_param_dict.get('exp', None)
        global_pose       = body_param_dict.get('global_pose', None)       # torch.Size([1, 1, 3, 3])
        body_pose         = body_param_dict.get('body_pose', None)         # torch.Size([1, 21, 3, 3])
        jaw_pose          = body_param_dict.get('jaw_pose', None)          # torch.Size([1, 1, 3, 3])
        left_hand_pose    = body_param_dict.get('left_hand_pose', None)    # torch.Size([1, 15, 3, 3])
        right_hand_pose   = body_param_dict.get('right_hand_pose', None)   # torch.Size([1, 15, 3, 3])
        eye_pose          = body_param_dict.get('eye_pose', None)          # torch.Size([1, 2, 3, 3])
        joints_offset     = body_param_dict.get('joints_offset',None)      # batch_size 55 3
        hand_scale        = body_param_dict.get('hand_scale', None)        # batch_size 3
        batch_size = shape_params.shape[0]

        if expression_params is None: expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None: global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)

        jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)

        if shape_params.shape[-1] < self.smplx.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.smplx.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.smplx.n_shape]

        shape_components = torch.cat([t_shape_params, expression_params], dim=1)
        full_pose = torch.cat([global_pose,
                               body_pose,
                               jaw_pose,
                               eye_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)

        template_vertices = self.smplx.v_template.clone().unsqueeze(0).expand(batch_size, -1, -1)
        new_template_vertices = template_vertices + blend_shapes(shape_components, self.smplx.shapedirs)
        if static_offset is not None: new_template_vertices = new_template_vertices + static_offset
        tbody_joints = vertices2joints(self.smplx.J_regressor, new_template_vertices)
        if joints_offset is not None: tbody_joints=tbody_joints+joints_offset

        if not hasattr(self, 'head_index'): self.head_index = np.unique(self.flame.head_index)
        if head_vertices is not None:
            selected_head = new_template_vertices[:, self.smplx.smplx2flame_ind]
            selected_head = head_vertices - head_joints[:, 3:5].mean(dim=1, keepdim=True) + tbody_joints[:, 23:25].mean(dim=1, keepdim=True)
            new_template_vertices[:, self.smplx.smplx2flame_ind] = selected_head

        if hand_scale is not None:
            left_hand_vert = new_template_vertices[:, self.smplx.smplx2mano_ind['left_hand']].clone()
            right_hand_vert = new_template_vertices[:, self.smplx.smplx2mano_ind['right_hand']].clone()
            left_hand_vert = left_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.smplx.left_hand_center[None,None]
            right_hand_vert = right_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.smplx.right_hand_center[None,None]
            new_template_vertices[:, self.smplx.smplx2mano_ind['left_hand']] = left_hand_vert
            new_template_vertices[:, self.smplx.smplx2mano_ind['right_hand']] = right_hand_vert

        vertices, joints_transform,joints,ver_transform_mat,joint_transform_mat = lbs_wobeta( full_pose, new_template_vertices,#
                                            self.smplx.posedirs,
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）

        ret_dict = {}
        prediction = {
            'vertices': vertices,
            'joints': joints,                  # tpose joints
            'joints_transform':joints_transform,#transformed joints
            'ver_transform_mat':ver_transform_mat, # transform matrix per vertex
            'joint_transform_mat':joint_transform_mat, # transofrm matrix per joint
            'head_vertices': vertices[:, self.smplx.smplx2flame_ind][:, self.head_index],
            'head_ref_joint': joints[:, 23:25].mean(dim=1, keepdim=True),

            'left_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['left_hand']],
            'left_hand_ref_joint': joints[:, 20:21, :],

            'right_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['right_hand']],
            'right_hand_ref_joint': joints[:, 21:22, :],
        }
        ret_dict.update(prediction)
        return ret_dict

    def get_transform_mat(self,body_param_dict:dict, flame_param_dict:dict,mano_param_dict:dict,joints=None):
        # body paramerters
        shape_params      = body_param_dict.get('shape')
        expression_params=None
        global_pose       = body_param_dict.get('global_pose', None)
        body_pose         = body_param_dict.get('body_pose', None)
        joints_offset     = body_param_dict.get('joints_offset',None)

        left_hand_pose    = mano_param_dict['left_hand'].get('hand_pose', None)
        right_hand_pose   = mano_param_dict['right_hand'].get('hand_pose', None)
        eye_pose          = flame_param_dict.get('eye_pose_params', None)
        jaw_pose          = flame_param_dict.get('jaw_params', None)

        batch_size = shape_params.shape[0]
        eye_pose=eye_pose.reshape(batch_size,2,3)
        jaw_pose=jaw_pose.reshape(batch_size,1,3).detach().clone()

        b, n = left_hand_pose.shape[:2]
        left_hand_pose=converter.batch_matrix2axis(left_hand_pose.flatten(0,1)).reshape(b, n*3)
        right_hand_pose=converter.batch_matrix2axis(right_hand_pose.flatten(0,1)).reshape(b, n, 3)
        left_hand_pose[:,1::3]*=-1
        left_hand_pose[:,2::3]*=-1
        left_hand_pose=left_hand_pose.reshape(b, n, 3)

        if expression_params is None: expression_params = self.smplx.expression_params.expand(batch_size, -1)
        if global_pose is None: global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if eye_pose is None: eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)
        if jaw_pose is None: jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)
        if len(jaw_pose.shape) == 2: jaw_pose = jaw_pose.unsqueeze(1)

        if shape_params.shape[-1] < self.smplx.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.smplx.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.smplx.n_shape]
        shape_components = torch.cat([t_shape_params, expression_params], dim=1)

        full_pose = torch.cat([global_pose,
                               body_pose,
                               jaw_pose,
                               eye_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)

        template_vertices = self.smplx.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        transform_mats, transform_joints = lbs_get_transform(shape_components, full_pose, template_vertices,
                                            self.smplx.shapedirs, self.smplx.posedirs,
                                            self.smplx.J_regressor, self.smplx.parents,
                                            self.smplx.lbs_weights,joints_offset=joints_offset,joints=joints, dtype=self.smplx.dtype)
        return transform_mats,transform_joints

    def transform_points3d(self, points3d, M):
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale

        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    def save_obj(filename, vertices, faces):
        """Saves a 3D mesh to an OBJ file."""
        with open(filename, 'w') as f:
            for v in vertices:
                f.write('v {:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
            for face in faces:
                # OBJ indices are 1-based, so we add 1 to each vertex index
                f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flame_assets_dir = "path_to/assets/FLAME"
    smplx_assets_dir = "path_to/assets/SMPLX"
    mano_assets_dir = "path_to/assets/MANO"
    ehm_model = EHM(flame_assets_dir, smplx_assets_dir, mano_assets_dir,add_teeth=True).to(device)
    batch_size = 2
    body_param_dict = {
        'shape': torch.randn(batch_size, 300).to(device),
        'exp': torch.randn(batch_size, 50).to(device),
        'global_pose': torch.randn(batch_size, 1, 3).to(device),
        'body_pose': torch.randn(batch_size, 21, 3).to(device),
        'joints_offset': torch.randn(batch_size, 55, 3).to(device),
        'head_scale': torch.randn(batch_size,3).to(device),
        'hand_scale': torch.randn(batch_size,3).to(device),
        'left_hand_pose': torch.randn(batch_size, 15,3).to(device),
        'right_hand_pose': torch.randn(batch_size, 15,3).to(device),
        }
    flame_param_dict = {
        'shape_params': torch.randn(batch_size, 300).to(device),
        'expression_params': torch.randn(batch_size, 50).to(device),
        'pose_params': torch.randn(batch_size, 3).to(device),
        'jaw_params': torch.randn(batch_size, 3).to(device),
        'eye_pose_params': torch.randn(batch_size, 6).to(device),
        'eyelid_params': torch.randn(batch_size, 2).to(device),
        }
    mano_param_dict={
                'left_hand':{
                        'betas': torch.randn(batch_size, 10).to(device),
                        'hand_pose': torch.randn(batch_size, 15,3,3).to(device),
                },
                'right_hand':{
                        'betas': torch.randn(batch_size, 10).to(device),
                        'hand_pose': torch.randn(batch_size, 15,3,3).to(device),
                }
        }
    output = ehm_model(body_param_dict, flame_param_dict,mano_param_dict)
    vertices=ehm_model.v_template.cpu().numpy()
    faces=ehm_model.smplx.faces_tensor.cpu().numpy()
    save_obj(f'path_to/z_temp/output_ehm_head.obj', vertices, faces)