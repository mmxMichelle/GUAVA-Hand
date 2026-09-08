"""
original from https://github.com/vchoutas/smplx
modified by Vassilis and Yao
"""
import os
import os.path as osp
from os.path import join
import torch
import pickle
import numpy as np
import torch.nn as nn
import cv2
from pytorch3d.structures import Meshes
from .lbs import Struct, to_tensor, to_np, lbs, JointsFromVerticesSelector

## SMPLX
J14_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head',
]
SMPLX_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplx', 'right_eye_smplx', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1', 'head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
extra_names = ['head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
SMPLX_names += extra_names

part_indices = {}
part_indices['body'] = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24, 123,
                                124, 125, 126, 127, 132, 134, 135, 136, 137, 138, 143])
part_indices['torso'] = np.array([  0,   1,   2,   3,   6,   9,  12,  13,  14,  15,  16,  17,  18,
                                19,  22,  23,  24,  55,  56,  57,  58,  59,  76,  77,  78,  79,
                                80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                                93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                            119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                            132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144])
part_indices['head'] = np.array([ 12,  15,  22,  23,  24,  55,  56,  57,  58,  59,  60,  61,  62,
                                63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                                76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
                                89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
                            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                            115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 134, 136,
                            137])
part_indices['face'] = np.array([ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['upper'] = np.array([ 12, 13, 14, 55, 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['hand'] = np.array([ 20,  21,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
                        36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                        49,  50,  51,  52,  53,  54, 128, 129, 130, 131, 133, 139, 140,
                        141, 142, 144])
part_indices['left_hand'] = np.array([ 20,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                        37,  38,  39, 128, 129, 130, 131, 133])
part_indices['right_hand'] = np.array([ 21,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                        52,  53,  54, 139, 140, 141, 142, 144])
# kinematic tree
head_kin_chain = [15,12,9,6,3,0]

#--smplx joints
# 00 - Global
# 01 - L_Thigh
# 02 - R_Thigh
# 03 - Spine
# 04 - L_Calf
# 05 - R_Calf
# 06 - Spine1
# 07 - L_Foot
# 08 - R_Foot
# 09 - Spine2
# 10 - L_Toes
# 11 - R_Toes
# 12 - Neck
# 13 - L_Shoulder
# 14 - R_Shoulder
# 15 - Head
# 16 - L_UpperArm
# 17 - R_UpperArm
# 18 - L_ForeArm
# 19 - R_ForeArm
# 20 - L_Hand
# 21 - R_Hand
# 22 - Jaw
# 23 - L_Eye
# 24 - R_Eye

class SMPLX(nn.Module):
    """
    Given smplx parameters, this class generates a differentiable SMPLX function
    which outputs a mesh and 3D joints
    """
    def __init__(self, smplx_assets_dir, n_shape=200, n_exp=50, with_texture=False, check_pose=True,add_teeth=False,uv_size=512):
        super(SMPLX, self).__init__()
        self.smplx_assets_dir=smplx_assets_dir
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.check_pose = check_pose
        self.with_texture = with_texture
        smplx_model_path = osp.join(smplx_assets_dir, 'SMPLX_NEUTRAL_2020.npz')
        ss = np.load(smplx_model_path, allow_pickle=True)
        smplx_model = Struct(**ss)

        # flame_model_path = osp.join(smplx_assets_dir, 'flame_generic_model.pkl')
        # with open(flame_model_path, 'rb') as f:
        #     ss = pickle.load(f, encoding='latin1')
        #     flame_model = Struct(**ss)
        #self.register_buffer('flame_faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(smplx_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(smplx_model.v_template), dtype=self.dtype))
        # The shape components and expression
        # expression space is the same as FLAME
        shapedirs = to_tensor(to_np(smplx_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = smplx_model.posedirs.shape[-1]
        posedirs = np.reshape(smplx_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(smplx_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(smplx_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(smplx_model.weights), dtype=self.dtype))
        # for face keypoints
        self.register_buffer('lmk_faces_idx', torch.tensor(smplx_model.lmk_faces_idx, dtype=torch.long))
        self.register_buffer('lmk_bary_coords', torch.tensor(smplx_model.lmk_bary_coords, dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', torch.tensor(smplx_model.dynamic_lmk_faces_idx, dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', torch.tensor(smplx_model.dynamic_lmk_bary_coords, dtype=self.dtype))
        # pelvis to head, to calculate head yaw angle, then find the dynamic landmarks
        self.register_buffer('head_kin_chain', torch.tensor(head_kin_chain, dtype=torch.long))

        #-- initialize parameters
        # shape and expression
        self.register_buffer('shape_params', nn.Parameter(torch.zeros([1, n_shape], dtype=self.dtype), requires_grad=False))
        self.register_buffer('expression_params', nn.Parameter(torch.zeros([1, n_exp], dtype=self.dtype), requires_grad=False))
        # pose: represented as rotation matrx [number of joints, 3, 3]
        self.register_buffer('global_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('head_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('neck_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('jaw_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('eye_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(2,1,1), requires_grad=False))
        self.register_buffer('body_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(21,1,1), requires_grad=False))
        self.register_buffer('left_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))
        self.register_buffer('right_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))

        extra_joint_path = osp.join(smplx_assets_dir, 'smplx_extra_joints.yaml')
        if osp.exists(extra_joint_path):
            self.extra_joint_selector = JointsFromVerticesSelector(
                fname=extra_joint_path)
        self.use_joint_regressor = True
        self.keypoint_names = SMPLX_names
        if self.use_joint_regressor:
            j14_regressor_path = osp.join(smplx_assets_dir, 'SMPLX_to_J14.pkl')
            with open(j14_regressor_path, 'rb') as f:
                j14_regressor = pickle.load(f, encoding='latin1')
            source = []
            target = []
            for idx, name in enumerate(self.keypoint_names):
                if name in J14_NAMES:
                    source.append(idx)
                    target.append(J14_NAMES.index(name))
            source = np.asarray(source)
            target = np.asarray(target)
            self.register_buffer('source_idxs', torch.from_numpy(source))
            self.register_buffer('target_idxs', torch.from_numpy(target))
            joint_regressor = torch.from_numpy(
                j14_regressor).to(dtype=torch.float32)
            self.register_buffer('extra_joint_regressor', joint_regressor)
            self.part_indices = part_indices

        self.smplx2flame_ind = np.load(osp.join(smplx_assets_dir, 'SMPL-X__FLAME_vertex_ids.npy'))
        self.register_buffer('face_l_eyelid', torch.from_numpy(np.load(osp.join(smplx_assets_dir, 'flame_l_eyelid.npy'))).to(self.dtype)[None])
        self.register_buffer('face_r_eyelid', torch.from_numpy(np.load(osp.join(smplx_assets_dir, 'flame_r_eyelid.npy'))).to(self.dtype)[None])

        lmk_embeddings_mp = np.load(osp.join(smplx_assets_dir, "mediapipe_landmark_embedding.npz"))
        self.register_buffer('mp_lmk_faces_idx', torch.from_numpy(lmk_embeddings_mp['lmk_face_idx'].astype('int32')).long())
        self.register_buffer('mp_lmk_bary_coords', torch.from_numpy(lmk_embeddings_mp['lmk_b_coords']).to(self.dtype))
        self.lmk_mp_indices = lmk_embeddings_mp['landmark_indices'].tolist()

        with open(osp.join(smplx_assets_dir, 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as fid:
            self.smplx2mano_ind = pickle.load(fid, encoding='latin1')

        # Hand semantic vertex indices in the global SMPL-X topology.
        left_hand_vertex_ids = torch.from_numpy(
            self.smplx2mano_ind['left_hand']
        ).long()

        right_hand_vertex_ids = torch.from_numpy(
            self.smplx2mano_ind['right_hand']
        ).long()

        self.register_buffer(
            'left_hand_vertex_ids',
            left_hand_vertex_ids
        )

        self.register_buffer(
            'right_hand_vertex_ids',
            right_hand_vertex_ids
        )

        # Binary semantic mask over all SMPL-X vertices:
        # 1 = hand vertex, 0 = non-hand vertex.
        hand_vertex_mask = torch.zeros(
            self.v_template.shape[0],
            dtype=self.dtype
        )

        hand_vertex_mask[left_hand_vertex_ids] = 1.0
        hand_vertex_mask[right_hand_vertex_ids] = 1.0

        self.register_buffer(
            'hand_vertex_mask',
            hand_vertex_mask
        )
        self.using_lmk203 = False
        lmk203_path = osp.join(smplx_assets_dir, "203_landmark_embeding.npz")
        if osp.exists(lmk203_path):
            self.using_lmk203 = True
            lmk_embeddings_203 = np.load(lmk203_path)
            self.register_buffer('lmk_203_faces_idx', torch.from_numpy(lmk_embeddings_203['lmk_face_idx'].astype('int32')).long())
            self.register_buffer('lmk_203_bary_coords', torch.from_numpy(lmk_embeddings_203['lmk_b_coords']).to(self.dtype))
            self.lmk_203_front_indices = lmk_embeddings_203['landmark_front_indices'].tolist()
            self.lmk_203_left_indices  = lmk_embeddings_203['landmark_left_indices']
            self.lmk_203_right_indices = lmk_embeddings_203['landmark_right_indices']

        head_center=self.v_template[self.smplx2flame_ind,:].mean(0)
        left_hand_center=self.v_template[self.smplx2mano_ind['left_hand'],:].mean(0)
        right_hand_center=self.v_template[self.smplx2mano_ind['right_hand'],:].mean(0)
        self.register_buffer('head_center', head_center)
        self.register_buffer('left_hand_center', left_hand_center)
        self.register_buffer('right_hand_center', right_hand_center)

        flist_uv, valid_idx, uv_coord_map ,uv_mask_faceid=load_masks(smplx_assets_dir)
        # query_lbs_path =join(smplx_assets_dir, f'lbs_map_smplx_{512}.npy')
        # query_lbs = torch.from_numpy(np.load(query_lbs_path)).reshape(512*512, 55)
        # self.query_lbs = query_lbs[valid_idx, :][None].expand(1, -1, -1).contiguous()
        self.uv_coord_map = uv_coord_map[:,[1,0]]#u(row-x) v(colum-y)
        self.flist_uv,self.uv_mask_faceid=flist_uv,uv_mask_faceid ##each uv_pixel-to-faces(3 vertex idx), uv_pixel-to-faces(face idx)
        self.uv_valid_idx = valid_idx #valid uv coord in uv map
        self.position_map=generate_position_map(self.flist_uv, self.uv_valid_idx, self.v_template,)

        smplx_obj=OBJLoader(osp.join(smplx_assets_dir, "smplx_uv.obj"))
        faces_uv_idx=torch.tensor(smplx_obj.faces[:,:,1],dtype=torch.int32) #face-to-texcoord_idx
        texcoords=torch.tensor(smplx_obj.texcoords,dtype=torch.float32) # contains uv coord
        texcoords[:,1]=1-texcoords[:,1]#flip v
        self.register_buffer('texcoords',texcoords)
        self.register_buffer('faces_uv_idx',faces_uv_idx)
        if add_teeth:
            self.add_teeth()

        uvmap_f_idx=get_uvmap_faces_index(self.faces_uv_idx.numpy(),self.texcoords.numpy(),uv_size=uv_size)
        uvmap_f_bary=get_uvmap_faces_barycoord(uvmap_f_idx,self.faces_uv_idx.numpy(),self.texcoords.numpy(),uv_size=uv_size)
        uvmap_mask=(uvmap_f_idx!=-1)

        self.register_buffer('uvmap_f_idx',torch.tensor(uvmap_f_idx,dtype=torch.int32))
        self.register_buffer('uvmap_f_bary',torch.tensor(uvmap_f_bary,dtype=torch.float32))
        self.register_buffer('uvmap_mask',torch.tensor(uvmap_mask,dtype=torch.bool))

        vertex_uv_coord=torch.tensor(get_vertex_uv_coord(self.v_template.numpy(),self.faces_tensor.numpy(),self.faces_uv_idx.numpy(),self.texcoords.numpy()),dtype=torch.float32)# vertex-to-uv_coord
        self.vertex_uv_coord=nn.Parameter(vertex_uv_coord,requires_grad=False)

        laplacian_matrix = Meshes(verts=[self.v_template], faces=[self.faces_tensor]).laplacian_packed().to_dense()
        self.register_buffer("laplacian_matrix", laplacian_matrix, persistent=False)
        D = torch.diag(laplacian_matrix)
        laplacian_matrix_negate_diag = laplacian_matrix - torch.diag(D) * 2
        self.register_buffer("laplacian_matrix_negate_diag", laplacian_matrix_negate_diag, persistent=False)
        # self.get_head_idx_from_pos()

    def forward(self, param_dict:dict,static_offset=None,):
        """
            Decode model parameters to smplx vertices & joints & texture
            Args:
                param_dict: smplx parameters
                which may contains:
                    shape_params: [N, number of shape parameters]
                    expression_params: [N, number of expression parameters]
                    global_pose: pelvis pose, [N, 1, 3, 3]
                    body_pose: [N, 21, 3, 3]
                    jaw_pose: [N, 1, 3, 3]
                    eye_pose: [N, 2, 3, 3]
                    left_hand_pose: [N, 15, 3, 3]
                    right_hand_pose: [N, 15, 3, 3]
                pose_type: matrot / aa for matrix rotations or axis angles
            Returns:
                predictions: smplx predictions,
                in which may contains:
                    vertices: [N, number of vertices, 3]
                    landmarks: [N, number of landmarks (68 face keypoints), 3]
                    joints: [N, number of smplx joints (145), 3]
        """

        shape_params      = param_dict.get('shape', None)             # torch.Size([1, 250])
        expression_params = param_dict.get('exp', None)               # torch.Size([1, 50])
        global_pose       = param_dict.get('global_pose', None)       # torch.Size([1, 1, 3, 3])
        body_pose         = param_dict.get('body_pose', None)         # torch.Size([1, 21, 3, 3])
        jaw_pose          = param_dict.get('jaw_pose', None)          # torch.Size([1, 1, 3, 3])
        left_hand_pose    = param_dict.get('left_hand_pose', None)    # torch.Size([1, 15, 3, 3])
        right_hand_pose   = param_dict.get('right_hand_pose', None)   # torch.Size([1, 15, 3, 3])
        eye_pose          = param_dict.get('eye_pose', None)          # torch.Size([1, 2, 3, 3])
        eyelid_params     = param_dict.get('eyelid_params', None)     # torch.Size([1, 2])
        joints_offset     = param_dict.get('joints_offset', None)
        head_scale        = param_dict.get('head_scale', None)
        hand_scale        = param_dict.get('hand_scale', None)
        face_offset       = param_dict.get('face_offset', None)

        if shape_params is None:
            batch_size = global_pose.shape[0]
            shape_params = self.shape_params.expand(batch_size, -1)
        else:
            batch_size = shape_params.shape[0]

        if expression_params is None: expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None:
            global_pose = self.global_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            global_pose = torch.zeros_like(global_pose[..., 0])
        if body_pose is None:
            body_pose = self.body_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            body_pose = torch.zeros_like(body_pose[..., 0])
        if jaw_pose is None:
            jaw_pose = self.jaw_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            jaw_pose = torch.zeros_like(jaw_pose[..., 0])
        if eye_pose is None:
            eye_pose = self.eye_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            eye_pose = torch.zeros_like(eye_pose[..., 0])
        if left_hand_pose is None:
            left_hand_pose = self.left_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            left_hand_pose = torch.zeros_like(left_hand_pose[..., 0])
        if right_hand_pose is None:
            right_hand_pose = self.right_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            right_hand_pose = torch.zeros_like(right_hand_pose[..., 0])

        if shape_params.shape[-1] < self.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.n_shape]

        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)
        if len(jaw_pose.shape) == 2: jaw_pose = jaw_pose.unsqueeze(1)
        if len(eye_pose.shape) == 2: eye_pose = eye_pose.unsqueeze(1)

        shape_components = torch.cat([t_shape_params, expression_params], dim=1)
        full_pose = torch.cat([global_pose,
                               body_pose,
                                jaw_pose,
                                eye_pose,
                                left_hand_pose,
                                right_hand_pose], dim=1)

        v_template=self.v_template.clone()
        if static_offset is not None:
            v_template=v_template+static_offset
        if face_offset is not None:
            v_template=v_template[self.smplx2flame_ind,:]+face_offset
        template_vertices = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1)

        #eyelid and head scale
        head_vert = template_vertices[:, self.smplx2flame_ind].clone()
        if eyelid_params is not None:
            head_vert = head_vert + self.face_r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]+ self.face_l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
        if head_scale is not None:
            head_vert = head_vert * head_scale[:, None]+ (1-head_scale[:, None])*self.head_center[None,None]
        template_vertices[:, self.smplx2flame_ind] = head_vert

        #hand scale
        if hand_scale is not None:
            left_hand_vert = template_vertices[:, self.smplx2mano_ind['left_hand']].clone()
            right_hand_vert = template_vertices[:, self.smplx2mano_ind['right_hand']].clone()
            left_hand_vert = left_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.left_hand_center[None,None]
            right_hand_vert = right_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.right_hand_center[None,None]
            template_vertices[:, self.smplx2mano_ind['left_hand']] = left_hand_vert
            template_vertices[:, self.smplx2mano_ind['right_hand']] = right_hand_vert

        # smplx，joints（55x3）
        vertices, joints_transformed,joints_tpose,joints_transform_mat,verts_transform_mat = lbs(shape_components, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,        # shapedirs[10475, 3, 20]
                          self.J_regressor, self.parents,       # J_regressor([55, 10475])
                          self.lbs_weights,joints_offset=joints_offset, dtype=self.dtype)   # template_vertices（10475x3）

        ret_dict = {}
        prediction = {
            'vertices': vertices,
            'joints': joints_tpose,  #tpose joints
            'joints_transformed': joints_transformed, #transformed joints
            'joints_transform_mat': joints_transform_mat,## transofrm matrix per joint
            'ver_transform_mat': verts_transform_mat,## transofrm matrix per vertex
            'shape': param_dict['shape'],
            'exp': param_dict['exp'],
        }

        ret_dict.update(prediction)

        return ret_dict

    def get_head_idx_from_pos(self,y_threshold=0.15):
        y_coordinates = self.v_template[:, 1]
        head_indices_y = torch.where(y_coordinates > y_threshold)[0]

        face_vertices=self.v_template[self.faces_tensor]#f k 3
        face_vertices_nn=face_vertices[self.uvmap_f_idx.reshape(-1)]# n k 3
        face_bary=self.uvmap_f_bary.reshape(-1,3)# n k

        face_center_nn= torch.einsum('nk,nkj->nj',face_bary,face_vertices_nn)# n 3

        head_indices_uv_flat = torch.where(face_center_nn[:,1] > y_threshold)[0]

        self.register_buffer('head_idxs_uv_flat',head_indices_uv_flat)
        self.register_buffer('head_idxs_temp',head_indices_y)

    def pose_rel2abs(self, global_pose, body_pose, abs_joint = 'head'):
        ''' change relative pose to absolute pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        '''
        full_pose = torch.cat([global_pose, body_pose], dim=1)

        if abs_joint == 'head':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == 'neck':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == 'right_wrist':
            # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
            # -> right elbow -> right wrist
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == 'left_wrist':
            # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
            # -> Left elbow -> Left wrist
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(
                f'pose_rel2abs does not support: {abs_joint}')
        rel_rot_mat = torch.eye(3, device=full_pose.device,
                                dtype=full_pose.dtype).unsqueeze_(dim=0)
        for idx in kin_chain:
            rel_rot_mat = torch.matmul(full_pose[:, idx], rel_rot_mat)
        abs_pose = rel_rot_mat[:,None,:,:]
        return abs_pose

    def add_teeth(self):
        from  ..flame.FLAME import FLAME
        flame=FLAME(osp.join(osp.dirname(self.smplx_assets_dir),'FLAME'),n_shape=self.n_shape,
                    n_exp=self.n_exp,add_teeth=True)
        vid_lip_outside_ring_upper = flame.mask.get_vid_by_region(['lip_outside_ring_upper'], keep_order=True)
        vid_lip_outside_ring_lower = flame.mask.get_vid_by_region(['lip_outside_ring_lower'], keep_order=True)
        vid_lip_outside_ring_upper = self.smplx2flame_ind[vid_lip_outside_ring_upper]
        vid_lip_outside_ring_lower = self.smplx2flame_ind[vid_lip_outside_ring_lower]
        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= mean_dist * 1.5  # how far the teeth are from the lips

        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]])*0.25 #0.1
        v_teeth_upper_edge += torch.tensor([[0, 0, mean_dist]]) * 0.4 # +0.0
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]])*0.2
        v_teeth_lower_edge += torch.tensor([[0, 0, mean_dist]]) * 0.2  #-0.4 slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # thickness = mean_dist * 0.5
        thickness = mean_dist * 0.75
        # upper, back
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_upper_edge_back[:, 2] -= thickness  # how thick the teeth are

        # lower, back
        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_lower_edge_back[:, 2] -= thickness  # how thick the teeth are

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
            v_teeth_upper_root_back,  # num_verts_orig + 60-74
            v_teeth_upper_edge_back,  # num_verts_orig + 75-89
            v_teeth_lower_root_back,  # num_verts_orig + 90-104
            v_teeth_lower_edge_back,  # num_verts_orig + 105-119
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120) + num_verts_orig
        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_root_back, vid_teeth_lower_edge_back], dim=0)
        vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)

        self.smplx2flame_ind=np.concatenate((self.smplx2flame_ind,vid_teeth.numpy()),axis=0)
        # u = torch.linspace(0.244, 0.342, 15)
        # v = torch.linspace(0.0097, 0.078, 7)
        u = torch.linspace(0.1328, 0.2695, 15)
        v = torch.linspace(0.94726, 0.9999, 7)
        v = v[[3, 2, 0, 1, 3, 4, 6, 5]]  # TODO: with this order, teeth_lower is not rendered correctly in the uv space
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)

        num_verts_uv_teeth = uv.shape[0]
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (self.shapedirs[vid_lip_outside_ring_upper, :, :self.n_shape] + self.shapedirs[vid_lip_outside_ring_lower, :, :self.n_shape]) / 2
        self.shapedirs[vid_teeth_upper_root, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_root_back, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root_back, :, :self.n_shape] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back, :, :self.n_shape] = shape_dirs_mean

        posedirs = self.posedirs.reshape(len(self.parents)-1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents)-1)*9, (num_verts_orig+num_verts_teeth)*3)  # (J*9, (V+num_verts_teeth)*3)
        #eyelid set to zero
        self.face_l_eyelid = torch.cat([self.face_l_eyelid, torch.zeros_like(self.face_l_eyelid[:,:num_verts_teeth])], dim=1)
        self.face_r_eyelid = torch.cat([self.face_r_eyelid, torch.zeros_like(self.face_r_eyelid[:,:num_verts_teeth])], dim=1)
        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)
        self.extra_joint_regressor=torch.cat([self.extra_joint_regressor, torch.zeros_like(self.extra_joint_regressor[:, :num_verts_teeth])], dim=1)
        # lbs_weights manually set
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 12] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 22] += 1  # move with jaw
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ])
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 115],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ])

        self.faces_tensor = torch.cat([self.faces_tensor, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], dim=0)
        num_uvcoord_orig=self.texcoords.shape[0]
        self.faces_uv_idx=torch.cat([self.faces_uv_idx, f_teeth_upper+num_uvcoord_orig, f_teeth_lower+num_uvcoord_orig])
        self.texcoords=torch.cat([self.texcoords,uv],dim=0)

def getIdxMap_torch(img, offset=False):
    # img has shape [channels, H, W]
    C, H, W = img.shape

    idx = torch.stack(torch.where(~torch.isnan(img[0])))
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = idx / (H-1) if not offset else idx / H
    return idx
def get_face_per_pixel(mask, flist):
    '''
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    '''
    mask2 = mask.clone()
    mask2[mask == -1] = 0 #remove the -1 in the mask, so that all mask elements can be seen as meaningful faceid
    flist_uv = flist[mask2]
    return flist_uv

def load_masks(PROJECT_DIR, posmap_size=512, body_model='smplx'):
    uv_mask_faceid = np.load(join(PROJECT_DIR, 'uv_masks', 'uv_mask{}_with_faceid_{}.npy'.format(posmap_size, body_model))).reshape(posmap_size, posmap_size)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long()

    smpl_faces = np.load(join(PROJECT_DIR, '{}_faces.npy'.format(body_model.lower()))) # faces = triangle list of the body mesh
    flist = torch.tensor(smpl_faces.astype(np.int32)).long()
    flist_uv = get_face_per_pixel(uv_mask_faceid, flist) # Each (valid) pixel on the uv map corresponds to a point on the SMPL body; flist_uv is a list of these triangles

    points_idx_from_posmap = (uv_mask_faceid!=-1).reshape(-1)

    uv_coord_map = getIdxMap_torch(torch.rand(3, posmap_size, posmap_size))
    uv_coord_map.requires_grad = True

    return flist_uv, points_idx_from_posmap, uv_coord_map,uv_mask_faceid

def generate_position_map(flist_uv, points_idx_from_posmap, v_template, posmap_size=512):
    """
    Generate position map from UV map and template vertices.

    :param flist_uv: Flat tensor of face indices for each pixel, shape (posmap_size * posmap_size, 3).
    :param points_idx_from_posmap: Boolean mask indicating valid pixels, shape (posmap_size * posmap_size).
    :param uv_coord_map: UV coordinate map, shape (3, posmap_size, posmap_size).
    :param flist: Tensor of shape (num_faces, 3) with vertex indices for each face.
    :param v_template: Template vertices of the model, shape (num_vertices, 3).
    :param posmap_size: Size of the position map.
    :return: Position map of shape (posmap_size, posmap_size, 3).
    """
    position_map = torch.zeros((posmap_size, posmap_size, 3), dtype=torch.float32)

    # Iterate over each pixel
    for i in range(posmap_size):
        for j in range(posmap_size):
            idx = i * posmap_size + j
            if not points_idx_from_posmap[idx]:
                continue

            face_indices = flist_uv[i,j]

            # Get the three vertices of the triangle
            v0_idx, v1_idx, v2_idx = face_indices
            v0 = v_template[v0_idx]
            v1 = v_template[v1_idx]
            v2 = v_template[v2_idx]

            # Calculate the average position of the three vertices
            avg_position = (v0 + v1 + v2) / 3.0
            position_map[i, j] = avg_position

    return position_map

def get_vertex_uv_coord(vertices,faces,faces_t,texcoords):
    vertex_texcoords = np.full((vertices.shape[0], 2), np.nan)

    for f_idx,face in enumerate(faces):
        for i,v_idx  in enumerate(face):
            vt_idx=faces_t[f_idx,i]
            vertex_texcoords[v_idx] = texcoords[vt_idx]
    return vertex_texcoords

def get_uvmap_faces_index(faces_uv,uv_coords,uv_size=512):
    uv_coords=np.round(uv_coords*uv_size).astype(np.int32)
    faces_uv=faces_uv.astype(np.int32)
    uvmap_faces_idx = np.ones((uv_size,uv_size), dtype=np.int32) * -1
    for f_idx in range(len(faces_uv)):
        cv2.drawContours(uvmap_faces_idx, [uv_coords[faces_uv[f_idx]]], 0, int(f_idx), -1)
    return uvmap_faces_idx

def get_uvmap_faces_barycoord(uvmap_faces_idx,faces_uv,uv_coords,uv_size=512):
    uv_coords=np.round(uv_coords*uv_size).astype(np.int32)
    uvmap_faces_barycoord = np.zeros((uv_size,uv_size,3), dtype=np.float32)
    for u_idx in range(uv_size):
        for v_idx in range(uv_size):
            f_idx=uvmap_faces_idx[v_idx,u_idx]
            if f_idx==-1:
                continue
            v_uvs=uv_coords[faces_uv[f_idx]]
            v_uv0=v_uvs[0]
            v_uv1=v_uvs[1]
            v_uv2=v_uvs[2]
            c_uv=np.array([u_idx,v_idx])

            c_0=c_uv-v_uv0
            c_1=c_uv-v_uv1
            c_2=c_uv-v_uv2
            area0=0.5*np.abs(np.cross(c_1,c_2))
            area1=0.5*np.abs(np.cross(c_0,c_2))
            area2=0.5*np.abs(np.cross(c_0,c_1))
            total_area=area0+area1+area2+1e-6
            uvmap_faces_barycoord[v_idx,u_idx]=np.array([area0,area1,area2])/total_area
    return uvmap_faces_barycoord

def cv2_triangle(img, p123):
    ''' draw triangles using OpenCV '''
    p1, p2, p3 = (tuple(i) for i in p123)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    cv2.line(img, p2, p3, (255, 0, 0), 1)
    cv2.line(img, p1, p3, (255, 0, 0), 1)
    return img

class OBJLoader:
    def __init__(self, filepath):
        self.vertices = []   #  v  vertex
        self.texcoords = []  #  vt texture coord
        self.faces = []      #  f  face

        self.load_obj(filepath)
        self.vertices=np.array(self.vertices)
        self.texcoords=np.array(self.texcoords)
        self.faces=np.array(self.faces)

    def load_obj(self, filepath):
        try:
            with open(filepath, 'r') as file:
                for line in file:

                    line = line.strip()
                    if line.startswith('v '):
                        self.vertices.append(list(map(float, line.split()[1:])))
                    elif line.startswith('vt '):
                        self.texcoords.append(list(map(float, line.split()[1:])))
                    elif line.startswith('f '):
                        face = []
                        for vert in line.split()[1:]:
                           #-1
                            parts = vert.split('/')
                            face.append(tuple(int(p)-1 if p else 0 for p in parts))
                        self.faces.append(face)
        except Exception as e:
            print(f"Error loading OBJ file: {e}")

if __name__=="__main__":

    def save_obj(filename, vertices, faces):
        """Saves a 3D mesh to an OBJ file."""
        with open(filename, 'w') as f:
            for v in vertices:
                f.write('v {:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
            for face in faces:
                # OBJ indices are 1-based, so we add 1 to each vertex index
                f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

    num_shape=200
    num_exp=50
    batch_size=1
    device='cpu'
    smplx=SMPLX("path_to/assets/SMPLX",n_shape=num_shape,
                n_exp=num_exp,add_teeth=True).to(device)

    jaw_pose=torch.randn(batch_size, 1, 3, device=device)*0.0
    jaw_pose[:,:,0]=0.3
    param_dict = {
        'shape': torch.randn(batch_size, num_shape, device=device)*0.2,
        'exp': torch.randn(batch_size, num_exp, device=device)*0.75,
        'global_pose': torch.zeros(batch_size, 1, 3, device=device),
        'body_pose': torch.zeros(batch_size, 21, 3, device=device),
        'jaw_pose': jaw_pose,
        'left_hand_pose': torch.zeros(batch_size, 15, 3, device=device),
        'right_hand_pose': torch.zeros(batch_size, 15, 3, device=device),
        'eye_pose': torch.zeros(batch_size, 2, 3, device=device),
    }
    out =smplx(param_dict)
    vertices = out['vertices'][0].detach().cpu().numpy()
    faces = smplx.faces_tensor.cpu().numpy()
    save_obj(f'path_to/z_temp/output_smplx_teeth.obj', vertices, faces)
    print(f"Saved output vertices.obj")

