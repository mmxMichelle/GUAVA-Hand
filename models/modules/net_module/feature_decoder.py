

import torch,pickle,os
import torch.nn as nn
import numpy as np
import lightning as L


class Vertex_GS_Decoder(L.LightningModule):
    # smplx vertices gaussian attributes predictor
    def __init__(self, in_dim=1024, dir_dim=27,color_out_dim=32):
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
        )
        layer_in_dim = in_dim//2 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, color_out_dim, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, cam_dirs):
        input_features = self.feature_layers(input_features)
        cam_dirs = cam_dirs[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, cam_dirs], dim=-1)
        # color
        colors = self.color_layers(input_features)
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        scales = torch.sigmoid(scales) * 0.05 #0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)

        res_dict={'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations,
                  'static_offsets':None}
        return  res_dict

class UV_Point_GS_Decoder(L.LightningModule):
    # Gaussian attributes predictor for uv points
    def __init__(self, in_dim=128, dir_dim=27, color_out_dim=27):
        super().__init__()
        color_out_dim = color_out_dim
        opacity_out_dim= 1
        scale_out_dim=3
        rotation_out_dim=4
        local_pos_dim=3
        hid_dim_1=max(in_dim,128)
        hid_dim_2=max(in_dim//2,64)

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, hid_dim_1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, stride=1, padding=1),
        )

        self.rot_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, rotation_out_dim, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, scale_out_dim, kernel_size=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, opacity_out_dim, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, color_out_dim, kernel_size=1),
        )
        self.local_pos_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, local_pos_dim, kernel_size=1),
        )
        # [NEW - Contribution 1]
        # Hand-specific residual branch for UV Gaussian local position.
        # The branch predicts an additional local geometric correction Delta p_hand.
        self.hand_local_pos_residual_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, local_pos_dim, kernel_size=1),
        )

        # Zero-initialize the final layer so that the new branch initially
        # predicts exactly zero and therefore preserves the original GUAVA geometry.
        nn.init.zeros_(self.hand_local_pos_residual_head[-1].weight)
        nn.init.zeros_(self.hand_local_pos_residual_head[-1].bias)


    def forward(self, input_features,cam_dirs):
        #assume img_height=img_width
        b,h,w=input_features.shape[0],input_features.shape[2],input_features.shape[3]
        cam_dirs = cam_dirs[:, :, None, None].expand(-1, -1, h, w)
        input_features = torch.cat([input_features, cam_dirs], dim=1)
        gaussian_feature = self.feature_conv(input_features)
        # color
        colors = self.color_head(gaussian_feature)
        # opacity
        opacities = self.opacity_head(gaussian_feature)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_head(gaussian_feature)
        scales = torch.exp(scales)  #0.05 * 0.05
        # rotation
        rotations = self.rot_head(gaussian_feature)
        rotations = nn.functional.normalize(rotations)

        # local position
        local_pos = self.local_pos_head(gaussian_feature)

        # [NEW - Contribution 1]
        # Hand-specific residual local-position prediction.
        hand_local_pos_residual = self.hand_local_pos_residual_head(
            gaussian_feature
        )

        results = {
            'colors': colors,
            'opacities': opacities,
            'scales': scales,
            'rotations': rotations,
            'local_pos': local_pos,
            'hand_local_pos_residual': hand_local_pos_residual,
        }

        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).contiguous()#.reshape(results[key].shape[0], -1, results[key].shape[1])

        return results