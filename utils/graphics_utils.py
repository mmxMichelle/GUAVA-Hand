
import torch
import math
import numpy as np
import lightning as L

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    CamerasBase,RasterizationSettings,
    PointLights, TexturesVertex, BlendParams,TexturesUV,
    SoftPhongShader, MeshRasterizer, MeshRenderer,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments,rasterize_meshes

def get_view_matrix(R, t):
    device=R.device
    Rt = torch.cat((R, t.view(3,1)),1)
    b_row=torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 4)
    #torch.FloatTensor([0,0,0,1],device=device).view(1,4)
    view_matrix = torch.cat((Rt, b_row))
    return view_matrix

def get_proj_matrix( tanfov,device, z_near=0.01, z_far=100, z_sign=1.0,):

    tanHalfFovY = tanfov
    tanHalfFovX = tanfov

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0

    proj_matrix = torch.zeros(4, 4).float().to(device)
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix

def get_full_proj_matrix(w2c_cam,tanfov):
    assert len(w2c_cam.shape)==2
    view_matrix=get_view_matrix(w2c_cam[:3,:3],w2c_cam[:3,3]).transpose(0,1).contiguous()
    proj_matrix=get_proj_matrix(tanfov,device=w2c_cam.device,z_near=0.01, z_far=100, z_sign=1.0).transpose(0,1).contiguous()
    full_proj_matrix = (view_matrix.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)#torch.mm(view_matrix, proj_matrix)

    return view_matrix,full_proj_matrix

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

class VertexPositionShader(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        """
        :param fragments: Fragments of the meshes that are rasterized.
        :param meshes: Meshes to render.
        :param kwargs: Additional arguments passed by the renderer.
        :return: The output colors, which in this case will be the vertex positions.
        """
        pixel_positions = fragments.pix_to_face  # shape (num_pixels, 3)
        batch_size, H, W = pixel_positions.shape[0], pixel_positions.shape[1], pixel_positions.shape[2]
        bary_coords=fragments.bary_coords.squeeze(-2)

        alpha = (pixel_positions!=-1)*1.0
        vertex_faces = meshes.faces_packed()[pixel_positions.squeeze(-1)]#  # shape (num_pixels, 3)
        vertex_positions=(meshes.verts_packed()[vertex_faces]*bary_coords[...,None]).sum(dim=-2)#.mean(dim=-2)
        results=torch.cat([vertex_positions,alpha],dim=-1)
        extra_result={"vertex_faces":vertex_faces,"bary_coords":bary_coords}
        return results,extra_result

class GS_Camera(CamerasBase):
    #still obey pytorch 3d coordinate system
    def __init__(
        self,
        focal_length=1.0,
        R: torch.Tensor = torch.eye(3)[None],
        T: torch.Tensor = torch.zeros(1, 3),
        principal_point=((0.0, 0.0),),
        device = "cpu",
        in_ndc: bool = True,
        image_size = None,
    ) -> None:

        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=((0.0, 0.0),),
            R=R,
            T=T,
            K=None,
            _in_ndc=in_ndc,
            **kwargs,
        )
        if image_size is not None:
            if (self.image_size < 1).any():
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)
        self.proj_mats=None

    def transform_points_to_view(self, points, eps = None, **kwargs):
        #from wold to view
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        if R.dim() == 2 :
            Tmat=torch.eye(4,device=R.device)[None]
            Tmat[:,:3,:3] = R
            Tmat[:,:3,3] = T
        else:

            Tmat=torch.eye(4,device=R.device)[None].repeat(R.shape[0],1,1)
            Tmat[:,:3,:3] = R
            Tmat[:,:3,3] = T

        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))
        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)
        points_out=torch.einsum('bij,bnj->bni',Tmat,points_batch)
        return points_out[:,:,:3]

    def get_projection_transform(self,device):
        if self.proj_mats is None:
            proj_mats=[]
            if  torch.unique(self.focal_length).numel()==1:
                invtanfov=self.focal_length[0,0]
                proj_mat=get_proj_matrix(1/invtanfov,device)
                proj_mats=proj_mat[None].repeat(self.focal_length.shape[0],1,1)
            else:
                for invtanfov in self.focal_length:
                    invtanfov=invtanfov[0]; assert invtanfov[0]==invtanfov[1]
                    proj_mat=get_proj_matrix(1/invtanfov,device)
                    proj_mats.append(proj_mat[None])
                proj_mats=torch.cat(proj_mats,dim=0)
            self.proj_mats=proj_mats
        else:
            proj_mats=self.proj_mats
        return proj_mats

    def transform_points_to_ndc(self, points, eps = None, **kwargs):
        #from wold to ndc
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        if R.dim() == 2 :
            Tmat=torch.eye(4,device=R.device)[None]
            Tmat[:,:3,:3] = R.clone()
            Tmat[:,:3,3] = T.clone()
        else:

            Tmat=torch.eye(4,device=R.device)[None].repeat(R.shape[0],1,1)
            Tmat[:,:3,:3] = R.clone()
            Tmat[:,:3,3] = T.clone()

        N, P, _3 = points.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=2)
        proj_mat=self.get_projection_transform(points.device)
        proj_mat=proj_mat.to(R.device)

        full_mat=torch.bmm(proj_mat,Tmat)
        points_ndc=torch.einsum('bij,bnj->bni',full_mat,points_h)
        points_ndc_xyz=points_ndc[:,:,:3]/(points_ndc[:,:,3:]+1e-7)
        return points_ndc_xyz

    def transform_points_view_to_ndc(self, points, eps = None, **kwargs):
        #from view to ndc
        points_view=points.clone()
        N, P, _3 = points_view.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points_view.device)
        points_view = torch.cat([points_view, ones], dim=2)

        proj_mat=self.get_projection_transform(points.device)
        points_ndc=torch.einsum('bij,bnj->bni',proj_mat,points_view)
        points_ndc_xyz=points_ndc[:,:,:3]/(points_ndc[:,:,3:]+1e-7)

        return points_ndc_xyz

    def transform_points_to_screen(self, points, with_xyflip = True, **kwargs):
        #from wold to screen
        'with_xyflip: obey pytroch 3d coordinate'
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T

        points_ndc=self.transform_points_to_ndc(points)
        N, P, _3 = points_ndc.shape
        image_size=self.image_size
        if not torch.is_tensor(image_size):
            image_size = torch.tensor(image_size, device=R.device)
        if image_size.dim()==2:
            image_size = image_size[:,None]
        image_size=image_size[:,:,[1,0]]#width height

        points_screen=points_ndc.clone()
        points_screen[...,:2]=points_ndc[...,:2]*image_size/2-image_size/2
        if with_xyflip:
            points_screen[...,:2]=points_screen[:,:,:2]*-1

        return points_screen

    def transform_points_screen(self, points, with_xyflip = True, **kwargs):
        return self.transform_points_to_screen(points, with_xyflip, **kwargs)


class GS_MeshRasterizer(MeshRasterizer):
    """
    adapted to GS_camera
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """

    def __init__(self, cameras:GS_Camera=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.transform_points_to_view(verts_world, eps=eps,**kwargs)
        verts_ndc =  cameras.transform_points_view_to_ndc(verts_view, eps=eps,**kwargs)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        perspective_correct=False
        z_clip = None
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = True
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )


class BaseMeshRenderer(L.LightningModule):
    def __init__(self, faces,image_size=512,lbs_weights=None, skin_color=[252, 224, 203], bg_color=[0, 0, 0],
                 faces_uvs=None,verts_uvs=None,focal_length=24,inverse_light=False):
        super(BaseMeshRenderer, self).__init__()

        self.image_size = image_size

        self.skin_color = np.array(skin_color)
        self.bg_color = bg_color
        self.focal_length = focal_length
        bin_size=None

        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                    bin_size=bin_size)
        if inverse_light:
            self.lights = PointLights( location=[[0.0, -1.0, -10.0]])
        else:
            self.lights = PointLights( location=[[0.0, 1.0, 10.0]])
        self.manual_lights = PointLights(
            location=((0.0, 0.0, 5.0), ),
            ambient_color=((0.5, 0.5, 0.5), ),
            diffuse_color=((0.5, 0.5, 0.5), ),
            specular_color=((0.01, 0.01, 0.01), )
        )
        self.blend = BlendParams(background_color=np.array(bg_color)/225.)
        self.faces = torch.nn.Parameter(faces, requires_grad=False)
        if faces_uvs is not None:
            self.faces_uvs = torch.nn.Parameter(faces_uvs, requires_grad=False)
        if verts_uvs is not None:
            self.verts_uvs = torch.nn.Parameter(verts_uvs.clone(), requires_grad=False)
            self.verts_uvs[:,1]=1-self.verts_uvs[:,1]
        self.lbs_weights = None
        if lbs_weights is not None: self.lbs_weights = torch.nn.Parameter(lbs_weights, requires_grad=False)

    def _build_cameras(self, transform_matrix, focal_length):
        device = transform_matrix.device
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=device).float(), 'focal_length': focal_length,
            'image_size': screen_size, 'device': device,
        }
        cameras = GS_Camera(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, ret_image=True):
        B, V = vertices.shape[:2]
        focal_length = self.focal_length if focal_length is None else focal_length
        if isinstance(cameras, torch.Tensor):
            cameras = cameras.clone()
        elif cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length)

        t_faces = faces[None].repeat(B, 1, 1)

        ret_vertices = cameras.transform_points_screen(vertices)
        ret_landmarks = {k: cameras.transform_points_screen(v) for k,v in landmarks.items()}

        images = None
        if ret_image:
            # Initialize each vertex to be white in color.
            verts_rgb = torch.from_numpy(self.skin_color/255).float().to(self.device)[None, None, :].repeat(B, V, 1)
            textures = TexturesVertex(verts_features=verts_rgb)
            mesh = Meshes(
                verts=vertices.to(self.device),
                faces=t_faces.to(self.device),
                textures=textures
            )
            renderer = MeshRenderer(
                rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
                shader=SoftPhongShader(cameras=cameras, lights=self.lights.to(vertices.device), device=self.device, blend_params=self.blend)
            )
            render_results = renderer(mesh).permute(0, 3, 1, 2)
            images = render_results[:, :3]
            alpha_images = render_results[:, 3:]
            images[alpha_images.expand(-1, 3, -1, -1)<0.5] = 0.0
            images = images * 255

        return ret_vertices, ret_landmarks, images

    def render_mesh(self, vertices,cameras=None,transform_matrix=None, faces=None,lights=None,reverse_camera=True):
        device = vertices.device
        B, V = vertices.shape[:2]

        if faces is None:
            faces = self.faces
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)
        t_faces = faces[None].repeat(B, 1, 1)
        mesh = Meshes(
            verts=vertices.to(device),
            faces=t_faces.to(device),
        )
        shader = VertexPositionShader().to(device)
        rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        #GS_MeshRasterizer MeshRasterizer
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        render_results,extra_result = renderer(mesh)
        render_lbs_weights=None
        if self.lbs_weights is not None:
            vertex_faces=extra_result['vertex_faces']
            bary_coords=extra_result['bary_coords']
            lbs_weights=self.lbs_weights[None].expand(B, -1, -1).reshape(-1,55)
            render_lbs_weights=(lbs_weights[vertex_faces]*bary_coords[...,None]).sum(dim=-2)
        return render_results,render_lbs_weights

    def render_fragments(self, vertices,cameras=None,transform_matrix=None, faces=None,reverse_camera=True):
        device = vertices.device
        B, V = vertices.shape[:2]

        if faces is None:
            faces = self.faces
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)
        t_faces = faces[None].repeat(B, 1, 1)
        mesh = Meshes(
            verts=vertices.to(device),
            faces=t_faces.to(device),
        )
        rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(mesh)
        #return visble faces
        return fragments.pix_to_face,fragments

    def render_textured_mesh(self,vertices,uvmap,fragments=None,faces_uvs=None,verts_uvs=None,faces=None,cameras=None,transform_matrix=None,reverse_camera=True):
        device = vertices.device
        B, V = vertices.shape[:2]
        if faces is None:
            faces = self.faces
        if faces_uvs is None:
            faces_uvs = self.faces_uvs
        if verts_uvs is None:
            verts_uvs = self.verts_uvs

        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)

        t_faces = faces[None].repeat(B, 1, 1)
        t_faces_uvs = faces_uvs[None].repeat(B, 1, 1)
        t_verts_uvs = verts_uvs[None].repeat(B, 1, 1)

        textures = TexturesUV(maps=uvmap,faces_uvs=t_faces_uvs,verts_uvs=t_verts_uvs)
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=t_faces.to(self.device),
            textures=textures
        )
        lights = PointLights( location=[[0.0, 0.0, 1000.0]])
        if fragments is None:
            rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
            fragments = rasterizer(mesh)
        shader=SoftPhongShader(cameras=cameras, lights=lights, device=device, blend_params=self.blend).to(device)
        images=shader(fragments, mesh)

        return images

if __name__=="__main__":
    pass