"""
Functions used for SE(3) transformations. (Torch implementation).
Has support for operations with batches.

Namely, converting quaternions to rotation matrices, getting an SE(3) transform from SE(3)
parameters, and applying the SE(3) transformation on a set of points.

Credit to Lewis J. Martin as this was adapted from
https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
and PyTorch's implementations.
"""
import torch
import torch.nn.functional as F
import numpy as np


def quaternion_to_SE3(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    q : (4,) torch tensor, already normalised (r,i,j,k)
    t : (3,) torch tensor, translation
    Returns: (4,4) torch.float32 on same device as q
    """
    r, i, j, k = q
    two = q.new_tensor(2.0)                      
    R = torch.stack(
        (
            1 - two * (j * j + k * k),
            two * (i * j - k * r),
            two * (i * k + j * r),
            two * (i * j + k * r),
            1 - two * (i * i + k * k),
            two * (j * k - i * r),
            two * (i * k - j * r),
            two * (j * k + i * r),
            1 - two * (i * i + j * j),
        ),
        dim=0,
    ).view(3, 3)
    SE3 = torch.eye(4, dtype=q.dtype, device=q.device)
    SE3[:3, :3] = R
    SE3[:3,  3] = t
    return SE3

def quaternions_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternion to a rotation matrix. Supports batched and non-batched inputs.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
    
    Parameters
    ----------
    quaternions : torch.Tensor (batch, 4) or (4,)
        Quaternion parameters in (r,i,j,k) order. Accepts single set of parameters or a batched
        set.
    
    Returns
    -------
    rotation_matrix : torch.Tensor (batch, 3, 3) or (3,3)
        Rotation matrix converted from quaternion in batched or single instance form.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Last dimension of "quaternions" must be length 4. Instead the shape given was: {quaternions.shape}')

    # Single instance
    if len(quaternions.shape) == 1:
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = torch.Tensor([2.0]).to(quaternions.device) / torch.sum(quaternions **2)

        rotation_matrix = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return rotation_matrix.reshape(quaternions.shape[:-1] + (3, 3))

    # Batched
    elif len(quaternions.shape) == 2:
        r, i, j, k = torch.unbind(quaternions, 1)
        two_s = torch.Tensor([2.0]).to(quaternions.device) / torch.sum(quaternions ** 2, dim=1)
        rotation_matrix = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            1
        ).reshape((-1, 3, 3))
        return rotation_matrix
    else:
        raise ValueError(f'Input "quaternions" must be a 1D Tensor of length 4 or a batched version of shape (batch_size,4). Instead the shape given was: {quaternions.shape}')


def get_SE3_transform(se3_params: torch.Tensor
                     ) -> torch.Tensor:
    """ Constructs an SE(3) transformtion matrix from parameters.
    Supports batched and non-batched inputs.
    
    Parameters
    ----------
    se3_params : torch.Tensor (batch, 7) or (7,)
        Parameters for SE(3) transformation.
        The first 4 values in the last dimension are quaternions of form (r,i,j,k)
        and the last 3 values of the last dimension are the translations in (x,y,z).
    
    Returns
    -------
    se3_matrix : torch.Tensor (batch, 4, 4) or (4, 4)
        se3_params converted to a 4x4 SE(3) transformation matrix.
    """
    if se3_params.shape[-1] != 7:
        raise ValueError(f'Last dimension of "se3_params" must be length 7. Instead the shape given was: {se3_params.shape}')

    # Single instance
    if len(se3_params.shape) == 1:
        # Extract quaternion and translation parameters
        quaternion_params = se3_params[:4]
        translation_params = se3_params[4:]

        # Normalize quaternion to ensure unit length
        quaternion_params = F.normalize(quaternion_params, p=2, dim=-1)
        rotation_matrix = quaternions_to_rotation_matrix(quaternion_params)

        # Construct SE(3) transformation matrix
        se3_matrix = torch.eye(4, dtype=torch.float32, device=se3_params.device)
        se3_matrix[:3, :3] = rotation_matrix
        se3_matrix[:3, 3] = translation_params
        return se3_matrix

    # Batched
    elif len(se3_params.shape) == 2:
        # Extract quaternion and translation parameters
        quaternion_params = se3_params[:, :4]
        translation_params = se3_params[:, 4:]

        # Normalize quaternion to ensure unit length
        quaternion_params = F.normalize(quaternion_params, p=2, dim=1)
        rotation_matrix = quaternions_to_rotation_matrix(quaternion_params)

        # Construct SE(3) transformation matrix
        se3_matrix = torch.eye(4, device=se3_params.device).repeat((quaternion_params.shape[0],1,1))
        se3_matrix[:, :3, :3] = rotation_matrix
        se3_matrix[:, :3, 3] = translation_params
        return se3_matrix

    else:
        raise ValueError(f'Input "se3_params" must be a 1D Tensor of length 7 or a batched version of shape (batch_size,7). Instead the shape given was: {se3_params.shape}')


def apply_SE3_transform(points, se3):
    """
    points : (R, M, 3) or (M, 3)
    se3    : (R, 7)  – quaternion + translation
             (R, 4, 4) – full SE(3) matrix
             matching batch size R (or broadcastable)
    returns: transformed points with same leading dims as `points`
    """
    # ---------------- single examples -> make them batched -----------------
    if se3.dim() == 1:                    # (7,)
        se3 = se3.unsqueeze(0)            # (1,7)
    if se3.dim() == 2 and se3.shape[-1] == 4 and se3.shape[-2] == 4:
        se3 = se3.unsqueeze(0)            # (1,4,4)
    if points.dim() == 2:                 # (M,3)
        points = points.unsqueeze(0)      # (1,M,3)

    R = points.shape[0]                   # batch / num-repeats

    # ----------------------------------------------------------------------
    #  CASE 1 – we were given a 4×4 matrix
    # ----------------------------------------------------------------------
    if se3.dim() == 3 and se3.shape[-2:] == (4, 4):
        rot   = se3[:, :3, :3]                 # (R,3,3)
        trans = se3[:, :3,  3].unsqueeze(1)    # (R,1,3)
        out = torch.baddbmm(trans, points, rot.transpose(1, 2))

    # ----------------------------------------------------------------------
    #  CASE 2 – we were given quaternion (r,i,j,k) + translation (x,y,z)
    # ----------------------------------------------------------------------
    elif se3.shape[-1] == 7:
        quat  = se3[:, :4]                     # (R,4)
        trans = se3[:, 4:].unsqueeze(1)        # (R,1,3)

        w, x, y, z = quat.unbind(-1)
        rot = torch.stack([
            1-2*(y*y+z*z),  2*(x*y - z*w),  2*(x*z + y*w),
            2*(x*y + z*w),  1-2*(x*x+z*z),  2*(y*z - x*w),
            2*(x*z - y*w),  2*(y*z + x*w),  1-2*(x*x+y*y)
        ], dim=-1).reshape(-1, 3, 3)           # (R,3,3)

        out = torch.baddbmm(trans, points, rot.transpose(1, 2))

    else:
        raise ValueError("`se3` must be (R,7) or (R,4,4)")

    # squeeze back if caller passed a single cloud
    return out[0] if out.shape[0] == 1 else out

# def apply_SE3_transform(points: torch.Tensor,
#                         SE3_transform: torch.Tensor
#                        ) -> torch.Tensor:
#     """
#     Takes a point cloud and transforms it according to the provided SE3 transformation matrix.
#     Supports batched and non-batched inputs.
    
#     Parameters
#     ----------
#     points : torch.Tensor (batch, N, 3) or (N, 3)
#         Set of coordinates representing a point cloud.
#     SE3_transform : torch.Tensor (batch, 4, 4) or (4, 4)
#         SE(3) transformation matrix. If 'points' argument is batched, this one should be too.
    
#     Returns
#     -------
#     transformed_points : torch.Tensor (batch, N, 3) or (N, 3)
#         Set of coordinates transformed by the corresponding SE(3) transformation.
#     """
#     if points.shape[-1] != 3:
#         raise ValueError(f'"points" should have shape (N_points, 3) or (batch, N_points, 3). Instead the shape given was: {points.shape}')
#     if SE3_transform.shape[-2:] != (4,4):
#         raise ValueError(f'"SE3_transform" should have shape (4, 4) or (batch, 4, 4). Instead the shape given was: {SE3_transform.shape}')
#     if len(SE3_transform.shape) != len(points.shape):
#         raise ValueError(f'Shapes of points and SE3_transform should be the same length. Instead {len(SE3_transform.shape)} and {len(points.shape)} were given.')

#     # ---------------------- single cloud ---------------------------
#     if SE3_transform.dim() == 2:                      # shapes (4,4)  / (N,3)
#         rot   = SE3_transform[:3, :3]                 # (3,3)
#         trans = SE3_transform[:3,  3]                 # (3,)
#         #  P·Rᵀ  +  t
#         return points @ rot.T + trans                 # (N,3)

#     # ---------------------- batched clouds -------------------------
#     # shapes:  (R,N,3)   (R,4,4)
#     rot   = SE3_transform[:, :3, :3]                  # (R,3,3)
#     trans = SE3_transform[:, :3, 3]                   # (R,3)

#     # torch.matmul broadcasts: (R,N,3) @ (R,3,3)ᵀ  → (R,N,3)
#     transformed = torch.matmul(points, rot.transpose(1, 2)) \
#                   + trans[:, None, :]                 # broadcast add
#     return transformed

    # else:
    #     raise ValueError(f'"points" and "SE3_transform" must be either batched or a single instance. \
    #     The expected length of shape for both should be 2 (single instance or 3 (batch) but {len(SE3_transform)} was given.')
    

def apply_SO3_transform(points: torch.Tensor,
                        SE3_transform: torch.Tensor
                        ) -> torch.Tensor:
    """
    Takes a point cloud and ONLY ROTATES it according to the provided SE3 transformation matrix.
    Supports batched and non-batched inputs.
    
    Parameters
    ----------
    points : torch.Tensor (batch, N, 3) or (N, 3)
        Set of coordinates representing a point cloud.
    SE3_transform : torch.Tensor (batch, 4, 4) or (4, 4)
        SE(3) transformation matrix. If 'points' argument is batched, this one should be too.
    
    Returns
    -------
    rotated_points : torch.Tensor (batch, N, 3) or (N, 3)
        Set of coordinates rotated by the rotation component of the SE(3) transformation.
    """
    if points.shape[-1] != 3:
        raise ValueError(f'"points" should have shape (N_points, 3) or (batch, N_points, 3). Instead the shape given was: {points.shape}')
    if SE3_transform.shape[-2:] != (4,4):
        raise ValueError(f'"SE3_transform" should have shape (4, 4) or (batch, 4, 4). Instead the shape given was: {SE3_transform.shape}')
    if len(SE3_transform.shape) != len(points.shape):
        raise ValueError(f'Shapes of points and SE3_transform should be the same length. Instead {len(SE3_transform.shape)} and {len(points.shape)} were given.')

    # Single instance
    if len(SE3_transform.shape) == 2:
        rotated_points = (SE3_transform[:3,:3] @ points.T).T
        return rotated_points

    # Batched
    elif len(SE3_transform.shape):
        rotated_points = torch.bmm(SE3_transform[:, :3,:3], points.permute(0,2,1)).permute(0,2,1)
        return rotated_points

    else:
        raise ValueError(f'"points" and "SE3_transform" must be either batched or a single instance. \
        The expected length of shape for both should be 2 (single instance or 3 (batch) but {len(SE3_transform)} was given.')
