import numpy as np
from torchmin import minimize
import trimesh
import torch

def rodrigues_to_unit_axis(r):
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-14:
        return np.array([0.0, 1.0, 0.0], dtype=float)  # default +Y
    return r / norm_r

def rotation_from_yaxis_to_vector(target_axis):
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    # if already close, return identity
    if np.allclose(target_axis, y_axis, atol=1e-14):
        return np.eye(3)
    # if nearly opposite, 180° about x or z
    if np.allclose(target_axis, -y_axis, atol=1e-14):
        return np.diag([1.0, -1.0, -1.0])  # rotate +Y->-Y around X by 180

    # general case
    v = np.cross(y_axis, target_axis)
    c = np.dot(y_axis, target_axis)  # cos(theta)
    s = np.linalg.norm(v)            # sin(theta)
    vx = np.array([
        [0.0,    -v[2],   v[1]],
        [v[2],    0.0,   -v[0]],
        [-v[1],   v[0],  0.0]
    ])
    I = np.eye(3)
    # Rodrigues formula
    R = I + vx + vx @ vx * ((1.0 - c)/(s*s))
    return R

def get_cylinder_transform_yaxis(cx, cy, cz, rx, ry, rz, radius, height):
    """
    Build a 4x4 transform that maps a canonical cylinder:
      - axis +Y
      - radius=1 in x,z
      - height=1 (centered around origin)
    into one with:
      - center (cx,cy,cz)
      - axis from (rx,ry,rz) after normalization
      - radius=<radius>, height=<height>.
    """
    # Scale matrix
    S = np.diag([radius, height, radius, 1.0])

    # Rotation: +Y -> target axis
    rod = np.array([rx, ry, rz], dtype=float)
    axis = rodrigues_to_unit_axis(rod)
    R_3x3 = rotation_from_yaxis_to_vector(axis)
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R_3x3

    # Translation
    T_4x4 = np.eye(4)
    T_4x4[0, 3] = cx
    T_4x4[1, 3] = cy
    T_4x4[2, 3] = cz

    # Final M = T * R * S
    M = T_4x4 @ R_4x4 @ S
    return M

def rodrigues_to_axis(r):
    """
    Convert a 3D Rodrigues vector r = (rx, ry, rz) to a UNIT axis vector.
    If ||r|| is near zero, returns (0, 0, 1).
    Otherwise, axis = r / ||r||
    """
    theta = torch.norm(r)
    if theta < 1e-14:
        # default axis = +Z if no rotation
        return torch.tensor([0.0, 0.0, 1.0], dtype=r.dtype, device=r.device)
    return r / theta


def distance_to_finite_cylinder(pts, c, axis_rod, radius, height):
    # 1) Convert Rodrigues to a unit axis
    axis = rodrigues_to_axis(axis_rod)  # shape (3,)

    # 2) For each point X, define t = dot(X - c, axis).
    #    But we "clamp" t to [-h/2, h/2] so points above or below
    #    measure distance to the side at the top or bottom edge.
    half_h = height * 0.5
    Xc = pts - c  # shape (N,3)

    t = torch.sum(Xc * axis, dim=1)  # (N,)
    # clamp t to the cylinder's finite height range
    t_clamped = torch.clamp(t, -half_h, half_h)

    # 3) Now compute the vector from X to the clamped axis point:
    #    axis_pt = c + t_clamped * axis
    #    radial_vec = X - axis_pt = Xc - t_clamped*axis
    radial_vec = Xc - t_clamped.unsqueeze(1)*axis  # (N,3)

    # radial distance from the cylinder axis
    radial_dist = torch.norm(radial_vec, dim=1)  # (N,)

    # 4) Distance to the side = |radial_dist - radius|
    dists = torch.abs(radial_dist - radius)
    return dists


def objective_function(p, pts, r, h):
    """
    Sum of squared distances from 'pts' to the finite cylinder
    parameterized by p:
       p = [cx, cy, cz, rx, ry, rz, radius, height]

    Returns a scalar (the sum of d^2).
    """
    cx, cy, cz, rx, ry, rz= p
    c = torch.stack([cx, cy, cz])
    rod = torch.stack([rx, ry, rz])
    # Distance for each point
    d = distance_to_finite_cylinder(pts, c, rod, r, h)
    
    com = torch.mean(pts, dim=0)
    d_com = torch.norm(com - c)
    return (d * d).sum() + 1e3 * d_com 


def fit_cylinder_torch(points_np):
    """
    Fits a finite cylinder to the given Nx3 'points_np' (numpy array)
    by minimizing sum of squared distances to the cylinder surface.

    The cylinder is parameterized by 8 params:
      [cx, cy, cz, rx, ry, rz, radius, height]
    with 'rx, ry, rz' = Rodrigues axis representation for the cylinder axis.

    Returns: (cx, cy, cz, axis_vec, radius, height, final_loss)
      axis_vec is the normalized axis in R^3
    """
    device = torch.device("cpu")

    # Convert data to a torch tensor
    pts = torch.tensor(points_np, dtype=torch.float64, device=device)

    # ========== 1) Initialize parameters ==========
    # We'll guess that the cylinder is near the bounding box center,
    # axis ~ +Z, radius ~ something from bounding box, height ~ bounding box size
    min_xyz = points_np.min(axis=0)
    max_xyz = points_np.max(axis=0)
    center_guess = (min_xyz + max_xyz)/2.0
    bbox_size = (max_xyz - min_xyz)
    guess_radius = 0.5*max(bbox_size[0], bbox_size[1])  # a rough guess
    guess_height = bbox_size[1]  # or whichever dimension is largest

    p_init = np.array([
        center_guess[0],
        center_guess[1],
        center_guess[2],
        0.0, 1.0, 0.0,  # rodrigues => axis near +Y
    ], dtype=np.float64)

    p = torch.tensor(p_init, dtype=torch.float64, device=device, requires_grad=True)

    # ========== 2) Define the closure for torchmin ==========
    def closure(x):
        return objective_function(x, pts, guess_radius, guess_height)

    # ========== 3) Run optimization (Newton's method or LBFGS, etc.) ==========
    # We'll do newton-exact. If it's unstable or not converging,
    # try method='dogleg' or method='lbfgs'.
    res = minimize(closure, p, method='newton-exact',
                   options={'disp': 2, 'xtol': 1e-5})

    p_opt = res.x.detach()  # final parameter vector
    final_loss = res.fun.item()

    # Unpack
    radius = guess_radius
    height = guess_height
    cx, cy, cz, rx, ry, rz = p_opt.tolist()
    rod = np.array([rx, ry, rz], dtype=float)
    axis_len = np.linalg.norm(rod)
    if axis_len < 1e-14:
        axis_vec = np.array([0, 0, 1], dtype=float)
    else:
        axis_vec = rod / axis_len

    return cx, cy, cz, axis_vec, radius, height, final_loss


if __name__ == "__main__":
    tgt_mesh = trimesh.load('mug.obj')
    cx, cy, cz, axis_vec, radius, height, _ = fit_cylinder_torch(np.array(tgt_mesh.vertices))
    M = get_cylinder_transform_yaxis(cx, cy, cz, axis_vec[0], axis_vec[1], axis_vec[2], radius, height)
    
    init_cylider = trimesh.load('cylinder.obj')
    # rotate to be y-axis aligned
    R = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    init_cylider.apply_transform(R)
    init_cylider.apply_transform(M)
    init_cylider.export('cylinder_fit.obj')
    
