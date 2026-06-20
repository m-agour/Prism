#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class CurvatureComputer(nn.Module):
    
    def __init__(self, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
    
    def compute_dyadic(self, fiber_dirs: torch.Tensor) -> torch.Tensor:
        fiber_dirs = F.normalize(fiber_dirs, dim=-1, eps=1e-8)
        
        T = torch.einsum('...i,...j->...ij', fiber_dirs, fiber_dirs)
        
        return T
    
    def compute_curvature_from_dyadic(self, T: torch.Tensor) -> torch.Tensor:
        X, Y, Z = T.shape[:3]
        device = T.device
        
        
        grad_norm_sq = torch.zeros(X, Y, Z, device=device)
        
        dT_dx = torch.zeros_like(T)
        dT_dx[1:-1, :, :] = (T[2:, :, :] - T[:-2, :, :]) / (2 * self.voxel_size[0])
        dT_dx[0, :, :] = (T[1, :, :] - T[0, :, :]) / self.voxel_size[0]
        dT_dx[-1, :, :] = (T[-1, :, :] - T[-2, :, :]) / self.voxel_size[0]
        
        dT_dy = torch.zeros_like(T)
        dT_dy[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * self.voxel_size[1])
        dT_dy[:, 0, :] = (T[:, 1, :] - T[:, 0, :]) / self.voxel_size[1]
        dT_dy[:, -1, :] = (T[:, -1, :] - T[:, -2, :]) / self.voxel_size[1]
        
        dT_dz = torch.zeros_like(T)
        dT_dz[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * self.voxel_size[2])
        dT_dz[:, :, 0] = (T[:, :, 1] - T[:, :, 0]) / self.voxel_size[2]
        dT_dz[:, :, -1] = (T[:, :, -1] - T[:, :, -2]) / self.voxel_size[2]
        
        grad_norm_sq = (dT_dx ** 2).sum(dim=(-2, -1)) + \
                       (dT_dy ** 2).sum(dim=(-2, -1)) + \
                       (dT_dz ** 2).sum(dim=(-2, -1))
        
        curvature = torch.sqrt(grad_norm_sq + 1e-10)
        
        return curvature
    
    def forward(self, fiber_dirs: torch.Tensor) -> torch.Tensor:
        if fiber_dirs.dim() == 4:
            T = self.compute_dyadic(fiber_dirs)
            return self.compute_curvature_from_dyadic(T)
        elif fiber_dirs.dim() == 5:
            X, Y, Z, K, _ = fiber_dirs.shape
            curvatures = []
            for k in range(K):
                T = self.compute_dyadic(fiber_dirs[:, :, :, k, :])
                c = self.compute_curvature_from_dyadic(T)
                curvatures.append(c)
            return torch.stack(curvatures, dim=-1)
        else:
            raise ValueError(f"Expected 4D or 5D fiber_dirs, got {fiber_dirs.dim()}D")


class GeometryPredictedDispersion(nn.Module):
    
    def __init__(
        self,
        alpha_init: float = 10000.0,
        epsilon: float = 0.2,
        min_kappa: float = 0.5,
        max_kappa: float = 100.0,
        d_parallel: float = 1.7,
        delta: float = 30.0,
    ):
        super().__init__()
        
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_init)))
        
        self.epsilon = epsilon
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        
        L = np.sqrt(2 * d_parallel * 1e-3 * delta)
        self.register_buffer('L', torch.tensor(L))
    
    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)
    
    def forward(
        self, 
        curvature: torch.Tensor,
        residual_kappa: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        variance = self.alpha * (curvature ** 2) * (self.L ** 2)
        
        kappa = 1.0 / (variance + self.epsilon)
        
        if residual_kappa is not None:
            kappa = kappa + F.softplus(residual_kappa)
        
        kappa = torch.clamp(kappa, self.min_kappa, self.max_kappa)
        
        return kappa


class CurvatureDependentDiffusivity(nn.Module):
    
    def __init__(
        self,
        d_parallel_0: float = 1.7,
        gamma_init: float = 1.0,
        min_d_parallel: float = 0.5,
    ):
        super().__init__()
        
        self.d_parallel_0 = d_parallel_0
        self.min_d_parallel = min_d_parallel
        
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init)))
    
    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)
    
    def forward(self, curvature: torch.Tensor) -> torch.Tensor:
        tortuosity = 1.0 / (1.0 + self.gamma * curvature ** 2)
        
        d_parallel_eff = self.d_parallel_0 * tortuosity
        
        d_parallel_eff = torch.clamp(d_parallel_eff, min=self.min_d_parallel)
        
        return d_parallel_eff


class StreamlineCurvatureMap(nn.Module):
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        affine: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        self.volume_shape = volume_shape
        self.voxel_size = voxel_size
        self.affine = affine if affine is not None else np.eye(4)
        self.inv_affine = np.linalg.inv(self.affine)
    
    def streamline_to_voxels(self, streamline: np.ndarray) -> np.ndarray:
        ones = np.ones((len(streamline), 1))
        pts_homo = np.hstack([streamline, ones])
        
        voxels = (self.inv_affine @ pts_homo.T).T[:, :3]
        
        return voxels
    
    def compute_streamline_curvature(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) < 3:
            return np.zeros(len(points)), np.zeros((len(points), 3))
        
        tangents = np.gradient(points, axis=0)
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.maximum(tangent_norms, 1e-8)
        unit_tangents = tangents / tangent_norms
        
        curvature_vec = np.gradient(unit_tangents, axis=0)
        curvature = np.linalg.norm(curvature_vec, axis=1)
        
        return curvature, unit_tangents
    
    def build_curvature_volume(self, streamlines: list) -> Tuple[np.ndarray, np.ndarray]:
        X, Y, Z = self.volume_shape
        
        curvature_sum = np.zeros((X, Y, Z))
        direction_sum = np.zeros((X, Y, Z, 3))
        count = np.zeros((X, Y, Z))
        
        for sl in streamlines:
            voxels = self.streamline_to_voxels(sl)
            
            curv, tangents = self.compute_streamline_curvature(voxels)
            
            for i, (vox, c, t) in enumerate(zip(voxels, curv, tangents)):
                ix, iy, iz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
                
                if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
                    curvature_sum[ix, iy, iz] += c
                    direction_sum[ix, iy, iz] += t
                    count[ix, iy, iz] += 1
        
        count = np.maximum(count, 1)
        curvature_vol = curvature_sum / count
        direction_vol = direction_sum / count[..., np.newaxis]
        
        dir_norms = np.linalg.norm(direction_vol, axis=-1, keepdims=True)
        dir_norms = np.maximum(dir_norms, 1e-8)
        direction_vol = direction_vol / dir_norms
        
        return curvature_vol, direction_vol


class CurvatureAwareSignalModel(nn.Module):
    
    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_geometry_kappa: bool = True,
        use_curvature_diffusivity: bool = False,
        allow_residual_kappa: bool = True,
        d_parallel: float = 1.7,
        d_perp: float = 0.4,
        delta: float = 30.0,
    ):
        super().__init__()
        
        self.use_geometry_kappa = use_geometry_kappa
        self.use_curvature_diffusivity = use_curvature_diffusivity
        self.allow_residual_kappa = allow_residual_kappa
        
        self.curvature_computer = CurvatureComputer(voxel_size)
        
        if use_geometry_kappa:
            self.kappa_predictor = GeometryPredictedDispersion(
                d_parallel=d_parallel,
                delta=delta,
            )
        
        if use_curvature_diffusivity:
            self.diffusivity_model = CurvatureDependentDiffusivity(
                d_parallel_0=d_parallel,
            )
        
        self.d_parallel = d_parallel
        self.d_perp = d_perp
    
    def get_curvature_and_params(
        self,
        fiber_dirs: torch.Tensor,
        residual_kappa: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        curvature = self.curvature_computer(fiber_dirs)
        
        if self.use_geometry_kappa:
            kappa = self.kappa_predictor(
                curvature,
                residual_kappa if self.allow_residual_kappa else None
            )
        else:
            if residual_kappa is not None:
                kappa = F.softplus(residual_kappa) + 0.5
            else:
                kappa = torch.ones_like(curvature) * 8.0
        
        if self.use_curvature_diffusivity:
            d_parallel = self.diffusivity_model(curvature)
        else:
            d_parallel = torch.ones_like(curvature) * self.d_parallel
        
        return {
            'curvature': curvature,
            'kappa': kappa,
            'd_parallel': d_parallel,
            'd_perp': torch.ones_like(curvature) * self.d_perp,
        }



class StreamlineForwardModel(nn.Module):
    
    def __init__(
        self,
        streamlines: list,
        volume_shape: Tuple[int, int, int],
        bvals: np.ndarray,
        bvecs: np.ndarray,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        affine: Optional[np.ndarray] = None,
        d_parallel: float = 1.7,
        d_perp: float = 0.4,
        f_intra: float = 0.7,
    ):
        super().__init__()
        
        self.volume_shape = volume_shape
        self.n_streamlines = len(streamlines)
        self.d_parallel = d_parallel
        self.d_perp = d_perp
        self.f_intra = f_intra
        
        occupancy, tangents, segment_lengths = self._build_mapping(
            streamlines, volume_shape, voxel_size, affine
        )
        
        self.register_buffer('occupancy', occupancy)
        self.register_buffer('tangents', tangents)
        self.register_buffer('segment_lengths', segment_lengths)
        
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        self.log_weights = nn.Parameter(torch.zeros(self.n_streamlines))
        
        self.log_f_intra = nn.Parameter(torch.zeros(self.n_streamlines))
    
    def _build_mapping(
        self,
        streamlines: list,
        volume_shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float],
        affine: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, Y, Z = volume_shape
        V = X * Y * Z
        S = len(streamlines)
        
        inv_affine = np.linalg.inv(affine) if affine is not None else np.eye(4)
        
        occupancy = torch.zeros(V, S)
        tangents = torch.zeros(V, S, 3)
        segment_lengths = torch.zeros(V, S)
        
        for s_idx, sl in enumerate(streamlines):
            ones = np.ones((len(sl), 1))
            pts_homo = np.hstack([sl, ones])
            voxels = (inv_affine @ pts_homo.T).T[:, :3]
            
            if len(voxels) < 2:
                continue
            tangent_vecs = np.diff(voxels, axis=0)
            tangent_norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
            tangent_norms = np.maximum(tangent_norms, 1e-8)
            unit_tangents = tangent_vecs / tangent_norms
            
            for i in range(len(voxels) - 1):
                p0, p1 = voxels[i], voxels[i+1]
                t = unit_tangents[i]
                seg_len = tangent_norms[i, 0]
                
                mid = (p0 + p1) / 2
                ix, iy, iz = int(round(mid[0])), int(round(mid[1])), int(round(mid[2]))
                
                if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
                    v_idx = ix * Y * Z + iy * Z + iz
                    occupancy[v_idx, s_idx] += seg_len
                    tangents[v_idx, s_idx] += torch.tensor(t * seg_len, dtype=torch.float32)
                    segment_lengths[v_idx, s_idx] += seg_len
        
        seg_len_safe = segment_lengths.unsqueeze(-1).clamp(min=1e-8)
        tangents = tangents / seg_len_safe
        tangents = F.normalize(tangents, dim=-1, eps=1e-8)
        
        
        return occupancy, tangents, segment_lengths
    
    @property
    def weights(self) -> torch.Tensor:
        return F.softplus(self.log_weights)
    
    @property
    def f_intra_per_streamline(self) -> torch.Tensor:
        return torch.sigmoid(self.log_f_intra)
    
    def compute_stick_signal(
        self,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        tangent: torch.Tensor,
        d_parallel: float,
    ) -> torch.Tensor:
        cos_theta = torch.einsum('md,...d->...m', bvecs, tangent)
        cos2_theta = cos_theta ** 2
        
        b_norm = bvals / 1000.0
        signal = torch.exp(-b_norm * d_parallel * cos2_theta)
        
        return signal
    
    def compute_zeppelin_signal(
        self,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        tangent: torch.Tensor,
        d_parallel: float,
        d_perp: float,
    ) -> torch.Tensor:
        cos_theta = torch.einsum('md,...d->...m', bvecs, tangent)
        cos2_theta = cos_theta ** 2
        
        D_eff = d_perp + (d_parallel - d_perp) * cos2_theta
        
        b_norm = bvals / 1000.0
        signal = torch.exp(-b_norm * D_eff)
        
        return signal
    
    def forward(
        self,
        f_csf: torch.Tensor,
        f_gm: torch.Tensor,
        s0: torch.Tensor,
        d_csf: float = 3.0,
        d_gm: float = 0.9,
    ) -> torch.Tensor:
        V = self.occupancy.shape[0]
        M = self.bvals.shape[0]
        device = self.occupancy.device
        
        b_norm = self.bvals / 1000.0
        S_csf = torch.exp(-b_norm * d_csf)
        S_gm = torch.exp(-b_norm * d_gm)
        
        
        weights = self.weights
        f_intra = self.f_intra_per_streamline
        
        
        S_stick = self.compute_stick_signal(
            self.bvals, self.bvecs, self.tangents, self.d_parallel
        )
        
        S_zep = self.compute_zeppelin_signal(
            self.bvals, self.bvecs, self.tangents, self.d_parallel, self.d_perp
        )
        
        S_wm_per_stream = (
            f_intra.view(1, -1, 1) * S_stick +
            (1 - f_intra.view(1, -1, 1)) * S_zep
        )
        
        weighted_occupancy = self.occupancy * weights.unsqueeze(0)
        
        S_wm = (weighted_occupancy.unsqueeze(-1) * S_wm_per_stream).sum(dim=1)
        
        total_weight = weighted_occupancy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        S_wm = S_wm / total_weight
        
        f_wm = weighted_occupancy.sum(dim=1)
        f_wm = f_wm / (f_wm.max() + 1e-8)
        
        total_f = f_csf + f_gm + f_wm
        f_csf_norm = f_csf / (total_f + 1e-8)
        f_gm_norm = f_gm / (total_f + 1e-8)
        f_wm_norm = f_wm / (total_f + 1e-8)
        
        signal = s0.unsqueeze(-1) * (
            f_csf_norm.unsqueeze(-1) * S_csf +
            f_gm_norm.unsqueeze(-1) * S_gm +
            f_wm_norm.unsqueeze(-1) * S_wm
        )
        
        return signal



def kappa_to_odi(kappa: torch.Tensor) -> torch.Tensor:
    return 2.0 / np.pi * torch.arctan(1.0 / kappa)


def odi_to_kappa(odi: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tan(np.pi / 2 * odi)


def validate_curvature_from_streamlines(
    streamlines: list,
    estimated_curvature: np.ndarray,
    volume_shape: Tuple[int, int, int],
    affine: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    mapper = StreamlineCurvatureMap(volume_shape, affine=affine)
    gt_curvature, _ = mapper.build_curvature_volume(streamlines)
    
    mask = (gt_curvature > 0) & (estimated_curvature > 0)
    
    if mask.sum() == 0:
        return {'correlation': 0.0, 'mae': 0.0, 'n_voxels': 0}
    
    gt_flat = gt_curvature[mask]
    est_flat = estimated_curvature[mask]
    
    correlation = np.corrcoef(gt_flat, est_flat)[0, 1]
    
    mae = np.abs(gt_flat - est_flat).mean()
    
    return {
        'correlation': float(correlation),
        'mae': float(mae),
        'n_voxels': int(mask.sum()),
        'gt_mean': float(gt_flat.mean()),
        'est_mean': float(est_flat.mean()),
    }
