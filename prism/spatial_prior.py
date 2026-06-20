
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialPrior3D(nn.Module):
    
    def __init__(self,
                 weight: float = 0.01,
                 connectivity: str = "6",
                 robust: bool = False,
                 huber_delta: float = 0.05,
                 fa_alpha: float = 2.0):
        super().__init__()
        self.weight = weight
        self.robust = robust
        self.huber_delta = huber_delta
        self.connectivity = connectivity
        self.fa_alpha = fa_alpha
        self.fa_threshold = 0.0
        
        self.register_buffer("fa_map", None)
        
        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        
        kernel[..., 1, 1, 1] = 1.0
        
        if connectivity == "6":
            neighbours = [
                (0, 1, 1), (2, 1, 1),
                (1, 0, 1), (1, 2, 1),
                (1, 1, 0), (1, 1, 2),
            ]
        elif connectivity == "26":
            neighbours = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        neighbours.append((1 + dz, 1 + dy, 1 + dx))
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}. Use '6' or '26'.")
        
        w = -1.0 / len(neighbours)
        for z, y, x in neighbours:
            kernel[..., z, y, x] = w
        
        self.register_buffer("kernel", kernel)
        self.n_neighbours = len(neighbours)
    
    def set_fa_map(self, fa: torch.Tensor, fa_threshold: float = 0.0) -> None:
        if fa.dim() == 4:
            fa = fa.unsqueeze(1)
        
        fa = fa.clamp(0.0, 1.0)
        
        fa_effective = torch.clamp(fa - fa_threshold, min=0.0)
        
        w_aniso = torch.exp(-self.fa_alpha * fa_effective)
        
        self.fa_threshold = fa_threshold
        
        self.register_buffer("fa_map", w_aniso)
    
    def clear_fa_map(self) -> None:
        self.register_buffer("fa_map", None)
        self.fa_threshold = 0.0
    
    def _robust_loss(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.robust:
            loss_per_voxel = x ** 2
        else:
            abs_x = x.abs()
            delta = self.huber_delta
            
            quad_mask = (abs_x <= delta).float()
            loss_quad = 0.5 * (x ** 2)
            
            loss_lin = delta * (abs_x - 0.5 * delta)
            
            loss_per_voxel = quad_mask * loss_quad + (1.0 - quad_mask) * loss_lin
        
        if mask is not None:
            n_valid = mask.sum().clamp(min=1.0)
            return (loss_per_voxel * mask).sum() / n_valid
        else:
            return loss_per_voxel.mean()
    
    def forward(self,
                frac: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = frac.unsqueeze(1)
        
        lap = F.conv3d(x, self.kernel, padding=1)
        
        if self.fa_map is not None:
            lap = lap * self.fa_map
        
        mask_5d = None
        if mask is not None:
            mask_5d = mask.unsqueeze(1).float()
            lap = lap * mask_5d
        
        loss = self._robust_loss(lap, mask=mask_5d)
        
        return self.weight * loss
    
    def compute_neighbour_diff_histogram(self,
                                         frac: torch.Tensor,
                                         mask: Optional[torch.Tensor] = None,
                                         n_bins: int = 50) -> tuple:
        with torch.no_grad():
            x = frac.unsqueeze(1)
            lap = F.conv3d(x, self.kernel, padding=1)
            
            if mask is not None:
                lap = lap[mask.unsqueeze(1) > 0]
            else:
                lap = lap.flatten()
            
            lap_np = lap.cpu().numpy().flatten()
            import numpy as np
            counts, edges = np.histogram(lap_np, bins=n_bins, range=(-0.5, 0.5))
            
            return edges, counts
    
    def __repr__(self) -> str:
        fa_status = "FA-aware" if self.fa_map is not None else "isotropic"
        threshold_str = f", fa_threshold={getattr(self, 'fa_threshold', 0.0)}" if self.fa_map is not None else ""
        return (f"SpatialPrior3D(weight={self.weight}, connectivity='{self.connectivity}', "
                f"robust={self.robust}, huber_delta={self.huber_delta}, "
                f"fa_alpha={self.fa_alpha}, mode={fa_status}{threshold_str})")


class TotalVariation3D(nn.Module):
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def forward(self,
                frac: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        dx = torch.abs(frac[:, 1:, :, :] - frac[:, :-1, :, :])
        dy = torch.abs(frac[:, :, 1:, :] - frac[:, :, :-1, :])
        dz = torch.abs(frac[:, :, :, 1:] - frac[:, :, :, :-1])
        
        if mask is not None:
            mask_x = (mask[:, 1:, :, :] * mask[:, :-1, :, :]).float()
            mask_y = (mask[:, :, 1:, :] * mask[:, :, :-1, :]).float()
            mask_z = (mask[:, :, :, 1:] * mask[:, :, :, :-1]).float()
            
            dx = dx * mask_x
            dy = dy * mask_y
            dz = dz * mask_z
        
        tv_loss = dx.mean() + dy.mean() + dz.mean()
        
        return self.weight * tv_loss
    
    def __repr__(self) -> str:
        return f"TotalVariation3D(weight={self.weight})"
