#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass



def get_sh_order_from_coeffs(n_coeffs: int) -> int:
    order_map = {1: 0, 6: 2, 15: 4, 28: 6, 45: 8, 66: 10}
    return order_map.get(n_coeffs, 4)


def compute_sh_matrix(directions: torch.Tensor, l_max: int = 8) -> torch.Tensor:
    device = directions.device
    dtype = directions.dtype
    
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)
    
    sh_list = []
    
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            Y_lm = _real_sph_harm(l, m, theta, phi)
            sh_list.append(Y_lm)
    
    return torch.stack(sh_list, dim=-1)


def _real_sph_harm(l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    abs_m = abs(m)
    
    cos_theta = torch.cos(theta)
    P_lm = _associated_legendre(l, abs_m, cos_theta)
    
    norm = math.sqrt((2*l + 1) / (4 * math.pi) * 
                     math.factorial(l - abs_m) / math.factorial(l + abs_m))
    
    if m > 0:
        return norm * math.sqrt(2) * P_lm * torch.cos(m * phi)
    elif m < 0:
        return norm * math.sqrt(2) * P_lm * torch.sin(abs_m * phi)
    else:
        return norm * P_lm


def _associated_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    if m > l:
        return torch.zeros_like(x)
    
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    
    if l == m:
        return pmm
    
    pmmp1 = x * (2 * m + 1) * pmm
    
    if l == m + 1:
        return pmmp1
    
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pll_new = ((2 * ll - 1) * x * pll - (ll + m - 1) * pmm) / (ll - m)
        pmm = pll
        pll = pll_new
    
    return pll



class IsotropicResponse(nn.Module):
    
    def __init__(
        self,
        n_atoms: int = 4,
        d_min: float = 0.1,
        d_max: float = 3.5,
        init_d_mean: float = 0.8,
        learnable: bool = True,
        name: str = 'isotropic',
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.d_min = d_min
        self.d_max = d_max
        self.name = name
        
        d_init = torch.linspace(
            max(d_min, init_d_mean - 0.5),
            min(d_max, init_d_mean + 0.5),
            n_atoms
        )
        
        log_d_init = torch.log(d_init)
        
        logit_w_init = torch.zeros(n_atoms)
        
        if learnable:
            self.log_d = nn.Parameter(log_d_init)
            self.logit_w = nn.Parameter(logit_w_init)
        else:
            self.register_buffer('log_d', log_d_init)
            self.register_buffer('logit_w', logit_w_init)
    
    def get_diffusivities(self) -> torch.Tensor:
        d = torch.exp(self.log_d)
        return torch.clamp(d, self.d_min, self.d_max)
    
    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.logit_w, dim=0)
    
    def forward(self, b: torch.Tensor) -> torch.Tensor:
        D = self.get_diffusivities()
        w = self.get_weights()
        
        b_norm = b / 1000.0
        
        exponents = -b_norm.unsqueeze(-1) * D.unsqueeze(0)
        R = (w.unsqueeze(0) * torch.exp(exponents)).sum(dim=-1)
        
        return R
    
    def get_mean_diffusivity(self) -> torch.Tensor:
        D = self.get_diffusivities()
        w = self.get_weights()
        return (w * D).sum()



class AnisotropicWMResponse(nn.Module):
    
    def __init__(
        self,
        n_atoms: int = 4,
        d_para_range: Tuple[float, float] = (1.0, 2.5),
        d_perp_range: Tuple[float, float] = (0.1, 0.8),
        learnable: bool = True,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.d_para_min, self.d_para_max = d_para_range
        self.d_perp_min, self.d_perp_max = d_perp_range
        
        d_para_init = torch.linspace(
            self.d_para_min + 0.2,
            self.d_para_max - 0.2,
            n_atoms
        )
        
        d_perp_init = torch.linspace(
            self.d_perp_min + 0.1,
            self.d_perp_max - 0.1,
            n_atoms
        )
        
        d_perp_init = torch.minimum(d_perp_init, d_para_init - 0.1)
        
        log_d_para_init = torch.log(d_para_init)
        log_d_perp_init = torch.log(torch.clamp(d_perp_init, min=0.01))
        
        logit_w_init = torch.zeros(n_atoms)
        
        if learnable:
            self.log_d_para = nn.Parameter(log_d_para_init)
            self.log_d_perp = nn.Parameter(log_d_perp_init)
            self.logit_w = nn.Parameter(logit_w_init)
        else:
            self.register_buffer('log_d_para', log_d_para_init)
            self.register_buffer('log_d_perp', log_d_perp_init)
            self.register_buffer('logit_w', logit_w_init)
    
    def get_diffusivities(self) -> Tuple[torch.Tensor, torch.Tensor]:
        d_para = torch.exp(self.log_d_para)
        d_perp = torch.exp(self.log_d_perp)
        
        d_para = torch.clamp(d_para, self.d_para_min, self.d_para_max)
        d_perp = torch.clamp(d_perp, self.d_perp_min, self.d_perp_max)
        
        d_perp = torch.minimum(d_perp, d_para - 0.01)
        
        return d_para, d_perp
    
    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.logit_w, dim=0)
    
    def forward(
        self,
        b: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        D_para, D_perp = self.get_diffusivities()
        w = self.get_weights()
        
        b_norm = b / 1000.0
        
        cos2 = cos_theta ** 2
        
        D_para_exp = D_para.view(*([1] * cos2.dim()), -1)
        D_perp_exp = D_perp.view(*([1] * cos2.dim()), -1)
        w_exp = w.view(*([1] * cos2.dim()), -1)
        
        cos2_exp = cos2.unsqueeze(-1)
        b_exp = b_norm.unsqueeze(-1)
        
        D_app = D_perp_exp + (D_para_exp - D_perp_exp) * cos2_exp
        
        R = (w_exp * torch.exp(-b_exp * D_app)).sum(dim=-1)
        
        return R
    
    def get_mean_fa(self) -> torch.Tensor:
        D_para, D_perp = self.get_diffusivities()
        w = self.get_weights()
        
        MD = (D_para + 2 * D_perp) / 3
        FA = torch.sqrt(0.5 * ((D_para - MD)**2 + 2*(D_perp - MD)**2)) / MD
        
        return (w * FA).sum()



class SphericalODF(nn.Module):
    
    def __init__(
        self,
        representation: str = 'discrete',
        n_directions: int = 60,
        l_max: int = 8,
        directions: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.representation = representation
        self.n_directions = n_directions
        self.l_max = l_max
        
        if representation == 'discrete':
            if directions is None:
                directions = self._fibonacci_sphere(n_directions)
            self.register_buffer('directions', directions)
            
            solid_angle = 4 * math.pi / n_directions
            self.register_buffer('solid_angle', torch.tensor(solid_angle))
            
        elif representation == 'sh':
            self.n_coeffs = (l_max // 2 + 1) * (l_max // 2 + 2)
            
    
    def _fibonacci_sphere(self, n: int) -> torch.Tensor:
        indices = torch.arange(n, dtype=torch.float32)
        
        phi = math.pi * (3.0 - math.sqrt(5.0))
        y = 1 - (indices / (n - 1)) * 2
        radius = torch.sqrt(1 - y * y)
        
        theta = phi * indices
        
        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius
        
        return torch.stack([x, y, z], dim=-1)
    
    def forward(
        self,
        odf_params: torch.Tensor,
        query_directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.representation == 'discrete':
            odf_values = F.softmax(odf_params, dim=-1)
            return odf_values, self.directions
            
        elif self.representation == 'sh':
            raise NotImplementedError("SH ODF not yet implemented")
    
    def integrate_with_kernel(
        self,
        odf_params: torch.Tensor,
        kernel_values: torch.Tensor,
    ) -> torch.Tensor:
        odf_values, _ = self.forward(odf_params)
        
        integral = (odf_values * kernel_values).sum(dim=-1)
        
        return integral



class MTCSDSignalModel(nn.Module):
    
    def __init__(
        self,
        n_atoms_iso: int = 4,
        n_atoms_wm: int = 4,
        n_odf_directions: int = 60,
        learnable_responses: bool = True,
        csf_d_mean: float = 3.0,
        gm_d_mean: float = 0.8,
    ):
        super().__init__()
        
        self.csf_response = IsotropicResponse(
            n_atoms=n_atoms_iso,
            d_min=2.0,
            d_max=3.5,
            init_d_mean=csf_d_mean,
            learnable=learnable_responses,
            name='CSF',
        )
        
        self.gm_response = IsotropicResponse(
            n_atoms=n_atoms_iso,
            d_min=0.3,
            d_max=1.5,
            init_d_mean=gm_d_mean,
            learnable=learnable_responses,
            name='GM',
        )
        
        self.wm_response = AnisotropicWMResponse(
            n_atoms=n_atoms_wm,
            d_para_range=(1.0, 2.5),
            d_perp_range=(0.1, 0.8),
            learnable=learnable_responses,
        )
        
        self.odf = SphericalODF(
            representation='discrete',
            n_directions=n_odf_directions,
        )
        
        self.n_odf_directions = n_odf_directions
    
    def forward(
        self,
        b: torch.Tensor,
        g: torch.Tensor,
        f_csf: torch.Tensor,
        f_gm: torch.Tensor,
        f_wm: torch.Tensor,
        odf_params: torch.Tensor,
        s0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N_meas = b.shape[0]
        spatial_shape = f_csf.shape
        
        
        R_csf = self.csf_response(b)
        S_csf = f_csf.unsqueeze(-1) * R_csf
        
        R_gm = self.gm_response(b)
        S_gm = f_gm.unsqueeze(-1) * R_gm
        
        odf_dirs = self.odf.directions
        
        cos_theta = torch.einsum('id,jd->ij', g, odf_dirs)
        
        b_expanded = b.unsqueeze(-1).expand(-1, self.n_odf_directions)
        R_wm = self.wm_response(b_expanded, cos_theta)
        
        odf_values, _ = self.odf(odf_params)
        
        S_wm_unnorm = torch.einsum('...j,ij->...i', odf_values, R_wm)
        S_wm = f_wm.unsqueeze(-1) * S_wm_unnorm
        
        signal = S_csf + S_gm + S_wm
        
        if s0 is not None:
            signal = s0.unsqueeze(-1) * signal
        
        return signal
    
    def get_response_info(self) -> Dict[str, float]:
        return {
            'csf_mean_d': self.csf_response.get_mean_diffusivity().item(),
            'gm_mean_d': self.gm_response.get_mean_diffusivity().item(),
            'wm_mean_fa': self.wm_response.get_mean_fa().item(),
        }



def create_mtcsd_model(
    protocol: 'AcquisitionProtocolV4',
    n_atoms: int = 4,
    n_odf_directions: int = 60,
    learnable_responses: bool = True,
) -> MTCSDSignalModel:
    return MTCSDSignalModel(
        n_atoms_iso=n_atoms,
        n_atoms_wm=n_atoms,
        n_odf_directions=n_odf_directions,
        learnable_responses=learnable_responses,
    )



if __name__ == '__main__':
    print("Testing MT-CSD Response Functions...")
    
    model = MTCSDSignalModel(
        n_atoms_iso=4,
        n_atoms_wm=4,
        n_odf_directions=60,
    )
    
    b = torch.tensor([0., 1000., 2000., 3000.], dtype=torch.float32)
    
    torch.manual_seed(42)
    g = torch.randn(4, 3)
    g = g / g.norm(dim=-1, keepdim=True)
    
    f_csf = torch.tensor([0.1])
    f_gm = torch.tensor([0.3])
    f_wm = torch.tensor([0.6])
    
    odf_params = torch.randn(1, 60)
    
    signal = model(b, g, f_csf, f_gm, f_wm, odf_params)
    
    print(f"Signal shape: {signal.shape}")
    print(f"Signal values: {signal.squeeze()}")
    print(f"Response info: {model.get_response_info()}")
    
    print("\nIsotropic responses:")
    print(f"  CSF: {model.csf_response(b)}")
    print(f"  GM:  {model.gm_response(b)}")
    
    print("\nWM response (b=2000, various angles):")
    cos_theta = torch.tensor([0., 0.5, 1.0])
    b_test = torch.full_like(cos_theta, 2000.)
    R_wm = model.wm_response(b_test, cos_theta)
    print(f"  cos(θ)=0 (perp): {R_wm[0]:.4f}")
    print(f"  cos(θ)=0.5:      {R_wm[1]:.4f}")
    print(f"  cos(θ)=1 (para): {R_wm[2]:.4f}")
    
    print("\n✓ MT-CSD model working!")
