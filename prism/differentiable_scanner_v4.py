#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import math

try:
    from .karger_exchange import KargerExchangeModule
    KARGER_AVAILABLE = True
except ImportError:
    KARGER_AVAILABLE = False
    KargerExchangeModule = None

try:
    from .curvature_physics import (
        CurvatureComputer,
        GeometryPredictedDispersion,
    )
    CURVATURE_PHYSICS_AVAILABLE = True
except ImportError:
    CURVATURE_PHYSICS_AVAILABLE = False
    CurvatureComputer = None
    GeometryPredictedDispersion = None



def soft_cap(x: torch.Tensor, cap: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    return cap - F.softplus(cap - x, beta=beta)


def soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    return floor + F.softplus(x - floor, beta=beta)


def kappa_to_odi(kappa: torch.Tensor) -> torch.Tensor:
    kappa = torch.clamp(kappa, min=0.1, max=100.0)
    odi = (2.0 / math.pi) * torch.atan(1.0 / kappa)
    return odi


def odi_to_kappa(odi: torch.Tensor) -> torch.Tensor:
    odi = torch.clamp(odi, min=0.01, max=0.99)
    kappa = 1.0 / torch.tan(math.pi * odi / 2.0)
    return kappa


def compute_dti_metrics(d_parallel: torch.Tensor, d_perp: torch.Tensor) -> Dict[str, torch.Tensor]:
    ad = d_parallel
    rd = d_perp
    md = (d_parallel + 2 * d_perp) / 3
    
    numerator = torch.sqrt((d_parallel - md) ** 2 + 2 * (d_perp - md) ** 2 + 1e-8)
    denominator = torch.sqrt(d_parallel ** 2 + 2 * d_perp ** 2 + 1e-8)
    fa = math.sqrt(3/2) * numerator / denominator
    fa = torch.clamp(fa, 0.0, 1.0)
    
    return {
        'FA': fa,
        'MD': md,
        'AD': ad,
        'RD': rd,
    }


@dataclass
class AcquisitionProtocolV4:
    bvals: torch.Tensor
    bvecs: torch.Tensor
    shell_ids: torch.Tensor
    shell_values: torch.Tensor
    te: float = 80.0
    tr: float = 5000.0
    delta: Optional[torch.Tensor] = None
    Delta: Optional[torch.Tensor] = None
    
    @classmethod
    def from_files(cls, bval_file: str, bvec_file: str, 
                   shell_threshold: float = 100.0,
                   Delta: Optional[float] = None,
                   delta: Optional[float] = None,
                   device: str = 'cpu'):
        bvals = np.loadtxt(bval_file).flatten()
        bvecs = np.loadtxt(bvec_file)
        if bvecs.shape[0] == 3:
            bvecs = bvecs.T
        
        bvals_t = torch.tensor(bvals, dtype=torch.float32, device=device)
        bvecs_t = torch.tensor(bvecs, dtype=torch.float32, device=device)
        bvecs_t = F.normalize(bvecs_t, dim=-1, eps=1e-8)
        
        shell_ids, shell_values = cls._assign_shells(bvals_t, shell_threshold)
        
        Delta_t = None
        delta_t = None
        if Delta is not None:
            if isinstance(Delta, (int, float)):
                Delta_t = torch.full_like(bvals_t, float(Delta))
            else:
                Delta_t = torch.as_tensor(Delta, dtype=torch.float32, device=device)
        if delta is not None:
            if isinstance(delta, (int, float)):
                delta_t = torch.full_like(bvals_t, float(delta))
            else:
                delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
        
        return cls(
            bvals=bvals_t, 
            bvecs=bvecs_t,
            shell_ids=shell_ids,
            shell_values=shell_values,
            Delta=Delta_t,
            delta=delta_t,
        )
    
    @classmethod
    def from_tensors(cls, bvals: torch.Tensor, bvecs: torch.Tensor,
                     shell_threshold: float = 100.0,
                     te: float = 80.0, tr: float = 5000.0,
                     Delta: Optional[torch.Tensor] = None,
                     delta: Optional[torch.Tensor] = None):
        shell_ids, shell_values = cls._assign_shells(bvals, shell_threshold)
        return cls(
            bvals=bvals,
            bvecs=bvecs,
            shell_ids=shell_ids,
            shell_values=shell_values,
            te=te,
            tr=tr,
            Delta=Delta,
            delta=delta,
        )
    
    @staticmethod
    def _assign_shells(bvals: torch.Tensor, threshold: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
        device = bvals.device
        bvals_np = bvals.cpu().numpy()
        
        sorted_bvals = np.sort(bvals_np)
        
        shells = []
        for b in sorted_bvals:
            assigned = False
            for i, shell_center in enumerate(shells):
                if abs(b - shell_center) < threshold:
                    assigned = True
                    break
            if not assigned:
                shells.append(b)
        
        shells = np.array(sorted(shells))
        
        shell_ids = np.zeros(len(bvals_np), dtype=np.int64)
        for i, b in enumerate(bvals_np):
            distances = np.abs(shells - b)
            shell_ids[i] = np.argmin(distances)
        
        return (
            torch.tensor(shell_ids, dtype=torch.long, device=device),
            torch.tensor(shells, dtype=torch.float32, device=device),
        )
    
    @property
    def num_measurements(self) -> int:
        return self.bvals.shape[0]
    
    @property
    def num_shells(self) -> int:
        return self.shell_values.shape[0]
    
    def __repr__(self) -> str:
        extra = ""
        if self.Delta is not None:
            delta_mean = self.Delta.mean().item()
            delta_min = self.Delta.min().item()
            delta_max = self.Delta.max().item()
            if delta_min == delta_max:
                extra = f", Delta={delta_mean:.1f}ms"
            else:
                extra = f", Delta={delta_min:.1f}-{delta_max:.1f}ms (mean={delta_mean:.1f})"
        return (f"AcquisitionProtocolV4("
                f"n_meas={self.num_measurements}, "
                f"shells={self.shell_values.cpu().numpy().tolist()}, "
                f"TE={self.te}ms{extra})")
    
    @classmethod
    def from_files_with_constant_timing(cls, bval_file: str, bvec_file: str,
                                        Delta_ms: float,
                                        delta_ms: Optional[float] = None,
                                        **kwargs) -> 'AcquisitionProtocolV4':
        proto = cls.from_files(bval_file, bvec_file, **kwargs)
        device = proto.bvals.device
        Delta = torch.full_like(proto.bvals, float(Delta_ms), device=device)
        delta = None if delta_ms is None else torch.full_like(proto.bvals, float(delta_ms), device=device)
        proto.Delta = Delta
        proto.delta = delta
        return proto


class MultiCompartmentSignalV4(nn.Module):
    
    def __init__(self,
                 n_fibers: int = 3,
                 n_shells: int = 4,
                 use_tortuosity: bool = False,
                 use_dispersion: bool = True,
                 use_t2_weighting: bool = False,
                 use_restricted: bool = True,
                 use_dot: bool = False,
                 use_kurtosis: bool = False,
                 use_dki: bool = False,
                 use_diffusivity_spectrum: bool = False,
                 n_spectrum_components: int = 5,
                 use_per_shell_modulation: bool = False,
                 use_biexp_gm: bool = False,
                 signal_clamp_factor: float = 1.2,
                 use_mtcsd: bool = False,
                 n_mtcsd_atoms: int = 4,
                 n_odf_directions: int = 60,
                 learnable_responses: bool = True,
                 learn_diffusivities: bool = False):
        super().__init__()
        
        self.n_fibers = n_fibers
        self.n_shells = n_shells
        self.use_tortuosity = use_tortuosity
        self.use_dispersion = use_dispersion
        self.use_t2_weighting = use_t2_weighting
        self.use_restricted = use_restricted
        self.use_dot = use_dot
        self.use_kurtosis = use_kurtosis
        self.use_dki = use_dki
        self.use_diffusivity_spectrum = use_diffusivity_spectrum
        self.n_spectrum_components = n_spectrum_components
        self.use_per_shell_modulation = use_per_shell_modulation
        self.use_biexp_gm = use_biexp_gm
        self.signal_clamp_factor = signal_clamp_factor
        
        self.use_mtcsd = use_mtcsd
        self.n_mtcsd_atoms = n_mtcsd_atoms
        self.n_odf_directions = n_odf_directions
        
        if use_mtcsd:
            import warnings
            warnings.warn(
                "MT-CSD mode is EXPERIMENTAL and may degrade fiber quality. "
                "Consider using standard model for production runs.",
                UserWarning
            )
            from .mtcsd_response import (
                IsotropicResponse, AnisotropicWMResponse, SphericalODF
            )
            
            self.csf_response = IsotropicResponse(
                n_atoms=n_mtcsd_atoms,
                d_min=2.0,
                d_max=3.5,
                init_d_mean=3.0,
                learnable=learnable_responses,
                name='CSF',
            )
            
            self.gm_response = IsotropicResponse(
                n_atoms=n_mtcsd_atoms,
                d_min=0.3,
                d_max=1.5,
                init_d_mean=0.8,
                learnable=learnable_responses,
                name='GM',
            )
            
            self.wm_response = AnisotropicWMResponse(
                n_atoms=n_mtcsd_atoms,
                d_para_range=(1.0, 2.5),
                d_perp_range=(0.1, 0.8),
                learnable=learnable_responses,
            )
            
            self.odf = SphericalODF(
                representation='discrete',
                n_directions=n_odf_directions,
            )
        
        self.log_deff = False
        self._log_counter = 0
        self._log_interval = 100
        
        self.learn_diffusivities = learn_diffusivities
        
        if use_diffusivity_spectrum:
            d_grid = torch.logspace(np.log10(0.1), np.log10(3.0), n_spectrum_components)
            self.register_buffer('spectrum_d_grid', d_grid)
        
        if learn_diffusivities:
            def inv_softplus(x):
                return torch.tensor(x).exp().sub(1).log()
            def logit(p):
                return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))
            
            self._raw_d_gm = nn.Parameter(inv_softplus(0.9))
            self._raw_d_gm_slow = nn.Parameter(inv_softplus(0.15))
            self._logit_f_gm_restricted = nn.Parameter(logit(0.45))
            self._raw_d_parallel = nn.Parameter(inv_softplus(1.7))
            self._raw_d_perp = nn.Parameter(inv_softplus(0.4))
            self.register_buffer('_d_csf_buf', torch.tensor(3.0))
        else:
            self.register_buffer('_d_csf_buf', torch.tensor(3.0))
            self.register_buffer('_d_gm_buf', torch.tensor(0.9))
            self.register_buffer('_d_gm_slow_buf', torch.tensor(0.15))
            self.register_buffer('_f_gm_restricted_buf', torch.tensor(0.45))
            self.register_buffer('_d_parallel_buf', torch.tensor(1.7))
            self.register_buffer('_d_perp_buf', torch.tensor(0.4))
        self.register_buffer('f_intra_prior', torch.tensor(0.5))
        self.register_buffer('kappa_prior', torch.tensor(8.0))
        self.register_buffer('d_restricted_prior', torch.tensor(0.2))
        self.register_buffer('d_dot_prior', torch.tensor(0.01))
        self.register_buffer('kurtosis_prior', torch.tensor(1.0))
        self.register_buffer('k_parallel_prior', torch.tensor(0.8))
        self.register_buffer('k_perp_prior', torch.tensor(1.2))
        self.register_buffer('k_cross_prior', torch.tensor(0.3))
        
        self.register_buffer('alpha_restricted_prior', torch.tensor(0.2))
        self.register_buffer('alpha_dot_prior', torch.tensor(0.3))
        self.register_buffer('Delta_ref', torch.tensor(40.0))
        
        self.register_buffer('t2_csf', torch.tensor(500.0))
        self.register_buffer('t2_gm', torch.tensor(80.0))
        self.register_buffer('t2_wm', torch.tensor(70.0))
        self.register_buffer('t2_restricted', torch.tensor(60.0))
        self.register_buffer('t2_dot', torch.tensor(50.0))
        
        if use_per_shell_modulation:
            self.log_d_modulation = nn.Parameter(torch.zeros(n_shells))
    
    
    @property
    def d_csf_prior(self) -> torch.Tensor:
        return self._d_csf_buf
    
    @property
    def d_gm_prior(self) -> torch.Tensor:
        if self.learn_diffusivities:
            return F.softplus(self._raw_d_gm)
        return self._d_gm_buf
    
    @property
    def d_gm_slow_prior(self) -> torch.Tensor:
        if self.learn_diffusivities:
            return F.softplus(self._raw_d_gm_slow)
        return self._d_gm_slow_buf
    
    @property
    def f_gm_restricted_prior(self) -> torch.Tensor:
        if self.learn_diffusivities:
            return torch.sigmoid(self._logit_f_gm_restricted)
        return self._f_gm_restricted_buf
    
    @property
    def d_parallel_prior(self) -> torch.Tensor:
        if self.learn_diffusivities:
            return F.softplus(self._raw_d_parallel)
        return self._d_parallel_buf
    
    @property
    def d_perp_prior(self) -> torch.Tensor:
        if self.learn_diffusivities:
            return F.softplus(self._raw_d_perp)
        return self._d_perp_buf
    
    def get_learned_diffusivities(self) -> dict:
        return {
            'd_gm': self.d_gm_prior.item() if self.d_gm_prior.numel() == 1 else self.d_gm_prior,
            'd_gm_slow': self.d_gm_slow_prior.item() if self.d_gm_slow_prior.numel() == 1 else self.d_gm_slow_prior,
            'f_gm_restricted': self.f_gm_restricted_prior.item() if self.f_gm_restricted_prior.numel() == 1 else self.f_gm_restricted_prior,
            'd_parallel': self.d_parallel_prior.item() if self.d_parallel_prior.numel() == 1 else self.d_parallel_prior,
            'd_perp': self.d_perp_prior.item() if self.d_perp_prior.numel() == 1 else self.d_perp_prior,
        }
    
    def _expand_per_fiber(self, param: torch.Tensor, f_wm: torch.Tensor) -> torch.Tensor:
        if param.dim() == 0:
            return param.expand_as(f_wm)
        elif param.dim() == 4:
            return param.unsqueeze(-1).expand_as(f_wm)
        elif param.dim() == 5:
            return param
        else:
            raise ValueError(f"Unexpected parameter shape: {param.shape}")
    
    def _get_d_modulation(self, shell_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_per_shell_modulation:
            return torch.ones(shell_ids.shape[0], device=shell_ids.device)
        
        modulation = torch.exp(self.log_d_modulation * 0.1)
        return modulation[shell_ids]
    
    def compute_isotropic_signal(
            self,
            bvals: torch.Tensor,
            d_iso: torch.Tensor,
            d_modulation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if d_modulation is not None:
            b_eff = bvals * d_modulation
        else:
            b_eff = bvals

        if d_iso.dim() == 0:
            return torch.exp(-b_eff * d_iso / 1000.0)

        N = b_eff.shape[0]

        if d_iso.shape[-1] == N:
            b_view = b_eff.view(*([1] * (d_iso.dim() - 1)), N)
            return torch.exp(-b_view * d_iso / 1000.0)

        b_view = b_eff.view(*([1] * d_iso.dim()), N)
        d_view = d_iso.unsqueeze(-1)
        return torch.exp(-b_view * d_view / 1000.0)
    
    def compute_spectrum_signal(
            self,
            bvals: torch.Tensor,
            spectrum_weights: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_diffusivity_spectrum:
            raise RuntimeError("Spectrum model not enabled. Set use_diffusivity_spectrum=True")
        
        weights = F.softmax(spectrum_weights, dim=-1)
        
        d_grid = self.spectrum_d_grid
        K = d_grid.shape[0]
        N = bvals.shape[0]
        
        b_exp = bvals.view(1, N) / 1000.0
        d_exp = d_grid.view(K, 1)
        exp_kernel = torch.exp(-b_exp * d_exp)
        
        signal = torch.einsum('...k,kn->...n', weights, exp_kernel)
        
        return signal
    
    def compute_stick_signal(self,
                              bvals: torch.Tensor,
                              bvecs: torch.Tensor,
                              fiber_dir: torch.Tensor,
                              d_parallel: torch.Tensor,
                              d_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        fiber_dir = F.normalize(fiber_dir, dim=-1, eps=1e-8)
        cos_theta = torch.einsum('...d,nd->...n', fiber_dir, bvecs)
        cos_theta_sq = cos_theta ** 2
        
        if d_parallel.dim() == 0:
            d_app = d_parallel * cos_theta_sq
        else:
            d_app = d_parallel.unsqueeze(-1) * cos_theta_sq
        
        if d_modulation is not None:
            d_app = d_app * d_modulation
        
        bvals_expanded = bvals.view(*([1] * (d_app.dim() - 1)), -1)
        return torch.exp(-bvals_expanded * d_app / 1000.0)
    
    def compute_zeppelin_signal(self,
                                 bvals: torch.Tensor,
                                 bvecs: torch.Tensor,
                                 fiber_dir: torch.Tensor,
                                 d_parallel: torch.Tensor,
                                 d_perp: torch.Tensor,
                                 d_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        fiber_dir = F.normalize(fiber_dir, dim=-1, eps=1e-8)
        cos_theta = torch.einsum('...d,nd->...n', fiber_dir, bvecs)
        cos_theta_sq = cos_theta ** 2
        sin_theta_sq = torch.clamp(1.0 - cos_theta_sq, min=0.0)
        
        if d_parallel.dim() == 0:
            d_app = d_parallel * cos_theta_sq + d_perp * sin_theta_sq
        else:
            d_app = (d_parallel.unsqueeze(-1) * cos_theta_sq + 
                     d_perp.unsqueeze(-1) * sin_theta_sq)
        
        if d_modulation is not None:
            d_app = d_app * d_modulation
        
        bvals_expanded = bvals.view(*([1] * (d_app.dim() - 1)), -1)
        return torch.exp(-bvals_expanded * d_app / 1000.0)
    
    def compute_dispersed_signal(self,
                                   bvals: torch.Tensor,
                                   bvecs: torch.Tensor,
                                   fiber_dir: torch.Tensor,
                                   d_parallel: torch.Tensor,
                                   d_perp: torch.Tensor,
                                   f_intra: torch.Tensor,
                                   kappa: torch.Tensor,
                                   d_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        s_intra = self.compute_stick_signal(bvals, bvecs, fiber_dir, d_parallel, d_modulation)
        s_extra = self.compute_zeppelin_signal(bvals, bvecs, fiber_dir, d_parallel, d_perp, d_modulation)
        
        if f_intra.dim() == 0:
            s_aligned = f_intra * s_intra + (1 - f_intra) * s_extra
        else:
            s_aligned = f_intra.unsqueeze(-1) * s_intra + (1 - f_intra).unsqueeze(-1) * s_extra
        
        if d_parallel.dim() == 0:
            d_iso = (d_parallel + 2 * d_perp) / 3
        else:
            d_iso = (d_parallel + 2 * d_perp) / 3
        s_isotropic = self.compute_isotropic_signal(bvals, d_iso, d_modulation)
        
        if kappa.dim() == 0:
            weight = 1.0 - torch.exp(-kappa / 4.0)
            s_dispersed = weight * s_aligned + (1 - weight) * s_isotropic
        else:
            weight = 1.0 - torch.exp(-kappa / 4.0)
            weight = weight.unsqueeze(-1)
            if s_isotropic.dim() < s_aligned.dim():
                s_isotropic = s_isotropic.expand_as(s_aligned)
            s_dispersed = weight * s_aligned + (1 - weight) * s_isotropic
        
        return s_dispersed
    
    def forward(self,
                params: Dict[str, torch.Tensor],
                bvals: torch.Tensor,
                bvecs: torch.Tensor,
                shell_ids: Optional[torch.Tensor] = None,
                te: float = 80.0,
                Delta: Optional[torch.Tensor] = None,
                normalize_fractions: bool = True) -> torch.Tensor:
        device = bvals.device
        
        if self.use_mtcsd:
            return self._forward_mtcsd(params, bvals, bvecs, te)
        
        f_csf = params['f_csf']
        f_gm = params['f_gm']
        f_wm = params['f_wm']
        fiber_dirs = params['fiber_dirs']
        
        spatial_shape = f_csf.shape
        n_meas = bvals.shape[0]
        
        d_csf = params.get('d_csf', self.d_csf_prior)
        d_gm = params.get('d_gm', self.d_gm_prior)
        d_parallel = params.get('d_parallel', self.d_parallel_prior)
        d_perp = params.get('d_perp', self.d_perp_prior)
        f_intra = params.get('f_intra', self.f_intra_prior)
        kappa = params.get('kappa', self.kappa_prior)
        d_restricted = params.get('d_restricted', self.d_restricted_prior)
        
        if 'd_csf' in params and hasattr(d_csf, 'requires_grad') and d_csf.requires_grad:
            d_csf = F.softplus(d_csf)
        if 'd_gm' in params and hasattr(d_gm, 'requires_grad') and d_gm.requires_grad:
            d_gm = F.softplus(d_gm)
        if 'd_parallel' in params and hasattr(d_parallel, 'requires_grad') and d_parallel.requires_grad:
            d_parallel = F.softplus(d_parallel)
        if 'd_perp' in params and hasattr(d_perp, 'requires_grad') and d_perp.requires_grad:
            d_perp = F.softplus(d_perp)
        if 'd_restricted' in params and hasattr(d_restricted, 'requires_grad') and d_restricted.requires_grad:
            d_restricted = F.softplus(d_restricted)
        if 'kappa' in params and hasattr(kappa, 'requires_grad') and kappa.requires_grad:
            kappa = F.softplus(kappa)
        kappa = torch.clamp(kappa, min=0.5, max=100.0)
        
        d_parallel_pf = self._expand_per_fiber(d_parallel, f_wm)
        d_perp_pf = self._expand_per_fiber(d_perp, f_wm)
        f_intra_pf = self._expand_per_fiber(f_intra, f_wm)
        kappa_pf = self._expand_per_fiber(kappa, f_wm)
        if self.use_tortuosity:
            d_perp_eff_pf = d_parallel_pf * (1 - f_intra_pf)
        else:
            d_perp_eff_pf = d_perp_pf
        
        s0 = params.get('s0', torch.ones(*spatial_shape, device=device))
        
        f_restricted = params.get('f_restricted', None)
        if f_restricted is None and self.use_restricted:
            f_restricted = torch.zeros_like(f_csf)
        
        f_dot = params.get('f_dot', None)
        if f_dot is None and self.use_dot:
            f_dot = torch.zeros_like(f_csf)
        d_dot = params.get('d_dot', self.d_dot_prior)
        
        if self.use_dki:
            k_parallel = params.get('k_parallel', self.k_parallel_prior)
            k_perp = params.get('k_perp', self.k_perp_prior)
            
            if hasattr(k_parallel, 'requires_grad') and k_parallel.requires_grad:
                k_parallel = F.softplus(k_parallel)
                k_parallel = torch.clamp(k_parallel, max=3.0)
            if hasattr(k_perp, 'requires_grad') and k_perp.requires_grad:
                k_perp = F.softplus(k_perp)
                k_perp = torch.clamp(k_perp, max=3.0)
            k_parallel = torch.clamp(k_parallel, min=0.0)
            k_perp = torch.clamp(k_perp, min=0.0)
            
            kurtosis = None
        elif self.use_kurtosis:
            kurtosis = params.get('kurtosis', self.kurtosis_prior)
            k_parallel = None
            k_perp = None
        else:
            kurtosis = None
            k_parallel = None
            k_perp = None
        
        spectrum_weights = None
        f_spectrum = None
        if self.use_diffusivity_spectrum:
            spectrum_weights = params.get('spectrum_weights', None)
            f_spectrum = params.get('f_spectrum', None)
        
        if normalize_fractions:
            total_f = f_csf + f_gm + f_wm.sum(dim=-1)
            if self.use_restricted and f_restricted is not None:
                total_f = total_f + f_restricted
            if self.use_dot and f_dot is not None:
                total_f = total_f + f_dot
            if self.use_diffusivity_spectrum and f_spectrum is not None:
                total_f = total_f + f_spectrum
            total_f = torch.clamp(total_f, min=1e-6)
            f_csf = f_csf / total_f
            f_gm = f_gm / total_f
            f_wm = f_wm / total_f.unsqueeze(-1)
            if self.use_restricted and f_restricted is not None:
                f_restricted = f_restricted / total_f
            if self.use_dot and f_dot is not None:
                f_dot = f_dot / total_f
            if self.use_diffusivity_spectrum and f_spectrum is not None:
                f_spectrum = f_spectrum / total_f
        
        d_modulation = None
        if self.use_per_shell_modulation and shell_ids is not None:
            d_modulation = self._get_d_modulation(shell_ids)

        t2_weight_csf = 1.0
        t2_weight_gm = 1.0
        t2_weight_wm = 1.0
        t2_weight_restricted = 1.0
        t2_weight_dot = 1.0
        if self.use_t2_weighting:
            te_t = torch.as_tensor(te, device=device, dtype=self.t2_csf.dtype)
            t2_weight_csf = torch.exp(-te_t / self.t2_csf)
            t2_weight_gm = torch.exp(-te_t / self.t2_gm)
            t2_weight_wm = torch.exp(-te_t / self.t2_wm)
            t2_weight_restricted = torch.exp(-te_t / self.t2_restricted)
            t2_weight_dot = torch.exp(-te_t / self.t2_dot)
        
        signal = torch.zeros(*spatial_shape, n_meas, device=device)
        signal_wm = torch.zeros_like(signal)
        
        s_csf = self.compute_isotropic_signal(bvals, d_csf, d_modulation)
        if s_csf.dim() == 1:
            s_csf = s_csf.expand(*spatial_shape, -1)
        s_csf = s_csf * t2_weight_csf
        signal = signal + f_csf.unsqueeze(-1) * s_csf
        
        if self.use_biexp_gm:
            d_gm_slow = self.d_gm_slow_prior
            f_gm_restricted = self.f_gm_restricted_prior
            s_gm_fast = self.compute_isotropic_signal(bvals, d_gm, d_modulation)
            s_gm_slow = self.compute_isotropic_signal(bvals, d_gm_slow, d_modulation)
            if s_gm_fast.dim() == 1:
                s_gm_fast = s_gm_fast.expand(*spatial_shape, -1)
            if s_gm_slow.dim() == 1:
                s_gm_slow = s_gm_slow.expand(*spatial_shape, -1)
            s_gm = ((1.0 - f_gm_restricted) * s_gm_fast + f_gm_restricted * s_gm_slow) * t2_weight_gm
        else:
            s_gm = self.compute_isotropic_signal(bvals, d_gm, d_modulation)
            if s_gm.dim() == 1:
                s_gm = s_gm.expand(*spatial_shape, -1)
            s_gm = s_gm * t2_weight_gm
        signal = signal + f_gm.unsqueeze(-1) * s_gm
        
        for i in range(self.n_fibers):
            f_fiber = f_wm[..., i]
            fiber_dir = fiber_dirs[..., i, :]
            d_par_i = d_parallel_pf[..., i]
            d_perp_i = d_perp_eff_pf[..., i]
            f_intra_i = f_intra_pf[..., i]
            kappa_i = kappa_pf[..., i]
            
            if self.use_dispersion:
                s_fiber = self.compute_dispersed_signal(
                    bvals, bvecs, fiber_dir,
                    d_par_i, d_perp_i, f_intra_i, kappa_i,
                    d_modulation
                )
            else:
                s_intra = self.compute_stick_signal(bvals, bvecs, fiber_dir, d_par_i, d_modulation)
                s_extra = self.compute_zeppelin_signal(bvals, bvecs, fiber_dir, d_par_i, d_perp_i, d_modulation)
                s_fiber = f_intra_i.unsqueeze(-1) * s_intra + (1 - f_intra_i).unsqueeze(-1) * s_extra
            
            signal_wm = signal_wm + f_fiber.unsqueeze(-1) * s_fiber * t2_weight_wm

        if self.use_dki and k_parallel is not None and k_perp is not None:
            bvals_sq = (bvals / 1000.0) ** 2

            kurtosis_factor = torch.ones(*spatial_shape, n_meas, device=device)

            total_f_wm = f_wm.sum(dim=-1)

            for i in range(self.n_fibers):
                fiber_dir = fiber_dirs[..., i, :]
                f_fiber = f_wm[..., i]
                d_par_i = d_parallel_pf[..., i]

                fiber_dir_norm = F.normalize(fiber_dir, dim=-1, eps=1e-8)
                cos_theta = torch.einsum('...d,nd->...n', fiber_dir_norm, bvecs)
                cos_theta_sq = cos_theta ** 2
                sin_theta_sq = 1.0 - cos_theta_sq

                cos_theta_4 = cos_theta_sq ** 2
                sin_theta_4 = sin_theta_sq ** 2
                cross_term = cos_theta_sq * sin_theta_sq
                
                k_cross = params.get('k_cross', self.k_cross_prior)

                if k_parallel.dim() == 0:
                    k_app = (k_parallel * cos_theta_4 + 
                             k_perp * sin_theta_4 + 
                             6.0 * k_cross * cross_term)
                else:
                    k_cross_exp = k_cross.unsqueeze(-1) if hasattr(k_cross, 'unsqueeze') and k_cross.dim() > 0 else k_cross
                    k_app = (k_parallel.unsqueeze(-1) * cos_theta_4 +
                             k_perp.unsqueeze(-1) * sin_theta_4 +
                             6.0 * k_cross_exp * cross_term)

                d_app = d_par_i.unsqueeze(-1) * cos_theta_sq + d_perp_eff_pf[..., i].unsqueeze(-1) * sin_theta_sq
                d_app_sq = (d_app / 1.0) ** 2

                fiber_k_factor = torch.exp(bvals_sq * d_app_sq * k_app / 6.0)
                fiber_k_factor = torch.clamp(fiber_k_factor, max=2.0)

                fiber_weight = f_fiber.unsqueeze(-1) / (total_f_wm.unsqueeze(-1) + 1e-6)
                kurtosis_factor = kurtosis_factor + fiber_weight * (fiber_k_factor - 1.0)

            signal_wm = signal_wm * kurtosis_factor

        elif self.use_kurtosis and kurtosis is not None:
            d_mean_pf = (d_parallel_pf + 2 * d_perp_eff_pf) / 3
            wm_sum = f_wm.sum(dim=-1, keepdim=True)
            wm_weights = f_wm / (wm_sum + 1e-6)
            d_mean = (d_mean_pf * wm_weights).sum(dim=-1)

            bvals_sq = (bvals / 1000.0) ** 2
            d_mean_sq = (d_mean / 1.0) ** 2

            if d_mean_sq.dim() == 0 and kurtosis.dim() == 0:
                kurtosis_factor = torch.exp(bvals_sq * d_mean_sq * kurtosis / 6.0)
            else:
                dim_base = max(d_mean_sq.dim(), kurtosis.dim())
                bvals_sq_exp = bvals_sq.view(*([1] * dim_base), -1)
                d_mean_sq_exp = d_mean_sq.unsqueeze(-1) if d_mean_sq.dim() > 0 else d_mean_sq
                kurtosis_exp = kurtosis.unsqueeze(-1) if kurtosis.dim() > 0 else kurtosis
                kurtosis_factor = torch.exp(bvals_sq_exp * d_mean_sq_exp * kurtosis_exp / 6.0)

            kurtosis_factor = torch.clamp(kurtosis_factor, max=2.0)
            signal_wm = signal_wm * kurtosis_factor

        signal = signal + signal_wm
        
        if self.use_restricted and f_restricted is not None:
            d_restricted_eff = d_restricted
            if Delta is not None:
                Delta_clamped = torch.clamp(Delta, min=5.0)
                alpha_restricted = params.get('alpha_restricted', self.alpha_restricted_prior)
                Delta_norm = (Delta_clamped / self.Delta_ref).view(1, 1, 1, 1, -1)
                if d_restricted.dim() == 0:
                    d_restricted_eff = d_restricted * (Delta_norm ** (-alpha_restricted))
                else:
                    alpha_exp = alpha_restricted.unsqueeze(-1) if alpha_restricted.dim() > 0 else alpha_restricted
                    d_restricted_eff = d_restricted.unsqueeze(-1) * (Delta_norm ** (-alpha_exp))
                
                if self.log_deff and self._log_counter % self._log_interval == 0:
                    with torch.no_grad():
                        d_eff_flat = d_restricted_eff.flatten()
                        print(f"[D_eff LOG step={self._log_counter}] Restricted D_eff: "
                              f"mean={d_eff_flat.mean().item():.4f}, "
                              f"min={d_eff_flat.min().item():.4f}, "
                              f"max={d_eff_flat.max().item():.4f}, "
                              f"D0={d_restricted.item():.4f}, "
                              f"alpha={alpha_restricted.item():.4f}, "
                              f"Delta_mean={Delta.mean().item():.1f}ms")
            
            s_restricted = self.compute_isotropic_signal(bvals, d_restricted_eff, d_modulation)
            if s_restricted.dim() == 1:
                s_restricted = s_restricted.expand(*spatial_shape, -1)
            s_restricted = s_restricted * t2_weight_restricted
            
            use_exchange = getattr(self, 'use_exchange', False)
            exchange_pairs = getattr(self, 'exchange_pairs', [])
            has_exchange_module = hasattr(self, 'exchange_module') and self.exchange_module is not None
            
            if use_exchange and 'restricted_gm' in exchange_pairs and has_exchange_module:
                
                Delta_ex = Delta if Delta is not None else torch.tensor(43.0, device=device)
                if Delta_ex.dim() > 0:
                    Delta_ex = Delta_ex.mean()
                
                f_total_exchange = f_restricted + f_gm
                f_rel_restricted = f_restricted / (f_total_exchange + 1e-8)

                if d_restricted_eff.dim() > 0:
                    D_restricted = d_restricted_eff.mean(dim=-1)
                else:
                    D_restricted = d_restricted_eff

                D_gm_val = d_gm if getattr(d_gm, 'dim', lambda: 0)() > 0 else d_gm

                s_exchange = self.exchange_module(
                    D1=D_restricted,
                    D2=D_gm_val,
                    f1=f_rel_restricted,
                    b=bvals,
                    Delta=Delta_ex,
                    pair_idx=0
                )
                
                if s_exchange.dim() == 1:
                    s_exchange = s_exchange.expand(*spatial_shape, -1)
                
                f_rel_gm = 1.0 - f_rel_restricted
                t2_weight_exchange = (f_rel_restricted.unsqueeze(-1) * t2_weight_restricted +
                                     f_rel_gm.unsqueeze(-1) * t2_weight_gm)
                s_exchange = s_exchange * t2_weight_exchange

                signal = signal - f_gm.unsqueeze(-1) * s_gm
                signal = signal + f_total_exchange.unsqueeze(-1) * s_exchange
            else:
                signal = signal + f_restricted.unsqueeze(-1) * s_restricted
        
        if self.use_dot and f_dot is not None:
            d_dot_eff = d_dot
            if Delta is not None:
                Delta_clamped = torch.clamp(Delta, min=5.0)
                alpha_dot = params.get('alpha_dot', self.alpha_dot_prior)
                Delta_norm = (Delta_clamped / self.Delta_ref).view(1, 1, 1, 1, -1)
                if d_dot.dim() == 0:
                    d_dot_eff = d_dot * (Delta_norm ** (-alpha_dot))
                else:
                    alpha_exp = alpha_dot.unsqueeze(-1) if alpha_dot.dim() > 0 else alpha_dot
                    d_dot_eff = d_dot.unsqueeze(-1) * (Delta_norm ** (-alpha_exp))
                
                if self.log_deff and self._log_counter % self._log_interval == 0:
                    with torch.no_grad():
                        d_eff_flat = d_dot_eff.flatten()
                        print(f"[D_eff LOG step={self._log_counter}] DOT D_eff: "
                              f"mean={d_eff_flat.mean().item():.4f}, "
                              f"min={d_eff_flat.min().item():.4f}, "
                              f"max={d_eff_flat.max().item():.4f}, "
                              f"D0={d_dot.item():.4f}, "
                              f"alpha={alpha_dot.item():.4f}")
                              
            s_dot = self.compute_isotropic_signal(bvals, d_dot_eff, d_modulation)
            if s_dot.dim() == 1:
                s_dot = s_dot.expand(*spatial_shape, -1)
            s_dot = s_dot * t2_weight_dot
            signal = signal + f_dot.unsqueeze(-1) * s_dot
        
        if self.use_diffusivity_spectrum and spectrum_weights is not None and f_spectrum is not None:
            s_spectrum = self.compute_spectrum_signal(bvals, spectrum_weights)
            s_spectrum = s_spectrum * t2_weight_gm
            signal = signal + f_spectrum.unsqueeze(-1) * s_spectrum
        
        signal = s0.unsqueeze(-1) * signal
        
        if self.signal_clamp_factor is not None:
            signal = torch.clamp(signal, max=s0.unsqueeze(-1) * self.signal_clamp_factor)
        
        if self.log_deff:
            self._log_counter += 1
        
        return signal
    
    def _forward_mtcsd(
        self,
        params: Dict[str, torch.Tensor],
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        te: float = 80.0,
    ) -> torch.Tensor:
        device = bvals.device
        
        f_csf = params['f_csf']
        f_gm = params['f_gm']
        f_wm = params['f_wm']
        f_restricted = params.get('f_restricted', torch.zeros_like(f_csf))
        fiber_dirs = params['fiber_dirs']
        
        spatial_shape = f_csf.shape
        n_meas = bvals.shape[0]
        n_fibers = fiber_dirs.shape[-2]
        
        fiber_dirs = F.normalize(fiber_dirs, dim=-1)
        
        s0 = params.get('s0', torch.ones(*spatial_shape, device=device))
        
        if f_wm.dim() > len(spatial_shape):
            f_wm_total = f_wm.sum(dim=-1)
        else:
            f_wm_total = f_wm
            f_wm = f_wm.unsqueeze(-1).expand(*spatial_shape, n_fibers) / n_fibers
        
        total_f = f_csf + f_gm + f_wm_total + f_restricted
        total_f = torch.clamp(total_f, min=1e-6)
        f_csf_norm = f_csf / total_f
        f_gm_norm = f_gm / total_f
        f_wm_norm = f_wm / total_f.unsqueeze(-1)
        f_restricted_norm = f_restricted / total_f
        
        t2_weight_csf = 1.0
        t2_weight_gm = 1.0
        t2_weight_wm = 1.0
        t2_weight_restricted = 1.0
        if self.use_t2_weighting:
            te_t = torch.as_tensor(te, device=device, dtype=self.t2_csf.dtype)
            t2_weight_csf = torch.exp(-te_t / self.t2_csf)
            t2_weight_gm = torch.exp(-te_t / self.t2_gm)
            t2_weight_wm = torch.exp(-te_t / self.t2_wm)
            t2_weight_restricted = torch.exp(-te_t / self.t2_wm)
        
        R_csf = self.csf_response(bvals)
        S_csf = f_csf_norm.unsqueeze(-1) * R_csf * t2_weight_csf
        
        R_gm = self.gm_response(bvals)
        S_gm = f_gm_norm.unsqueeze(-1) * R_gm * t2_weight_gm
        
        d_restricted = params.get('d_restricted', torch.tensor(0.2, device=device))
        b_norm = bvals / 1000.0
        R_restricted = torch.exp(-b_norm * d_restricted)
        S_restricted = f_restricted_norm.unsqueeze(-1) * R_restricted * t2_weight_restricted
        
        
        d_parallel = params.get('d_parallel', torch.ones(*spatial_shape, device=device) * 1.7)
        d_perp = params.get('d_perp', torch.ones(*spatial_shape, device=device) * 0.4)
        
        if hasattr(d_parallel, 'requires_grad') and d_parallel.requires_grad:
            d_parallel = F.softplus(d_parallel)
        if hasattr(d_perp, 'requires_grad') and d_perp.requires_grad:
            d_perp = F.softplus(d_perp)
        
        d_parallel = torch.clamp(d_parallel, min=0.5, max=3.0)
        d_perp = torch.clamp(d_perp, min=0.1, max=1.5)
        d_perp = torch.minimum(d_perp, d_parallel - 0.1)
        
        cos_theta = torch.einsum('md,...fd->...fm', bvecs, fiber_dirs)
        
        kappa = params.get('kappa', None)
        if kappa is not None and self.use_dispersion:
            kappa = torch.clamp(F.softplus(kappa), min=0.5, max=100.0)
            kappa_expanded = kappa.unsqueeze(-1).unsqueeze(-1)
            dispersion_factor = torch.sigmoid((kappa_expanded - 2.0) / 2.0)
            cos2_effective = dispersion_factor * (cos_theta ** 2) + (1 - dispersion_factor) * (1.0 / 3.0)
        else:
            cos2_effective = cos_theta ** 2
        
        d_para_exp = d_parallel.unsqueeze(-1).unsqueeze(-1)
        d_perp_exp = d_perp.unsqueeze(-1).unsqueeze(-1)
        
        D_eff = d_perp_exp + (d_para_exp - d_perp_exp) * cos2_effective
        
        b_norm = bvals / 1000.0
        S_wm_fibers = torch.exp(-b_norm.unsqueeze(0) * D_eff)
        
        S_wm = (f_wm_norm.unsqueeze(-1) * S_wm_fibers).sum(dim=-2)
        S_wm = S_wm * t2_weight_wm
        
        signal = S_csf + S_gm + S_wm + S_restricted
        
        signal = s0.unsqueeze(-1) * signal
        
        if self.signal_clamp_factor is not None:
            signal = torch.clamp(signal, max=s0.unsqueeze(-1) * self.signal_clamp_factor)
        
        return signal
    
    def get_mtcsd_response_info(self) -> Dict[str, float]:
        if not self.use_mtcsd:
            return {}
        
        return {
            'csf_mean_d': self.csf_response.get_mean_diffusivity().item(),
            'gm_mean_d': self.gm_response.get_mean_diffusivity().item(),
            'wm_mean_fa': self.wm_response.get_mean_fa().item(),
        }



class DifferentiableArtifactsV4(nn.Module):
    
    def __init__(self,
                 n_channels: int,
                 n_shells: int,
                 learn_noise: bool = True,
                 learn_eddy: bool = True,
                 learn_shell_gain: bool = True,
                 learn_bias_field: bool = True,
                 learn_warps: bool = False,
                 use_spatial_noise: bool = False,
                 use_psf: bool = False,
                 psf_kernel_size: int = 3,
                 psf_sigma: float = 0.8,
                 psf_learnable: bool = False,
                 use_ghosting: bool = False,
                 ghost_axis: int = 2,
                 use_qspace_mixing: bool = False,
                 bvecs: Optional[torch.Tensor] = None,
                 bvals: Optional[torch.Tensor] = None,
                 qspace_k_neighbors: int = 6,
                 n_steps_qspace: int = 1,
                 learn_channel_gain: bool = False,
                 channel_gain_reg: float = 0.1,
                 low_rank_channel_gain: bool = False,
                 channel_gain_rank: int = 5,
                 bias_field_res: Tuple[int, int, int] = (8, 8, 8),
                 noise_field_res: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_shells = n_shells
        self.bias_field_res = bias_field_res
        self.noise_field_res = noise_field_res
        self.use_spatial_noise = use_spatial_noise
        
        self.use_psf = use_psf
        if use_psf:
            self.psf = PSFBlur3D(
                kernel_size=psf_kernel_size,
                sigma=psf_sigma,
                learnable=psf_learnable,
            )
        
        self.use_ghosting = use_ghosting
        self.ghost_axis = ghost_axis
        if use_ghosting:
            self.ghost_amplitudes = nn.Parameter(torch.zeros(n_channels))
            self.ghost_reg_weight = 0.5
        
        self.use_qspace_mixing = use_qspace_mixing
        self.n_steps_qspace = n_steps_qspace
        if use_qspace_mixing and bvecs is not None:
            L = self._build_qspace_laplacian(bvecs, bvals=bvals, k_neighbors=qspace_k_neighbors)
            self.register_buffer('L_qspace', L)
            self.log_eps_qspace = nn.Parameter(torch.tensor(-3.0))
            self.qspace_reg_weight = 0.5
        else:
            self.use_qspace_mixing = False
        
        self.learn_shell_gain = learn_shell_gain
        if learn_shell_gain:
            self.log_shell_gain = nn.Parameter(torch.zeros(n_shells))
        
        self.learn_channel_gain = learn_channel_gain
        self.channel_gain_reg = channel_gain_reg
        self.low_rank_channel_gain = low_rank_channel_gain
        self.channel_gain_rank = channel_gain_rank
        if low_rank_channel_gain:
            U = torch.randn(n_channels, channel_gain_rank)
            U, _ = torch.linalg.qr(U)
            self.register_buffer('channel_gain_basis', U)
            self.channel_gain_coeffs = nn.Parameter(torch.zeros(channel_gain_rank))
            self.log_channel_gain = nn.Parameter(torch.zeros(1), requires_grad=False)
        elif learn_channel_gain:
            self.log_channel_gain = nn.Parameter(torch.zeros(n_channels))
        else:
            self.log_channel_gain = nn.Parameter(torch.zeros(n_channels), requires_grad=False)
        
        self.learn_noise = learn_noise
        if learn_noise:
            if use_spatial_noise:
                self.log_noise_sigma_lr = nn.Parameter(torch.zeros(1, 1, *noise_field_res) - 3.0)
            else:
                self.log_noise_sigma = nn.Parameter(torch.tensor(-3.0))
        
        self.learn_eddy = learn_eddy
        if learn_eddy:
            self.log_eddy_scale = nn.Parameter(torch.zeros(n_channels))
            self.eddy_bias = nn.Parameter(torch.zeros(n_channels))
        
        self.learn_bias_field = learn_bias_field
        if learn_bias_field:
            self.bias_field_lr = nn.Parameter(torch.zeros(1, 1, *bias_field_res))
        
        self.learn_warps = learn_warps
        if learn_warps:
            warp_res_y = bias_field_res[1]
            self.warp_field_lr = nn.Parameter(torch.zeros(n_channels, 1, warp_res_y, 1))
    
    def _build_qspace_laplacian(self, bvecs: torch.Tensor, bvals: Optional[torch.Tensor] = None,
                                  k_neighbors: int = 6) -> torch.Tensor:
        bvecs = bvecs.detach().cpu()
        N = bvecs.shape[0]
        
        bvecs = F.normalize(bvecs, dim=-1, eps=1e-8)
        
        L = torch.zeros(N, N)
        
        if bvals is not None:
            bvals = bvals.detach().cpu()
            
            unique_bvals = []
            shell_ids = torch.zeros(N, dtype=torch.long)
            tol = 100
            
            for i, b in enumerate(bvals):
                assigned = False
                for s, ub in enumerate(unique_bvals):
                    if abs(b - ub) < tol:
                        shell_ids[i] = s
                        assigned = True
                        break
                if not assigned:
                    shell_ids[i] = len(unique_bvals)
                    unique_bvals.append(b.item())
            
            for s in range(len(unique_bvals)):
                shell_mask = shell_ids == s
                shell_indices = torch.where(shell_mask)[0]
                n_shell = len(shell_indices)
                
                if n_shell <= 1:
                    continue
                
                bvecs_shell = bvecs[shell_indices]
                
                similarity = torch.abs(bvecs_shell @ bvecs_shell.T)
                
                W_shell = torch.zeros(n_shell, n_shell)
                for i in range(n_shell):
                    sim_i = similarity[i].clone()
                    sim_i[i] = -1
                    k_use = min(k_neighbors, n_shell - 1)
                    _, topk_idx = torch.topk(sim_i, k=k_use)
                    
                    for j in topk_idx:
                        weight = torch.exp(-2 * (1 - similarity[i, j]))
                        W_shell[i, j] = weight
                        W_shell[j, i] = weight
                
                D_shell = torch.diag(W_shell.sum(dim=1))
                L_shell = D_shell - W_shell
                
                for i, gi in enumerate(shell_indices):
                    for j, gj in enumerate(shell_indices):
                        L[gi, gj] = L_shell[i, j]
        else:
            similarity = torch.abs(bvecs @ bvecs.T)
            W = torch.zeros(N, N)
            
            for i in range(N):
                sim_i = similarity[i].clone()
                sim_i[i] = -1
                _, topk_idx = torch.topk(sim_i, k=min(k_neighbors, N-1))
                
                for j in topk_idx:
                    weight = torch.exp(-2 * (1 - similarity[i, j]))
                    W[i, j] = weight
                    W[j, i] = weight
            
            D = torch.diag(W.sum(dim=1))
            L = D - W
        
        L = L / (L.abs().max() + 1e-6)
        
        return L
    
    @property
    def shell_gain(self) -> torch.Tensor:
        if self.learn_shell_gain:
            log_gain_centered = self.log_shell_gain - self.log_shell_gain.mean()
            return torch.exp(log_gain_centered)
        return torch.ones(self.n_shells, device=self.log_channel_gain.device)
    
    @property
    def channel_gain(self) -> torch.Tensor:
        if self.low_rank_channel_gain:
            log_gain = self.channel_gain_basis @ self.channel_gain_coeffs
        else:
            log_gain = self.log_channel_gain
        
        log_gain_centered = log_gain - log_gain.mean()
        return torch.exp(log_gain_centered)
    
    @property
    def noise_sigma(self) -> float:
        if self.learn_noise:
            if self.use_spatial_noise:
                return torch.exp(self.log_noise_sigma_lr).mean().item()
            else:
                return torch.exp(self.log_noise_sigma).item()
        return 0.0
    
    def get_noise_field(self, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        if not self.learn_noise:
            return torch.zeros(1, 1, *spatial_shape, device=self.log_channel_gain.device)
        
        if self.use_spatial_noise:
            sigma_field = F.interpolate(
                self.log_noise_sigma_lr,
                size=spatial_shape,
                mode='trilinear',
                align_corners=False,
            )
            return torch.exp(sigma_field)
        else:
            sigma = torch.exp(self.log_noise_sigma)
            return sigma.expand(1, 1, *spatial_shape)
    
    def apply_shell_gain(self, signal: torch.Tensor, shell_ids: torch.Tensor) -> torch.Tensor:
        if not self.learn_shell_gain:
            return signal
        
        gain_per_meas = self.shell_gain[shell_ids]
        
        return signal * gain_per_meas
    
    def apply_channel_gain(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.learn_channel_gain and not self.low_rank_channel_gain:
            return signal
        return signal * self.channel_gain
    
    def apply_eddy(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.learn_eddy:
            return signal
        scale = torch.exp(self.log_eddy_scale)
        return signal * scale + self.eddy_bias
    
    def apply_bias_field(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.learn_bias_field:
            return signal
        
        B, X, Y, Z, N = signal.shape
        
        bias = F.interpolate(
            self.bias_field_lr,
            size=(X, Y, Z),
            mode='trilinear',
            align_corners=False,
        )
        bias = torch.exp(bias)
        
        return signal * bias.squeeze(0).squeeze(0).unsqueeze(-1)
    
    def apply_warps(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.learn_warps:
            return signal
        
        B, X, Y, Z, N = signal.shape
        device = signal.device
        output = torch.zeros_like(signal)
        
        for c in range(N):
            vol = signal[..., c]
            
            warp_c = self.warp_field_lr[c]
            warp_c = F.interpolate(
                warp_c.unsqueeze(0),
                size=(Y, 1),
                mode='bilinear',
                align_corners=True,
            ).squeeze(0).squeeze(0).squeeze(-1)
            
            grid_x = torch.linspace(-1, 1, X, device=device)
            grid_y = torch.linspace(-1, 1, Y, device=device) + warp_c * 0.1
            grid_z = torch.linspace(-1, 1, Z, device=device)
            
            gx, gy, gz = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
            grid = torch.stack([gz, gy, gx], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
            
            vol_5d = vol.unsqueeze(1)
            warped = F.grid_sample(
                vol_5d, grid, mode='bilinear', padding_mode='border', align_corners=True
            ).squeeze(1)
            
            output[..., c] = warped
        
        return output
    
    def apply_ghosting(self, signal: torch.Tensor, bvals: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_ghosting:
            return signal
        
        B, X, Y, Z, N = signal.shape
        
        dim_map = {1: 1, 2: 2, 3: 3}
        axis = dim_map.get(self.ghost_axis, 2)
        size = signal.shape[axis]
        shift = size // 2
        
        alpha = torch.tanh(self.ghost_amplitudes)
        
        if bvals is not None:
            b_max = bvals.max() + 1e-6
            b_weight = (bvals / b_max) ** 0.5
            alpha = alpha * b_weight
        
        alpha = alpha.view(1, 1, 1, 1, N)
        
        shifted = torch.roll(signal, shifts=shift, dims=axis)
        
        ghosted = signal + alpha * shifted
        
        return ghosted
    
    def apply_qspace_mixing(self, signal: torch.Tensor, bvals: Optional[torch.Tensor] = None,
                             n_steps: int = 1) -> torch.Tensor:
        if not self.use_qspace_mixing:
            return signal
        
        B, X, Y, Z, N = signal.shape
        device = signal.device
        
        eps = torch.exp(self.log_eps_qspace)
        I = torch.eye(N, device=device)
        
        if bvals is not None:
            b_max = bvals.max() + 1e-6
            b_weight = (bvals / b_max) ** 0.5
            w = b_weight.view(-1, 1)
            L_scaled = self.L_qspace.to(device) * w * w.transpose(0, 1)
            A = I - eps * L_scaled
        else:
            A = I - eps * self.L_qspace.to(device)
        
        sig_flat = signal.view(B * X * Y * Z, N)
        
        for _ in range(n_steps):
            sig_flat = sig_flat @ A.T
        
        mixed = sig_flat.view(B, X, Y, Z, N)
        
        return mixed
    
    def add_rician_noise(self, signal: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.dim() == 0 or sigma.numel() == 1:
            noise_real = torch.randn_like(signal) * sigma
            noise_imag = torch.randn_like(signal) * sigma
        else:
            sigma_expanded = sigma.squeeze(0).squeeze(0).unsqueeze(-1)
            noise_real = torch.randn_like(signal) * sigma_expanded
            noise_imag = torch.randn_like(signal) * sigma_expanded
        
        return torch.sqrt((signal + noise_real) ** 2 + noise_imag ** 2)
    
    def forward(self,
                signal: torch.Tensor,
                shell_ids: torch.Tensor,
                bvals: Optional[torch.Tensor] = None,
                add_noise: bool = True) -> torch.Tensor:
        x = signal
        
        x = self.apply_bias_field(x)
        
        x = self.apply_warps(x)
        
        x = self.apply_shell_gain(x, shell_ids)
        
        x = self.apply_channel_gain(x)
        
        x = self.apply_eddy(x)
        
        x = self.apply_qspace_mixing(x, bvals=bvals, n_steps=self.n_steps_qspace)
        
        x = self.apply_ghosting(x, bvals=bvals)
        
        if self.use_psf:
            x = self.psf(x)
        
        if add_noise and self.learn_noise and self.training:
            B, X, Y, Z, N = x.shape
            sigma = self.get_noise_field((X, Y, Z))
            x = self.add_rician_noise(x, sigma)
        
        return x
    
    def get_regularization_loss(self) -> torch.Tensor:
        device = self.log_channel_gain.device
        loss = torch.tensor(0.0, device=device)
        
        if self.learn_shell_gain:
            loss = loss + 0.1 * (self.log_shell_gain ** 2).mean()
        
        if self.low_rank_channel_gain:
            loss = loss + self.channel_gain_reg * (self.channel_gain_coeffs ** 2).mean()
        elif self.learn_channel_gain:
            loss = loss + self.channel_gain_reg * (self.log_channel_gain ** 2).mean()
        
        if self.learn_eddy:
            loss = loss + 0.1 * (self.log_eddy_scale ** 2).mean()
            loss = loss + 0.1 * (self.eddy_bias ** 2).mean()
        
        if self.learn_bias_field:
            loss = loss + 0.1 * (self.bias_field_lr ** 2).mean()
            loss = loss + 0.05 * self._tv_loss(self.bias_field_lr)
        
        if self.learn_warps:
            loss = loss + 0.1 * (self.warp_field_lr ** 2).mean()
        
        
        if self.use_ghosting:
            ghost_reg_weight = getattr(self, 'ghost_reg_weight', 0.5)
            loss = loss + ghost_reg_weight * (torch.tanh(self.ghost_amplitudes) ** 2).mean()
        
        if self.use_qspace_mixing:
            qspace_reg_weight = getattr(self, 'qspace_reg_weight', 0.5)
            eps = torch.exp(self.log_eps_qspace)
            loss = loss + qspace_reg_weight * (eps ** 2)
        
        return loss
    
    def freeze_novel_artifacts(self):
        if self.use_ghosting:
            self.ghost_amplitudes.requires_grad_(False)
            self.ghost_amplitudes.data.zero_()
        
        if self.use_qspace_mixing:
            self.log_eps_qspace.requires_grad_(False)
            self.log_eps_qspace.data.fill_(-10.0)
    
    def unfreeze_novel_artifacts(self, small_init: bool = True):
        if self.use_ghosting:
            if small_init:
                self.ghost_amplitudes.data.uniform_(-0.1, 0.1)
            self.ghost_amplitudes.requires_grad_(True)
        
        if self.use_qspace_mixing:
            if small_init:
                self.log_eps_qspace.data.fill_(-5.0)
            self.log_eps_qspace.requires_grad_(True)
    
    def get_artifact_stats(self) -> Dict[str, float]:
        stats = {}
        
        if self.learn_shell_gain:
            stats['shell_gain_mean'] = self.shell_gain.mean().item()
            stats['shell_gain_std'] = self.shell_gain.std().item()
        
        stats['channel_gain_mean'] = self.channel_gain.mean().item()
        stats['channel_gain_std'] = self.channel_gain.std().item()
        if self.low_rank_channel_gain:
            stats['channel_gain_rank'] = self.channel_gain_rank
            stats['channel_gain_coeffs_norm'] = self.channel_gain_coeffs.norm().item()
        
        if self.use_ghosting:
            ghost = torch.tanh(self.ghost_amplitudes)
            stats['ghost_amp_mean'] = ghost.mean().item()
            stats['ghost_amp_std'] = ghost.std().item()
            stats['ghost_amp_max'] = ghost.abs().max().item()
        
        if self.use_qspace_mixing:
            eps = torch.exp(self.log_eps_qspace).item()
            stats['qspace_eps'] = eps
        
        return stats
    
    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[..., 1:, :, :] - x[..., :-1, :, :]
        dy = x[..., :, 1:, :] - x[..., :, :-1, :]
        dz = x[..., :, :, 1:] - x[..., :, :, :-1]
        return dx.abs().mean() + dy.abs().mean() + dz.abs().mean()


class DifferentiableScannerV4(nn.Module):
    
    def __init__(self,
                 protocol: AcquisitionProtocolV4,
                 n_fibers: int = 3,
                 learn_artifacts: bool = True,
                 use_dispersion: bool = True,
                 use_tortuosity: bool = False,
                 use_t2_weighting: bool = False,
                 use_restricted: bool = True,
                 use_dot: bool = False,
                 use_kurtosis: bool = False,
                 use_dki: bool = False,
                 use_diffusivity_spectrum: bool = False,
                 n_spectrum_components: int = 5,
                 use_per_shell_modulation: bool = False,
                 use_biexp_gm: bool = False,
                 use_mtcsd: bool = False,
                 n_odf_directions: int = 60,
                 n_mtcsd_atoms: int = 5,
                 learnable_responses: bool = True,
                 learn_eddy: bool = True,
                 learn_shell_gain: bool = True,
                 learn_bias_field: bool = True,
                 learn_warps: bool = False,
                 use_spatial_noise: bool = False,
                 use_psf: bool = False,
                 psf_kernel_size: int = 3,
                 psf_sigma: float = 0.68,
                 psf_learnable: bool = False,
                 use_ghosting: bool = False,
                 ghost_axis: int = 2,
                 use_qspace_mixing: bool = False,
                 qspace_k_neighbors: int = 6,
                 n_steps_qspace: int = 1,
                 learn_channel_gain: bool = False,
                 channel_gain_reg: float = 0.1,
                 low_rank_channel_gain: bool = False,
                 channel_gain_rank: int = 5,
                 noise_field_res: Tuple[int, int, int] = (4, 4, 4),
                 bias_field_res: Tuple[int, int, int] = (8, 8, 8),
                 signal_clamp_factor: Optional[float] = 1.2,
                 learn_diffusivities: bool = False,
                 use_curvature_physics: bool = False,
                 curvature_alpha: float = 10000.0,
                 curvature_epsilon: float = 0.2,
                 voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        
        self.register_buffer('bvals', protocol.bvals)
        self.register_buffer('bvecs', protocol.bvecs)
        self.register_buffer('shell_ids', protocol.shell_ids)
        self.register_buffer('shell_values', protocol.shell_values)
        if protocol.Delta is not None:
            self.register_buffer('Delta', protocol.Delta)
        else:
            self.Delta = None
        if protocol.delta is not None:
            self.register_buffer('delta_pulse', protocol.delta)
        else:
            self.delta_pulse = None
        self.te = protocol.te
        self.n_fibers = n_fibers
        self.protocol = protocol
        
        self.use_dot = use_dot
        self.use_kurtosis = use_kurtosis
        self.use_dki = use_dki
        self.use_diffusivity_spectrum = use_diffusivity_spectrum
        self.n_spectrum_components = n_spectrum_components
        self.use_spatial_noise = use_spatial_noise
        self.use_psf = use_psf
        self.use_ghosting = use_ghosting
        self.use_qspace_mixing = use_qspace_mixing
        self.use_restricted = use_restricted
        
        self.use_curvature_physics = use_curvature_physics
        self.curvature_alpha = curvature_alpha
        self.curvature_epsilon = curvature_epsilon
        self.voxel_size = voxel_size
        
        if use_curvature_physics:
            if not CURVATURE_PHYSICS_AVAILABLE:
                raise ImportError(
                    "Curvature physics requested but module not available. "
                    "Ensure models/curvature_physics.py exists."
                )
            self.curvature_computer = CurvatureComputer(voxel_size=voxel_size)
            self.geometry_dispersion = GeometryPredictedDispersion(
                alpha_init=curvature_alpha,
                epsilon=curvature_epsilon,
            )
            print(f"[Curvature Physics] Enabled: α={curvature_alpha}, ε={curvature_epsilon}")
        else:
            self.curvature_computer = None
            self.geometry_dispersion = None
        
        self.use_tissue_priors = True
        self.learnable_tissue_priors = False
        self.gm_fraction_prior = 0.30
        self.csf_fraction_prior = 0.10
        self.wm_fraction_prior = 0.50
        self.gm_prior_weight = 0.005
        self.csf_prior_weight = 0.005
        self.wm_prior_weight = 0.005
        
        self._tissue_prior_logits = None
        
        n_meas = protocol.num_measurements
        n_shells = protocol.num_shells
        
        self.use_mtcsd = use_mtcsd
        
        self.signal_model = MultiCompartmentSignalV4(
            n_fibers=n_fibers,
            n_shells=n_shells,
            use_tortuosity=use_tortuosity,
            use_dispersion=use_dispersion,
            use_t2_weighting=use_t2_weighting,
            use_restricted=use_restricted,
            use_dot=use_dot,
            use_kurtosis=use_kurtosis,
            use_dki=use_dki,
            use_diffusivity_spectrum=use_diffusivity_spectrum,
            n_spectrum_components=n_spectrum_components,
            use_per_shell_modulation=use_per_shell_modulation,
            use_biexp_gm=use_biexp_gm,
            signal_clamp_factor=signal_clamp_factor,
            use_mtcsd=use_mtcsd,
            n_odf_directions=n_odf_directions,
            n_mtcsd_atoms=n_mtcsd_atoms,
            learnable_responses=learnable_responses,
            learn_diffusivities=learn_diffusivities,
        )
        
        self.learn_diffusivities = learn_diffusivities
        
        self.learn_artifacts = learn_artifacts
        if learn_artifacts:
            self.artifacts = DifferentiableArtifactsV4(
                n_channels=n_meas,
                n_shells=n_shells,
                learn_eddy=learn_eddy,
                learn_shell_gain=learn_shell_gain,
                learn_bias_field=learn_bias_field,
                learn_warps=learn_warps,
                use_spatial_noise=use_spatial_noise,
                use_psf=use_psf,
                psf_kernel_size=psf_kernel_size,
                psf_sigma=psf_sigma,
                psf_learnable=psf_learnable,
                use_ghosting=use_ghosting,
                ghost_axis=ghost_axis,
                use_qspace_mixing=use_qspace_mixing,
                bvecs=protocol.bvecs,
                qspace_k_neighbors=qspace_k_neighbors,
                n_steps_qspace=n_steps_qspace,
                learn_channel_gain=learn_channel_gain,
                channel_gain_reg=channel_gain_reg,
                low_rank_channel_gain=low_rank_channel_gain,
                channel_gain_rank=channel_gain_rank,
                bias_field_res=bias_field_res,
                noise_field_res=noise_field_res,
            )
        
        self.use_spatial_prior = False
        self.spatial_prior = None
        
        self.use_wm_weighted_orientation = True
        self.use_curvature_prior = False
        self.curvature_prior_weight = 0.01
        self.use_permutation_invariant_orientation = False
        self.perm_inv_orientation_weight = 0.01
    
    def get_learned_diffusivities(self) -> dict:
        return self.signal_model.get_learned_diffusivities()
    
    def compute_geometry_kappa(self, 
                               fiber_dirs: torch.Tensor,
                               bvals: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_curvature_physics:
            raise RuntimeError("use_curvature_physics=False, call compute_geometry_kappa only when enabled")
        
        has_batch = False
        if fiber_dirs.dim() == 6:
            has_batch = True
            B, X, Y, Z = fiber_dirs.shape[:4]
            dirs_for_curv = fiber_dirs[0, :, :, :, 0, :]
            spatial_shape = (X, Y, Z)
        elif fiber_dirs.dim() == 5:
            if fiber_dirs.shape[-1] == 3 and fiber_dirs.shape[0] == 1:
                has_batch = True
                B = fiber_dirs.shape[0]
                X, Y, Z = fiber_dirs.shape[1:4]
                dirs_for_curv = fiber_dirs[0]
                spatial_shape = (X, Y, Z)
            else:
                X, Y, Z = fiber_dirs.shape[:3]
                dirs_for_curv = fiber_dirs[:, :, :, 0, :]
                spatial_shape = (X, Y, Z)
        elif fiber_dirs.dim() == 4:
            dirs_for_curv = fiber_dirs
            spatial_shape = fiber_dirs.shape[:-1]
        else:
            raise ValueError(f"Expected 4-6D fiber_dirs, got {fiber_dirs.dim()}D")
        
        min_spatial_size = min(spatial_shape)
        
        if min_spatial_size < 3:
            kappa_default = 1.0 / self.geometry_dispersion.epsilon
            if has_batch:
                kappa = torch.full((1,) + spatial_shape, kappa_default, device=fiber_dirs.device, dtype=fiber_dirs.dtype)
            else:
                kappa = torch.full(spatial_shape, kappa_default, device=fiber_dirs.device, dtype=fiber_dirs.dtype)
            return kappa
        
        curvature = self.curvature_computer(dirs_for_curv)
        
        kappa = self.geometry_dispersion(curvature)
        
        if has_batch:
            kappa = kappa.unsqueeze(0)
        
        return kappa
    
    def enable_curvature_physics(self,
                                  alpha: float = 10000.0,
                                  epsilon: float = 0.2,
                                  voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> 'DifferentiableScannerV4':
        if not CURVATURE_PHYSICS_AVAILABLE:
            raise ImportError(
                "Curvature physics module not available. "
                "Ensure models/curvature_physics.py exists."
            )
        
        self.use_curvature_physics = True
        self.curvature_alpha = alpha
        self.curvature_epsilon = epsilon
        self.voxel_size = voxel_size
        
        self.curvature_computer = CurvatureComputer(voxel_size=voxel_size).to(self.bvals.device)
        self.geometry_dispersion = GeometryPredictedDispersion(
            alpha_init=alpha,
            epsilon=epsilon,
        ).to(self.bvals.device)
        
        print(f"[Curvature Physics] Enabled: α={alpha}, ε={epsilon}")
        return self
    
    def set_curved_fiber_priors(self,
                                 curvature_weight: float = 0.01,
                                 perm_invariant_weight: float = 0.01,
                                 wm_weighted: bool = True) -> 'DifferentiableScannerV4':
        self.use_curvature_prior = True
        self.curvature_prior_weight = curvature_weight
        
        self.use_permutation_invariant_orientation = True
        self.perm_inv_orientation_weight = perm_invariant_weight
        
        self.use_wm_weighted_orientation = wm_weighted
        
        print(f"[Curved Fiber Priors] Enabled:")
        print(f"  Curvature prior: weight={curvature_weight}")
        print(f"  Permutation-invariant: weight={perm_invariant_weight}")
        print(f"  WM-weighted orientation: {wm_weighted}")
        
        return self
    
    def set_spatial_prior(self,
                          weight: float = 0.01,
                          connectivity: str = "6",
                          robust: bool = True,
                          huber_delta: float = 0.05,
                          fa_alpha: float = 2.0):
        from .spatial_prior import SpatialPrior3D
        
        self.use_spatial_prior = True
        self.spatial_prior = SpatialPrior3D(
            weight=weight,
            connectivity=connectivity,
            robust=robust,
            huber_delta=huber_delta,
            fa_alpha=fa_alpha,
        ).to(self.bvals.device)
        
        return self
    
    def set_fa_map(self, fa: torch.Tensor, fa_threshold: float = 0.0):
        if self.spatial_prior is None:
            raise RuntimeError("Must call set_spatial_prior() before set_fa_map()")
        self.spatial_prior.set_fa_map(fa, fa_threshold=fa_threshold)
        return self
    
    def set_learnable_tissue_priors(self,
                                     init_wm: float = 0.50,
                                     init_gm: float = 0.30,
                                     init_csf: float = 0.10,
                                     prior_weight: float = 0.001,
                                     spatial_mode: str = 'global') -> 'DifferentiableScannerV4':
        import torch.nn as nn
        
        self.learnable_tissue_priors = True
        self.tissue_prior_mode = spatial_mode
        self.tissue_prior_weight = prior_weight
        
        init_logits = torch.tensor([init_wm, init_gm, init_csf]).log()
        self._tissue_prior_logits = nn.Parameter(init_logits)
        
        self._init_tissue_priors = {'wm': init_wm, 'gm': init_gm, 'csf': init_csf}
        
        print(f"[Learnable Priors] Enabled with mode='{spatial_mode}'")
        print(f"  Initial: WM={init_wm:.0%}, GM={init_gm:.0%}, CSF={init_csf:.0%}")
        print(f"  Prior weight: {prior_weight}")
        
        return self
    
    def set_signal_aware_priors(self, 
                                signal: torch.Tensor,
                                wm_threshold: float = 0.18,
                                csf_threshold: float = 0.05,
                                weight: float = 0.25) -> 'DifferentiableScannerV4':
        self.use_signal_aware_priors = True
        self.signal_aware_weight = weight
        
        bvals = self.bvals.cpu().numpy()
        max_b_idx = bvals > 2500
        
        if signal.dim() == 4:
            signal = signal.unsqueeze(0)
        
        b0_idx = bvals < 100
        s0 = signal[..., b0_idx].mean(dim=-1, keepdim=True).clamp(min=1e-6)
        high_b_signal = signal[..., max_b_idx].mean(dim=-1) / s0.squeeze(-1)
        
        wm_prob = torch.sigmoid((high_b_signal - wm_threshold) * 20)
        csf_prob = torch.sigmoid(-(high_b_signal - csf_threshold) * 20)
        gm_prob = 1 - wm_prob - csf_prob
        gm_prob = gm_prob.clamp(min=0)
        
        total = wm_prob + gm_prob + csf_prob
        self._signal_wm_prior = (wm_prob / total).detach()
        self._signal_gm_prior = (gm_prob / total).detach()
        self._signal_csf_prior = (csf_prob / total).detach()
        
        print(f"[Signal-Aware Priors] Enabled:")
        print(f"  WM threshold:  {wm_threshold:.2f}")
        print(f"  CSF threshold: {csf_threshold:.2f}")
        print(f"  Prior weight:  {weight}")
        print(f"  WM prior range: {self._signal_wm_prior.min():.2f} - {self._signal_wm_prior.max():.2f}")
        
        return self
    
    def get_learned_tissue_priors(self) -> dict:
        if self._tissue_prior_logits is None:
            return {
                'wm': self.wm_fraction_prior,
                'gm': self.gm_fraction_prior,
                'csf': self.csf_fraction_prior,
            }
        
        priors = torch.softmax(self._tissue_prior_logits, dim=0)
        return {
            'wm': priors[0].item(),
            'gm': priors[1].item(),
            'csf': priors[2].item(),
        }
    
    def set_topology_prior(self,
                           lambda_orphan: float = 0.01,
                           lambda_continuity: float = 0.01,
                           lambda_endpoint: float = 0.01):
        self.use_topology_prior = True
        self.topology_lambda_orphan = lambda_orphan
        self.topology_lambda_continuity = lambda_continuity
        self.topology_lambda_endpoint = lambda_endpoint
        return self
    
    def set_exchange(self,
                     tau_ex: float = 50.0,
                     exchange_pairs: List[str] = ['restricted_gm'],
                     learnable: bool = True) -> 'DifferentiableScannerV4':
        from .karger_exchange import KargerExchangeModule
        
        self.signal_model.use_exchange = True
        self.signal_model.exchange_pairs = exchange_pairs
        self.signal_model.exchange_module = KargerExchangeModule(
            compartment_pairs=[('restricted', 'gm')],
            init_tau_ex=tau_ex,
            learnable=learnable,
        ).to(self.bvals.device)
        
        return self
    
    def get_exchange_times(self) -> Dict[str, float]:
        if hasattr(self.signal_model, 'exchange_module') and self.signal_model.exchange_module is not None:
            tau_ex = self.signal_model.exchange_module.tau_ex
            return {'tau_ex': tau_ex[0].item()}
        return {}
    
    @classmethod
    def from_files(cls, bval_file: str, bvec_file: str, 
                   device: str = 'cuda', **kwargs) -> 'DifferentiableScannerV4':
        protocol = AcquisitionProtocolV4.from_files(bval_file, bvec_file, device=device)
        return cls(protocol, **kwargs)
    
    @classmethod
    def from_tensors(cls, bvals: torch.Tensor, bvecs: torch.Tensor,
                     **kwargs) -> 'DifferentiableScannerV4':
        protocol = AcquisitionProtocolV4.from_tensors(bvals, bvecs)
        return cls(protocol, **kwargs)
    
    def forward(self,
                params: Dict[str, torch.Tensor],
                add_noise: bool = False,
                normalize_fractions: bool = True,
                use_geometry_kappa: Optional[bool] = None) -> torch.Tensor:
        params = self._clamp_params(params)
        
        use_geo = use_geometry_kappa if use_geometry_kappa is not None else self.use_curvature_physics
        if use_geo and self.curvature_computer is not None:
            fiber_dirs = params.get('fiber_dirs')
            if fiber_dirs is not None:
                geometry_kappa = self.compute_geometry_kappa(fiber_dirs)
                params = dict(params)
                params['kappa'] = geometry_kappa
        
        signal = self.signal_model(
            params,
            self.bvals,
            self.bvecs,
            self.shell_ids,
            self.te,
            Delta=getattr(self, 'Delta', None),
            normalize_fractions=normalize_fractions,
        )
        
        if self.learn_artifacts:
            signal = self.artifacts(signal, self.shell_ids, bvals=self.bvals, add_noise=add_noise)
        
        return signal
    
    def _clamp_params(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        params = dict(params)
        
        if params.get('f_restricted') is not None:
            params['f_restricted'] = params['f_restricted'].clamp(0.0, 0.3)
        if params.get('f_dot') is not None:
            params['f_dot'] = params['f_dot'].clamp(0.0, 0.2)
        if params.get('f_intra') is not None:
            params['f_intra'] = params['f_intra'].clamp(0.05, 0.95)
        if params.get('kurtosis') is not None:
            params['kurtosis'] = params['kurtosis'].clamp(0.0, 3.0)
        
        return params
    
    def get_regularization_loss(self, 
                                params: Optional[Dict[str, torch.Tensor]] = None,
                                w_artifacts: float = 1.0,
                                w_micro: float = 1.0,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.bvals.device
        loss = torch.tensor(0.0, device=device)
        
        if self.learn_artifacts:
            loss = loss + w_artifacts * self.artifacts.get_regularization_loss()
        
        if params is not None:
            loss = loss + w_micro * self._microstructure_regularization(params, mask=mask)
        
        return loss
    
    def _microstructure_regularization(self, 
                                       params: Dict[str, torch.Tensor],
                                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.bvals.device
        loss = torch.tensor(0.0, device=device)
        
        priors = {
            'd_csf': self.signal_model.d_csf_prior,
            'd_gm': self.signal_model.d_gm_prior,
            'd_parallel': self.signal_model.d_parallel_prior,
            'd_perp': self.signal_model.d_perp_prior,
            'd_restricted': self.signal_model.d_restricted_prior,
            'd_dot': self.signal_model.d_dot_prior,
        }
        
        for name, prior in priors.items():
            if name in params:
                d = params[name]
                if d.requires_grad:
                    loss = loss + 0.01 * ((d - prior) ** 2).mean()
        
        if 'f_intra' in params and params['f_intra'].requires_grad:
            loss = loss + 0.01 * ((params['f_intra'] - 0.7) ** 2).mean()
        
        if 'kappa' in params and params['kappa'].requires_grad:
            kappa_phys = F.softplus(params['kappa']).clamp(0.5, 100.0)
            loss = loss + 0.01 * ((kappa_phys - 8.0) ** 2).mean()
        
        if getattr(self, 'use_kappa_wm_coupling', True):
            if 'kappa' in params and 'f_wm' in params:
                kappa = params['kappa']
                f_wm = params['f_wm']
                if f_wm.dim() > 4:
                    f_wm_total = f_wm.sum(dim=-1)
                else:
                    f_wm_total = f_wm
                
                if kappa.requires_grad:
                    kappa_phys = F.softplus(kappa).clamp(0.5, 100.0)
                    
                    kappa_min = 2.0
                    kappa_max = 50.0
                    expected_kappa = kappa_min + (kappa_max - kappa_min) * f_wm_total.detach()
                    
                    kappa_deficit = F.relu(expected_kappa - kappa_phys)
                    coupling_weight = getattr(self, 'kappa_wm_coupling_weight', 0.01)
                    loss = loss + coupling_weight * (kappa_deficit ** 2).mean()
        
        if 'f_restricted' in params and params['f_restricted'].requires_grad:
            loss = loss + 0.01 * ((params['f_restricted'] - 0.05) ** 2).mean()
        
        if 'f_dot' in params and params['f_dot'].requires_grad:
            loss = loss + 0.01 * ((params['f_dot'] - 0.05) ** 2).mean()
        
        if 'kurtosis' in params and params['kurtosis'].requires_grad:
            loss = loss + 0.01 * ((params['kurtosis'] - 1.0) ** 2).mean()
        
        if getattr(self, 'use_tissue_priors', True):
            if getattr(self, 'learnable_tissue_priors', False) and self._tissue_prior_logits is not None:
                learned_priors = torch.softmax(self._tissue_prior_logits, dim=0)
                wm_prior = learned_priors[0]
                gm_prior = learned_priors[1]
                csf_prior = learned_priors[2]
                prior_weight = getattr(self, 'tissue_prior_weight', 0.001)
                
                if 'f_wm' in params and params['f_wm'].requires_grad:
                    f_wm = params['f_wm']
                    if f_wm.dim() > 4:
                        f_wm_total = f_wm.sum(dim=-1)
                    else:
                        f_wm_total = f_wm
                    loss = loss + prior_weight * (f_wm_total.mean() - wm_prior) ** 2
                
                if 'f_gm' in params and params['f_gm'].requires_grad:
                    loss = loss + prior_weight * (params['f_gm'].mean() - gm_prior) ** 2
                
                if 'f_csf' in params and params['f_csf'].requires_grad:
                    loss = loss + prior_weight * (params['f_csf'].mean() - csf_prior) ** 2
            else:
                gm_prior = getattr(self, 'gm_fraction_prior', 0.30)
                gm_prior_weight = getattr(self, 'gm_prior_weight', 0.005)
                if 'f_gm' in params and params['f_gm'].requires_grad:
                    loss = loss + gm_prior_weight * (params['f_gm'].mean() - gm_prior) ** 2
                
                csf_prior = getattr(self, 'csf_fraction_prior', 0.10)
                csf_prior_weight = getattr(self, 'csf_prior_weight', 0.005)
                if 'f_csf' in params and params['f_csf'].requires_grad:
                    loss = loss + csf_prior_weight * (params['f_csf'].mean() - csf_prior) ** 2
                
                wm_prior = getattr(self, 'wm_fraction_prior', 0.50)
                wm_prior_weight = getattr(self, 'wm_prior_weight', 0.005)
                if 'f_wm' in params and params['f_wm'].requires_grad:
                    f_wm = params['f_wm']
                    if f_wm.dim() > 4:
                        f_wm_total = f_wm.sum(dim=-1)
                    else:
                        f_wm_total = f_wm
                    loss = loss + wm_prior_weight * (f_wm_total.mean() - wm_prior) ** 2
        
        if getattr(self, 'use_signal_aware_priors', False):
            weight = getattr(self, 'signal_aware_weight', 0.01)
            
            if 'f_wm' in params and params['f_wm'].requires_grad:
                f_wm = params['f_wm']
                if f_wm.dim() > 4:
                    f_wm_total = f_wm.sum(dim=-1)
                else:
                    f_wm_total = f_wm
                wm_prior = self._signal_wm_prior.to(f_wm_total.device)
                loss = loss + weight * ((f_wm_total - wm_prior) ** 2).mean()
            
            if 'f_gm' in params and params['f_gm'].requires_grad:
                gm_prior = self._signal_gm_prior.to(params['f_gm'].device)
                loss = loss + weight * ((params['f_gm'] - gm_prior) ** 2).mean()
            
            if 'f_csf' in params and params['f_csf'].requires_grad:
                csf_prior = self._signal_csf_prior.to(params['f_csf'].device)
                loss = loss + weight * ((params['f_csf'] - csf_prior) ** 2).mean()
        
        if not getattr(self, 'use_spatial_prior', False):
            for name in ['f_csf', 'f_gm', 'f_wm', 'f_restricted', 'f_dot']:
                if name in params and params[name].requires_grad and params[name].dim() >= 4:
                    loss = loss + 0.001 * self._spatial_tv(params[name])
            
            for name in ['f_intra', 'kappa', 'kurtosis']:
                if name in params and params[name].requires_grad and params[name].dim() >= 4:
                    loss = loss + 0.0005 * self._spatial_tv(params[name])
        
        if 'fiber_dirs' in params and params['fiber_dirs'].requires_grad:
            f_wm_for_weight = params.get('f_wm') if getattr(self, 'use_wm_weighted_orientation', True) else None
            loss = loss + 0.0005 * self._orientation_smoothness(params['fiber_dirs'], f_wm_for_weight)
        
        if getattr(self, 'use_curvature_prior', False):
            curvature_weight = getattr(self, 'curvature_prior_weight', 0.01)
            if 'fiber_dirs' in params and params['fiber_dirs'].requires_grad:
                f_wm_for_weight = params.get('f_wm') if params.get('f_wm') is not None and params['f_wm'].dim() > 4 else None
                loss = loss + curvature_weight * self._curvature_loss(params['fiber_dirs'], f_wm_for_weight)
        
        if getattr(self, 'use_permutation_invariant_orientation', False):
            perm_inv_weight = getattr(self, 'perm_inv_orientation_weight', 0.01)
            if 'fiber_dirs' in params and 'f_wm' in params:
                if params['fiber_dirs'].requires_grad and params['f_wm'].dim() > 4:
                    loss = loss + perm_inv_weight * self._permutation_invariant_orientation_loss(
                        params['fiber_dirs'], params['f_wm']
                    )
        
        if getattr(self, 'use_spatial_prior', False) and self.spatial_prior is not None:
            if 'f_csf' in params and params['f_csf'].requires_grad:
                loss = loss + self.spatial_prior(params['f_csf'], mask=mask)
            
            if 'f_gm' in params and params['f_gm'].requires_grad:
                loss = loss + self.spatial_prior(params['f_gm'], mask=mask)
            
            if 'f_wm' in params and params['f_wm'].requires_grad:
                f_wm = params['f_wm']
                if f_wm.dim() > 4:
                    f_wm_total = f_wm.sum(dim=-1)
                else:
                    f_wm_total = f_wm
                loss = loss + self.spatial_prior(f_wm_total, mask=mask)
            
            if 'f_restricted' in params and params['f_restricted'].requires_grad:
                loss = loss + 0.5 * self.spatial_prior(params['f_restricted'], mask=mask)
            
            if 'd_restricted' in params and params['d_restricted'].dim() >= 4:
                d_restricted = params['d_restricted']
                if d_restricted.requires_grad or d_restricted.shape[-1] > 1 or d_restricted.shape[-2] > 1:
                    loss = loss + 0.5 * self.spatial_prior(d_restricted, mask=mask)
        
        if getattr(self, 'use_topology_prior', False):
            topology_losses = self.get_topology_losses(
                params,
                lambda_orphan=getattr(self, 'topology_lambda_orphan', 0.01),
                lambda_continuity=getattr(self, 'topology_lambda_continuity', 0.01),
                lambda_endpoint=getattr(self, 'topology_lambda_endpoint', 0.01),
            )
            loss = loss + topology_losses['total']
        
        return loss
    
    def _orientation_smoothness(self, fiber_dirs: torch.Tensor, 
                                  f_wm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if fiber_dirs.dim() < 5:
            return torch.tensor(0.0, device=fiber_dirs.device)
        
        device = fiber_dirs.device
        loss = torch.tensor(0.0, device=device)
        
        fd = F.normalize(fiber_dirs, dim=-1, eps=1e-8)
        
        
        if fd.shape[1] > 1:
            dot_x = (fd[:, 1:] * fd[:, :-1]).sum(dim=-1).abs()
            loss_x = (1.0 - dot_x)
            if f_wm is not None and f_wm.dim() >= 5:
                w_x = torch.sqrt(f_wm[:, 1:] * f_wm[:, :-1] + 1e-8)
                loss_x = loss_x * w_x
            loss = loss + loss_x.mean()
        
        if fd.shape[2] > 1:
            dot_y = (fd[:, :, 1:] * fd[:, :, :-1]).sum(dim=-1).abs()
            loss_y = (1.0 - dot_y)
            if f_wm is not None and f_wm.dim() >= 5:
                w_y = torch.sqrt(f_wm[:, :, 1:] * f_wm[:, :, :-1] + 1e-8)
                loss_y = loss_y * w_y
            loss = loss + loss_y.mean()
        
        if fd.shape[3] > 1:
            dot_z = (fd[:, :, :, 1:] * fd[:, :, :, :-1]).sum(dim=-1).abs()
            loss_z = (1.0 - dot_z)
            if f_wm is not None and f_wm.dim() >= 5:
                w_z = torch.sqrt(f_wm[:, :, :, 1:] * f_wm[:, :, :, :-1] + 1e-8)
                loss_z = loss_z * w_z
            loss = loss + loss_z.mean()
        
        return loss
    
    def _spatial_tv(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 4:
            return torch.tensor(0.0, device=x.device)
        
        device = x.device
        loss = torch.tensor(0.0, device=device)
        
        if x.shape[1] > 1:
            loss = loss + (x[:, 1:] - x[:, :-1]).abs().mean()
        if x.shape[2] > 1:
            loss = loss + (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        if x.shape[3] > 1:
            loss = loss + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        
        return loss
    
    
    def _curvature_loss(self, 
                        fiber_dirs: torch.Tensor,
                        f_wm: Optional[torch.Tensor] = None,
                        eps: float = 0.5) -> torch.Tensor:
        if fiber_dirs.dim() < 6:
            return torch.tensor(0.0, device=fiber_dirs.device)
        
        B, X, Y, Z, K, _ = fiber_dirs.shape
        device = fiber_dirs.device
        
        if X < 3 or Y < 3 or Z < 3:
            return torch.tensor(0.0, device=device)
        
        xs = torch.linspace(-1, 1, X, device=device)
        ys = torch.linspace(-1, 1, Y, device=device)
        zs = torch.linspace(-1, 1, Z, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        base_grid = torch.stack([gz, gy, gx], dim=-1)
        
        voxel_to_norm = torch.tensor(
            [2.0/(max(Z-1,1)), 2.0/(max(Y-1,1)), 2.0/(max(X-1,1))], 
            device=device
        )
        
        loss = torch.tensor(0.0, device=device)
        
        for k in range(K):
            dir_k = fiber_dirs[..., k, :]
            dir_k = F.normalize(dir_k, dim=-1, eps=1e-8)
            
            dir_k_grid = dir_k[..., [2, 1, 0]] * voxel_to_norm
            
            grid_fwd = base_grid.unsqueeze(0) + eps * dir_k_grid
            
            dir_k_5d = dir_k.permute(0, 4, 1, 2, 3)
            
            dir_fwd = F.grid_sample(
                dir_k_5d, grid_fwd,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            dir_fwd = dir_fwd.permute(0, 2, 3, 4, 1)
            dir_fwd = F.normalize(dir_fwd, dim=-1, eps=1e-8)
            
            dot = (dir_k * dir_fwd).sum(dim=-1).abs()
            curvature_k = 1.0 - dot
            
            if f_wm is not None and f_wm.dim() >= 5:
                w_k = f_wm[..., k]
                curvature_k = curvature_k * w_k
            
            loss = loss + curvature_k.mean()
        
        return loss / max(K, 1)
    
    def _permutation_invariant_orientation_loss(
        self,
        fiber_dirs: torch.Tensor,
        f_wm: torch.Tensor,
        tau: float = 0.05,
        temperature: float = 0.1
    ) -> torch.Tensor:
        if fiber_dirs.dim() < 6 or f_wm.dim() < 5:
            return torch.tensor(0.0, device=fiber_dirs.device)
        
        B, X, Y, Z, K, _ = fiber_dirs.shape
        device = fiber_dirs.device
        
        if K <= 1:
            return self._orientation_smoothness(fiber_dirs, f_wm)
        
        loss = torch.tensor(0.0, device=device)
        n_axes = 0
        
        fd = F.normalize(fiber_dirs, dim=-1, eps=1e-8)
        
        axes_pairs = [
            (1, slice(1, None), slice(None, -1)),
            (2, slice(1, None), slice(None, -1)),
            (3, slice(1, None), slice(None, -1)),
        ]
        
        for axis, fwd_slice, bwd_slice in axes_pairs:
            if fiber_dirs.shape[axis] <= 1:
                continue
            
            if axis == 1:
                fd_A = fd[:, fwd_slice, :, :, :, :]
                fd_B = fd[:, bwd_slice, :, :, :, :]
                f_A = f_wm[:, fwd_slice, :, :, :]
                f_B = f_wm[:, bwd_slice, :, :, :]
            elif axis == 2:
                fd_A = fd[:, :, fwd_slice, :, :, :]
                fd_B = fd[:, :, bwd_slice, :, :, :]
                f_A = f_wm[:, :, fwd_slice, :, :]
                f_B = f_wm[:, :, bwd_slice, :, :]
            else:
                fd_A = fd[:, :, :, fwd_slice, :, :]
                fd_B = fd[:, :, :, bwd_slice, :, :]
                f_A = f_wm[:, :, :, fwd_slice, :]
                f_B = f_wm[:, :, :, bwd_slice, :]
            
            spatial_size = fd_A.shape[1] * fd_A.shape[2] * fd_A.shape[3]
            fd_A_flat = fd_A.reshape(B * spatial_size, K, 3)
            fd_B_flat = fd_B.reshape(B * spatial_size, K, 3)
            f_A_flat = f_A.reshape(B * spatial_size, K)
            f_B_flat = f_B.reshape(B * spatial_size, K)
            
            S = torch.bmm(fd_A_flat, fd_B_flat.transpose(1, 2)).abs()
            
            assignment = F.softmax(S / temperature, dim=-1)
            
            matched_sim = (assignment * S).sum(dim=-1)
            
            f_A_active = (f_A_flat > tau).float()
            f_B_max = (assignment * f_B_flat.unsqueeze(1)).sum(dim=-1)
            weight = torch.sqrt(f_A_flat * f_B_max + 1e-8) * f_A_active
            
            axis_loss = ((1.0 - matched_sim) * weight).sum() / (weight.sum() + 1e-8)
            loss = loss + axis_loss
            n_axes += 1
        
        return loss / max(n_axes, 1)
    
    
    def _orphan_wm_loss(self, f_wm_total: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
        if f_wm_total.dim() < 4:
            return torch.tensor(0.0, device=f_wm_total.device)
        
        B, X, Y, Z = f_wm_total.shape
        device = f_wm_total.device
        
        w_face = 1.0
        w_edge = 0.707
        w_corner = 0.577
        
        kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
        
        kernel[0, 0, 1, 1, 0] = w_face
        kernel[0, 0, 1, 1, 2] = w_face
        kernel[0, 0, 1, 0, 1] = w_face
        kernel[0, 0, 1, 2, 1] = w_face
        kernel[0, 0, 0, 1, 1] = w_face
        kernel[0, 0, 2, 1, 1] = w_face
        
        kernel[0, 0, 0, 0, 1] = w_edge
        kernel[0, 0, 0, 2, 1] = w_edge
        kernel[0, 0, 2, 0, 1] = w_edge
        kernel[0, 0, 2, 2, 1] = w_edge
        kernel[0, 0, 0, 1, 0] = w_edge
        kernel[0, 0, 0, 1, 2] = w_edge
        kernel[0, 0, 2, 1, 0] = w_edge
        kernel[0, 0, 2, 1, 2] = w_edge
        kernel[0, 0, 1, 0, 0] = w_edge
        kernel[0, 0, 1, 0, 2] = w_edge
        kernel[0, 0, 1, 2, 0] = w_edge
        kernel[0, 0, 1, 2, 2] = w_edge
        
        kernel[0, 0, 0, 0, 0] = w_corner
        kernel[0, 0, 0, 0, 2] = w_corner
        kernel[0, 0, 0, 2, 0] = w_corner
        kernel[0, 0, 0, 2, 2] = w_corner
        kernel[0, 0, 2, 0, 0] = w_corner
        kernel[0, 0, 2, 0, 2] = w_corner
        kernel[0, 0, 2, 2, 0] = w_corner
        kernel[0, 0, 2, 2, 2] = w_corner
        
        total_weight = 6 * w_face + 12 * w_edge + 8 * w_corner
        
        f_wm_5d = f_wm_total.unsqueeze(1)
        neighbor_sum = F.conv3d(f_wm_5d, kernel, padding=1)
        neighbor_mean = neighbor_sum.squeeze(1) / total_weight
        
        is_wm = f_wm_total > tau
        neighbors_empty = neighbor_mean < tau
        orphan_mask = is_wm & neighbors_empty
        
        if orphan_mask.any():
            loss = (f_wm_total[orphan_mask] ** 2).mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def _orphan_wm_loss_6conn(self, f_wm_total: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
        if f_wm_total.dim() < 4:
            return torch.tensor(0.0, device=f_wm_total.device)
        
        B, X, Y, Z = f_wm_total.shape
        device = f_wm_total.device
        
        kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
        
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        
        total_weight = 6.0
        
        f_wm_5d = f_wm_total.unsqueeze(1)
        neighbor_sum = F.conv3d(f_wm_5d, kernel, padding=1)
        neighbor_mean = neighbor_sum.squeeze(1) / total_weight
        
        is_wm = f_wm_total > tau
        neighbors_empty = neighbor_mean < tau
        orphan_mask = is_wm & neighbors_empty
        
        if orphan_mask.any():
            loss = (f_wm_total[orphan_mask] ** 2).mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss

    def _fiber_ordering_loss(self, f_wm: torch.Tensor) -> torch.Tensor:
        if f_wm.dim() < 5 or f_wm.shape[-1] <= 1:
            return torch.tensor(0.0, device=f_wm.device)
        
        K = f_wm.shape[-1]
        device = f_wm.device
        loss = torch.tensor(0.0, device=device)
        
        for k in range(K - 1):
            violation = F.relu(f_wm[..., k+1] - f_wm[..., k])
            loss = loss + violation.mean()
        
        return loss / max(K - 1, 1)
    
    def _fiber_repulsion_loss(self, f_wm: torch.Tensor, fiber_dirs: torch.Tensor) -> torch.Tensor:
        if f_wm.dim() < 5 or f_wm.shape[-1] <= 1 or fiber_dirs is None:
            return torch.tensor(0.0, device=f_wm.device)
        
        K = f_wm.shape[-1]
        device = f_wm.device
        loss = torch.tensor(0.0, device=device)
        
        fiber_dirs_norm = F.normalize(fiber_dirs, dim=-1, eps=1e-8)
        
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dot_ij = (fiber_dirs_norm[..., i, :] * fiber_dirs_norm[..., j, :]).sum(dim=-1)
                dot_ij = torch.abs(dot_ij)
                
                f_prod = f_wm[..., i] * f_wm[..., j]
                
                loss = loss + (f_prod * dot_ij).mean()
                n_pairs += 1
        
        return loss / max(n_pairs, 1)
    
    def _directional_continuity_loss(self, 
                                      f_wm: torch.Tensor, 
                                      fiber_dirs: torch.Tensor,
                                      eps: float = 0.5) -> torch.Tensor:
        if f_wm.dim() < 5 or fiber_dirs.dim() < 6:
            return torch.tensor(0.0, device=f_wm.device)
        
        B, X, Y, Z, K = f_wm.shape
        device = f_wm.device
        
        xs = torch.linspace(-1, 1, X, device=device)
        ys = torch.linspace(-1, 1, Y, device=device)
        zs = torch.linspace(-1, 1, Z, device=device)
        
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        base_grid = torch.stack([gz, gy, gx], dim=-1)
        
        voxel_to_norm = torch.tensor([2.0/(max(Z-1,1)), 2.0/(max(Y-1,1)), 2.0/(max(X-1,1))], device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        for k in range(K):
            f_k = f_wm[..., k]
            dir_k = fiber_dirs[..., k, :]
            
            dir_k_grid = dir_k[..., [2, 1, 0]] * voxel_to_norm
            
            grid_fwd = base_grid.unsqueeze(0) + eps * dir_k_grid
            grid_bwd = base_grid.unsqueeze(0) - eps * dir_k_grid
            
            f_k_5d = f_k.unsqueeze(1)
            
            f_fwd = F.grid_sample(
                f_k_5d, grid_fwd, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
            ).squeeze(1)
            
            f_bwd = F.grid_sample(
                f_k_5d, grid_bwd,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(1)
            
            diff = (f_k - f_fwd).abs() + (f_k - f_bwd).abs()
            
            weighted_diff = diff * f_k
            loss = loss + weighted_diff.mean()
        
        return loss / max(K, 1)
    
    def _endpoint_gm_loss(self,
                          f_wm: torch.Tensor,
                          fiber_dirs: torch.Tensor,
                          f_gm: torch.Tensor,
                          eps: float = 0.5,
                          tau: float = 0.05) -> torch.Tensor:
        if f_wm.dim() < 5 or fiber_dirs.dim() < 6 or f_gm.dim() < 4:
            return torch.tensor(0.0, device=f_wm.device)
        
        B, X, Y, Z, K = f_wm.shape
        device = f_wm.device
        
        xs = torch.linspace(-1, 1, X, device=device)
        ys = torch.linspace(-1, 1, Y, device=device)
        zs = torch.linspace(-1, 1, Z, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        base_grid = torch.stack([gz, gy, gx], dim=-1)
        
        voxel_to_norm = torch.tensor([2.0/(max(Z-1,1)), 2.0/(max(Y-1,1)), 2.0/(max(X-1,1))], device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        for k in range(K):
            f_k = f_wm[..., k]
            dir_k = fiber_dirs[..., k, :]
            
            dir_k_grid = dir_k[..., [2, 1, 0]] * voxel_to_norm
            
            grid_fwd = base_grid.unsqueeze(0) + eps * dir_k_grid
            grid_bwd = base_grid.unsqueeze(0) - eps * dir_k_grid
            
            f_k_5d = f_k.unsqueeze(1)
            
            f_fwd = F.grid_sample(
                f_k_5d, grid_fwd,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(1)
            
            f_bwd = F.grid_sample(
                f_k_5d, grid_bwd,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(1)
            
            f_neighbor_max = torch.maximum(f_fwd, f_bwd)
            endpoint_indicator = F.relu(f_k - f_neighbor_max - tau)
            
            penalty = endpoint_indicator * (1.0 - f_gm)
            loss = loss + penalty.mean()
        
        return loss / max(K, 1)
    
    def get_topology_losses(self,
                            params: Dict[str, torch.Tensor],
                            lambda_orphan: float = 0.01,
                            lambda_continuity: float = 0.01,
                            lambda_endpoint: float = 0.01,
                            lambda_ordering: float = 0.01,
                            lambda_repulsion: float = 0.01,
                            lambda_curvature: float = 0.0,
                            lambda_perm_invariant: float = 0.0,
                            connectivity: int = 26) -> Dict[str, torch.Tensor]:
        device = self.bvals.device
        losses = {
            'orphan': torch.tensor(0.0, device=device),
            'continuity': torch.tensor(0.0, device=device),
            'endpoint': torch.tensor(0.0, device=device),
            'ordering': torch.tensor(0.0, device=device),
            'repulsion': torch.tensor(0.0, device=device),
            'curvature': torch.tensor(0.0, device=device),
            'perm_invariant': torch.tensor(0.0, device=device),
            'total': torch.tensor(0.0, device=device),
        }
        
        f_wm = params.get('f_wm')
        fiber_dirs = params.get('fiber_dirs')
        f_gm = params.get('f_gm')
        
        if f_wm is None:
            return losses
        
        if f_wm.dim() > 4:
            f_wm_total = f_wm.sum(dim=-1)
        else:
            f_wm_total = f_wm
        
        if lambda_orphan > 0:
            if connectivity == 6:
                losses['orphan'] = lambda_orphan * self._orphan_wm_loss_6conn(f_wm_total)
            else:
                losses['orphan'] = lambda_orphan * self._orphan_wm_loss(f_wm_total)
        
        if lambda_continuity > 0 and f_wm.dim() > 4 and fiber_dirs is not None:
            losses['continuity'] = lambda_continuity * self._directional_continuity_loss(f_wm, fiber_dirs)
        
        if lambda_endpoint > 0 and f_wm.dim() > 4 and fiber_dirs is not None and f_gm is not None:
            losses['endpoint'] = lambda_endpoint * self._endpoint_gm_loss(f_wm, fiber_dirs, f_gm)
        
        if lambda_ordering > 0 and f_wm.dim() > 4:
            losses['ordering'] = lambda_ordering * self._fiber_ordering_loss(f_wm)
        
        if lambda_repulsion > 0 and f_wm.dim() > 4 and fiber_dirs is not None:
            losses['repulsion'] = lambda_repulsion * self._fiber_repulsion_loss(f_wm, fiber_dirs)
        
        if lambda_curvature > 0 and fiber_dirs is not None:
            losses['curvature'] = lambda_curvature * self._curvature_loss(
                fiber_dirs, f_wm if f_wm.dim() > 4 else None
            )
        
        if lambda_perm_invariant > 0 and f_wm.dim() > 4 and fiber_dirs is not None:
            losses['perm_invariant'] = lambda_perm_invariant * self._permutation_invariant_orientation_loss(
                fiber_dirs, f_wm
            )
        
        losses['total'] = (losses['orphan'] + losses['continuity'] + losses['endpoint'] +
                          losses['ordering'] + losses['repulsion'] + 
                          losses['curvature'] + losses['perm_invariant'])
        
        return losses

    def get_shell_statistics(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        stats = {}
        for s in range(self.protocol.num_shells):
            mask = self.shell_ids == s
            shell_signal = signal[..., mask]
            stats[f'shell_{int(self.shell_values[s].item())}_mean'] = shell_signal.mean()
            stats[f'shell_{int(self.shell_values[s].item())}_std'] = shell_signal.std()
        return stats
    
    
    def get_bochner_loss(self, 
                         predicted_signal: torch.Tensor,
                         params: Dict[str, torch.Tensor],
                         m: int = 16,
                         b_shells: Optional[List[float]] = None,
                         penalty_type: str = 'min_eigenvalue',
                         subsample_voxels: int = 500) -> torch.Tensor:
        device = predicted_signal.device
        
        if b_shells is None:
            b_shells = [b.item() for b in self.shell_values if b > 500][:2]
        
        if len(b_shells) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b_val in b_shells:
            loss = self._compute_bochner_loss_single_shell(
                predicted_signal, params, m, b_val, penalty_type, subsample_voxels
            )
            total_loss = total_loss + loss
        
        return total_loss / len(b_shells)
    
    def _compute_bochner_loss_single_shell(self,
                                            signal: torch.Tensor,
                                            params: Dict[str, torch.Tensor],
                                            m: int,
                                            b_shell: float,
                                            penalty_type: str,
                                            subsample_voxels: int) -> torch.Tensor:
        device = signal.device
        
        q_vectors = self._fibonacci_sphere(m, device)
        
        dots = torch.mm(q_vectors, q_vectors.T)
        
        b_eff = 2 * b_shell * (1 - dots)
        
        q_diff = q_vectors.unsqueeze(0) - q_vectors.unsqueeze(1)
        q_diff_norm = torch.norm(q_diff, dim=-1, keepdim=True).clamp(min=1e-8)
        q_diff_dirs = q_diff / q_diff_norm
        
        diagonal_mask = b_eff < 1e-6
        q_diff_dirs[diagonal_mask] = q_vectors[0]
        
        b_eff_flat = b_eff.reshape(-1)
        dirs_flat = q_diff_dirs.reshape(-1, 3)
        
        gram_values = self._eval_signal_at_q(params, b_eff_flat, dirs_flat, subsample_voxels)
        
        if gram_values is None:
            return torch.tensor(0.0, device=device)
        
        n_voxels = gram_values.shape[0]
        K = gram_values.reshape(n_voxels, m, m)
        
        return self._bochner_psd_penalty(K, penalty_type)
    
    def _fibonacci_sphere(self, n: int, device: torch.device) -> torch.Tensor:
        indices = torch.arange(n, device=device, dtype=torch.float32)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        
        y = 1 - (indices / max(n - 1, 1)) * 2
        radius = torch.sqrt((1 - y * y).clamp(min=0))
        theta = phi * indices
        
        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius
        
        return torch.stack([x, y, z], dim=-1)
    
    def _eval_signal_at_q(self,
                          params: Dict[str, torch.Tensor],
                          bvals: torch.Tensor,
                          bvecs: torch.Tensor,
                          max_voxels: int) -> Optional[torch.Tensor]:
        device = bvals.device
        
        f_csf = params.get('f_csf')
        if f_csf is None:
            return None
        
        spatial_shape = f_csf.shape
        f_csf_flat = f_csf.reshape(-1)
        n_voxels = f_csf_flat.shape[0]
        
        if n_voxels > max_voxels:
            idx = torch.randperm(n_voxels, device=device)[:max_voxels]
        else:
            idx = torch.arange(n_voxels, device=device)
        
        n_selected = len(idx)
        
        sub_params = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                v_flat = v.reshape(-1, *v.shape[len(spatial_shape):])
                sub_params[k] = v_flat[idx]
            else:
                sub_params[k] = v
        
        signal = self._simple_signal_model(sub_params, bvals, bvecs)
        
        return signal
    
    def _simple_signal_model(self,
                             params: Dict[str, torch.Tensor],
                             bvals: torch.Tensor,
                             bvecs: torch.Tensor) -> torch.Tensor:
        device = bvals.device
        n_meas = bvals.shape[0]
        
        f_csf = params.get('f_csf', torch.zeros(1, device=device))
        f_gm = params.get('f_gm', torch.zeros(1, device=device))
        d_csf = params.get('d_csf', 3.0)
        d_gm = params.get('d_gm', 0.8)
        d_parallel = params.get('d_parallel', 1.7)
        d_perp = params.get('d_perp', 0.3)
        
        if isinstance(d_csf, (int, float)):
            d_csf = torch.tensor(d_csf, device=device)
        if isinstance(d_gm, (int, float)):
            d_gm = torch.tensor(d_gm, device=device)
        if isinstance(d_parallel, (int, float)):
            d_parallel = torch.tensor(d_parallel, device=device)
        if isinstance(d_perp, (int, float)):
            d_perp = torch.tensor(d_perp, device=device)
        
        if f_csf.dim() >= 1 and f_csf.numel() > 1:
            n_voxels = f_csf.shape[0]
        else:
            n_voxels = 1
            f_csf = f_csf.reshape(1)
            f_gm = f_gm.reshape(1) if isinstance(f_gm, torch.Tensor) else torch.tensor([f_gm], device=device)
        
        b_norm = bvals / 1000.0
        
        s_csf = f_csf.unsqueeze(-1) * torch.exp(-b_norm.unsqueeze(0) * d_csf)
        
        if f_gm.dim() == 0:
            f_gm = f_gm.unsqueeze(0)
        
        d_gm_slow = params.get('d_gm_slow', getattr(self, 'd_gm_slow_prior', torch.tensor(0.1, device=device)))
        f_gm_restricted = params.get('f_gm_restricted', getattr(self, 'f_gm_restricted_prior', torch.tensor(0.4, device=device)))
        
        f_gm_fast = 1.0 - f_gm_restricted
        s_gm_fast = torch.exp(-b_norm.unsqueeze(0) * d_gm)
        s_gm_slow = torch.exp(-b_norm.unsqueeze(0) * d_gm_slow)
        s_gm = f_gm.unsqueeze(-1) * (f_gm_fast * s_gm_fast + f_gm_restricted * s_gm_slow)
        
        f_wm = params.get('f_wm')
        fiber_dirs = params.get('fiber_dirs')
        
        if f_wm is not None and fiber_dirs is not None:
            if f_wm.dim() >= 2:
                f_wm_total = f_wm.sum(dim=-1)
            else:
                f_wm_total = f_wm
            if f_wm_total.dim() == 0:
                f_wm_total = f_wm_total.unsqueeze(0)
            
            if fiber_dirs.dim() == 1:
                v = fiber_dirs.unsqueeze(0)
            elif fiber_dirs.dim() == 2:
                if fiber_dirs.shape[-1] == 3:
                    v = fiber_dirs
                else:
                    v = fiber_dirs[:, :3]
            elif fiber_dirs.dim() == 3:
                v = fiber_dirs[:, 0, :]
            else:
                v = fiber_dirs.reshape(fiber_dirs.shape[0], -1, 3)[:, 0, :]
            
            v = F.normalize(v, dim=-1, eps=1e-8)
            
            dot = torch.mm(v, bvecs.T)
            dot_sq = dot ** 2
            
            f_intra = params.get('f_intra', 0.7)
            if isinstance(f_intra, (int, float)):
                f_intra = torch.tensor(f_intra, device=device)
            if f_intra.dim() >= 1 and f_intra.shape[0] == n_voxels:
                f_intra = f_intra.unsqueeze(-1)
            
            D_app = d_perp + (d_parallel - d_perp) * dot_sq
            
            s_wm = f_wm_total.unsqueeze(-1) * (
                f_intra * torch.exp(-b_norm.unsqueeze(0) * D_app) +
                (1 - f_intra) * torch.exp(-b_norm.unsqueeze(0) * d_perp)
            )
        else:
            s_wm = torch.zeros(n_voxels, n_meas, device=device)
        
        signal = s_csf + s_gm + s_wm
        
        signal = signal.clamp(min=1e-6, max=1.0)
        
        return signal
    
    def _bochner_psd_penalty(self, K: torch.Tensor, penalty_type: str) -> torch.Tensor:
        device = K.device
        
        K = 0.5 * (K + K.transpose(-2, -1))
        
        m = K.shape[-1]
        K = K + 1e-6 * torch.eye(m, device=device).unsqueeze(0)
        
        try:
            eigenvalues = torch.linalg.eigvalsh(K)
        except RuntimeError:
            return torch.tensor(0.0, device=device)
        
        if penalty_type == 'min_eigenvalue':
            min_eig = eigenvalues.min(dim=-1)[0]
            penalty = F.relu(-min_eig).mean()
        
        elif penalty_type == 'relu':
            neg_eig = F.relu(-eigenvalues)
            penalty = neg_eig.sum(dim=-1).mean()
        
        elif penalty_type == 'logdet':
            log_eig = torch.log(eigenvalues.clamp(min=1e-8))
            penalty = -log_eig.sum(dim=-1).mean()
        
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")
        
        return penalty
    
    
    def get_bernstein_loss(self,
                           predicted_signal: torch.Tensor,
                           n_orders: int = 3,
                           mask: Optional[torch.Tensor] = None,
                           penalty_weight: float = 1.0) -> torch.Tensor:
        device = predicted_signal.device
        
        shell_means, shell_bvals = self._compute_spherical_means(predicted_signal)
        
        n_shells = len(shell_bvals)
        
        if n_shells < 2:
            return torch.tensor(0.0, device=device)
        
        sorted_idx = torch.argsort(shell_bvals)
        shell_means_sorted = shell_means[..., sorted_idx]
        
        batch_shape = shell_means_sorted.shape[:-1]
        shell_means_flat = shell_means_sorted.reshape(-1, n_shells)
        
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            shell_means_masked = shell_means_flat[mask_flat]
        else:
            shell_means_masked = shell_means_flat
        
        if shell_means_masked.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        
        total_penalty = torch.tensor(0.0, device=device)
        current_diff = shell_means_masked
        
        for k in range(1, min(n_orders + 1, n_shells)):
            next_diff = current_diff[..., 1:] - current_diff[..., :-1]
            
            
            if k % 2 == 1:
                violation = F.relu(next_diff)
            else:
                violation = F.relu(-next_diff)
            
            order_weight = 1.0 / k
            total_penalty = total_penalty + order_weight * violation.mean()
            
            current_diff = next_diff
            
            if next_diff.shape[-1] < 2:
                break
        
        return penalty_weight * total_penalty
    
    def _compute_spherical_means(self, 
                                  signal: torch.Tensor,
                                  shell_threshold: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
        device = signal.device
        
        n_shells = self.protocol.num_shells
        shell_means_list = []
        
        for s in range(n_shells):
            shell_mask = (self.protocol.shell_ids == s)
            shell_signal = signal[..., shell_mask]
            shell_mean = shell_signal.mean(dim=-1)
            shell_means_list.append(shell_mean)
        
        shell_means = torch.stack(shell_means_list, dim=-1)
        shell_bvals = self.protocol.shell_values
        
        return shell_means, shell_bvals
    
    def get_bernstein_diagnostic(self, 
                                  predicted_signal: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> dict:
        device = predicted_signal.device
        
        shell_means, shell_bvals = self._compute_spherical_means(predicted_signal)
        
        n_shells = len(shell_bvals)
        sorted_idx = torch.argsort(shell_bvals)
        shell_means_sorted = shell_means[..., sorted_idx]
        shell_bvals_sorted = shell_bvals[sorted_idx]
        
        shell_means_flat = shell_means_sorted.reshape(-1, n_shells)
        
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            shell_means_flat = shell_means_flat[mask_flat]
        
        first_diff = shell_means_flat[..., 1:] - shell_means_flat[..., :-1]
        
        if n_shells >= 3:
            second_diff = first_diff[..., 1:] - first_diff[..., :-1]
        else:
            second_diff = torch.tensor([], device=device)
        
        return {
            'shell_means': shell_means_sorted.mean(dim=tuple(range(shell_means_sorted.dim()-1))),
            'shell_bvals': shell_bvals_sorted,
            'first_diff_mean': first_diff.mean(dim=0) if first_diff.numel() > 0 else None,
            'second_diff_mean': second_diff.mean(dim=0) if second_diff.numel() > 0 else None,
            'n_monotonicity_violations': (first_diff > 0).sum().item(),
            'n_convexity_violations': (second_diff < 0).sum().item() if second_diff.numel() > 0 else 0,
            'total_voxels': shell_means_flat.shape[0],
        }
    
    
    def get_dki_realizability_loss(self,
                                    k_parallel: torch.Tensor,
                                    k_perp: torch.Tensor,
                                    k_cross: Optional[torch.Tensor] = None,
                                    constraint_type: str = 'strong',
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = k_parallel.device
        loss = torch.tensor(0.0, device=device)
        
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            k_par = k_parallel.reshape(-1)[mask_flat]
            k_prp = k_perp.reshape(-1)[mask_flat]
            if k_cross is not None:
                k_x = k_cross.reshape(-1)[mask_flat]
            else:
                k_x = None
        else:
            k_par = k_parallel.reshape(-1)
            k_prp = k_perp.reshape(-1)
            k_x = k_cross.reshape(-1) if k_cross is not None else None
        
        if k_par.numel() == 0:
            return loss
        
        loss = loss + F.relu(-k_par).mean()
        loss = loss + F.relu(-k_prp).mean()
        
        if k_x is not None:
            k_prod = torch.sqrt(F.relu(k_par) * F.relu(k_prp) + 1e-8)
            cs_lower = -k_prod / 3.0
            loss = loss + F.relu(cs_lower - k_x).mean()
            
            if constraint_type in ['strong', 'mixture']:
                strong_sum = k_par + k_prp + 2 * k_x
                loss = loss + F.relu(-strong_sum).mean()
            
            if constraint_type == 'mixture':
                loss = loss + F.relu(k_par - 3.0).mean() * 0.1
                loss = loss + F.relu(k_prp - 3.0).mean() * 0.1
                loss = loss + F.relu(k_x - 2.0).mean() * 0.1
        
        return loss
    
    def get_dki_diagnostic(self,
                           k_parallel: torch.Tensor,
                           k_perp: torch.Tensor,
                           k_cross: Optional[torch.Tensor] = None,
                           mask: Optional[torch.Tensor] = None) -> dict:
        device = k_parallel.device
        
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            k_par = k_parallel.reshape(-1)[mask_flat]
            k_prp = k_perp.reshape(-1)[mask_flat]
            if k_cross is not None:
                k_x = k_cross.reshape(-1)[mask_flat]
            else:
                k_x = None
            n_voxels = mask_flat.sum().item()
        else:
            k_par = k_parallel.reshape(-1)
            k_prp = k_perp.reshape(-1)
            k_x = k_cross.reshape(-1) if k_cross is not None else None
            n_voxels = k_par.numel()
        
        diag = {
            'k_parallel_mean': k_par.mean().item(),
            'k_parallel_std': k_par.std().item(),
            'k_perp_mean': k_prp.mean().item(),
            'k_perp_std': k_prp.std().item(),
            'k_parallel_negative_frac': (k_par < 0).float().mean().item(),
            'k_perp_negative_frac': (k_prp < 0).float().mean().item(),
            'n_voxels': n_voxels,
        }
        
        if k_x is not None:
            diag['k_cross_mean'] = k_x.mean().item()
            diag['k_cross_std'] = k_x.std().item()
            
            k_prod = torch.sqrt(F.relu(k_par) * F.relu(k_prp) + 1e-8)
            cs_lower = -k_prod / 3.0
            diag['cauchy_schwarz_violation_frac'] = (k_x < cs_lower).float().mean().item()
            
            strong_sum = k_par + k_prp + 2 * k_x
            diag['strong_violation_frac'] = (strong_sum < 0).float().mean().item()
        
        return diag
    
    @staticmethod
    def create_hcp_like_protocol(device: str = 'cuda') -> AcquisitionProtocolV4:
        shells = [0, 1000, 2000, 3000]
        dirs_per_shell = [18, 90, 90, 90]
        
        total = sum(dirs_per_shell)
        bvals = torch.zeros(total, device=device)
        bvecs = torch.zeros(total, 3, device=device)
        
        np.random.seed(42)
        idx = 0
        for b, n in zip(shells, dirs_per_shell):
            bvals[idx:idx+n] = b
            theta = np.random.uniform(0, np.pi, n)
            phi = np.random.uniform(0, 2*np.pi, n)
            bvecs[idx:idx+n, 0] = torch.tensor(np.sin(theta) * np.cos(phi), device=device)
            bvecs[idx:idx+n, 1] = torch.tensor(np.sin(theta) * np.sin(phi), device=device)
            bvecs[idx:idx+n, 2] = torch.tensor(np.cos(theta), device=device)
            idx += n
        
        bvecs = F.normalize(bvecs, dim=-1, eps=1e-8)
        return AcquisitionProtocolV4.from_tensors(bvals, bvecs)



def broadcast_sigma(sigma: Union[float, torch.Tensor],
                    target_shape: Tuple[int, ...],
                    device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    if isinstance(sigma, (int, float)):
        return torch.tensor(sigma, device=device, dtype=dtype)
    
    sigma = sigma.to(device=device, dtype=dtype)
    
    if sigma.dim() == 0:
        return sigma
    
    if sigma.dim() == 3 and len(target_shape) == 5:
        return sigma.unsqueeze(0).unsqueeze(-1)
    
    if sigma.dim() == 4 and len(target_shape) == 5:
        return sigma.unsqueeze(-1)
    
    if sigma.dim() == 2 and len(target_shape) == 4:
        return sigma.unsqueeze(0).unsqueeze(-1)
    
    if sigma.dim() == 5 or sigma.dim() == len(target_shape):
        return sigma
    
    return sigma


def rician_nll(y: torch.Tensor, 
               s: torch.Tensor, 
               sigma: Union[float, torch.Tensor], 
               eps: float = 1e-8) -> torch.Tensor:
    y = torch.clamp(y, min=eps)
    s = torch.clamp(s, min=eps)
    sigma2 = torch.clamp(sigma ** 2, min=eps)
    
    z = (y * s) / sigma2
    
    log_i0 = torch.log(torch.special.i0e(z) + eps) + torch.abs(z)
    
    nll = torch.log(sigma2) + (y ** 2 + s ** 2) / (2.0 * sigma2) - log_i0
    
    return nll


def rician_loss(pred: torch.Tensor,
                target: torch.Tensor,
                sigma: Union[float, torch.Tensor],
                mask: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
    sigma = broadcast_sigma(sigma, pred.shape, pred.device, pred.dtype)
    
    nll = rician_nll(target, pred, sigma)
    
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.unsqueeze(-1)
        nll = nll * mask
        if reduction == 'mean':
            return nll.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
        elif reduction == 'sum':
            return nll.sum()
        else:
            return nll
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll



def estimate_noise_sigma_from_background(signal: torch.Tensor,
                                         mask: Optional[torch.Tensor] = None,
                                         percentile: float = 5.0,
                                         method: str = "median") -> torch.Tensor:
    if mask is not None:
        if signal.dim() == 5:
            bg_mask = ~mask
            bg_signal = signal[bg_mask.unsqueeze(-1).expand_as(signal)]
        else:
            bg_signal = signal[~mask]
    else:
        threshold = torch.quantile(signal.flatten(), percentile / 100.0)
        bg_signal = signal[signal < threshold]
    
    if method == "median":
        stat = torch.median(bg_signal)
        sigma = stat / math.sqrt(2.0 * math.log(2.0))
    else:
        stat = bg_signal.mean()
        sigma = stat / math.sqrt(math.pi / 2.0)
    
    return sigma


def rician_bias_correction(y: torch.Tensor,
                           sigma: Union[float, torch.Tensor],
                           eps: float = 1e-8) -> torch.Tensor:
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, dtype=y.dtype, device=y.device)
    
    s_corrected = torch.sqrt(torch.relu(y ** 2 - sigma ** 2))
    
    return s_corrected


def hybrid_mse_rician_loss(pred: torch.Tensor,
                           target: torch.Tensor,
                           sigma: Union[float, torch.Tensor],
                           mse_weight: float = 0.7,
                           rician_weight: float = 0.3,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sigma = broadcast_sigma(sigma, pred.shape, pred.device, pred.dtype)
    
    if mask is not None:
        if mask.dim() == 4 and pred.dim() == 5:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        else:
            mask_exp = mask
    
    if mask is not None:
        mse = ((pred - target) ** 2)[mask_exp].mean()
    else:
        mse = F.mse_loss(pred, target)
    
    nll = rician_nll(target, pred, sigma)
    if mask is not None:
        rician = nll[mask_exp].mean()
    else:
        rician = nll.mean()
    
    return mse_weight * mse + rician_weight * rician


def b_dependent_hybrid_loss(pred: torch.Tensor,
                            target: torch.Tensor,
                            sigma: Union[float, torch.Tensor],
                            bvals: torch.Tensor,
                            b_switch: float = 2000.0,
                            b_slope: float = 0.002,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sigma = broadcast_sigma(sigma, pred.shape, pred.device, pred.dtype)
    
    if bvals.device != pred.device:
        bvals = bvals.to(pred.device)
    
    w_mse = torch.sigmoid((b_switch - bvals) * b_slope)
    w_rician = 1.0 - w_mse
    
    while w_mse.dim() < pred.dim():
        w_mse = w_mse.unsqueeze(0)
        w_rician = w_rician.unsqueeze(0)
    
    if mask is not None:
        if mask.dim() == 4 and pred.dim() == 5:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        elif mask.dim() == 3 and pred.dim() == 4:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        else:
            mask_exp = mask
    
    sq_error = (pred - target) ** 2
    
    nll = rician_nll(target, pred, sigma)
    
    loss_per_meas = w_mse * sq_error + w_rician * nll
    
    if mask is not None:
        loss = loss_per_meas[mask_exp].mean()
    else:
        loss = loss_per_meas.mean()
    
    return loss


def snr_dependent_hybrid_loss(pred: torch.Tensor,
                              target: torch.Tensor,
                              sigma: Union[float, torch.Tensor],
                              snr_switch: float = 3.0,
                              snr_slope: float = 1.0,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sigma_bc = broadcast_sigma(sigma, pred.shape, pred.device, pred.dtype)
    
    pred_detached = pred.detach()
    r = pred_detached / (sigma_bc + 1e-8)
    
    w_mse = torch.sigmoid((r - snr_switch) * snr_slope)
    w_rician = 1.0 - w_mse
    
    if mask is not None:
        if mask.dim() == 4 and pred.dim() == 5:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        elif mask.dim() == 3 and pred.dim() == 4:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        else:
            mask_exp = mask
    
    sq_error = (pred - target) ** 2
    
    nll = rician_nll(target, pred, sigma_bc)
    
    loss_per_meas = w_mse * sq_error + w_rician * nll
    
    if mask is not None:
        loss = loss_per_meas[mask_exp].mean()
    else:
        loss = loss_per_meas.mean()
    
    return loss


def bias_corrected_mse_loss(pred: torch.Tensor,
                            target: torch.Tensor,
                            sigma: Union[float, torch.Tensor],
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    target_corr = rician_bias_correction(target, sigma)
    
    if mask is not None:
        if mask.dim() == 4 and pred.dim() == 5:
            mask_exp = mask.unsqueeze(-1).expand_as(pred)
        else:
            mask_exp = mask
        mse = ((pred - target_corr) ** 2)[mask_exp].mean()
    else:
        mse = F.mse_loss(pred, target_corr)
    
    return mse

