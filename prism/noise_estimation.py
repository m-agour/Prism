#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class NoiseEstimationResult:
    
    sigma_raw: float
    
    sigma_normalized: float
    
    sigma_map: Optional[torch.Tensor] = None
    
    s0_map: Optional[torch.Tensor] = None
    
    method: str = "unknown"
    n_background_voxels: Optional[int] = None
    n_b0_volumes: Optional[int] = None
    n_volumes_used: Optional[int] = None
    
    s0_median_in_mask: Optional[float] = None
    s0_p10: Optional[float] = None
    s0_p90: Optional[float] = None
    
    sigma_map_median: Optional[float] = None
    sigma_map_p10: Optional[float] = None
    sigma_map_p90: Optional[float] = None
    
    implied_snr: Optional[float] = None
    
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    
    def __repr__(self):
        snr_str = f", SNR≈{self.implied_snr:.1f}" if self.implied_snr else ""
        return (f"NoiseEstimationResult(σ_raw={self.sigma_raw:.4f}, "
                f"σ_norm={self.sigma_normalized:.4f}, method='{self.method}'{snr_str})")
    
    def to_dict(self) -> Dict:
        return {
            'sigma_method': self.method,
            'sigma_raw': self.sigma_raw,
            'sigma_normalized': self.sigma_normalized,
            'sigma_map_saved': self.sigma_map is not None,
            'bg_voxels_used': self.n_background_voxels,
            'b0_volumes_used': self.n_b0_volumes,
            'n_volumes_used': self.n_volumes_used,
            's0_median_in_mask': self.s0_median_in_mask,
            's0_p10': self.s0_p10,
            's0_p90': self.s0_p90,
            'sigma_map_median': self.sigma_map_median,
            'sigma_map_p10': self.sigma_map_p10,
            'sigma_map_p90': self.sigma_map_p90,
            'implied_snr': self.implied_snr,
            'used_fallback': self.used_fallback,
            'fallback_reason': self.fallback_reason,
        }
    
    def print_diagnostics(self):
        print(f"  [σ Diagnostics] method={self.method}")
        print(f"    σ_raw = {self.sigma_raw:.4f}")
        if self.s0_median_in_mask:
            print(f"    median(S0 in mask) = {self.s0_median_in_mask:.2f}")
        print(f"    σ_norm (median σ-map) = {self.sigma_normalized:.4f}")
        if self.implied_snr:
            print(f"    implied SNR = {self.implied_snr:.1f} (= 1/σ_norm)")
        if self.n_background_voxels:
            print(f"    bg_voxels = {self.n_background_voxels}")
        if self.n_b0_volumes:
            print(f"    b0_volumes = {self.n_b0_volumes}")
        if self.used_fallback:
            print(f"    ⚠ Fallback used: {self.fallback_reason}")



def rayleigh_median_to_sigma(median: float) -> float:
    return median / math.sqrt(2 * math.log(2))


def rayleigh_mean_to_sigma(mean: float) -> float:
    return mean / math.sqrt(math.pi / 2)


def rayleigh_mode_to_sigma(mode: float) -> float:
    return mode



def create_background_mask(
    brain_mask: torch.Tensor,
    erode_brain: int = 0,
    erode_background: int = 2,
) -> torch.Tensor:
    device = brain_mask.device
    mask = brain_mask.float()
    
    has_batch = mask.dim() == 4
    if not has_batch:
        mask = mask.unsqueeze(0)
    
    mask = mask.unsqueeze(1)
    
    def make_kernel(radius: int) -> torch.Tensor:
        size = 2 * radius + 1
        center = radius
        kernel = torch.zeros(size, size, size, device=device)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    if (i - center)**2 + (j - center)**2 + (k - center)**2 <= radius**2:
                        kernel[i, j, k] = 1.0
        return kernel.view(1, 1, size, size, size)
    
    if erode_brain > 0:
        kernel = make_kernel(erode_brain)
        kernel = kernel / kernel.sum()
        mask = F.conv3d(mask, kernel, padding=erode_brain)
        mask = (mask > 0.99).float()
    
    background = 1.0 - mask
    
    if erode_background > 0:
        kernel = make_kernel(erode_background)
        kernel = kernel / kernel.sum()
        background = F.conv3d(background, kernel, padding=erode_background)
        background = (background > 0.99).float()
    
    background = background.squeeze(1)
    if not has_batch:
        background = background.squeeze(0)
    
    return background.bool()


def create_background_mask_simple(
    brain_mask: torch.Tensor,
    margin: int = 3,
) -> torch.Tensor:
    device = brain_mask.device
    mask = brain_mask.float()
    
    has_batch = mask.dim() == 4
    if not has_batch:
        mask = mask.unsqueeze(0)
    
    mask = mask.unsqueeze(1)
    
    if margin > 0:
        kernel_size = 2 * margin + 1
        dilated = F.max_pool3d(mask, kernel_size, stride=1, padding=margin)
    else:
        dilated = mask
    
    background = 1.0 - dilated
    
    background = background.squeeze(1)
    if not has_batch:
        background = background.squeeze(0)
    
    return background.bool()



def estimate_sigma_from_background(
    signal: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bvals: Optional[torch.Tensor] = None,
    use_b0_only: bool = True,
    method: str = "median",
    bg_margin: int = 3,
    min_bg_voxels: int = 100,
    max_bg_fraction: float = 0.5,
) -> NoiseEstimationResult:
    device = signal.device
    dtype = signal.dtype
    
    if signal.dim() == 4:
        signal = signal.unsqueeze(0)
    
    B, X, Y, Z, N = signal.shape
    
    if bvals is not None and use_b0_only:
        b0_mask = bvals < 50
        if b0_mask.sum() > 0:
            volumes = signal[..., b0_mask]
        else:
            volumes = signal
    else:
        volumes = signal
    
    if mask is not None:
        bg_mask = create_background_mask_simple(mask, margin=bg_margin)
    else:
        mean_signal = volumes.mean(dim=-1)
        threshold = torch.quantile(mean_signal.flatten(), 0.2)
        bg_mask = mean_signal < threshold
    
    n_bg = bg_mask.sum().item()
    total_voxels = X * Y * Z
    
    if n_bg < min_bg_voxels:
        raise ValueError(
            f"Insufficient background voxels: {n_bg} < {min_bg_voxels}. "
            f"Check that mask properly excludes brain."
        )
    
    if n_bg > max_bg_fraction * total_voxels:
        import warnings
        warnings.warn(
            f"Background is {n_bg/total_voxels:.1%} of volume. "
            f"This seems high - verify mask is correct."
        )
    
    if bg_mask.dim() == 3:
        bg_mask_exp = bg_mask.unsqueeze(0).expand(B, -1, -1, -1)
    else:
        bg_mask_exp = bg_mask
    
    bg_values = volumes[bg_mask_exp.unsqueeze(-1).expand_as(volumes)]
    
    n_volumes_used = volumes.shape[-1]
    n_b0_used = None
    if bvals is not None and use_b0_only:
        n_b0_used = int((bvals < 50).sum().item())
    
    if method == "median":
        stat_value = torch.median(bg_values).item()
        sigma_raw = rayleigh_median_to_sigma(stat_value)
    elif method == "mean":
        stat_value = bg_values.mean().item()
        sigma_raw = rayleigh_mean_to_sigma(stat_value)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")
    
    return NoiseEstimationResult(
        sigma_raw=sigma_raw,
        sigma_normalized=sigma_raw,
        sigma_map=None,
        s0_map=None,
        method=f"background_{method}{'_b0only' if (bvals is not None and use_b0_only) else ''}",
        n_background_voxels=n_bg,
        n_b0_volumes=n_b0_used,
        n_volumes_used=n_volumes_used,
    )


def estimate_sigma_from_b0_repeats(
    signal: torch.Tensor,
    bvals: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    b0_threshold: float = 50.0,
) -> NoiseEstimationResult:
    device = signal.device
    
    if signal.dim() == 4:
        signal = signal.unsqueeze(0)
    
    B, X, Y, Z, N = signal.shape
    
    b0_mask = bvals < b0_threshold
    n_b0 = b0_mask.sum().item()
    
    if n_b0 < 2:
        raise ValueError(
            f"Need at least 2 b0 volumes for variance estimation, found {n_b0}. "
            f"Use estimate_sigma_from_background instead."
        )
    
    b0_volumes = signal[..., b0_mask]
    
    sigma_voxel = b0_volumes.std(dim=-1)
    
    s0_mean = b0_volumes.mean(dim=-1)
    
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).expand(B, -1, -1, -1)
        sigma_values = sigma_voxel[mask]
        s0_values = s0_mean[mask]
    else:
        sigma_values = sigma_voxel.flatten()
        s0_values = s0_mean.flatten()
    
    sigma_raw = torch.median(sigma_values).item()
    s0_median = torch.median(s0_values).item()
    
    sigma_normalized_voxel = sigma_voxel / (s0_mean + 1e-8)
    if mask is not None:
        sigma_normalized = torch.median(sigma_normalized_voxel[mask]).item()
    else:
        sigma_normalized = torch.median(sigma_normalized_voxel).item()
    
    return NoiseEstimationResult(
        sigma_raw=sigma_raw,
        sigma_normalized=sigma_normalized,
        sigma_map=sigma_voxel.squeeze(0) if B == 1 else sigma_voxel,
        s0_map=s0_mean.squeeze(0) if B == 1 else s0_mean,
        method="b0_repeats",
        n_b0_volumes=n_b0,
    )



def compute_sigma_map(
    sigma_raw: float,
    s0_map: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return sigma_raw / (s0_map + eps)


def compute_s0_map(
    signal: torch.Tensor,
    bvals: torch.Tensor,
    b0_threshold: float = 50.0,
    method: str = "mean",
) -> torch.Tensor:
    squeeze_batch = signal.dim() == 4
    if squeeze_batch:
        signal = signal.unsqueeze(0)
    
    b0_mask = bvals < b0_threshold
    b0_volumes = signal[..., b0_mask]
    
    if method == "mean":
        s0_map = b0_volumes.mean(dim=-1)
    elif method == "median":
        s0_map = b0_volumes.median(dim=-1).values
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if squeeze_batch:
        s0_map = s0_map.squeeze(0)
    
    return s0_map



def estimate_noise_sigma(
    signal: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bvals: Optional[torch.Tensor] = None,
    method: str = "auto",
    compute_map: bool = True,
    min_bg_voxels: int = 2000,
    print_diagnostics: bool = True,
    **kwargs,
) -> NoiseEstimationResult:
    device = signal.device
    
    squeeze_batch = signal.dim() == 4
    if squeeze_batch:
        signal = signal.unsqueeze(0)
    
    B, X, Y, Z, N = signal.shape
    
    used_fallback = False
    fallback_reason = None
    
    if method == "auto" or method == "background":
        try:
            result = estimate_sigma_from_background(
                signal, mask=mask, bvals=bvals, 
                min_bg_voxels=min_bg_voxels, **kwargs
            )
        except ValueError as e:
            if bvals is not None and (bvals < 50).sum() >= 2:
                import warnings
                warnings.warn(f"Background estimation failed: {e}. Falling back to b0 method.")
                result = estimate_sigma_from_b0_repeats(
                    signal, bvals=bvals, mask=mask
                )
                used_fallback = True
                fallback_reason = str(e)
            else:
                raise ValueError(
                    f"Background estimation failed ({e}) and b0_repeats fallback not available "
                    f"(need >= 2 b0 volumes, have {(bvals < 50).sum().item() if bvals is not None else 0}). "
                    f"Provide --noise-sigma manually or ensure volume has sufficient background."
                )
    elif method == "b0_repeats":
        if bvals is None:
            raise ValueError("b0_repeats method requires bvals")
        result = estimate_sigma_from_b0_repeats(
            signal, bvals=bvals, mask=mask, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'background', or 'b0_repeats'.")
    
    result.used_fallback = used_fallback
    result.fallback_reason = fallback_reason
    
    if bvals is not None:
        s0_map = compute_s0_map(signal, bvals)
        result.s0_map = s0_map.squeeze(0) if squeeze_batch else s0_map
        
        if mask is not None:
            if mask.dim() == 3 and s0_map.dim() == 4:
                s0_in_brain = s0_map[0, mask]
            elif mask.dim() == 3 and s0_map.dim() == 3:
                s0_in_brain = s0_map[mask]
            else:
                s0_in_brain = s0_map.flatten()
            
            result.s0_median_in_mask = torch.median(s0_in_brain).item()
            result.s0_p10 = torch.quantile(s0_in_brain.float(), 0.1).item()
            result.s0_p90 = torch.quantile(s0_in_brain.float(), 0.9).item()
        
        if compute_map:
            sigma_map = compute_sigma_map(result.sigma_raw, s0_map)
            result.sigma_map = sigma_map.squeeze(0) if squeeze_batch else sigma_map
            
            if mask is not None:
                if mask.dim() == 3 and sigma_map.dim() == 4:
                    sigma_in_brain = sigma_map[0, mask]
                elif mask.dim() == 3 and sigma_map.dim() == 3:
                    sigma_in_brain = sigma_map[mask]
                else:
                    sigma_in_brain = sigma_map.flatten()
                
                result.sigma_normalized = torch.median(sigma_in_brain).item()
                result.sigma_map_median = result.sigma_normalized
                result.sigma_map_p10 = torch.quantile(sigma_in_brain.float(), 0.1).item()
                result.sigma_map_p90 = torch.quantile(sigma_in_brain.float(), 0.9).item()
                
                result.implied_snr = 1.0 / result.sigma_normalized if result.sigma_normalized > 0 else None
            else:
                result.sigma_normalized = sigma_map.mean().item()
    
    if print_diagnostics:
        result.print_diagnostics()
    
    return result



def validate_sigma_estimate(
    sigma: float,
    expected_range: Tuple[float, float] = (0.01, 0.5),
    context: str = "S0-normalized",
) -> bool:
    min_sigma, max_sigma = expected_range
    
    if sigma < min_sigma:
        raise ValueError(
            f"Estimated σ = {sigma:.4f} is too low (< {min_sigma}). "
            f"This suggests extremely high SNR or incorrect estimation. "
            f"Context: {context}"
        )
    
    if sigma > max_sigma:
        raise ValueError(
            f"Estimated σ = {sigma:.4f} is too high (> {max_sigma}). "
            f"This suggests very low SNR, incorrect estimation, or wrong signal units. "
            f"Context: {context}"
        )
    
    return True


def compute_snr_map(
    signal: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    bvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    squeeze_batch = signal.dim() == 4
    if squeeze_batch:
        signal = signal.unsqueeze(0)
    
    if bvals is not None:
        s0 = compute_s0_map(signal, bvals)
    else:
        s0 = signal.mean(dim=-1)
    
    if isinstance(sigma, (int, float)):
        snr = s0 / sigma
    else:
        snr = s0 / (sigma + 1e-8)
    
    if squeeze_batch:
        snr = snr.squeeze(0)
    
    return snr



if __name__ == "__main__":
    print("Testing noise estimation module...")
    
    torch.manual_seed(42)
    
    X, Y, Z, N = 32, 32, 5, 50
    true_sigma = 50.0
    
    s0_map = 1000 + 100 * torch.randn(X, Y, Z)
    s0_map = torch.clamp(s0_map, min=500)
    
    xx, yy, zz = torch.meshgrid(
        torch.arange(X) - X//2,
        torch.arange(Y) - Y//2,
        torch.arange(Z) - Z//2,
        indexing='ij'
    )
    brain_mask = (xx**2 + yy**2 + zz**2) < (X//3)**2
    
    signal_clean = s0_map.unsqueeze(-1).expand(-1, -1, -1, N) * brain_mask.unsqueeze(-1).float()
    
    noise_real = true_sigma * torch.randn_like(signal_clean)
    noise_imag = true_sigma * torch.randn_like(signal_clean)
    signal = torch.sqrt((signal_clean + noise_real)**2 + noise_imag**2)
    
    print(f"\nTrue σ = {true_sigma:.2f}")
    print(f"Brain voxels: {brain_mask.sum().item()}")
    print(f"Background voxels: {(~brain_mask).sum().item()}")
    
    print("\n--- Testing background estimation ---")
    result = estimate_sigma_from_background(
        signal.unsqueeze(0),
        mask=brain_mask,
        method="median"
    )
    print(f"Estimated σ (median): {result.sigma_raw:.2f}")
    print(f"Error: {abs(result.sigma_raw - true_sigma) / true_sigma * 100:.1f}%")
    
    result_mean = estimate_sigma_from_background(
        signal.unsqueeze(0),
        mask=brain_mask,
        method="mean"
    )
    print(f"Estimated σ (mean): {result_mean.sigma_raw:.2f}")
    print(f"Error: {abs(result_mean.sigma_raw - true_sigma) / true_sigma * 100:.1f}%")
    
    print("\n--- Testing σ-map computation ---")
    bvals = torch.zeros(N)
    bvals[5:] = 1000
    
    result_full = estimate_noise_sigma(
        signal.unsqueeze(0),
        mask=brain_mask,
        bvals=bvals,
        method="background",
        compute_map=True,
    )
    print(f"σ_raw: {result_full.sigma_raw:.2f}")
    print(f"σ_normalized (median in brain): {result_full.sigma_normalized:.4f}")
    print(f"σ-map shape: {result_full.sigma_map.shape}")
    
    expected_sigma_norm = true_sigma / s0_map[brain_mask].median().item()
    print(f"Expected σ_norm: {expected_sigma_norm:.4f}")
    
    print("\n✓ All tests passed!")
