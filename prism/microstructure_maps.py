#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class MicrostructureMaps:
    
    f_wm: np.ndarray
    f_gm: np.ndarray
    f_csf: np.ndarray
    f_intra: np.ndarray
    kappa: np.ndarray
    fiber_dirs: np.ndarray
    f_restricted: Optional[np.ndarray] = None
    
    signal_pred: Optional[np.ndarray] = None
    signal_obs: Optional[np.ndarray] = None
    
    wm_mask: Optional[np.ndarray] = None
    brain_mask: Optional[np.ndarray] = None
    
    r2_dti: Optional[np.ndarray] = None
    fa_dti: Optional[np.ndarray] = None
    evecs_dti: Optional[np.ndarray] = None
    
    _cache: Dict = field(default_factory=dict)
    
    @classmethod
    def from_fitted_params(
        cls,
        params: Dict[str, torch.Tensor],
        signal_pred: Optional[torch.Tensor] = None,
        signal_obs: Optional[torch.Tensor] = None,
        wm_mask: Optional[np.ndarray] = None,
        brain_mask: Optional[np.ndarray] = None,
        r2_dti: Optional[np.ndarray] = None,
        fa_dti: Optional[np.ndarray] = None,
        evecs_dti: Optional[np.ndarray] = None,
    ) -> 'MicrostructureMaps':
        
        def to_numpy(t, squeeze_batch=True):
            if t is None:
                return None
            arr = t.detach().cpu().numpy()
            if squeeze_batch and arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
        
        return cls(
            f_wm=to_numpy(params.get('f_wm')),
            f_gm=to_numpy(params.get('f_gm')),
            f_csf=to_numpy(params.get('f_csf')),
            f_intra=to_numpy(params.get('f_intra')),
            kappa=to_numpy(params.get('kappa')),
            fiber_dirs=to_numpy(params.get('fiber_dirs')),
            f_restricted=to_numpy(params.get('f_restricted')),
            signal_pred=to_numpy(signal_pred),
            signal_obs=to_numpy(signal_obs),
            wm_mask=wm_mask,
            brain_mask=brain_mask,
            r2_dti=r2_dti,
            fa_dti=fa_dti,
            evecs_dti=evecs_dti,
        )
    
    def set_dti_comparison(
        self,
        r2_dti: Optional[np.ndarray] = None,
        fa_dti: Optional[np.ndarray] = None,
        evecs_dti: Optional[np.ndarray] = None,
    ) -> None:
        if r2_dti is not None:
            self.r2_dti = r2_dti
        if fa_dti is not None:
            self.fa_dti = fa_dti
        if evecs_dti is not None:
            self.evecs_dti = evecs_dti
        
        for key in ['complexity_need', 'complexity_gain', 'fa_disagree', 'dir_disagree']:
            self._cache.pop(key, None)
    
    def compute_dti_from_signal(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
    ) -> None:
        if self.signal_obs is None:
            raise ValueError("signal_obs required for DTI computation")
        
        try:
            from dipy.reconst.dti import TensorModel
            import dipy.core.gradients as dpg
        except ImportError:
            raise ImportError("dipy required for DTI computation: pip install dipy")
        
        gtab = dpg.gradient_table(bvals, bvecs=bvecs)
        dti_model = TensorModel(gtab)
        
        signal = self.signal_obs
        original_shape = signal.shape[:-1]
        
        dti_fit = dti_model.fit(signal)
        
        self.fa_dti = dti_fit.fa
        self.evecs_dti = dti_fit.evecs[..., 0]
        
        dti_pred = dti_fit.predict(gtab)
        ss_res = np.sum((signal - dti_pred) ** 2, axis=-1)
        ss_tot = np.sum((signal - signal.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
        self.r2_dti = 1 - ss_res / (ss_tot + 1e-8)
        
        print(f"DTI computed: R²={np.mean(self.r2_dti):.3f}, FA={np.mean(self.fa_dti):.3f}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.f_gm.shape
    
    @property
    def n_fibers(self) -> int:
        return self.f_wm.shape[-1]
    
    @property
    def f_wm_total(self) -> np.ndarray:
        if 'f_wm_total' not in self._cache:
            self._cache['f_wm_total'] = self.f_wm.sum(axis=-1)
        return self._cache['f_wm_total']
    
    
    def get_ndi(self) -> np.ndarray:
        if 'ndi' not in self._cache:
            self._cache['ndi'] = self.f_intra
        return self._cache['ndi']
    
    def get_absolute_ndi(self) -> np.ndarray:
        if 'absolute_ndi' not in self._cache:
            self._cache['absolute_ndi'] = self.f_wm_total * self.f_intra
        return self._cache['absolute_ndi']
    
    def get_odi(self) -> np.ndarray:
        if 'odi' not in self._cache:
            kappa_safe = np.maximum(self.kappa, 0.1)
            self._cache['odi'] = (2 / np.pi) * np.arctan(1.0 / kappa_safe)
        return self._cache['odi']
    
    def get_odi_wm_weighted(self) -> np.ndarray:
        if 'odi_wm_weighted' not in self._cache:
            self._cache['odi_wm_weighted'] = self.get_odi() * self.f_wm_total
        return self._cache['odi_wm_weighted']
    
    def get_free_water_fraction(self) -> np.ndarray:
        return self.f_csf
    
    def get_wm_free_water_contamination(self) -> np.ndarray:
        if 'fw_contam' not in self._cache:
            denom = self.f_wm_total + self.f_gm + 1e-6
            contam = self.f_csf / denom
            if self.wm_mask is not None:
                contam = np.where(self.wm_mask, contam, 0)
            self._cache['fw_contam'] = contam
        return self._cache['fw_contam']
    
    def get_restricted_fraction(self) -> np.ndarray:
        if self.f_restricted is not None:
            return self.f_restricted
        return self.get_ndi()
    
    def get_micro_fa(self) -> np.ndarray:
        if 'micro_fa' not in self._cache:
            fa_stick = 1.0
            fa_zeppelin = 0.5
            
            mu_fa_wm = self.f_intra * fa_stick + (1 - self.f_intra) * fa_zeppelin
            
            self._cache['micro_fa'] = self.f_wm_total * mu_fa_wm
        return self._cache['micro_fa']
    
    def get_intra_axonal_fraction(self) -> np.ndarray:
        return self.f_intra * self.f_wm_total
    
    def get_extra_axonal_fraction(self) -> np.ndarray:
        return (1 - self.f_intra) * self.f_wm_total
    
    def get_restricted_density_index(self) -> np.ndarray:
        if self.f_restricted is not None:
            return self.f_restricted * self.f_wm_total
        return self.get_intra_axonal_fraction()
    
    
    def get_fiber_count(self, threshold: float = 0.1) -> np.ndarray:
        cache_key = f'fiber_count_{threshold}'
        if cache_key not in self._cache:
            self._cache[cache_key] = (self.f_wm > threshold).sum(axis=-1).astype(np.int32)
        return self._cache[cache_key]
    
    def get_is_crossing(self, threshold: float = 0.1) -> np.ndarray:
        return self.get_fiber_count(threshold) >= 2
    
    def get_complexity_index(self) -> np.ndarray:
        if 'complexity' not in self._cache:
            f_sum = self.f_wm.sum(axis=-1, keepdims=True)
            f_sum = np.maximum(f_sum, 1e-8)
            p = self.f_wm / f_sum
            
            p_safe = np.maximum(p, 1e-10)
            entropy = -np.sum(p * np.log(p_safe), axis=-1)
            
            max_entropy = np.log(self.n_fibers)
            self._cache['complexity'] = entropy / max_entropy
        return self._cache['complexity']
    
    def get_crossing_angles(self, threshold: float = 0.05) -> np.ndarray:
        if 'crossing_angle' not in self._cache:
            shape = self.shape
            angles = np.full(shape, np.nan)
            
            for idx in np.ndindex(shape):
                f_wm_voxel = self.f_wm[idx]
                dirs_voxel = self.fiber_dirs[idx]
                
                valid = f_wm_voxel > threshold
                if valid.sum() < 2:
                    continue
                
                order = np.argsort(-f_wm_voxel)
                k1, k2 = order[0], order[1]
                
                if f_wm_voxel[k1] < threshold or f_wm_voxel[k2] < threshold:
                    continue
                
                d1 = dirs_voxel[k1]
                d2 = dirs_voxel[k2]
                
                d1 = d1 / (np.linalg.norm(d1) + 1e-8)
                d2 = d2 / (np.linalg.norm(d2) + 1e-8)
                
                dot = np.abs(np.dot(d1, d2))
                dot = np.clip(dot, 0, 1)
                angles[idx] = np.arccos(dot) * 180 / np.pi
            
            self._cache['crossing_angle'] = angles
        return self._cache['crossing_angle']
    
    def get_curvature(self) -> np.ndarray:
        if 'curvature' not in self._cache:
            shape = self.shape
            curvature = np.zeros(shape)
            
            dom_idx = self.f_wm.argmax(axis=-1)
            dom_dir = np.zeros((*shape, 3))
            for idx in np.ndindex(shape):
                k = dom_idx[idx]
                dom_dir[idx] = self.fiber_dirs[idx][k]
            
            dom_dir = dom_dir / (np.linalg.norm(dom_dir, axis=-1, keepdims=True) + 1e-8)
            
            for axis in range(len(shape)):
                if shape[axis] < 3:
                    continue
                shifted_fwd = np.roll(dom_dir, -1, axis=axis)
                shifted_bwd = np.roll(dom_dir, 1, axis=axis)
                
                diff_fwd = 1 - np.abs((dom_dir * shifted_fwd).sum(axis=-1))
                diff_bwd = 1 - np.abs((dom_dir * shifted_bwd).sum(axis=-1))
                
                curvature += (diff_fwd + diff_bwd) / 2
            
            curvature *= self.f_wm_total
            
            curvature /= len(shape)
            
            self._cache['curvature'] = curvature
        return self._cache['curvature']
    
    
    def get_orientation_coherence(self, neighborhood: str = '6') -> np.ndarray:
        cache_key = f'coherence_{neighborhood}'
        if cache_key not in self._cache:
            shape = self.shape
            coherence = np.zeros(shape)
            
            dom_idx = self.f_wm.argmax(axis=-1)
            dom_dir = np.zeros((*shape, 3))
            for idx in np.ndindex(shape):
                k = dom_idx[idx]
                dom_dir[idx] = self.fiber_dirs[idx][k]
            
            dom_dir = dom_dir / (np.linalg.norm(dom_dir, axis=-1, keepdims=True) + 1e-8)
            
            if neighborhood == '6':
                offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
            else:
                offsets = [(dx, dy, dz) 
                           for dx in [-1, 0, 1] 
                           for dy in [-1, 0, 1] 
                           for dz in [-1, 0, 1] 
                           if not (dx == 0 and dy == 0 and dz == 0)]
            
            for idx in np.ndindex(shape):
                if self.f_wm_total[idx] < 0.2:
                    continue
                    
                my_dir = dom_dir[idx]
                alignments = []
                
                for offset in offsets:
                    ni = tuple(idx[i] + offset[i] for i in range(len(shape)))
                    
                    if any(ni[i] < 0 or ni[i] >= shape[i] for i in range(len(shape))):
                        continue
                    
                    if self.f_wm_total[ni] < 0.2:
                        continue
                    
                    neighbor_dir = dom_dir[ni]
                    alignment = np.abs(np.dot(my_dir, neighbor_dir))
                    alignments.append(alignment)
                
                if alignments:
                    coherence[idx] = np.mean(alignments)
            
            self._cache[cache_key] = coherence
        return self._cache[cache_key]
    
    def get_direction_certainty(self) -> np.ndarray:
        if 'certainty' not in self._cache:
            shape = self.shape
            certainty = np.zeros(shape)
            
            for idx in np.ndindex(shape):
                weights = self.f_wm[idx]
                sorted_w = np.sort(weights)[::-1]
                w1, w2 = sorted_w[0], sorted_w[1] if len(sorted_w) > 1 else 0
                
                if w1 + w2 > 1e-6:
                    certainty[idx] = (w1 - w2) / (w1 + w2)
                else:
                    certainty[idx] = 0
            
            self._cache['certainty'] = certainty
        return self._cache['certainty']
    
    def get_direction_ambiguity(self) -> np.ndarray:
        return 1 - self.get_direction_certainty()
    
    
    def get_partial_volume_severity(self) -> np.ndarray:
        if 'pv_severity' not in self._cache:
            max_frac = np.maximum.reduce([
                self.f_csf, 
                self.f_gm, 
                self.f_wm_total
            ])
            self._cache['pv_severity'] = 1 - max_frac
        return self._cache['pv_severity']
    
    def get_wm_gm_pv_index(self) -> np.ndarray:
        return np.minimum(self.f_wm_total, self.f_gm)
    
    def get_wm_csf_pv_index(self) -> np.ndarray:
        return np.minimum(self.f_wm_total, self.f_csf)
    
    def get_deep_wm_mask(
        self, 
        wm_threshold: float = 0.6,
        gm_threshold: float = 0.1,
        csf_threshold: float = 0.1,
    ) -> np.ndarray:
        mask = (
            (self.f_wm_total > wm_threshold) &
            (self.f_gm < gm_threshold) &
            (self.f_csf < csf_threshold)
        )
        return mask
    
    def get_interface_mask(
        self,
        pv_threshold: float = 0.15,
    ) -> np.ndarray:
        wm_gm = self.get_wm_gm_pv_index()
        wm_csf = self.get_wm_csf_pv_index()
        return (wm_gm > pv_threshold) | (wm_csf > pv_threshold)
    
    
    def get_orphan_map(self, alignment_threshold: float = 0.7) -> np.ndarray:
        if 'orphan' not in self._cache:
            shape = self.shape
            orphan = np.zeros(shape, dtype=bool)
            
            dom_idx = self.f_wm.argmax(axis=-1)
            dom_dir = np.zeros((*shape, 3))
            dom_frac = np.zeros(shape)
            
            for idx in np.ndindex(shape):
                k = dom_idx[idx]
                dom_dir[idx] = self.fiber_dirs[idx][k]
                dom_frac[idx] = self.f_wm[idx][k]
            
            dom_dir = dom_dir / (np.linalg.norm(dom_dir, axis=-1, keepdims=True) + 1e-8)
            
            wm_threshold = 0.2
            
            for idx in np.ndindex(shape):
                if dom_frac[idx] < wm_threshold:
                    continue
                
                my_dir = dom_dir[idx]
                has_aligned_neighbor = False
                
                for axis in range(len(shape)):
                    for delta in [-1, 1]:
                        neighbor_idx = list(idx)
                        neighbor_idx[axis] += delta
                        
                        if neighbor_idx[axis] < 0 or neighbor_idx[axis] >= shape[axis]:
                            continue
                        
                        neighbor_idx = tuple(neighbor_idx)
                        
                        if dom_frac[neighbor_idx] < wm_threshold:
                            continue
                        
                        neighbor_dir = dom_dir[neighbor_idx]
                        alignment = np.abs(np.dot(my_dir, neighbor_dir))
                        
                        if alignment > alignment_threshold:
                            has_aligned_neighbor = True
                            break
                    
                    if has_aligned_neighbor:
                        break
                
                if not has_aligned_neighbor:
                    orphan[idx] = True
            
            self._cache['orphan'] = orphan
        return self._cache['orphan']
    
    def get_orphan_density(self) -> float:
        orphan = self.get_orphan_map()
        wm = self.f_wm_total > 0.2
        if wm.sum() == 0:
            return 0.0
        return orphan.sum() / wm.sum()
    
    def get_endpoint_map(self, gm_threshold: float = 0.3) -> np.ndarray:
        if 'endpoint' not in self._cache:
            shape = self.shape
            endpoint = np.zeros(shape, dtype=bool)
            
            dom_idx = self.f_wm.argmax(axis=-1)
            dom_frac = np.zeros(shape)
            
            for idx in np.ndindex(shape):
                k = dom_idx[idx]
                dom_frac[idx] = self.f_wm[idx][k]
            
            wm_threshold = 0.2
            
            for idx in np.ndindex(shape):
                if dom_frac[idx] < wm_threshold:
                    continue
                
                adjacent_to_gm = False
                for axis in range(len(shape)):
                    for delta in [-1, 1]:
                        neighbor_idx = list(idx)
                        neighbor_idx[axis] += delta
                        
                        if neighbor_idx[axis] < 0 or neighbor_idx[axis] >= shape[axis]:
                            continue
                        
                        neighbor_idx = tuple(neighbor_idx)
                        if self.f_gm[neighbor_idx] > gm_threshold:
                            adjacent_to_gm = True
                            break
                    
                    if adjacent_to_gm:
                        break
                
                if adjacent_to_gm:
                    endpoint[idx] = True
            
            self._cache['endpoint'] = endpoint
        return self._cache['endpoint']
    
    def get_endpoint_gm_score(self) -> np.ndarray:
        endpoint = self.get_endpoint_map()
        return endpoint.astype(float) * self.f_gm
    
    def get_topology_score(self) -> np.ndarray:
        if 'topo_score' not in self._cache:
            orphan = self.get_orphan_map().astype(float)
            endpoint = self.get_endpoint_map().astype(float)
            
            orphan_penalty = orphan
            
            endpoint_gm_score = endpoint * self.f_gm
            endpoint_wm_penalty = endpoint * (1 - self.f_gm) * 0.5
            
            penalty = orphan_penalty + endpoint_wm_penalty - endpoint_gm_score
            penalty = np.clip(penalty, 0, 1)
            
            wm_mask = self.f_wm_total > 0.2
            score = 1 - penalty
            score = np.where(wm_mask, score, np.nan)
            
            self._cache['topo_score'] = score
        return self._cache['topo_score']
    
    
    def get_residual_norm(self) -> Optional[np.ndarray]:
        if self.signal_pred is None or self.signal_obs is None:
            return None
        
        if 'residual_norm' not in self._cache:
            diff = self.signal_pred - self.signal_obs
            self._cache['residual_norm'] = np.linalg.norm(diff, axis=-1)
        return self._cache['residual_norm']
    
    def get_relative_residual(self) -> Optional[np.ndarray]:
        if self.signal_pred is None or self.signal_obs is None:
            return None
        
        if 'relative_residual' not in self._cache:
            diff = self.signal_pred - self.signal_obs
            residual_norm = np.linalg.norm(diff, axis=-1)
            signal_norm = np.linalg.norm(self.signal_obs, axis=-1)
            signal_norm = np.maximum(signal_norm, 1e-6)
            self._cache['relative_residual'] = residual_norm / signal_norm
        return self._cache['relative_residual']
    
    def get_relative_residual_mask(self, threshold: float = 0.15, 
                                    largest_component: bool = True,
                                    fill_holes: bool = True) -> np.ndarray:
        rel_res = self.get_relative_residual()
        if rel_res is None:
            return np.ones(self.shape[:3], dtype=bool)
        
        mask = rel_res < threshold
        
        if largest_component or fill_holes:
            from scipy import ndimage
            
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask_2d = mask[:, :, 0]
                
                if largest_component:
                    labeled, n_features = ndimage.label(mask_2d)
                    if n_features > 0:
                        sizes = ndimage.sum(mask_2d, labeled, range(1, n_features + 1))
                        largest_label = np.argmax(sizes) + 1
                        mask_2d = labeled == largest_label
                
                if fill_holes:
                    mask_2d = ndimage.binary_fill_holes(mask_2d)
                
                mask = mask_2d[:, :, np.newaxis]
            else:
                for z in range(mask.shape[2]):
                    slice_mask = mask[:, :, z]
                    
                    if largest_component:
                        labeled, n_features = ndimage.label(slice_mask)
                        if n_features > 0:
                            sizes = ndimage.sum(slice_mask, labeled, range(1, n_features + 1))
                            largest_label = np.argmax(sizes) + 1
                            slice_mask = labeled == largest_label
                    
                    if fill_holes:
                        slice_mask = ndimage.binary_fill_holes(slice_mask)
                    
                    mask[:, :, z] = slice_mask
        
        return mask.astype(bool)

    def get_r_squared(self) -> Optional[np.ndarray]:
        if self.signal_pred is None or self.signal_obs is None:
            return None
        
        if 'r_squared' not in self._cache:
            ss_res = np.sum((self.signal_pred - self.signal_obs) ** 2, axis=-1)
            
            obs_mean = self.signal_obs.mean(axis=-1, keepdims=True)
            ss_tot = np.sum((self.signal_obs - obs_mean) ** 2, axis=-1)
            
            valid_mask = ss_tot > 8.0
            r2 = np.where(valid_mask, 1 - ss_res / (ss_tot + 1e-8), np.nan)
            
            self._cache['r_squared'] = r2
        return self._cache['r_squared']
    
    def get_r2_brain_mask(
        self, 
        threshold: float = 0.5,
        min_ss_tot: float = 8.0,
        largest_component: bool = False,
        fill_holes: bool = False,
        convex_hull: bool = False,
    ) -> np.ndarray:
        from scipy import ndimage
        
        r2 = self.get_r_squared()
        if r2 is None:
            return np.ones(self.shape, dtype=bool)
        
        valid = ~np.isnan(r2)
        brain_mask = valid & (r2 >= threshold)
        
        if largest_component or convex_hull:
            labeled, n_components = ndimage.label(brain_mask)
            if n_components > 0:
                component_sizes = ndimage.sum(brain_mask, labeled, range(1, n_components + 1))
                largest_label = np.argmax(component_sizes) + 1
                brain_mask = (labeled == largest_label)
        
        if fill_holes:
            if brain_mask.ndim == 3:
                for z in range(brain_mask.shape[2]):
                    brain_mask[:, :, z] = ndimage.binary_fill_holes(brain_mask[:, :, z])
            else:
                brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        if convex_hull:
            from scipy.spatial import ConvexHull
            from skimage.draw import polygon
            
            if brain_mask.ndim == 2 or (brain_mask.ndim == 3 and brain_mask.shape[2] == 1):
                mask_2d = brain_mask.squeeze() if brain_mask.ndim == 3 else brain_mask
                coords = np.array(np.where(mask_2d)).T
                
                if len(coords) >= 3:
                    try:
                        hull = ConvexHull(coords)
                        hull_points = coords[hull.vertices]
                        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], mask_2d.shape)
                        hull_mask = np.zeros_like(mask_2d, dtype=bool)
                        hull_mask[rr, cc] = True
                        brain_mask = hull_mask[..., np.newaxis] if brain_mask.ndim == 3 else hull_mask
                    except Exception:
                        pass
            else:
                for z in range(brain_mask.shape[2]):
                    slice_mask = brain_mask[:, :, z]
                    coords = np.array(np.where(slice_mask)).T
                    
                    if len(coords) >= 3:
                        try:
                            hull = ConvexHull(coords)
                            hull_points = coords[hull.vertices]
                            rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], slice_mask.shape)
                            hull_slice = np.zeros_like(slice_mask, dtype=bool)
                            hull_slice[rr, cc] = True
                            brain_mask[:, :, z] = hull_slice
                        except Exception:
                            pass
        
        return brain_mask
        
        return brain_mask
    
    def get_r2_quality_mask(
        self,
        threshold: float = 0.8,
    ) -> np.ndarray:
        return self.get_r2_brain_mask(threshold=threshold)
    
    def get_adc_correlation(
        self, 
        bvals: np.ndarray, 
        bvecs: np.ndarray
    ) -> np.ndarray:
        if 'adc_corr' not in self._cache:
            shape = self.shape
            corr = np.full(shape, np.nan)
            
            if self.signal_obs is None:
                return corr
            
            b_threshold = 500
            high_b = bvals > b_threshold
            bvals_high = bvals[high_b]
            bvecs_high = bvecs[high_b]
            
            signal_high = self.signal_obs[..., high_b]
            signal_safe = np.maximum(signal_high, 1e-6)
            adc_obs = -np.log(signal_safe) / (bvals_high[np.newaxis, np.newaxis, np.newaxis, :] + 1)
            
            for idx in np.ndindex(shape):
                if self.f_wm_total[idx] < 0.2:
                    continue
                
                g_dot_mu_sq = np.zeros(len(bvecs_high))
                total_weight = 0
                
                for k in range(self.n_fibers):
                    frac = self.f_wm[idx][k]
                    if frac < 0.05:
                        continue
                    
                    d = self.fiber_dirs[idx][k]
                    d = d / (np.linalg.norm(d) + 1e-8)
                    
                    dot_products = np.dot(bvecs_high, d)
                    g_dot_mu_sq += frac * (dot_products ** 2)
                    total_weight += frac
                
                if total_weight < 0.1:
                    continue
                
                g_dot_mu_sq /= total_weight
                
                adc_voxel = adc_obs[idx]
                valid = np.isfinite(adc_voxel) & np.isfinite(g_dot_mu_sq)
                
                if valid.sum() > 10:
                    c = np.corrcoef(adc_voxel[valid], g_dot_mu_sq[valid])[0, 1]
                    corr[idx] = c
            
            self._cache['adc_corr'] = corr
        return self._cache['adc_corr']
    
    
    def get_fa_disagreement(
        self, 
        fa_dti: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if fa_dti is None:
            return None
        
        micro_fa = self.get_micro_fa()
        return np.abs(fa_dti - micro_fa)
    
    def get_direction_disagreement(
        self,
        dti_evec: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if dti_evec is None:
            return None
        
        if 'dir_disagree' not in self._cache:
            shape = self.shape
            angles = np.full(shape, np.nan)
            
            dom_idx = self.f_wm.argmax(axis=-1)
            
            for idx in np.ndindex(shape):
                if self.f_wm_total[idx] < 0.2:
                    continue
                
                k = dom_idx[idx]
                d_model = self.fiber_dirs[idx][k]
                d_model = d_model / (np.linalg.norm(d_model) + 1e-8)
                
                d_dti = dti_evec[idx]
                d_dti = d_dti / (np.linalg.norm(d_dti) + 1e-8)
                
                dot = np.abs(np.dot(d_model, d_dti))
                angles[idx] = np.arccos(np.clip(dot, 0, 1)) * 180 / np.pi
            
            self._cache['dir_disagree'] = angles
        return self._cache['dir_disagree']
    
    
    def get_safe_wm_mask(
        self,
        f_wm_threshold: float = 0.3,
        r2_threshold: float = 0.8,
    ) -> np.ndarray:
        mask = self.f_wm_total > f_wm_threshold
        
        r2 = self.get_r_squared()
        if r2 is not None:
            mask = mask & (r2 > r2_threshold)
        
        regime = self.get_fiber_regime_labels()
        mask = mask & (regime > 0)
        
        return mask
    
    def get_uncertain_mask(
        self,
        r2_threshold: float = 0.5,
        residual_threshold: float = 2.0,
    ) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        
        r2 = self.get_r_squared()
        if r2 is not None:
            mask |= (r2 < r2_threshold)
        
        residual = self.get_residual_norm()
        if residual is not None:
            mask |= (residual > residual_threshold)
        
        return mask
    
    def get_crossing_mask(
        self,
        min_fibers: int = 2,
        min_angle: float = 30.0,
        f_wm_threshold: float = 0.3,
    ) -> np.ndarray:
        mask = self.f_wm_total > f_wm_threshold
        mask &= self.get_fiber_count() >= min_fibers
        
        angles = self.get_crossing_angles()
        mask &= (angles >= min_angle) | np.isnan(angles)
        
        return mask
    
    
    def get_micro_ad(self) -> np.ndarray:
        if 'micro_ad' not in self._cache:
            d_parallel = 1.7e-3
            
            
            self._cache['micro_ad'] = d_parallel * self.f_wm_total
        return self._cache['micro_ad']
    
    def get_micro_rd(self) -> np.ndarray:
        if 'micro_rd' not in self._cache:
            d_parallel = 1.7e-3
            
            d_perp = d_parallel * (1 - self.f_intra)
            
            self._cache['micro_rd'] = d_perp * self.f_wm_total
        return self._cache['micro_rd']
    
    def get_ad_mismatch(
        self,
        ad_dti: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if ad_dti is None:
            return None
        return ad_dti - self.get_micro_ad()
    
    def get_rd_mismatch(
        self,
        rd_dti: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if rd_dti is None:
            return None
        return rd_dti - self.get_micro_rd()
    
    
    def get_model_complexity_need(
        self,
        r2_dti: Optional[np.ndarray] = None,
        threshold: float = 0.05,
    ) -> np.ndarray:
        if r2_dti is None:
            r2_dti = self.r2_dti
        
        labels = np.zeros(self.shape, dtype=np.int32)
        
        r2_ours = self.get_r_squared()
        fiber_count = self.get_fiber_count()
        
        non_wm = self.f_wm_total < 0.2
        labels[non_wm] = 0
        
        if r2_dti is None or r2_ours is None:
            labels[~non_wm] = 1
            return labels
        
        delta_r2 = r2_ours - r2_dti
        
        dti_ok = (~non_wm) & (delta_r2 < threshold) & (fiber_count <= 1)
        
        multi_needed = (~non_wm) & (delta_r2 >= threshold) & (fiber_count >= 2)
        
        mismatch = (~non_wm) & (r2_ours < 0.5) & (r2_dti < 0.5)
        
        labels[dti_ok] = 1
        labels[multi_needed] = 2
        labels[mismatch] = 3
        
        return labels
    
    def get_complexity_gain(
        self,
        r2_dti: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if r2_dti is None:
            r2_dti = self.r2_dti
            
        r2_ours = self.get_r_squared()
        if r2_dti is None or r2_ours is None:
            return None
        return r2_ours - r2_dti
    
    
    def get_wm_integrity_index(
        self,
        alpha_ndi: float = 0.25,
        alpha_fa: float = 0.25,
        alpha_topo: float = 0.25,
        alpha_quality: float = 0.25,
    ) -> np.ndarray:
        if 'integrity' not in self._cache:
            ndi = np.clip(self.get_ndi(), 0, 1)
            micro_fa = np.clip(self.get_micro_fa(), 0, 1)
            topo = np.clip(np.nan_to_num(self.get_topology_score(), nan=0.5), 0, 1)
            
            r2 = self.get_r_squared()
            if r2 is not None:
                r2 = np.clip(np.nan_to_num(r2, nan=0.5), 0, 1)
            else:
                r2 = np.full(self.shape, 0.5)
            
            health = (
                alpha_ndi * ndi +
                alpha_fa * micro_fa +
                alpha_topo * topo +
                alpha_quality * r2
            )
            
            health = np.where(self.f_wm_total > 0.2, health, np.nan)
            
            self._cache['integrity'] = health
        return self._cache['integrity']
    
    def get_topology_aware_integrity(self) -> np.ndarray:
        micro_fa = np.clip(self.get_micro_fa(), 0, 1)
        orphan = self.get_orphan_map().astype(float)
        
        r2 = self.get_r_squared()
        if r2 is None:
            r2 = np.ones(self.shape)
        r2 = np.clip(np.nan_to_num(r2, nan=0.5), 0, 1)
        
        integrity = micro_fa * (1 - orphan) * r2
        
        return np.where(self.f_wm_total > 0.2, integrity, np.nan)
    
    def get_crossing_robust_integrity(self) -> np.ndarray:
        integrity = self.get_wm_integrity_index()
        crossing = self.get_crossing_mask()
        
        return np.where(crossing, integrity, np.nan)
    
    
    def get_fiber_regime_labels(self, threshold: float = 0.1) -> np.ndarray:
        if 'regime' not in self._cache:
            labels = np.zeros(self.shape, dtype=np.int32)
            
            fiber_count = self.get_fiber_count(threshold)
            odi = self.get_odi()
            wm_total = self.f_wm_total
            
            no_wm = wm_total < threshold
            
            single = (fiber_count == 1) & ~no_wm
            
            multi = (fiber_count >= 2) & ~no_wm
            
            dispersed = (odi > 0.5) & (fiber_count <= 1) & ~no_wm
            
            labels[no_wm] = 0
            labels[single & ~dispersed] = 1
            labels[multi] = 2
            labels[dispersed] = 3
            
            self._cache['regime'] = labels
        return self._cache['regime']
    
    def get_topology_labels(self) -> np.ndarray:
        if 'topo_labels' not in self._cache:
            labels = np.zeros(self.shape, dtype=np.int32)
            
            wm_mask = self.f_wm_total > 0.2
            orphan = self.get_orphan_map()
            endpoint = self.get_endpoint_map()
            
            labels[~wm_mask] = 0
            
            labels[wm_mask & ~orphan & ~endpoint] = 1
            
            endpoint_gm = endpoint & (self.f_gm > 0.2)
            labels[endpoint_gm] = 2
            
            endpoint_bad = endpoint & (self.f_gm <= 0.2)
            labels[endpoint_bad] = 3
            
            labels[orphan] = 4
            
            self._cache['topo_labels'] = labels
        return self._cache['topo_labels']
    
    
    def _get_sphere(self, n_dirs: int = 642) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = f'sphere_{n_dirs}'
        if cache_key not in self._cache:
            try:
                from dipy.data import get_sphere
                if n_dirs <= 362:
                    sphere = get_sphere('symmetric362')
                elif n_dirs <= 642:
                    sphere = get_sphere(name='symmetric642')
                else:
                    sphere = get_sphere('repulsion724')
                vertices = sphere.vertices
                weights = np.ones(len(vertices)) / len(vertices)
            except ImportError:
                indices = np.arange(n_dirs, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / n_dirs)
                theta = np.pi * (1 + 5**0.5) * indices
                vertices = np.column_stack([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                weights = np.ones(n_dirs) / n_dirs
            
            self._cache[cache_key] = (vertices, weights)
        return self._cache[cache_key]
    
    def _watson_kernel(
        self, 
        u: np.ndarray,
        mu: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        dot = np.dot(u, mu)
        return np.exp(kappa * dot**2)
    
    def _vmf_kernel(
        self,
        u: np.ndarray,
        mu: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        dot = np.dot(u, mu)
        return np.exp(kappa * dot)
    
    def get_fod_sphere(
        self,
        n_dirs: int = 642,
        kernel: str = 'watson',
        compartment: str = 'wm',
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = f'fod_sphere_{n_dirs}_{kernel}'
        if cache_key not in self._cache:
            vertices, weights = self._get_sphere(n_dirs)
            shape = self.shape
            n_verts = len(vertices)
            
            fod = np.zeros((*shape, n_verts))
            
            kernel_fn = self._watson_kernel if kernel == 'watson' else self._vmf_kernel
            
            for idx in np.ndindex(shape):
                if self.f_wm_total[idx] < 0.1:
                    continue
                
                for k in range(self.n_fibers):
                    f_k = self.f_wm[idx][k]
                    if f_k < 0.05:
                        continue
                    
                    mu_k = self.fiber_dirs[idx][k]
                    mu_k = mu_k / (np.linalg.norm(mu_k) + 1e-8)
                    
                    kappa_k = float(self.kappa[idx]) if np.isscalar(self.kappa[idx]) else self.kappa[idx]
                    
                    fod[idx] += f_k * kernel_fn(vertices, mu_k, kappa_k)
                
                if fod[idx].sum() > 0:
                    fod[idx] /= (fod[idx] * weights).sum()
            
            self._cache[cache_key] = fod
            self._cache[f'{cache_key}_vertices'] = vertices
        
        return self._cache[f'{cache_key}_vertices'], self._cache[cache_key]
    
    def get_fod_sh(
        self,
        lmax: int = 8,
        kernel: str = 'watson',
    ) -> np.ndarray:
        cache_key = f'fod_sh_{lmax}_{kernel}'
        if cache_key not in self._cache:
            try:
                from dipy.reconst.shm import sph_harm_ind_list
                from dipy.core.sphere import Sphere
                from dipy.reconst.shm import sf_to_sh
            except ImportError:
                raise ImportError("dipy required for SH computation: pip install dipy")
            
            vertices, fod = self.get_fod_sphere(kernel=kernel)
            
            sphere = Sphere(xyz=vertices)
            
            m_list, l_list = sph_harm_ind_list(lmax)
            n_coeffs = len(l_list)
            
            shape = self.shape
            fod_flat = fod.reshape(-1, fod.shape[-1])
            
            sh_flat = sf_to_sh(fod_flat, sphere, sh_order=lmax, basis_type='descoteaux07')
            
            sh_coeffs = sh_flat.reshape(*shape, -1)
            
            self._cache[cache_key] = sh_coeffs
            self._cache[f'{cache_key}_lmax'] = lmax
            self._cache[f'{cache_key}_l_list'] = l_list
            self._cache[f'{cache_key}_m_list'] = m_list
        
        return self._cache[cache_key]
    
    def get_fod_peaks(
        self,
        lmax: int = 8,
        peak_thresh: float = 0.1,
        min_separation_angle: float = 25.0,
        max_peaks: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from dipy.direction import peak_directions
        except ImportError:
            raise ImportError("dipy required for peak extraction: pip install dipy")
        
        vertices, fod = self.get_fod_sphere()
        
        from dipy.data import get_sphere
        sphere = get_sphere(name='symmetric642')
        
        shape = self.shape
        peak_dirs = np.full((*shape, max_peaks, 3), np.nan)
        peak_values = np.full((*shape, max_peaks), np.nan)
        
        for idx in np.ndindex(shape):
            if self.f_wm_total[idx] < 0.1:
                continue
            
            odf = fod[idx]
            if odf.max() < 1e-6:
                continue
            
            dirs, vals, indices = peak_directions(
                odf, sphere, 
                relative_peak_threshold=peak_thresh,
                min_separation_angle=min_separation_angle
            )
            
            n_peaks = min(len(dirs), max_peaks)
            if n_peaks > 0:
                peak_dirs[idx][:n_peaks] = dirs[:n_peaks]
                peak_values[idx][:n_peaks] = vals[:n_peaks]
        
        return peak_dirs, peak_values
    
    def get_gfa(self, n_dirs: int = 642) -> np.ndarray:
        if 'gfa' not in self._cache:
            _, fod = self.get_fod_sphere(n_dirs=n_dirs)
            
            fod_mean = fod.mean(axis=-1, keepdims=True)
            fod_std = np.sqrt(np.mean((fod - fod_mean)**2, axis=-1))
            fod_rms = np.sqrt(np.mean(fod**2, axis=-1))
            
            gfa = np.where(fod_rms > 1e-8, fod_std / fod_rms, 0)
            
            self._cache['gfa'] = gfa
        return self._cache['gfa']
    
    def get_odf_entropy(self, n_dirs: int = 642) -> np.ndarray:
        if 'odf_entropy' not in self._cache:
            _, fod = self.get_fod_sphere(n_dirs=n_dirs)
            
            fod_sum = fod.sum(axis=-1, keepdims=True)
            p = np.where(fod_sum > 1e-8, fod / fod_sum, 1.0 / fod.shape[-1])
            
            p_safe = np.maximum(p, 1e-10)
            entropy = -np.sum(p * np.log(p_safe), axis=-1)
            
            max_entropy = np.log(fod.shape[-1])
            entropy /= max_entropy
            
            self._cache['odf_entropy'] = entropy
        return self._cache['odf_entropy']
    
    def get_sh_power_spectrum(self, lmax: int = 8, kernel: str = 'watson') -> np.ndarray:
        sh = self.get_fod_sh(lmax=lmax, kernel=kernel)
        cache_key = f'fod_sh_{lmax}_{kernel}'
        l_list = self._cache[f'{cache_key}_l_list']
        
        orders = np.arange(0, lmax + 1, 2)
        n_orders = len(orders)
        
        power = np.zeros((*self.shape, n_orders))
        
        for i, l in enumerate(orders):
            mask = l_list == l
            power[..., i] = np.sum(sh[..., mask]**2, axis=-1)
        
        return power
    
    def compare_with_csd(
        self,
        csd_sh: np.ndarray,
        lmax: int = 8,
    ) -> Dict[str, np.ndarray]:
        model_sh = self.get_fod_sh(lmax=lmax)
        
        model_norm = np.linalg.norm(model_sh, axis=-1, keepdims=True) + 1e-8
        csd_norm = np.linalg.norm(csd_sh, axis=-1, keepdims=True) + 1e-8
        
        angular_corr = np.sum(model_sh * csd_sh, axis=-1) / (
            model_norm.squeeze() * csd_norm.squeeze()
        )
        
        
        return {
            'angular_correlation': angular_corr,
        }
    
    
    def get_summary(self, roi_mask: Optional[np.ndarray] = None) -> Dict:
        if roi_mask is not None:
            mask = roi_mask
        elif self.brain_mask is not None:
            mask = self.brain_mask > 0
        elif self.wm_mask is not None:
            mask = self.wm_mask > 0
        else:
            mask = self.f_wm_total > 0.2
        
        wm_mask = self.f_wm_total > 0.3
        
        if mask.ndim < self.f_wm_total.ndim:
            mask = mask[..., np.newaxis]
        if wm_mask.ndim < self.f_wm_total.ndim:
            wm_mask = wm_mask[..., np.newaxis]
        
        def masked_stats(arr, use_mask=None):
            m = use_mask if use_mask is not None else mask
            valid = arr[m]
            valid = valid[np.isfinite(valid)]
            if len(valid) == 0:
                return {'mean': np.nan, 'std': np.nan, 'median': np.nan}
            return {
                'mean': np.mean(valid),
                'std': np.std(valid),
                'median': np.median(valid),
            }
        
        return {
            'ndi': masked_stats(self.get_ndi(), wm_mask),
            'odi': masked_stats(self.get_odi(), wm_mask),
            'free_water': masked_stats(self.get_free_water_fraction()),
            'micro_fa': masked_stats(self.get_micro_fa(), wm_mask),
            
            'fiber_count': masked_stats(self.get_fiber_count().astype(float), wm_mask),
            'complexity': masked_stats(self.get_complexity_index(), wm_mask),
            'crossing_angle': masked_stats(self.get_crossing_angles(), wm_mask),
            'curvature': masked_stats(self.get_curvature(), wm_mask),
            
            'orphan_density': self.get_orphan_density(),
            'topo_score': masked_stats(self.get_topology_score()),
            
            'r_squared': masked_stats(self.get_r_squared()) if self.signal_obs is not None else None,
            
            'regime_counts': {
                i: (self.get_fiber_regime_labels() == i).sum()
                for i in range(4)
            },
        }
    
    def print_summary(self, name: str = "Patch"):
        stats = self.get_summary()
        
        wm_voxels = (self.f_wm_total > 0.3).sum()
        total_voxels = np.prod(self.shape)
        
        print(f"\n{'='*60}")
        print(f"MICROSTRUCTURE SUMMARY: {name}")
        print(f"{'='*60}")
        
        print(f"\n--- NODDI-like Indices (WM-specific, {wm_voxels} voxels) ---")
        print(f"  NDI:        {stats['ndi']['mean']:.3f} ± {stats['ndi']['std']:.3f}")
        print(f"  ODI:        {stats['odi']['mean']:.3f} ± {stats['odi']['std']:.3f}")
        print(f"  Free Water: {stats['free_water']['mean']:.3f} ± {stats['free_water']['std']:.3f}")
        print(f"  Micro FA:   {stats['micro_fa']['mean']:.3f} ± {stats['micro_fa']['std']:.3f}")
        
        print(f"\n--- Fiber Geometry (WM-specific) ---")
        print(f"  Fiber Count: {stats['fiber_count']['mean']:.1f} ± {stats['fiber_count']['std']:.1f}")
        print(f"  Complexity:  {stats['complexity']['mean']:.3f} ± {stats['complexity']['std']:.3f}")
        print(f"  Crossing °:  {stats['crossing_angle']['mean']:.1f} ± {stats['crossing_angle']['std']:.1f}")
        print(f"  Curvature:   {stats['curvature']['mean']:.4f} ± {stats['curvature']['std']:.4f}")
        
        print("\n--- Topology ---")
        print(f"  Orphan Density: {stats['orphan_density']:.3f}")
        print(f"  Topo Score:     {stats['topo_score']['mean']:.3f} ± {stats['topo_score']['std']:.3f}")
        
        if stats['r_squared'] is not None:
            print("\n--- Fit Quality ---")
            print(f"  R²: {stats['r_squared']['mean']:.3f} ± {stats['r_squared']['std']:.3f}")
            
            r2 = self.get_r_squared()
            if r2 is not None:
                total_voxels = np.prod(self.shape)
                for thresh in [0.5, 0.7, 0.9]:
                    r2_mask = self.get_r2_brain_mask(threshold=thresh)
                    n_brain = r2_mask.sum()
                    pct = 100 * n_brain / total_voxels
                    print(f"  R²>{thresh:.1f} mask: {n_brain:,} voxels ({pct:.1f}%)")
        
        print("\n--- Fiber Regime Counts ---")
        regime_names = ['No WM', 'Single Fiber', 'Multi-Fiber', 'Dispersed']
        for i, name in enumerate(regime_names):
            print(f"  {name}: {stats['regime_counts'][i]}")


def extract_all_maps(
    params: Dict[str, torch.Tensor],
    signal_obs: Optional[np.ndarray] = None,
    signal_pred: Optional[np.ndarray] = None,
    bvals: Optional[np.ndarray] = None,
    bvecs: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    maps = MicrostructureMaps.from_fitted_params(
        params, 
        signal_pred=torch.tensor(signal_pred) if signal_pred is not None else None,
        signal_obs=torch.tensor(signal_obs) if signal_obs is not None else None,
    )
    
    result = {
        'ndi': maps.get_ndi(),
        'odi': maps.get_odi(),
        'free_water': maps.get_free_water_fraction(),
        'micro_fa': maps.get_micro_fa(),
        
        'fiber_count': maps.get_fiber_count(),
        'complexity': maps.get_complexity_index(),
        'crossing_angle': maps.get_crossing_angles(),
        'curvature': maps.get_curvature(),
        
        'orphan_map': maps.get_orphan_map(),
        'endpoint_map': maps.get_endpoint_map(),
        'topology_score': maps.get_topology_score(),
        
        'fiber_regime': maps.get_fiber_regime_labels(),
        'topology_labels': maps.get_topology_labels(),
    }
    
    if signal_obs is not None and signal_pred is not None:
        result['residual_norm'] = maps.get_residual_norm()
        result['r_squared'] = maps.get_r_squared()
        result['r2_brain_mask'] = maps.get_r2_brain_mask(threshold=0.5)
    
    if bvals is not None and bvecs is not None and signal_obs is not None:
        maps.signal_obs = signal_obs
        result['adc_correlation'] = maps.get_adc_correlation(bvals, bvecs)
    
    return result


def compute_r2_brain_mask(
    signal_obs: np.ndarray,
    signal_pred: np.ndarray,
    threshold: float = 0.5,
    min_ss_tot: float = 8.0,
) -> np.ndarray:
    ss_res = np.sum((signal_pred - signal_obs) ** 2, axis=-1)
    
    obs_mean = signal_obs.mean(axis=-1, keepdims=True)
    ss_tot = np.sum((signal_obs - obs_mean) ** 2, axis=-1)
    
    valid_mask = ss_tot > min_ss_tot
    r2 = np.where(valid_mask, 1 - ss_res / (ss_tot + 1e-8), np.nan)
    
    brain_mask = valid_mask & (r2 >= threshold)
    
    return brain_mask


def export_roi_summary_table(
    maps: MicrostructureMaps,
    roi_masks: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
) -> str:
    import csv
    from io import StringIO
    
    columns = [
        'ROI', 'n_voxels',
        'NDI_mean', 'NDI_std',
        'ODI_mean', 'ODI_std',
        'FW_mean', 'FW_std',
        'microFA_mean', 'microFA_std',
        'fiber_count_mean', 'fiber_count_std',
        'crossing_angle_mean', 'crossing_angle_std',
        'complexity_mean', 'complexity_std',
        'curvature_mean', 'curvature_std',
        'orphan_count', 'endpoint_count',
        'topo_score_mean', 'topo_score_std',
        'R2_mean', 'R2_std',
        'regime_single', 'regime_multi', 'regime_dispersed',
    ]
    
    rows = []
    for roi_name, mask in roi_masks.items():
        stats = maps.get_summary(roi_mask=mask)
        
        regime = maps.get_fiber_regime_labels()
        orphan = maps.get_orphan_map()
        endpoint = maps.get_endpoint_map()
        
        row = {
            'ROI': roi_name,
            'n_voxels': mask.sum(),
            'NDI_mean': stats['ndi']['mean'],
            'NDI_std': stats['ndi']['std'],
            'ODI_mean': stats['odi']['mean'],
            'ODI_std': stats['odi']['std'],
            'FW_mean': stats['free_water']['mean'],
            'FW_std': stats['free_water']['std'],
            'microFA_mean': stats['micro_fa']['mean'],
            'microFA_std': stats['micro_fa']['std'],
            'fiber_count_mean': stats['fiber_count']['mean'],
            'fiber_count_std': stats['fiber_count']['std'],
            'crossing_angle_mean': stats['crossing_angle']['mean'],
            'crossing_angle_std': stats['crossing_angle']['std'],
            'complexity_mean': stats['complexity']['mean'],
            'complexity_std': stats['complexity']['std'],
            'curvature_mean': stats['curvature']['mean'],
            'curvature_std': stats['curvature']['std'],
            'orphan_count': (orphan & mask).sum(),
            'endpoint_count': (endpoint & mask).sum(),
            'topo_score_mean': stats['topo_score']['mean'],
            'topo_score_std': stats['topo_score']['std'],
            'R2_mean': stats['r_squared']['mean'] if stats['r_squared'] else np.nan,
            'R2_std': stats['r_squared']['std'] if stats['r_squared'] else np.nan,
            'regime_single': ((regime == 1) & mask).sum(),
            'regime_multi': ((regime == 2) & mask).sum(),
            'regime_dispersed': ((regime == 3) & mask).sum(),
        }
        rows.append(row)
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    
    csv_str = output.getvalue()
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(csv_str)
    
    return csv_str


def create_figure2_panel(
    maps: MicrostructureMaps,
    slice_idx: int = 0,
    output_path: Optional[str] = None,
    dpi: int = 150,
    use_r2_mask: bool = True,
    r2_threshold: float = 0.9,
    brain_mask: Optional[np.ndarray] = None,
    use_relative_residual_mask: bool = False,
    rel_residual_threshold: float = 0.15,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy import ndimage
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Derived Microstructure Maps (Single Fit)', fontsize=16, fontweight='bold')
    
    if brain_mask is not None:
        if brain_mask.ndim == 2:
            r2_mask_2d = brain_mask.astype(bool)
        elif brain_mask.ndim == 3 and brain_mask.shape[2] == 1:
            r2_mask_2d = brain_mask[:, :, 0].astype(bool)
        else:
            r2_mask_2d = brain_mask[:, :, slice_idx].astype(bool)
    elif use_relative_residual_mask:
        rel_mask_3d = maps.get_relative_residual_mask(
            threshold=rel_residual_threshold,
            largest_component=True,
            fill_holes=True
        )
        if rel_mask_3d.ndim == 2:
            r2_mask_2d = rel_mask_3d
        elif rel_mask_3d.ndim == 3 and rel_mask_3d.shape[2] == 1:
            r2_mask_2d = rel_mask_3d[:, :, 0]
        else:
            r2_mask_2d = rel_mask_3d[:, :, slice_idx]
    elif use_r2_mask:
        r2_mask_3d = maps.get_r2_brain_mask(
            threshold=r2_threshold, 
            largest_component=True, 
            fill_holes=True,
            convex_hull=False
        )
        if r2_mask_3d.ndim == 2:
            r2_mask_2d = r2_mask_3d
        elif r2_mask_3d.ndim == 3 and r2_mask_3d.shape[2] == 1:
            r2_mask_2d = r2_mask_3d[:, :, 0]
        else:
            r2_mask_2d = r2_mask_3d[:, :, slice_idx]
    else:
        r2_mask_2d = None
    
    def get_slice(arr, apply_mask=True):
        if arr is None:
            return np.zeros(maps.shape[:2])
        if arr.ndim == 2:
            sl = arr
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                sl = arr[:, :, 0]
            else:
                sl = arr[:, :, slice_idx]
        elif arr.ndim == 4:
            if arr.shape[2] == 1:
                sl = arr[:, :, 0, 0]
            else:
                sl = arr[:, :, slice_idx, 0]
        else:
            sl = arr
        
        if apply_mask and r2_mask_2d is not None:
            return np.ma.masked_where(~r2_mask_2d, sl)
        return sl
    
    def normalize_to_01(arr, low_pct=2, high_pct=98):
        arr_flat = arr.compressed() if hasattr(arr, 'compressed') else arr[arr > 0]
        if len(arr_flat) == 0:
            return arr
        vmin = np.percentile(arr_flat, low_pct)
        vmax = np.percentile(arr_flat, high_pct)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        normalized = (arr - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)
    
    imshow_kwargs = {'origin': 'lower', 'interpolation': 'nearest'}
    
    def get_cmap_masked(cmap_name):
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap(cmap_name).copy()
        cmap.set_bad(color='white', alpha=0)
        return cmap
    
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    row = 0
    ndi_slice = get_slice(maps.get_absolute_ndi())
    ndi_scaled = normalize_to_01(ndi_slice)
    axes[row, 0].imshow(ndi_scaled.T, cmap=get_cmap_masked('plasma'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 0].set_title('NDI (scaled 0-1)', fontsize=10)
    
    odi_slice = get_slice(maps.get_odi_wm_weighted())
    odi_scaled = normalize_to_01(odi_slice)
    axes[row, 1].imshow(odi_scaled.T, cmap=get_cmap_masked('plasma'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 1].set_title('ODI (scaled 0-1)', fontsize=10)
    
    axes[row, 2].imshow(get_slice(maps.get_free_water_fraction()).T, cmap=get_cmap_masked('Blues'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 2].set_title('Free Water Fraction', fontsize=10)
    
    microfa_slice = get_slice(maps.get_micro_fa())
    microfa_scaled = normalize_to_01(microfa_slice)
    axes[row, 3].imshow(microfa_scaled.T, cmap=get_cmap_masked('cividis'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 3].set_title('Micro FA (scaled 0-1)', fontsize=10)

    row = 1
    axes[row, 0].imshow(get_slice(maps.get_fiber_count()).T, cmap=get_cmap_masked('jet'), vmin=0, vmax=3, **imshow_kwargs)
    axes[row, 0].set_title('Fiber Count', fontsize=10)
    
    ca = get_slice(maps.get_crossing_angles())
    axes[row, 1].imshow(ca.T, cmap=get_cmap_masked('coolwarm'), vmin=0, vmax=90, **imshow_kwargs)
    axes[row, 1].set_title('Crossing Angle (°)', fontsize=10)
    
    axes[row, 2].imshow(get_slice(maps.get_complexity_index()).T, cmap=get_cmap_masked('inferno'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 2].set_title('Complexity (Entropy)', fontsize=10)
    
    axes[row, 3].imshow(get_slice(maps.get_curvature()).T, cmap=get_cmap_masked('YlOrRd'), vmin=0, vmax=0.1, **imshow_kwargs)
    axes[row, 3].set_title('Curvature', fontsize=10)
    
    row = 2
    axes[row, 0].imshow(get_slice(maps.get_orphan_map()).T, cmap=get_cmap_masked('Reds'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 0].set_title('Orphan Voxels', fontsize=10)
    
    axes[row, 1].imshow(get_slice(maps.get_endpoint_map()).T, cmap=get_cmap_masked('Greens'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 1].set_title('Fiber Endpoints', fontsize=10)
    
    ts = get_slice(maps.get_topology_score())
    axes[row, 2].imshow(ts.T, cmap=get_cmap_masked('RdYlGn'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 2].set_title('Topology Score', fontsize=10)
    
    regime_cmap = ListedColormap(['gray', 'green', 'orange', 'purple'])
    regime_cmap.set_bad(color='white', alpha=0)
    axes[row, 3].imshow(get_slice(maps.get_fiber_regime_labels()).T, cmap=regime_cmap, vmin=0, vmax=3, **imshow_kwargs)
    axes[row, 3].set_title('Fiber Regime', fontsize=10)
    
    row = 3
    r2 = get_slice(maps.get_r_squared())
    if r2 is not None and not np.all(r2 == 0):
        axes[row, 0].imshow(r2.T, cmap=get_cmap_masked('RdYlGn'), vmin=0.5, vmax=1, **imshow_kwargs)
    axes[row, 0].set_title('R² (Fit Quality)', fontsize=10)
    
    residual = get_slice(maps.get_residual_norm())
    if residual is not None and not np.all(residual == 0):
        axes[row, 1].imshow(residual.T, cmap=get_cmap_masked('Reds'), vmin=0, vmax=3, **imshow_kwargs)
    axes[row, 1].set_title('Residual Norm', fontsize=10)
    
    safe = get_slice(maps.get_safe_wm_mask())
    axes[row, 2].imshow(safe.T, cmap=get_cmap_masked('Greens'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 2].set_title('Safe WM Mask', fontsize=10)
    
    uncertain = get_slice(maps.get_uncertain_mask())
    axes[row, 3].imshow(uncertain.T, cmap=get_cmap_masked('Reds'), vmin=0, vmax=1, **imshow_kwargs)
    axes[row, 3].set_title('Uncertain Regions', fontsize=10)
    
    for ax in axes.flat:
        ax.axis('off')
    
    row_labels = ['Microstructure', 'Fiber Geometry', 'Topology', 'Quality / Confidence']
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.875 - i*0.235, label, fontsize=12, fontweight='bold', 
                 rotation=90, va='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_figure2b_panel(
    maps: MicrostructureMaps,
    slice_idx: int = 0,
    r2_dti: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    if r2_dti is None:
        r2_dti = maps.r2_dti
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Model Complexity & WM Health Maps', fontsize=16, fontweight='bold')
    
    def get_slice(arr):
        if arr is None:
            return np.zeros(maps.shape[:2])
        if arr.ndim == 3:
            return arr[:, :, slice_idx]
        return arr[:, :, slice_idx, 0] if arr.ndim == 4 else arr
    
    row = 0
    axes[row, 0].imshow(get_slice(maps.get_orientation_coherence()), cmap='viridis', vmin=0, vmax=1)
    axes[row, 0].set_title('Orientation Coherence', fontsize=10)
    
    axes[row, 1].imshow(get_slice(maps.get_direction_certainty()), cmap='plasma', vmin=0, vmax=1)
    axes[row, 1].set_title('Direction Certainty', fontsize=10)
    
    axes[row, 2].imshow(get_slice(maps.get_direction_ambiguity()), cmap='Reds', vmin=0, vmax=1)
    axes[row, 2].set_title('Direction Ambiguity', fontsize=10)
    
    axes[row, 3].imshow(get_slice(maps.get_complexity_index()), cmap='inferno', vmin=0, vmax=1)
    axes[row, 3].set_title('Fiber Complexity', fontsize=10)
    
    row = 1
    axes[row, 0].imshow(get_slice(maps.get_partial_volume_severity()), cmap='YlOrRd', vmin=0, vmax=0.5)
    axes[row, 0].set_title('PV Severity', fontsize=10)
    
    axes[row, 1].imshow(get_slice(maps.get_wm_gm_pv_index()), cmap='Purples', vmin=0, vmax=0.5)
    axes[row, 1].set_title('WM-GM Interface', fontsize=10)
    
    axes[row, 2].imshow(get_slice(maps.get_wm_csf_pv_index()), cmap='Blues', vmin=0, vmax=0.5)
    axes[row, 2].set_title('WM-CSF Interface', fontsize=10)
    
    axes[row, 3].imshow(get_slice(maps.get_deep_wm_mask()), cmap='Greens', vmin=0, vmax=1)
    axes[row, 3].set_title('Deep WM Mask', fontsize=10)
    
    row = 2
    if r2_dti is not None:
        complexity_labels = get_slice(maps.get_model_complexity_need(r2_dti=r2_dti))
        complexity_cmap = ListedColormap(['gray', 'green', 'orange', 'red'])
        axes[row, 0].imshow(complexity_labels, cmap=complexity_cmap, vmin=0, vmax=3)
        axes[row, 0].set_title('Model Complexity Need', fontsize=10)
        
        gain = get_slice(maps.get_complexity_gain(r2_dti=r2_dti))
        if gain is not None:
            axes[row, 1].imshow(gain, cmap='RdYlGn', vmin=-0.2, vmax=0.2)
        axes[row, 1].set_title('ΔR² (Ours - DTI)', fontsize=10)
        
        dti_ok = complexity_labels == 1
        multi_needed = complexity_labels == 2
        axes[row, 2].imshow(dti_ok, cmap='Greens', vmin=0, vmax=1)
        axes[row, 2].set_title('DTI Sufficient', fontsize=10)
        
        axes[row, 3].imshow(multi_needed, cmap='Oranges', vmin=0, vmax=1)
        axes[row, 3].set_title('Multi-Fiber Needed', fontsize=10)
    else:
        for i in range(4):
            axes[row, i].text(0.5, 0.5, 'Needs DTI R²', ha='center', va='center', 
                             transform=axes[row, i].transAxes)
            axes[row, i].set_title(['Model Complexity', 'ΔR²', 'DTI-OK', 'Multi-Fiber'][i], fontsize=10)
    
    row = 3
    integrity = get_slice(maps.get_wm_integrity_index())
    axes[row, 0].imshow(integrity, cmap='RdYlGn', vmin=0, vmax=1)
    axes[row, 0].set_title('WM Integrity Index', fontsize=10)
    
    topo_integrity = get_slice(maps.get_topology_aware_integrity())
    axes[row, 1].imshow(topo_integrity, cmap='RdYlGn', vmin=0, vmax=1)
    axes[row, 1].set_title('Topology-Aware Integrity', fontsize=10)
    
    crossing_integrity = get_slice(maps.get_crossing_robust_integrity())
    axes[row, 2].imshow(np.nan_to_num(crossing_integrity, nan=0), cmap='RdYlGn', vmin=0, vmax=1)
    axes[row, 2].set_title('Crossing-Robust Integrity', fontsize=10)
    
    healthy = get_slice(maps.get_safe_wm_mask()) & (integrity > 0.6)
    axes[row, 3].imshow(healthy, cmap='Greens', vmin=0, vmax=1)
    axes[row, 3].set_title('Healthy WM Mask', fontsize=10)
    
    for ax in axes.flat:
        ax.axis('off')
    
    row_labels = ['Orientation', 'Partial Volume', 'Model Complexity', 'WM Health']
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.875 - i*0.235, label, fontsize=12, fontweight='bold', 
                 rotation=90, va='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig
