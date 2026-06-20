#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

HCP_DIR = "/home/moabouag/Music/biophysical_sphereflow/data/100307"
PRISM_DIR = os.path.join(PROJECT, "outputs", "hcp_mse_slab20_overlap10")
CACHED_DIR = os.path.join(PROJECT, "outputs", "hcp_bundle_5methods")
TC_CACHED_DIR = os.path.join(PROJECT, "outputs", "hcp_tractcloud_5methods")
OUT_DIR = os.path.join(PROJECT, "outputs", "hcp_tracking_benchmark")

TC_PROJECT = os.path.join(PROJECT, "TractCloud")
TC_MODEL_PATH = os.path.join(TC_PROJECT, "TrainedModel", "best_tract_f1_model.pth")
TC_MASS_CENTER = os.path.join(TC_PROJECT, "TrainData_800clu800ol", "HCP_mass_center.npy")
TC_ANNOT_PATH = os.path.join(TC_PROJECT, "datasets",
                              "FiberClusterAnnotation_Updated20230110.xlsx")

METHODS = ['prism', 'csd', 'odffp']
METHOD_LABELS = {
    'prism': 'PRISM', 'csd': 'CSD', 'odffp': 'ODF-FP', 'atlas': 'Atlas (GT)',
}
METHOD_COLORS_RGB = {
    'prism': np.array([0.13, 0.59, 0.95]),
    'csd':   np.array([1.00, 0.60, 0.00]),
    'odffp': np.array([0.61, 0.15, 0.69]),
    'atlas': np.array([0.50, 0.50, 0.50]),
}
METHOD_COLORS_HEX = {
    'prism': '#2196F3', 'csd': '#FF9800', 'odffp': '#9C27B0', 'atlas': '#9E9E9E',
}

BUNDLE_NAMES = [
    "CST_L", "CST_R", "AF_L", "AF_R",
    "CC_ForcepsMajor", "CC_ForcepsMinor",
    "ILF_L", "ILF_R", "SLF_L", "SLF_R",
    "UF_L", "UF_R",
]
BUNDLE_COLORS = {
    "CST_L": (1.0, 0.2, 0.2),   "CST_R": (0.8, 0.15, 0.15),
    "AF_L":  (0.2, 0.6, 1.0),   "AF_R":  (0.15, 0.45, 0.85),
    "CC_ForcepsMajor": (0.2, 0.85, 0.2),
    "CC_ForcepsMinor": (0.6, 1.0, 0.3),
    "ILF_L": (1.0, 0.6, 0.0),   "ILF_R": (0.85, 0.5, 0.0),
    "SLF_L": (0.8, 0.2, 1.0),   "SLF_R": (0.65, 0.15, 0.85),
    "UF_L":  (0.0, 0.85, 0.85), "UF_R":  (0.0, 0.65, 0.65),
}

TRACT_NAMES_ORDERED = [
    'AF', 'CB', 'EC', 'EmC', 'ILF', 'IOFF', 'MdLF',
    'SLF-I', 'SLF-II', 'SLF-III', 'UF',
    'CST', 'CR-F', 'CR-P', 'SF', 'SO', 'SP',
    'TF', 'TO', 'TT', 'TP', 'PLIC',
    'CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7',
    'CPC', 'ICP', 'Intra-CBLM-I-P', 'Intra-CBLM-PaT', 'MCP',
    'Sup-F', 'Sup-FP', 'Sup-O', 'Sup-OT',
    'Sup-P', 'Sup-PO', 'Sup-PT', 'Sup-T',
    'Other',
]
TRACT_IDX = {n: i for i, n in enumerate(TRACT_NAMES_ORDERED)}

TC_BUNDLES = {
    'CST': {'tc_indices': [TRACT_IDX['CST']],
            'atlas_files': ['CST_L.trk', 'CST_R.trk'],
            'color': np.array([1.0, 0.2, 0.2])},
    'AF':  {'tc_indices': [TRACT_IDX['AF']],
            'atlas_files': ['AF_L.trk', 'AF_R.trk'],
            'color': np.array([0.2, 0.6, 1.0])},
    'CC':  {'tc_indices': [TRACT_IDX[f'CC{i}'] for i in range(1, 8)],
            'atlas_files': ['CC_ForcepsMajor.trk', 'CC_ForcepsMinor.trk'],
            'color': np.array([0.2, 0.85, 0.2])},
    'ILF': {'tc_indices': [TRACT_IDX['ILF']],
            'atlas_files': ['ILF_L.trk', 'ILF_R.trk'],
            'color': np.array([1.0, 0.6, 0.0])},
    'SLF': {'tc_indices': [TRACT_IDX['SLF-I'], TRACT_IDX['SLF-II'],
                           TRACT_IDX['SLF-III']],
            'atlas_files': ['SLF_L.trk', 'SLF_R.trk'],
            'color': np.array([0.8, 0.2, 1.0])},
    'UF':  {'tc_indices': [TRACT_IDX['UF']],
            'atlas_files': ['UF_L.trk', 'UF_R.trk'],
            'color': np.array([0.0, 0.85, 0.85])},
    'CB':  {'tc_indices': [TRACT_IDX['CB']],
            'atlas_files': ['CB_L.trk', 'CB_R.trk'],
            'color': np.array([0.9, 0.9, 0.1])},
}

TRACK_PARAMS = dict(
    seed_density=1,
    step_size=0.5,
    max_angle=30,
    min_length=20,
    max_length=250,
    fa_threshold=0.1,
)

BUAN_N_DISKS = 100



def build_prism_odf(prism_dir, affine, kappa=12.0, cache_path=None):
    from dipy.data import default_sphere

    if cache_path and os.path.exists(cache_path):
        print("  [PRISM] Loading cached ODF...")
        return nib.load(cache_path).get_fdata(dtype=np.float32)

    fiber_dirs = nib.load(os.path.join(prism_dir, "fiber_dirs.nii.gz")).get_fdata(np.float32)
    f_wm = nib.load(os.path.join(prism_dir, "f_wm_fibers.nii.gz")).get_fdata(np.float32)

    if fiber_dirs.ndim == 4 and fiber_dirs.shape[-1] != 3:
        n_fib = fiber_dirs.shape[-1] // 3
        fiber_dirs = fiber_dirs.reshape(fiber_dirs.shape[:3] + (n_fib, 3))

    norms = np.linalg.norm(fiber_dirs, axis=-1, keepdims=True)
    fiber_dirs = fiber_dirs / np.where(norms > 0, norms, 1.0)

    sphere = default_sphere
    C = kappa / (4 * np.pi * np.sinh(kappa))
    odf = np.zeros(fiber_dirs.shape[:3] + (len(sphere.vertices),), dtype=np.float64)
    verts = sphere.vertices.T

    for f in range(fiber_dirs.shape[3]):
        w = f_wm[..., f]
        mask = w > 0.01
        if not mask.any():
            continue
        cos_theta = np.abs(np.einsum('...d,dv->...v', fiber_dirs[..., f, :], verts))
        np.clip(cos_theta, 0, 1, out=cos_theta)
        odf += w[..., None] * C * np.exp(kappa * cos_theta ** 2)

    odf = np.maximum(odf, 0).astype(np.float32)
    if cache_path:
        nib.save(nib.Nifti1Image(odf, affine), cache_path)
    return odf


def fit_csd_odf(data, affine, bvals, bvecs, mask, cache_path=None):
    from dipy.data import default_sphere
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.direction import peaks_from_model
    from dipy.reconst.shm import sh_to_sf_matrix

    if cache_path and os.path.exists(cache_path):
        print("  [CSD] Loading cached ODF...")
        return nib.load(cache_path).get_fdata(dtype=np.float32)

    gtab = gradient_table(bvals, bvecs)
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    print(f"  [CSD] Response ratio = {ratio:.3f}")

    csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    peaks = peaks_from_model(csd, data, default_sphere,
                             relative_peak_threshold=0.15,
                             min_separation_angle=25, mask=mask,
                             npeaks=5, parallel=True, num_processes=8)

    B, _ = sh_to_sf_matrix(default_sphere, sh_order_max=8,
                            basis_type=None, return_inv=True)
    odf = np.maximum(np.dot(peaks.shm_coeff, B.T), 0).astype(np.float32)

    if cache_path:
        nib.save(nib.Nifti1Image(odf, affine), cache_path)
    return odf


def fit_odffp_odf(data, affine, bvals, bvecs, mask, cache_path=None):
    from dipy.data import default_sphere
    from dipy.core.gradients import gradient_table

    if cache_path and os.path.exists(cache_path):
        print("  [ODF-FP] Loading cached ODF...")
        return nib.load(cache_path).get_fdata(dtype=np.float32)

    try:
        from dipy.reconst.odf_fingerprinting import (
            OdffpDictionary, OdffpModel, OdffpTessellation)
    except ImportError:
        print("  [ODF-FP] ⚠ dipy.reconst.odf_fingerprinting not available!")
        return None

    gtab = gradient_table(bvals, bvecs)
    tess = OdffpTessellation()
    odf_dict = OdffpDictionary(gtab, tessellation=tess)
    odf_dict.generate(dict_size=50000, max_peaks_num=3,
                      D_a=[1.2, 2.2], D_e=[0.1, 0.6], D_r=[0.1, 0.6])

    model = OdffpModel(gtab, odf_dict)
    fit = model.fit(data, mask=mask)
    raw_odf = fit.odf()

    n_default = len(default_sphere.vertices)
    if raw_odf.shape[-1] == n_default:
        odf = raw_odf.astype(np.float32)
    else:
        n_tess = len(tess.vertices)
        if raw_odf.shape[-1] < n_tess:
            full_odf = np.concatenate([raw_odf, raw_odf], axis=-1)
        else:
            full_odf = raw_odf
        from scipy.spatial import cKDTree
        tree = cKDTree(tess.vertices[:full_odf.shape[-1]])
        _, nn_idx = tree.query(default_sphere.vertices)
        odf = full_odf[..., nn_idx].astype(np.float32)

    odf = np.maximum(odf, 0)
    if cache_path:
        nib.save(nib.Nifti1Image(odf, affine), cache_path)
    return odf



def compute_gfa(odf):
    mean_odf = odf.mean(axis=-1, keepdims=True)
    n = odf.shape[-1]
    var = np.sum((odf - mean_odf) ** 2, axis=-1)
    total = np.sum(odf ** 2, axis=-1)
    return np.sqrt(n * var / ((n - 1) * np.maximum(total, 1e-12)))


def run_tracking(odf, affine, mask, method_name, cache_trk=None, ref_img=None):
    from dipy.data import default_sphere
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking import utils as tracking_utils
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram, load_tractogram
    from nibabel.streamlines import ArraySequence as Streamlines

    if cache_trk and os.path.exists(cache_trk):
        print(f"  [{method_name}] Loading cached tractogram...")
        sft = load_tractogram(cache_trk, ref_img, bbox_valid_check=False)
        print(f"  [{method_name}] {len(sft.streamlines):,} streamlines")
        return sft.streamlines

    p = TRACK_PARAMS
    gfa = compute_gfa(odf)
    stopping = np.where(mask > 0, gfa, 0.0).astype(np.float64)
    sc = ThresholdStoppingCriterion(stopping, p['fa_threshold'])

    pmf = odf.clip(min=0).astype(np.float64)
    pmf_sum = pmf.sum(axis=-1, keepdims=True)
    pmf = np.where(pmf_sum > 1e-6, pmf / pmf_sum, 0.0)

    dg = DeterministicMaximumDirectionGetter.from_pmf(
        pmf, max_angle=p['max_angle'], sphere=default_sphere)

    seeds = tracking_utils.seeds_from_mask(mask, affine, density=p['seed_density'])
    print(f"  [{method_name}] {len(seeds):,} seeds")

    t0 = time.time()
    streamlines = Streamlines(LocalTracking(
        dg, sc, seeds, affine,
        step_size=p['step_size'],
        maxlen=int(p['max_length'] / p['step_size']),
        minlen=int(p['min_length'] / p['step_size']),
        return_all=False, random_seed=42,
    ))
    print(f"  [{method_name}] {len(streamlines):,} streamlines in {time.time()-t0:.1f}s")

    if cache_trk and ref_img is not None:
        sft = StatefulTractogram(streamlines, ref_img, Space.RASMM)
        save_tractogram(sft, cache_trk, bbox_valid_check=False)
        print(f"  [{method_name}] Saved: {cache_trk}")

    return streamlines



def extract_bundles(streamlines, ref_img, method_name, cache_dir=None,
                    fallback_cache=None):
    from dipy.align.streamlinear import whole_brain_slr
    from dipy.segment.bundles import RecoBundles
    from dipy.data import get_bundle_atlas_hcp842
    from dipy.io.streamline import load_tractogram, save_tractogram
    from dipy.io.stateful_tractogram import Space, StatefulTractogram

    extracted = {}
    all_cached = True

    for bname in BUNDLE_NAMES:
        found = False
        for cdir in [cache_dir, fallback_cache]:
            if cdir is None:
                continue
            bpath = os.path.join(cdir, f"{method_name}_{bname}.trk")
            if os.path.exists(bpath):
                sft = load_tractogram(bpath, ref_img, bbox_valid_check=False)
                extracted[bname] = sft.streamlines
                found = True
                break
        if not found:
            all_cached = False

    if all_cached:
        print(f"  [{method_name}] All 12 bundles loaded from cache")
        for bname in BUNDLE_NAMES:
            print(f"    {bname}: {len(extracted[bname]):,}")
        return extracted

    atlas_file, all_bundles_pattern = get_bundle_atlas_hcp842()
    atlas_sft = load_tractogram(str(atlas_file), 'same', bbox_valid_check=False)

    print(f"  [{method_name}] Whole-brain SLR alignment...")
    t0 = time.time()
    moved, transform, _, _ = whole_brain_slr(
        atlas_sft.streamlines, streamlines,
        x0='affine', verbose=False, progressive=True)
    print(f"  [{method_name}] SLR done in {time.time()-t0:.1f}s")

    rb = RecoBundles(moved, greater_than=50, less_than=1000000, verbose=False)
    bundles_dir = os.path.dirname(str(all_bundles_pattern).replace('*.trk', ''))

    for bname in BUNDLE_NAMES:
        model_file = os.path.join(bundles_dir, f"{bname}.trk")
        if not os.path.exists(model_file):
            print(f"  [{method_name}] ⚠ {bname}: model not found")
            continue
        model_sft = load_tractogram(model_file, 'same', bbox_valid_check=False)
        try:
            recognized, labels = rb.recognize(
                model_sft.streamlines,
                model_clust_thr=5, reduction_thr=10, pruning_thr=8, slr=True)
            if len(recognized) > 0:
                extracted[bname] = streamlines[labels]
                print(f"  [{method_name}] {bname}: {len(recognized):,}")
                save_dir = fallback_cache or cache_dir
                if save_dir:
                    sft = StatefulTractogram(extracted[bname], ref_img, Space.RASMM)
                    save_tractogram(sft, os.path.join(
                        save_dir, f"{method_name}_{bname}.trk"),
                        bbox_valid_check=False)
            else:
                print(f"  [{method_name}] {bname}: 0 (not found)")
        except Exception as e:
            print(f"  [{method_name}] {bname}: failed ({e})")

    return extracted


def load_atlas_bundles():
    from dipy.data import get_bundle_atlas_hcp842
    from dipy.io.streamline import load_tractogram

    atlas_file, all_bundles_pattern = get_bundle_atlas_hcp842()
    bundles_dir = os.path.dirname(str(all_bundles_pattern).replace('*.trk', ''))

    atlas = {}
    for bname in BUNDLE_NAMES:
        model_file = os.path.join(bundles_dir, f"{bname}.trk")
        if os.path.exists(model_file):
            sft = load_tractogram(model_file, 'same', bbox_valid_check=False)
            atlas[bname] = sft.streamlines
            print(f"  [Atlas] {bname}: {len(sft.streamlines):,}")
    return atlas



def compute_mdf_to_atlas(method_bundles, atlas_bundles, n_points=20, max_sl=2000):
    from dipy.tracking.streamline import (set_number_of_points,
                                           bundles_distances_mdf)

    results = {}
    for bname in BUNDLE_NAMES:
        if bname not in method_bundles or bname not in atlas_bundles:
            continue
        sl_m = method_bundles[bname]
        sl_a = atlas_bundles[bname]
        if len(sl_m) == 0 or len(sl_a) == 0:
            continue

        idx_m = np.random.choice(len(sl_m), min(max_sl, len(sl_m)), replace=False)
        idx_a = np.random.choice(len(sl_a), min(max_sl, len(sl_a)), replace=False)
        sl_m_sub = set_number_of_points([sl_m[i] for i in idx_m], n_points)
        sl_a_sub = set_number_of_points([sl_a[i] for i in idx_a], n_points)

        dist = bundles_distances_mdf(sl_m_sub, sl_a_sub)
        mean_mdf = float(dist.min(axis=1).mean())
        results[bname] = mean_mdf

    return results


def compute_bundle_shape_similarity(method_bundles, atlas_bundles, max_sl=3000):
    from dipy.segment.bundles import bundle_shape_similarity

    rng = np.random.default_rng(42)
    results = {}
    for bname in BUNDLE_NAMES:
        if bname not in method_bundles or bname not in atlas_bundles:
            continue
        sl_m = method_bundles[bname]
        sl_a = atlas_bundles[bname]
        if len(sl_m) < 10 or len(sl_a) < 10:
            continue

        if len(sl_m) > max_sl:
            idx = np.random.choice(len(sl_m), max_sl, replace=False)
            sl_m = [sl_m[i] for i in idx]
        if len(sl_a) > max_sl:
            idx = np.random.choice(len(sl_a), max_sl, replace=False)
            sl_a = [sl_a[i] for i in idx]

        try:
            ba = bundle_shape_similarity(sl_m, sl_a, rng,
                                          clust_thr=(5, 3, 1.5), threshold=6)
            results[bname] = float(ba)
        except Exception as e:
            print(f"    ⚠ {bname} BA failed: {e}")

    return results


def compute_buan_profiles(method_bundles, atlas_bundles, n_disks=100, n_points=20):
    from dipy.stats.analysis import assignment_map
    from dipy.tracking.streamline import set_number_of_points

    profiles = {}
    for bname in BUNDLE_NAMES:
        if bname not in method_bundles or bname not in atlas_bundles:
            continue
        sl_m = method_bundles[bname]
        sl_a = atlas_bundles[bname]
        if len(sl_m) < 5 or len(sl_a) < 5:
            continue

        sl_m_rs = set_number_of_points(sl_m, n_points)
        sl_a_rs = set_number_of_points(sl_a, n_points)

        try:
            indx = assignment_map(sl_m_rs, sl_a_rs, n_disks)

            disk_counts = np.zeros(n_disks)
            for sl_idx in range(len(sl_m_rs)):
                if sl_idx >= len(indx):
                    continue
                val = indx[sl_idx]
                val = np.atleast_1d(np.asarray(val))
                for disk_idx in val:
                    d = int(disk_idx)
                    if 0 <= d < n_disks:
                        disk_counts[d] += 1

            if disk_counts.max() > 0:
                density = disk_counts / disk_counts.max()
            else:
                density = disk_counts

            profiles[bname] = {
                'density': density.tolist(),
                'n_streamlines': len(sl_m),
                'n_disks': n_disks,
            }
        except Exception as e:
            print(f"    ⚠ BUAN {bname}: {e}")

    return profiles



def load_tractcloud_model(device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def tract_knn(x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pw = -xx - inner - xx.transpose(2, 1)
        return pw.topk(k=k, dim=-1)[1]

    def get_graph_feat(x, k_point_level=15, dev=device):
        nf, nd, np_ = x.size()
        if k_point_level >= np_:
            idx = torch.arange(0, np_)[None, None, :].repeat(nf, np_, 1).to(dev)
        else:
            idx = tract_knn(x, k=k_point_level)
        idx_base = torch.arange(0, nf, device=dev).view(-1, 1, 1) * np_
        idx = (idx + idx_base).view(-1)
        x = x.transpose(2, 1).contiguous()
        feat = x.view(nf * np_, -1)[idx, :]
        feat = feat.view(nf, np_, k_point_level, nd)
        x = x.view(nf, np_, 1, nd).repeat(1, 1, k_point_level, 1)
        return torch.cat((feat - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    class DGCNN(nn.Module):
        def __init__(self, num_classes=1600, k=20, k_global=500,
                     k_point_level=5, emb_dims=1024, dropout=0.5):
            super().__init__()
            self.fiber_level_k = k
            self.fiber_level_k_global = k_global
            self.k_point_level = k_point_level

            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(emb_dims)

            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False),
                                        self.bn1, nn.LeakyReLU(0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False),
                                        self.bn2, nn.LeakyReLU(0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                        self.bn3, nn.LeakyReLU(0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                        self.bn4, nn.LeakyReLU(0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False),
                                        self.bn5, nn.LeakyReLU(0.2))

            self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(dropout)
            self.linear3 = nn.Linear(256, num_classes)

        def forward(self, x, info):
            nf = x.size(0)
            if self.fiber_level_k + self.fiber_level_k_global == 0:
                x = get_graph_feat(x, self.k_point_level)
            else:
                x = x[:, :, :, None].repeat(
                    1, 1, 1, self.fiber_level_k + self.fiber_level_k_global)
                x = torch.cat((info - x, x), dim=1)

            x = self.conv1(x)
            x1 = x.max(dim=-1)[0]
            x = get_graph_feat(x1, self.k_point_level)
            x = self.conv2(x)
            x2 = x.max(dim=-1)[0]
            x = get_graph_feat(x2, self.k_point_level)
            x = self.conv3(x)
            x3 = x.max(dim=-1)[0]
            x = get_graph_feat(x3, self.k_point_level)
            x = self.conv4(x)
            x4 = x.max(dim=-1)[0]

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = self.conv5(x)
            x1 = F.adaptive_max_pool1d(x, 1).view(nf, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(nf, -1)
            x = torch.cat((x1, x2), 1)

            x = F.leaky_relu(self.bn6(self.linear1(x)), 0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), 0.2)
            x = self.dp2(x)
            return F.log_softmax(self.linear3(x), dim=1)

    model = DGCNN(num_classes=1600, k=20, k_global=500)
    weights = torch.load(TC_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    print(f"  TractCloud model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def run_tractcloud(streamlines, model, device, method_name,
                   cache_path=None, k=20, k_global=500, batch_size=2048):
    import torch
    from dipy.tracking.streamline import set_number_of_points
    from collections import defaultdict
    import pandas as pd

    if cache_path and os.path.exists(cache_path):
        print(f"  [{method_name}] Loading cached TractCloud predictions...")
        return np.load(cache_path)

    feat_ras = np.array(set_number_of_points(streamlines, 15), dtype=np.float32)
    hcp_center = np.load(TC_MASS_CENTER)
    subj_center = feat_ras.mean(axis=0)
    feat_ras = feat_ras + (hcp_center - subj_center)

    n_fiber = len(feat_ras)
    g_idx = np.random.randint(0, n_fiber, k_global)
    global_feat = feat_ras[g_idx].transpose(1, 2, 0)[None].astype(np.float32)
    global_feat_t = torch.from_numpy(global_feat)

    all_preds = []
    model.eval()
    chunk_size = min(10000, n_fiber)
    n_chunks = (n_fiber + chunk_size - 1) // chunk_size
    print(f"  [{method_name}] TractCloud inference: {n_chunks} chunks...")

    t0 = time.time()
    for ci in range(n_chunks):
        s, e = ci * chunk_size, min((ci + 1) * chunk_size, n_fiber)
        chunk = feat_ras[s:e]
        cn = e - s
        chunk_t = torch.from_numpy(chunk.transpose(0, 2, 1))

        def _mdf_dist(a, b):
            s1 = a.reshape(a.shape[0], -1)
            s2 = b.reshape(b.shape[0], -1)
            d = (s1 ** 2).sum(1).view(-1, 1) + (s2 ** 2).sum(1).view(1, -1) \
                - 2.0 * torch.mm(s1, s2.t())
            return torch.sqrt(torch.clamp(d, 0.0)) / 15

        n_ds = max(k + 1, int(cn * 0.1))
        ds_idx = np.random.choice(cn, min(n_ds, cn), replace=False)
        ds_feat = chunk_t[ds_idx]
        ds_flip = torch.flip(ds_feat, dims=[-1])

        d1 = _mdf_dist(chunk_t, ds_feat)
        d2 = _mdf_dist(chunk_t, ds_flip)
        mdf = torch.minimum(d1, d2)
        flip_mask = (d2 < d1).int()

        topk_idx = mdf.topk(k=k, largest=False, dim=-1)[1]
        near_flip = torch.gather(flip_mask, 1, topk_idx)

        near_org = ds_feat[topk_idx.reshape(-1)]
        near_eq = ds_flip[topk_idx.reshape(-1)]
        nf_flat = near_flip.reshape(-1)[:, None, None]
        cur_local = near_org * (1 - nf_flat) + near_eq * nf_flat
        local_feat = cur_local.reshape(cn, k, 3, 15).permute(0, 2, 3, 1).float()

        with torch.no_grad():
            for bi in range(0, cn, batch_size):
                be = min(bi + batch_size, cn)
                bs = be - bi
                pts = chunk_t[bi:be].float().to(device)
                kl = local_feat[bi:be].to(device)
                g = global_feat_t.repeat(bs, 1, 1, 1).permute(0, 2, 1, 3).to(device)
                info = torch.cat([kl, g], dim=-1)
                pred = model(pts, info)
                all_preds.extend(pred.argmax(dim=1).cpu().numpy().tolist())

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci+1}/{n_chunks} ({e:,}/{n_fiber:,}, {time.time()-t0:.1f}s)")

    df = pd.read_excel(TC_ANNOT_PATH)
    cluster_map = {int(row['Cluster Index']): str(row['Final'])
                   for _, row in df.iterrows()}

    ordered_names = TRACT_NAMES_ORDERED[:-1]
    name_to_idx = {n: i for i, n in enumerate(TRACT_NAMES_ORDERED)}

    tract_cluster_dict = defaultdict(list)
    for clu, tra in cluster_map.items():
        tract_cluster_dict[tra].append(clu)

    ordered_mapping = {}
    all_clusters = list(df['Cluster Index'])
    used = []
    for tn in ordered_names:
        ordered_mapping[tn] = tract_cluster_dict[tn]
        used.extend(tract_cluster_dict[tn])
    ordered_mapping['Other'] = sorted(list(set(all_clusters) - set(used)))

    preds = np.array(all_preds)
    org = preds.copy()
    n_tracts = len(ordered_mapping)
    for idx_t, tname in enumerate(ordered_mapping.keys()):
        for cname in ordered_mapping[tname]:
            cidx = int(str(cname).split('_')[1][2:]) - 1 if isinstance(cname, str) \
                else int(cname)
            preds[org == cidx] = idx_t
            preds[org == cidx + 800] = n_tracts - 1

    if cache_path:
        np.save(cache_path, preds)
        print(f"  [{method_name}] Saved predictions: {cache_path}")

    return preds



def render_bundle_panel(streamlines, color, cam_func, size=(450, 450)):
    from fury import actor, window

    scene = window.Scene()
    scene.background((1, 1, 1))

    if len(streamlines) > 0:
        colors = np.tile(color, (len(streamlines), 1))
        sl_actor = actor.line(streamlines, colors=colors,
                               opacity=0.65, linewidth=2.0,
                               fake_tube=True, depth_cue=True)
        scene.add(sl_actor)
        cam_func(scene, [streamlines])

    arr = window.snapshot(scene, size=size, offscreen=True)
    scene.clear()
    return arr


def set_camera_coronal(scene, sl_sets):
    pts = np.concatenate([np.concatenate([np.asarray(s) for s in ss], axis=0)
                          for ss in sl_sets if len(ss) > 0], axis=0)
    c = pts.mean(0)
    d = max(pts.max(0) - pts.min(0)) * 2.8
    scene.set_camera(position=(c[0], c[1] - d, c[2]),
                     focal_point=c, view_up=(0, 0, 1))


def set_camera_sagittal(scene, sl_sets):
    pts = np.concatenate([np.concatenate([np.asarray(s) for s in ss], axis=0)
                          for ss in sl_sets if len(ss) > 0], axis=0)
    c = pts.mean(0)
    d = max(pts.max(0) - pts.min(0)) * 2.8
    scene.set_camera(position=(c[0] + d, c[1], c[2]),
                     focal_point=c, view_up=(0, 0, 1))


def set_camera_axial(scene, sl_sets):
    pts = np.concatenate([np.concatenate([np.asarray(s) for s in ss], axis=0)
                          for ss in sl_sets if len(ss) > 0], axis=0)
    c = pts.mean(0)
    d = max(pts.max(0) - pts.min(0)) * 2.8
    scene.set_camera(position=(c[0], c[1], c[2] + d),
                     focal_point=c, view_up=(0, 1, 0))


def render_recobundles_mosaic(all_bundles, atlas_bundles, out_dir):
    from PIL import Image, ImageDraw, ImageFont

    PW, PH = 400, 400
    methods_present = [m for m in METHODS if m in all_bundles]
    n_cols = 1 + len(methods_present)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for view_name, cam in [('coronal', set_camera_coronal),
                           ('sagittal', set_camera_sagittal),
                           ('axial', set_camera_axial)]:
        rows = []
        valid_bundles = []
        for bname in BUNDLE_NAMES:
            color = np.array(BUNDLE_COLORS.get(bname, (0.5, 0.5, 0.5)))
            row_imgs = []

            sl_a = atlas_bundles.get(bname, [])
            if len(sl_a) == 0:
                continue
            valid_bundles.append(bname)
            row_imgs.append(render_bundle_panel(sl_a, color * 0.6, cam, (PW, PH)))

            for method in methods_present:
                sl_m = all_bundles[method].get(bname, [])
                row_imgs.append(render_bundle_panel(
                    sl_m if len(sl_m) > 0 else [], color, cam, (PW, PH)))

            rows.append(np.concatenate(row_imgs, axis=1))

        if not rows:
            continue
        mosaic = np.concatenate(rows, axis=0)

        label_h, label_w = 35, 90
        total_w = label_w + n_cols * PW
        total_h = label_h + len(valid_bundles) * PH

        canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        col_labels = ['Atlas GT'] + [METHOD_LABELS[m] for m in methods_present]
        for ci, label in enumerate(col_labels):
            x = label_w + ci * PW + PW // 2
            draw.text((x, 8), label, fill=(0, 0, 0), font=font, anchor='mt')

        for ri, bname in enumerate(valid_bundles):
            y = label_h + ri * PH + PH // 2
            draw.text((label_w // 2, y), bname.replace('_', '\n'),
                      fill=(0, 0, 0), font=font_sm, anchor='mm')

        mosaic_img = Image.fromarray(mosaic)
        canvas.paste(mosaic_img, (label_w, label_h))

        for ci in range(n_cols + 1):
            x = label_w + ci * PW
            draw.line([(x, label_h), (x, total_h)], fill=(200, 200, 200), width=1)
        for ri in range(len(valid_bundles) + 1):
            y = label_h + ri * PH
            draw.line([(label_w, y), (total_w, y)], fill=(200, 200, 200), width=1)

        out_path = os.path.join(out_dir, f"recobundles_{view_name}.png")
        canvas.save(out_path, dpi=(150, 150))
        print(f"  ✓ Saved: {out_path}")


def render_tractcloud_mosaic(streamlines_dict, predictions_dict, out_dir):
    from dipy.data import get_bundle_atlas_hcp842
    from dipy.io.streamline import load_tractogram
    from PIL import Image, ImageDraw, ImageFont

    PW, PH = 400, 400
    methods_present = [m for m in METHODS if m in predictions_dict]
    n_cols = 1 + len(methods_present)

    atlas_file, all_bundles_pattern = get_bundle_atlas_hcp842()
    atlas_dir = os.path.dirname(str(all_bundles_pattern).replace('*.trk', ''))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for view_name, cam in [('coronal', set_camera_coronal)]:
        rows = []
        valid_tracts = []

        for tract_disp, info in TC_BUNDLES.items():
            color = info['color']
            row_imgs = []

            atlas_sl = []
            for af in info['atlas_files']:
                fpath = os.path.join(atlas_dir, af)
                if os.path.exists(fpath):
                    sft = load_tractogram(fpath, 'same', bbox_valid_check=False)
                    atlas_sl.extend(sft.streamlines)

            if len(atlas_sl) == 0:
                continue
            valid_tracts.append(tract_disp)

            if len(atlas_sl) > 5000:
                idx = np.random.choice(len(atlas_sl), 5000, replace=False)
                atlas_sl = [atlas_sl[i] for i in idx]
            row_imgs.append(render_bundle_panel(atlas_sl, color * 0.6, cam, (PW, PH)))

            for method in methods_present:
                preds = predictions_dict[method]
                sl = streamlines_dict[method]
                mask = np.isin(preds, info['tc_indices'])
                indices = np.where(mask)[0]
                if len(indices) > 5000:
                    indices = np.random.choice(indices, 5000, replace=False)
                tc_sl = [sl[i] for i in indices] if len(indices) > 0 else []
                row_imgs.append(render_bundle_panel(tc_sl, color, cam, (PW, PH)))

            rows.append(np.concatenate(row_imgs, axis=1))

        if not rows:
            return
        mosaic = np.concatenate(rows, axis=0)

        label_h, label_w = 35, 80
        total_w = label_w + n_cols * PW
        total_h = label_h + len(valid_tracts) * PH

        canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        col_labels = ['Atlas GT'] + [METHOD_LABELS[m] for m in methods_present]
        for ci, label in enumerate(col_labels):
            x = label_w + ci * PW + PW // 2
            draw.text((x, 8), label, fill=(0, 0, 0), font=font, anchor='mt')

        for ri, tname in enumerate(valid_tracts):
            y = label_h + ri * PH + PH // 2
            draw.text((label_w // 2, y), tname,
                      fill=(0, 0, 0), font=font_sm, anchor='mm')

        canvas.paste(Image.fromarray(mosaic), (label_w, label_h))

        for ci in range(n_cols + 1):
            x = label_w + ci * PW
            draw.line([(x, label_h), (x, total_h)], fill=(200, 200, 200), width=1)
        for ri in range(len(valid_tracts) + 1):
            y = label_h + ri * PH
            draw.line([(label_w, y), (total_w, y)], fill=(200, 200, 200), width=1)

        out_path = os.path.join(out_dir, f"tractcloud_{view_name}.png")
        canvas.save(out_path, dpi=(150, 150))
        print(f"  ✓ Saved: {out_path}")



def plot_metrics_summary(all_mdf, all_ba, all_profiles, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods_present = [m for m in METHODS if m in all_mdf]
    if not methods_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    bundles_with_data = [b for b in BUNDLE_NAMES
                         if any(b in all_mdf.get(m, {}) for m in methods_present)]
    n_groups = len(bundles_with_data)
    n_bars = len(methods_present)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, method in enumerate(methods_present):
        offset = (i - n_bars / 2 + 0.5) * width
        vals = [all_mdf.get(method, {}).get(b, 0) for b in bundles_with_data]
        ax.bar(x + offset, vals, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS_HEX[method], alpha=0.85, edgecolor='#333',
               linewidth=0.4)

    ax.set_ylabel('Mean MDF Distance (mm)', fontsize=11)
    ax.set_title('MDF Distance to Atlas (lower = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in bundles_with_data],
                        rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    bundles_ba = [b for b in BUNDLE_NAMES
                  if any(b in all_ba.get(m, {}) for m in methods_present)]
    n_groups = len(bundles_ba)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    for i, method in enumerate(methods_present):
        offset = (i - n_bars / 2 + 0.5) * width
        vals = [all_ba.get(method, {}).get(b, 0) for b in bundles_ba]
        ax.bar(x + offset, vals, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS_HEX[method], alpha=0.85, edgecolor='#333',
               linewidth=0.4)

    ax.set_ylabel('Bundle Adjacency Score', fontsize=11)
    ax.set_title('Shape Similarity to Atlas (higher = better)', fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in bundles_ba],
                        rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics_mdf_ba.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: metrics_mdf_ba.png")

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, bname in enumerate(BUNDLE_NAMES):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.set_title(bname, fontsize=10, fontweight='bold')

        for method in methods_present:
            prof = all_profiles.get(method, {}).get(bname, None)
            if prof is None:
                continue
            density = np.array(prof['density'])
            x = np.linspace(0, 100, len(density))
            ax.plot(x, density, label=METHOD_LABELS[method],
                    color=METHOD_COLORS_HEX[method], linewidth=1.5, alpha=0.85)
            ax.fill_between(x, 0, density, color=METHOD_COLORS_HEX[method],
                            alpha=0.12)

        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Along-Tract Position (%)', fontsize=7)
        ax.set_ylabel('Normalized Density', fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)

    plt.suptitle('BUAN Along-Tract Density Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'buan_profiles.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: buan_profiles.png")


def plot_tractcloud_counts(predictions_dict, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods_present = [m for m in METHODS if m in predictions_dict]
    key_tracts = ['CST', 'AF', 'ILF', 'SLF-II', 'UF', 'CB', 'IOFF']
    display = key_tracts + ['CC (all)', 'SLF (all)', 'Other']

    def _count(preds, disp):
        if disp == 'CC (all)':
            return sum(int(np.sum(preds == TRACT_IDX.get(f'CC{i}', -1)))
                       for i in range(1, 8))
        elif disp == 'SLF (all)':
            return sum(int(np.sum(preds == TRACT_IDX.get(f'SLF-{x}', -1)))
                       for x in ['I', 'II', 'III'])
        else:
            idx = TRACT_IDX.get(disp, -1)
            return int(np.sum(preds == idx)) if idx >= 0 else 0

    n_groups = len(display)
    n_bars = len(methods_present)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, method in enumerate(methods_present):
        offset = (i - n_bars / 2 + 0.5) * width
        counts = [_count(predictions_dict[method], d) for d in display]
        ax.bar(x + offset, counts, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS_HEX[method], alpha=0.85, edgecolor='#333',
               linewidth=0.3)

    ax.set_ylabel('Streamline Count', fontsize=11)
    ax.set_title('TractCloud Parcellation - Registration-Free',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tractcloud_counts.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: tractcloud_counts.png")



def main():
    parser = argparse.ArgumentParser(
        description='HCP Tracking Benchmark - PRISM vs CSD vs ODF-FP')
    parser.add_argument('--skip-tractcloud', action='store_true',
                        help='Skip TractCloud parcellation step')
    parser.add_argument('--viz-only', action='store_true',
                        help='Only render visuals from cached data')
    parser.add_argument('--no-fury', action='store_true',
                        help='Skip FURY rendering (headless servers)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("=" * 70)
    print("  HCP TRACKING BENCHMARK")
    print("  Methods: PRISM, CSD, ODF-FP")
    print("  Subject: HCP 100307")
    print("=" * 70)

    print("\n[1/7] Loading HCP data...")
    ref_img = nib.load(os.path.join(HCP_DIR, "data.nii.gz"))
    affine = ref_img.affine

    if not args.viz_only:
        all_trk_cached = all(
            os.path.exists(os.path.join(CACHED_DIR, f"{m}_tractogram.trk"))
            for m in METHODS)
        if not all_trk_cached:
            data = ref_img.get_fdata(dtype=np.float32)
            bvals = np.loadtxt(os.path.join(HCP_DIR, "bvals"))
            bvecs = np.loadtxt(os.path.join(HCP_DIR, "bvecs"))
            if bvecs.shape[0] == 3:
                bvecs = bvecs.T
            mask_path = os.path.join(HCP_DIR, "nodif_brain_mask.nii.gz")
            if os.path.exists(mask_path):
                mask = nib.load(mask_path).get_fdata().astype(bool)
            else:
                mask = data[..., 0] > 50
            print(f"  Data: {data.shape}, Mask voxels: {mask.sum():,}")
        else:
            print("  All tractograms cached, skipping full data load")

    all_streamlines = {}
    if not args.viz_only:
        print("\n[2/7] Building ODFs and tracking...")
        os.makedirs(CACHED_DIR, exist_ok=True)

        for method in METHODS:
            print(f"\n--- {METHOD_LABELS[method]} ---")
            trk_cache = os.path.join(CACHED_DIR, f"{method}_tractogram.trk")

            if os.path.exists(trk_cache):
                from dipy.io.streamline import load_tractogram
                sft = load_tractogram(trk_cache, ref_img, bbox_valid_check=False)
                all_streamlines[method] = sft.streamlines
                print(f"  [{METHOD_LABELS[method]}] {len(sft.streamlines):,} streamlines (cached)")
                continue

            if method == 'prism':
                odf = build_prism_odf(PRISM_DIR, affine,
                                       cache_path=os.path.join(CACHED_DIR, "prism_odf.nii.gz"))
            elif method == 'csd':
                odf = fit_csd_odf(data, affine, bvals, bvecs, mask,
                                   cache_path=os.path.join(CACHED_DIR, "csd_odf.nii.gz"))
            elif method == 'odffp':
                odf = fit_odffp_odf(data, affine, bvals, bvecs, mask,
                                     cache_path=os.path.join(CACHED_DIR, "odffp_odf.nii.gz"))

            if odf is None:
                print(f"  [{method}] ⚠ ODF not available, skipping")
                continue

            sl = run_tracking(odf, affine, mask, METHOD_LABELS[method],
                              cache_trk=trk_cache, ref_img=ref_img)
            all_streamlines[method] = sl
            del odf
    else:
        from dipy.io.streamline import load_tractogram
        for method in METHODS:
            trk = os.path.join(CACHED_DIR, f"{method}_tractogram.trk")
            if os.path.exists(trk):
                sft = load_tractogram(trk, ref_img, bbox_valid_check=False)
                all_streamlines[method] = sft.streamlines
                print(f"  [{method}] {len(sft.streamlines):,} streamlines (cached)")

    print("\n[3/7] RecoBundles extraction...")
    all_bundles = {}
    bundle_cache = OUT_DIR
    os.makedirs(bundle_cache, exist_ok=True)
    for method in METHODS:
        if method not in all_streamlines:
            continue
        all_bundles[method] = extract_bundles(
            all_streamlines[method], ref_img, method,
            cache_dir=CACHED_DIR, fallback_cache=bundle_cache)

    print("\n  Loading atlas reference bundles...")
    atlas_bundles = load_atlas_bundles()

    print("\n[4/7] Computing metrics...")
    all_mdf = {}
    all_ba = {}
    all_profiles = {}

    for method in METHODS:
        if method not in all_bundles:
            continue
        print(f"\n  --- {METHOD_LABELS[method]} ---")

        print("  Computing MDF distances...")
        all_mdf[method] = compute_mdf_to_atlas(all_bundles[method], atlas_bundles)

        print("  Computing bundle shape similarity (BA)...")
        all_ba[method] = compute_bundle_shape_similarity(
            all_bundles[method], atlas_bundles)

        print("  Computing BUAN along-tract profiles...")
        all_profiles[method] = compute_buan_profiles(
            all_bundles[method], atlas_bundles, n_disks=BUAN_N_DISKS)

    print(f"\n{'='*80}")
    print(f"  METRICS SUMMARY")
    print(f"{'='*80}")
    methods_present = [m for m in METHODS if m in all_mdf]

    header = f"  {'Bundle':<22}"
    for m in methods_present:
        header += f"  MDF-{METHOD_LABELS[m]:>5}  BA-{METHOD_LABELS[m]:>5}"
    print(header)
    print(f"  {'-'*75}")

    for bname in BUNDLE_NAMES:
        row = f"  {bname:<22}"
        for m in methods_present:
            mdf = all_mdf[m].get(bname, float('nan'))
            ba = all_ba[m].get(bname, float('nan'))
            row += f"  {mdf:>8.2f}  {ba:>8.3f}"
        print(row)

    print(f"  {'-'*75}")
    row = f"  {'MEAN':<22}"
    for m in methods_present:
        mdf_vals = [v for v in all_mdf[m].values()]
        ba_vals = [v for v in all_ba[m].values()]
        row += f"  {np.mean(mdf_vals):>8.2f}  {np.mean(ba_vals):>8.3f}"
    print(row)

    predictions_dict = {}
    if not args.skip_tractcloud:
        print("\n[5/7] TractCloud parcellation...")
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        tc_model = None
        for method in METHODS:
            if method not in all_streamlines:
                continue
            cache_pred = os.path.join(TC_CACHED_DIR, f"{method}_tract_predictions.npy")
            if os.path.exists(cache_pred):
                predictions_dict[method] = np.load(cache_pred)
                print(f"  [{method}] Loaded cached predictions ({len(predictions_dict[method]):,})")
            else:
                if tc_model is None:
                    os.makedirs(TC_CACHED_DIR, exist_ok=True)
                    tc_model = load_tractcloud_model(device)
                predictions_dict[method] = run_tractcloud(
                    all_streamlines[method], tc_model, device,
                    METHOD_LABELS[method], cache_path=cache_pred)

        if predictions_dict:
            print(f"\n  TractCloud tract counts:")
            for method in METHODS:
                if method not in predictions_dict:
                    continue
                preds = predictions_dict[method]
                total = len(preds)
                other_idx = TRACT_IDX.get('Other', len(TRACT_NAMES_ORDERED) - 1)
                other = int(np.sum(preds == other_idx))
                assigned = total - other
                print(f"    {METHOD_LABELS[method]}: {assigned:,}/{total:,} assigned "
                      f"({100*assigned/total:.1f}%)")
    else:
        for method in METHODS:
            cache_pred = os.path.join(TC_CACHED_DIR, f"{method}_tract_predictions.npy")
            if os.path.exists(cache_pred):
                predictions_dict[method] = np.load(cache_pred)

    print("\n[6/7] Generating figures...")
    fig_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_metrics_summary(all_mdf, all_ba, all_profiles, fig_dir)

    if predictions_dict:
        plot_tractcloud_counts(predictions_dict, fig_dir)

    if not args.no_fury:
        print("\n[7/7] FURY rendering...")
        try:
            print("  Rendering RecoBundles mosaic...")
            render_recobundles_mosaic(all_bundles, atlas_bundles, fig_dir)

            if predictions_dict and all_streamlines:
                print("  Rendering TractCloud mosaic...")
                render_tractcloud_mosaic(all_streamlines, predictions_dict, fig_dir)
        except Exception as e:
            print(f"  ⚠ FURY rendering failed: {e}")
            print("  (Try running with a display or use --no-fury)")
    else:
        print("\n[7/7] FURY rendering skipped (--no-fury)")

    results = {
        'subject': 'HCP_100307',
        'methods': METHODS,
        'tracking_params': TRACK_PARAMS,
        'streamline_counts': {m: len(all_streamlines[m])
                               for m in METHODS if m in all_streamlines},
        'mdf_distances': {m: {k: round(v, 3) for k, v in d.items()}
                           for m, d in all_mdf.items()},
        'bundle_adjacency': {m: {k: round(v, 4) for k, v in d.items()}
                              for m, d in all_ba.items()},
        'recobundle_counts': {m: {k: len(v) for k, v in all_bundles[m].items()}
                               for m in METHODS if m in all_bundles},
    }
    if predictions_dict:
        results['tractcloud_assigned'] = {}
        for m in METHODS:
            if m not in predictions_dict:
                continue
            p = predictions_dict[m]
            other_idx = TRACT_IDX.get('Other', len(TRACT_NAMES_ORDERED) - 1)
            results['tractcloud_assigned'][m] = {
                'total': int(len(p)),
                'assigned': int(np.sum(p != other_idx)),
                'pct': round(100 * float(np.sum(p != other_idx)) / len(p), 1),
            }

    json_path = os.path.join(OUT_DIR, "tracking_benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {json_path}")

    print(f"\n{'='*70}")
    print(f"  TRACKING BENCHMARK COMPLETE")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
