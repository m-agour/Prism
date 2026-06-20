#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
from dipy.data import default_sphere
from dipy.reconst.shm import sf_to_sh
from fury import actor, window

AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def load_fit(fit_dir, n_fibers):
    fit_dir = Path(fit_dir)
    dirs = nib.load(str(fit_dir / "fiber_dirs.nii.gz")).get_fdata()
    dirs = dirs.reshape(dirs.shape[:3] + (n_fibers, 3))
    f_fib = nib.load(str(fit_dir / "f_wm_fibers.nii.gz")).get_fdata()
    f_wm = nib.load(str(fit_dir / "f_wm.nii.gz")).get_fdata()
    return dirs, f_fib, f_wm


def pick_slice(dirs, f_fib, axis, peak_thr):
    in_plane = (np.abs(dirs[..., axis]) < 0.6) & (f_fib > peak_thr)
    crossings = in_plane.sum(-1) >= 2
    counts = [crossings.take(s, axis=axis).sum() for s in range(crossings.shape[axis])]
    return int(np.argmax(counts))


def build_glyphs(dirs_s, f_fib_s, f_wm_s, wm_thr, peak_thr, kappa, radius=0.5):
    verts = default_sphere.vertices
    centers, amps = [], []
    for ia in range(f_wm_s.shape[0]):
        for ib in range(f_wm_s.shape[1]):
            if f_wm_s[ia, ib] < wm_thr:
                continue
            f = f_fib_s[ia, ib]
            keep = f > peak_thr
            if keep.sum() == 0:
                continue
            sf = (np.exp(kappa * ((verts @ dirs_s[ia, ib][keep].T) ** 2 - 1.0)) * f[keep][None, :]).sum(1)
            sf = sf / (sf.max() + 1e-9) * radius
            centers.append([ia, ib, 0.0])
            amps.append(sf)
    return np.array(centers, np.float32), np.array(amps, np.float32)


def render(centers, amps, out, box, zoom, size, sh_order):
    coeffs = sf_to_sh(amps, default_sphere, sh_order_max=sh_order, basis_type="descoteaux07").astype(np.float32)
    cx, cy = centers[:, 0].mean(), centers[:, 1].mean()
    glyphs = actor.odf(centers=centers.copy(), coeffs=coeffs, sh_basis="descoteaux", scales=box)
    scene = window.Scene()
    scene.background((0, 0, 0))
    scene.add(glyphs)
    scene.set_camera(position=(cx, cy, 300), focal_point=(cx, cy, 0), view_up=(0, 1, 0))
    scene.reset_camera()
    scene.zoom(zoom)
    window.record(scene=scene, out_path=out, size=(size, size), reset_camera=False)


def main():
    p = argparse.ArgumentParser(description="Render fiber-peak ODF glyphs from a PRISM fit directory.")
    p.add_argument("--fit-dir", required=True, help="directory holding fiber_dirs/f_wm_fibers/f_wm nii.gz")
    p.add_argument("--n-fibers", type=int, default=5)
    p.add_argument("--axis", choices=list(AXES), default="axial")
    p.add_argument("--slice", type=int, default=-1, help="slice index; -1 picks the slice with most in-plane crossings")
    p.add_argument("--wm-thr", type=float, default=0.1)
    p.add_argument("--peak-thr", type=float, default=0.04)
    p.add_argument("--kappa", type=float, default=7.0)
    p.add_argument("--sh-order", type=int, default=8)
    p.add_argument("--box", type=float, default=2.0)
    p.add_argument("--zoom", type=float, default=1.6)
    p.add_argument("--size", type=int, default=1600)
    p.add_argument("--out", default="odf.png")
    args = p.parse_args()

    axis = AXES[args.axis]
    dirs, f_fib, f_wm = load_fit(args.fit_dir, args.n_fibers)
    idx = args.slice if args.slice >= 0 else pick_slice(dirs, f_fib, axis, args.peak_thr)

    dirs_s = np.take(dirs, idx, axis=axis)
    f_fib_s = np.take(f_fib, idx, axis=axis)
    f_wm_s = np.take(f_wm, idx, axis=axis)

    centers, amps = build_glyphs(dirs_s, f_fib_s, f_wm_s, args.wm_thr, args.peak_thr, args.kappa)
    if len(centers) == 0:
        sys.exit("no glyphs above threshold; lower --wm-thr or --peak-thr")

    render(centers, amps, args.out, args.box, args.zoom, args.size, args.sh_order)
    print(f"{args.axis} slice {idx}: {len(centers)} glyphs -> {args.out}")


if __name__ == "__main__":
    main()
