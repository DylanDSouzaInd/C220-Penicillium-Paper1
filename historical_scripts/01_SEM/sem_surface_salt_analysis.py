#!/usr/bin/env python3
"""
SEM bright salt-lining analysis for fungal mycelia.

Purpose
-------
Quantify bright deposits / lining associated with the mycelial surface in SEM images.
This is intended as a *supportive morphology analysis* for salt-associated extracellular
coating, not a direct chemical measurement of EPS.

Core idea
---------
1. Read JEOL TIFF + .txt metadata.
2. Remove the footer/annotation strip using JEOL metadata.
3. Build a structural foreground mask (mycelial scaffold) from a smoothed image.
4. Define a narrow boundary/rim zone around the scaffold in physical units (µm).
5. Detect bright small-scale deposits using a white top-hat / high-pass transform.
6. Restrict bright deposits to the mycelial rim and quantify:
   - bright rim fraction
   - coated perimeter fraction
   - bright rim area per 100 µm perimeter
   - bright puncta per 100 µm perimeter
   - rim high-pass mean
7. Optionally combine these into a descriptive composite:
   salt_lining_brightness_index

Statistics
----------
- Kruskal-Wallis across groups
- pairwise Mann-Whitney U
- Benjamini-Hochberg FDR correction

Important caution
-----------------
If multiple images come from the same biological sample/culture, image-level p-values are
optimistic. Use --aggregate-column biorep (or another replicate column) with a sample sheet
to aggregate images first and avoid pseudoreplication.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from scipy import ndimage as ndi
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from skimage import exposure, filters, measure, morphology, segmentation, util
from skimage.morphology import disk


# ---------------------------- utilities ----------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def make_odd(n: int) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def robust_norm(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_q, hi_q])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y.astype(np.float32)


def robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - med) / scale


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(list(pvals), dtype=float)
    if len(pvals) == 0:
        return np.array([], dtype=float)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    return qvals


def safe_ratio(num: float, den: float, eps: float = 1e-9) -> float:
    return float(num / (den + eps))


def infer_group_from_name(name: str) -> Optional[str]:
    """
    Infer 0 / 10 / 20 group from filenames like:
      0% DSW_1.tif
      10%DSW_1.tif
      20%DSW_12_50um.tif
    """
    m = re.search(r'(?i)(^|[^0-9])(0|10|20)\s*%?\s*DSW', name)
    if m:
        return m.group(2)
    # fallback: plain leading 0/10/20 before DSW
    m = re.search(r'(?i)(0|10|20).*DSW', name)
    if m:
        return m.group(1)
    return None


# ---------------------------- metadata ----------------------------

@dataclass
class JeolMeta:
    image_width_px: Optional[int] = None
    image_height_px: Optional[int] = None
    footer_height_px: Optional[int] = None
    mag: Optional[float] = None
    fov_width_um: Optional[float] = None
    fov_height_um: Optional[float] = None
    detector: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

    @property
    def um_per_px(self) -> Optional[float]:
        if self.fov_width_um is None or self.image_width_px is None:
            return None
        return self.fov_width_um / self.image_width_px


def parse_jeol_txt(path: Path) -> JeolMeta:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    meta = JeolMeta()
    def grab(pattern: str, cast=str):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    meta.image_width_px = grab(r"\$CM_IMAGE_SIZE\s+(\d+)\s+\d+", int)
    meta.image_height_px = grab(r"\$CM_IMAGE_SIZE\s+\d+\s+(\d+)", int)
    meta.footer_height_px = grab(r"\$SM_PNU_HEIGHT\s+(\d+)", int)
    meta.mag = grab(r"\$CM_MAG\s+([0-9.]+)", float)
    meta.fov_width_um = grab(r"\$CM_FIELD_OF_VIEW\s+([0-9.]+)µm\s+[0-9.]+µm", float)
    meta.fov_height_um = grab(r"\$CM_FIELD_OF_VIEW\s+[0-9.]+µm\s+([0-9.]+)µm", float)
    meta.detector = grab(r"\$SM_DETECTOR\s+([^\n\r]+)", str)
    meta.date = grab(r"\$CM_DATE\s+([^\n\r]+)", str)
    meta.time = grab(r"\$CM_TIME\s+([^\n\r]+)", str)
    return meta


# ---------------------------- image handling ----------------------------

def read_gray_image(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # JEOL tifs often read as RGBA with identical RGB channels.
        gray = arr[..., 0].astype(np.float32)
    else:
        gray = arr.astype(np.float32)
    return gray


def crop_footer(gray: np.ndarray, meta: JeolMeta) -> np.ndarray:
    """
    Prefer explicit JEOL image size / footer height. Fallback to raw height - 64.
    """
    h_raw, w_raw = gray.shape[:2]

    if meta.image_height_px is not None:
        h_keep = meta.image_height_px
        if 0 < h_keep <= h_raw:
            return gray[:h_keep, :]
    if meta.footer_height_px is not None:
        h_keep = h_raw - meta.footer_height_px
        if 0 < h_keep <= h_raw:
            return gray[:h_keep, :]
    # fallback
    return gray[: max(1, h_raw - 64), :]


def px_from_um(um: float, um_per_px: float, minimum: int = 1) -> int:
    if um_per_px <= 0:
        return minimum
    return max(minimum, int(round(um / um_per_px)))


def area_px_from_um2(area_um2: float, um_per_px: float, minimum: int = 1) -> int:
    if um_per_px <= 0:
        return minimum
    return max(minimum, int(round(area_um2 / (um_per_px ** 2))))


# ---------------------------- core analysis ----------------------------

@dataclass
class Params:
    body_blur_um: float = 0.70
    body_close_um: float = 0.35
    min_object_um2: float = 0.20
    fill_hole_um2: float = 0.20
    rim_in_um: float = 0.50
    rim_out_um: float = 0.80
    core_min_um: float = 0.80
    bg_min_um: float = 1.20
    bg_max_um: float = 3.00
    bright_tophat_um: float = 0.60
    bright_threshold_mad: float = 2.75
    bright_threshold_q: float = 0.995
    puncta_min_um2: float = 0.03
    coated_neighborhood_um: float = 0.60
    save_overlay_limit: int = 999999


def structural_mask(gray_norm: np.ndarray, um_per_px: float, p: Params) -> np.ndarray:
    """
    Segment the broader mycelial scaffold, not the tiny salt granules.

    We deliberately smooth in physical units before thresholding so that the
    mask follows the hyphal body/coarse scaffold rather than every bright speck.
    """
    sigma_px = max(0.5, p.body_blur_um / um_per_px)
    blur = filters.gaussian(gray_norm, sigma=sigma_px, preserve_range=True)

    # Global threshold is usually more stable across SEM than local Sauvola here.
    try:
        thr = filters.threshold_otsu(blur)
    except Exception:
        thr = float(np.median(blur))

    mask = blur > thr

    close_px = px_from_um(p.body_close_um, um_per_px, minimum=1)
    if close_px > 0:
        mask = morphology.closing(mask, morphology.disk(close_px))

    min_obj_px = area_px_from_um2(p.min_object_um2, um_per_px, minimum=4)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_obj_px)

    fill_hole_px = area_px_from_um2(p.fill_hole_um2, um_per_px, minimum=4)
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=fill_hole_px)

    # Keep only reasonably large connected regions.
    if mask.any():
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        if props:
            keep = np.zeros_like(mask, dtype=bool)
            for rp in props:
                if rp.area >= min_obj_px:
                    keep[labels == rp.label] = True
            mask = keep

    return mask.astype(bool)


def bright_deposit_map(gray_norm: np.ndarray, um_per_px: float, p: Params) -> np.ndarray:
    """
    High-pass / top-hat image highlighting bright small-scale deposits and bright rims.
    """
    radius_px = px_from_um(p.bright_tophat_um, um_per_px, minimum=1)
    hp = morphology.white_tophat(gray_norm, footprint=disk(radius_px))
    hp = hp.astype(np.float32)
    return hp


def analyze_one_image(tif_path: Path, txt_path: Path, group: str, sample_id: Optional[str], p: Params) -> Dict[str, float]:
    meta = parse_jeol_txt(txt_path)
    gray_raw = read_gray_image(tif_path)
    gray = crop_footer(gray_raw, meta).astype(np.float32)

    if meta.um_per_px is None:
        raise ValueError(f"Could not compute µm/px from metadata: {txt_path}")
    um_per_px = float(meta.um_per_px)

    # normalize after footer crop; avoid absolute brightness because JEOL brightness settings vary
    gray_norm = robust_norm(gray)

    # build structural foreground
    mask = structural_mask(gray_norm, um_per_px, p)

    if mask.sum() < 20:
        raise ValueError(f"Structural mask too small for {tif_path.name}")

    # distances in µm
    dist_in_um = ndi.distance_transform_edt(mask) * um_per_px
    dist_out_um = ndi.distance_transform_edt(~mask) * um_per_px

    rim_mask = ((mask) & (dist_in_um <= p.rim_in_um)) | ((~mask) & (dist_out_um > 0) & (dist_out_um <= p.rim_out_um))
    core_mask = (mask) & (dist_in_um >= p.core_min_um)
    if core_mask.sum() < 20:
        # fallback for thin structures
        core_mask = mask & (dist_in_um >= max(um_per_px, p.core_min_um * 0.5))
    bg_mask = (~mask) & (dist_out_um >= p.bg_min_um) & (dist_out_um <= p.bg_max_um)
    if bg_mask.sum() < 50:
        bg_mask = (~mask) & (dist_out_um >= p.bg_min_um)

    hp = bright_deposit_map(gray_norm, um_per_px, p)

    hp_bg = hp[bg_mask] if bg_mask.any() else hp.ravel()
    hp_bg_med = float(np.median(hp_bg))
    hp_bg_mad = float(1.4826 * np.median(np.abs(hp_bg - np.median(hp_bg))))
    hp_q = float(np.quantile(hp_bg, min(0.9999, max(0.5, p.bright_threshold_q))))
    bright_thr = max(hp_q, hp_bg_med + p.bright_threshold_mad * max(hp_bg_mad, 1e-6))
    bright_mask = hp > bright_thr

    # restrict to mycelial rim
    bright_rim_mask = bright_mask & rim_mask

    puncta_min_px = area_px_from_um2(p.puncta_min_um2, um_per_px, minimum=2)
    if puncta_min_px > 1:
        bright_rim_mask = morphology.remove_small_objects(bright_rim_mask.astype(bool), min_size=puncta_min_px)

    boundary = segmentation.find_boundaries(mask, mode="inner")
    perimeter_px = int(boundary.sum())
    perimeter_um = perimeter_px * um_per_px
    mask_area_um2 = float(mask.sum() * (um_per_px ** 2))
    bright_rim_area_um2 = float(bright_rim_mask.sum() * (um_per_px ** 2))
    rim_area_um2 = float(rim_mask.sum() * (um_per_px ** 2))

    # coated perimeter fraction: what fraction of the structural boundary has bright deposit nearby?
    coat_neigh_px = px_from_um(p.coated_neighborhood_um, um_per_px, minimum=1)
    bright_near_boundary = ndi.binary_dilation(bright_rim_mask, structure=disk(coat_neigh_px)) & boundary
    coated_perimeter_fraction = safe_ratio(float(bright_near_boundary.sum()), float(boundary.sum()))

    # connected puncta statistics
    puncta_labels = measure.label(bright_rim_mask)
    puncta_props = [rp for rp in measure.regionprops(puncta_labels) if rp.area >= puncta_min_px]
    bright_rim_puncta_n = len(puncta_props)
    bright_rim_puncta_per_100um = safe_ratio(bright_rim_puncta_n * 100.0, perimeter_um)

    # intensities
    rim_mean = float(gray_norm[rim_mask].mean()) if rim_mask.any() else np.nan
    core_mean = float(gray_norm[core_mask].mean()) if core_mask.any() else np.nan
    bg_mean = float(gray_norm[bg_mask].mean()) if bg_mask.any() else np.nan

    rim_hp_mean = float(hp[rim_mask].mean()) if rim_mask.any() else np.nan
    core_hp_mean = float(hp[core_mask].mean()) if core_mask.any() else np.nan
    bg_hp_mean = float(hp[bg_mask].mean()) if bg_mask.any() else np.nan

    out = {
        "sample_id": sample_id or tif_path.stem,
        "file_name": tif_path.name,
        "tif_path": str(tif_path),
        "txt_path": str(txt_path),
        "group": str(group),
        "magnification": meta.mag,
        "fov_width_um": meta.fov_width_um,
        "fov_height_um": meta.fov_height_um,
        "um_per_px": um_per_px,
        "detector": meta.detector,
        "date": meta.date,
        "time": meta.time,
        "image_height_px_cropped": gray.shape[0],
        "image_width_px": gray.shape[1],
        "mask_fraction": float(mask.mean()),
        "mask_area_um2": mask_area_um2,
        "perimeter_um": perimeter_um,
        "rim_area_um2": rim_area_um2,
        "bright_threshold_hp": bright_thr,
        "bright_rim_fraction": safe_ratio(float(bright_rim_mask.sum()), float(rim_mask.sum())),
        "bright_rim_area_per_100um_perimeter": safe_ratio(bright_rim_area_um2 * 100.0, perimeter_um),
        "coated_perimeter_fraction": coated_perimeter_fraction,
        "bright_rim_puncta_n": bright_rim_puncta_n,
        "bright_rim_puncta_per_100um": bright_rim_puncta_per_100um,
        "rim_mean_intensity": rim_mean,
        "core_mean_intensity": core_mean,
        "bg_mean_intensity": bg_mean,
        "rim_to_core_ratio": safe_ratio(rim_mean, core_mean) if np.isfinite(core_mean) else np.nan,
        "rim_to_background_ratio": safe_ratio(rim_mean, bg_mean) if np.isfinite(bg_mean) else np.nan,
        "rim_minus_core": (rim_mean - core_mean) if np.isfinite(core_mean) else np.nan,
        "rim_minus_background": (rim_mean - bg_mean) if np.isfinite(bg_mean) else np.nan,
        "rim_tophat_mean": rim_hp_mean,
        "core_tophat_mean": core_hp_mean,
        "bg_tophat_mean": bg_hp_mean,
        "rim_tophat_to_bg_ratio": safe_ratio(rim_hp_mean, bg_hp_mean) if np.isfinite(bg_hp_mean) else np.nan,
        "_gray_norm": gray_norm,
        "_mask": mask,
        "_rim_mask": rim_mask,
        "_bright_rim_mask": bright_rim_mask,
        "_boundary": boundary,
    }
    return out


# ---------------------------- plotting ----------------------------

def plot_feature_scatter(df: pd.DataFrame, feature: str, out_png: Path, group_order: List[str]):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    xpos = {g: i for i, g in enumerate(group_order)}
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    for i, g in enumerate(group_order):
        sub = df[df["group"] == g][feature].dropna().astype(float)
        if sub.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(sub))
        if len(sub) > 1:
            rng = np.random.default_rng(12345 + i)
            rng.shuffle(jitter)
        x = np.full(len(sub), xpos[g], dtype=float) + jitter
        plt.scatter(x, sub, s=55, alpha=0.8, color=colors[i % len(colors)])
        med = float(np.median(sub))
        q1 = float(np.quantile(sub, 0.25))
        q3 = float(np.quantile(sub, 0.75))
        plt.hlines(med, xpos[g] - 0.22, xpos[g] + 0.22, lw=2.5, color=colors[i % len(colors)])
        plt.vlines(xpos[g], q1, q3, lw=2.5, color=colors[i % len(colors)])

    plt.xticks(list(xpos.values()), group_order)
    plt.xlabel("Group")
    plt.ylabel(feature)
    plt.title(feature)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_overlay(gray_norm: np.ndarray, mask: np.ndarray, rim_mask: np.ndarray, bright_rim_mask: np.ndarray, out_png: Path):
    import matplotlib.pyplot as plt

    base = np.dstack([gray_norm, gray_norm, gray_norm])
    overlay = base.copy()
    # structural boundary cyan
    boundary = segmentation.find_boundaries(mask, mode="inner")
    overlay[boundary] = [0.1, 0.95, 0.95]
    # rim band yellow (subtle)
    rim_only = rim_mask & (~boundary) & (~bright_rim_mask)
    overlay[rim_only] = 0.7 * overlay[rim_only] + 0.3 * np.array([1.0, 1.0, 0.0])
    # bright rim deposits red
    overlay[bright_rim_mask] = [1.0, 0.15, 0.15]

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------- statistics ----------------------------

FEATURES_FOR_STATS = [
    "bright_rim_fraction",
    "bright_rim_area_per_100um_perimeter",
    "coated_perimeter_fraction",
    "bright_rim_puncta_per_100um",
    "rim_tophat_mean",
    "rim_tophat_to_bg_ratio",
    "rim_to_background_ratio",
    "rim_minus_background",
    "rim_to_core_ratio",
    "rim_minus_core",
    "mask_fraction",
]

PRIMARY_FEATURES = [
    "salt_lining_brightness_index",
    "rim_mean_intensity",
    "rim_to_background_ratio",
    "rim_minus_background",
    "rim_tophat_to_bg_ratio",
    "bright_rim_fraction",
    "bright_rim_area_per_100um_perimeter",
    "coated_perimeter_fraction",
    "bright_rim_puncta_per_100um",
    "mask_fraction",
]


def aggregate_units(df: pd.DataFrame, group_col: str, aggregate_col: Optional[str]) -> pd.DataFrame:
    if not aggregate_col or aggregate_col not in df.columns:
        x = df.copy()
        x["analysis_unit"] = x["sample_id"]
        return x

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("_")]
    keep_first = [c for c in df.columns if c not in num_cols and c != aggregate_col]
    grouped = []
    for (grp, unit), sub in df.groupby([group_col, aggregate_col], dropna=False):
        row = {group_col: grp, "analysis_unit": unit}
        for c in num_cols:
            row[c] = float(np.nanmedian(sub[c].astype(float)))
        for c in keep_first:
            if c not in row and c in sub.columns:
                row[c] = sub.iloc[0][c]
        grouped.append(row)
    return pd.DataFrame(grouped)


def kruskal_by_feature(df: pd.DataFrame, group_col: str, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    pvals = []
    kept_rows_idx = []

    groups = [g for g in sorted(df[group_col].dropna().astype(str).unique(), key=lambda x: (float(x) if str(x).replace(".","",1).isdigit() else x))]
    for feature in features:
        sub = df[[group_col, feature]].dropna()
        grouped = [sub.loc[sub[group_col] == g, feature].astype(float).values for g in groups if (sub[group_col] == g).any()]
        group_sizes = [len(x) for x in grouped]
        if len(grouped) < 2 or min(group_sizes) < 2:
            continue
        try:
            H, p = kruskal(*grouped)
        except Exception:
            continue
        n_total = int(sum(group_sizes))
        k = len(grouped)
        eps2 = (H - k + 1) / (n_total - k) if n_total > k else np.nan
        row = {
            "feature": feature,
            "n_total": n_total,
            "n_groups": k,
            "group_sizes": ",".join(map(str, group_sizes)),
            "kruskal_H": float(H),
            "p_value": float(p),
            "epsilon_squared": float(eps2),
        }
        kept_rows_idx.append(len(rows))
        pvals.append(float(p))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_value"].values)
    return out


def pairwise_mw(df: pd.DataFrame, group_col: str, features: Sequence[str]) -> pd.DataFrame:
    groups = [g for g in sorted(df[group_col].dropna().astype(str).unique(), key=lambda x: (float(x) if str(x).replace(".","",1).isdigit() else x))]
    rows = []
    pvals = []
    for feature in features:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                x = df.loc[df[group_col] == g1, feature].dropna().astype(float).values
                y = df.loc[df[group_col] == g2, feature].dropna().astype(float).values
                if len(x) < 2 or len(y) < 2:
                    continue
                try:
                    U, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                except TypeError:
                    U, p = mannwhitneyu(x, y, alternative="two-sided")
                row = {
                    "feature": feature,
                    "group1": g1,
                    "group2": g2,
                    "n1": len(x),
                    "n2": len(y),
                    "u_stat": float(U),
                    "p_value": float(p),
                    "median1": float(np.median(x)),
                    "median2": float(np.median(y)),
                    "delta_median": float(np.median(y) - np.median(x)),
                }
                rows.append(row)
                pvals.append(float(p))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_value"].values)
    return out


def group_summary(df: pd.DataFrame, group_col: str, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        for g, sub in df.groupby(group_col):
            vals = sub[feature].dropna().astype(float)
            if vals.empty:
                continue
            rows.append({
                "feature": feature,
                "group": g,
                "n": int(vals.shape[0]),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "q1": float(np.quantile(vals, 0.25)),
                "q3": float(np.quantile(vals, 0.75)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })
    return pd.DataFrame(rows)


def spearman_trend(df: pd.DataFrame, group_col: str, features: Sequence[str], group_order: Sequence[str]) -> pd.DataFrame:
    order_map = {g: i for i, g in enumerate(group_order)}
    if not set(df[group_col].dropna()).issubset(set(group_order)):
        return pd.DataFrame()
    x = df[group_col].map(order_map).astype(float)
    rows = []
    pvals = []
    for feature in features:
        y = df[feature].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 4:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        rows.append({"feature": feature, "spearman_rho": float(rho), "p_value": float(p), "n": int(ok.sum())})
        pvals.append(float(p))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_value"].values)
    return out


# ---------------------------- discovery / input ----------------------------

def discover_samples(input_dir: Path) -> pd.DataFrame:
    rows = []
    tif_paths = sorted(input_dir.glob("*.tif"))
    for tif_path in tif_paths:
        txt_path = tif_path.with_suffix(".txt")
        if not txt_path.exists():
            eprint(f"[WARN] Missing metadata .txt for {tif_path.name}; skipping.")
            continue
        group = infer_group_from_name(tif_path.name)
        if group is None:
            eprint(f"[WARN] Could not infer group from {tif_path.name}; skipping. Use a sample sheet if needed.")
            continue
        rows.append({
            "sample_id": tif_path.stem,
            "tif_path": str(tif_path),
            "txt_path": str(txt_path),
            "group": str(group),
        })
    return pd.DataFrame(rows)


def read_sample_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    required = {"tif_path", "txt_path", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample sheet missing required columns: {sorted(missing)}")
    if "sample_id" not in df.columns:
        df["sample_id"] = [Path(x).stem for x in df["tif_path"]]
    return df.copy()


# ---------------------------- main ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quantify SEM bright salt lining around fungal mycelia.")
    ap.add_argument("--input-dir", type=Path, default=None, help="Directory containing JEOL .tif and matching .txt files.")
    ap.add_argument("--sample-sheet", type=Path, default=None, help="TSV/CSV/XLSX with columns: sample_id(optional), tif_path, txt_path, group, biorep(optional).")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--aggregate-column", default=None, help="Optional column in sample sheet used to aggregate images before stats (recommended for biological replicates).")
    ap.add_argument("--group-order", default="0,10,20", help="Comma-separated ordered groups for plots/trend tests.")
    ap.add_argument("--max-overlay", type=int, default=999999, help="Max number of overlay images to save.")
    ap.add_argument("--body-blur-um", type=float, default=0.70)
    ap.add_argument("--body-close-um", type=float, default=0.35)
    ap.add_argument("--min-object-um2", type=float, default=0.20)
    ap.add_argument("--fill-hole-um2", type=float, default=0.20)
    ap.add_argument("--rim-in-um", type=float, default=0.50)
    ap.add_argument("--rim-out-um", type=float, default=0.80)
    ap.add_argument("--core-min-um", type=float, default=0.80)
    ap.add_argument("--bg-min-um", type=float, default=1.20)
    ap.add_argument("--bg-max-um", type=float, default=3.00)
    ap.add_argument("--bright-tophat-um", type=float, default=0.60)
    ap.add_argument("--bright-threshold-mad", type=float, default=2.75)
    ap.add_argument("--bright-threshold-q", type=float, default=0.995)
    ap.add_argument("--puncta-min-um2", type=float, default=0.03)
    ap.add_argument("--coated-neighborhood-um", type=float, default=0.60)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    overlay_dir = outdir / "qc_overlays"
    overlay_dir.mkdir(exist_ok=True)

    if args.sample_sheet is None and args.input_dir is None:
        raise SystemExit("Provide either --input-dir or --sample-sheet")

    if args.sample_sheet is not None:
        samples = read_sample_sheet(args.sample_sheet)
    else:
        samples = discover_samples(args.input_dir)

    if samples.empty:
        raise SystemExit("No valid samples found.")

    params = Params(
        body_blur_um=args.body_blur_um,
        body_close_um=args.body_close_um,
        min_object_um2=args.min_object_um2,
        fill_hole_um2=args.fill_hole_um2,
        rim_in_um=args.rim_in_um,
        rim_out_um=args.rim_out_um,
        core_min_um=args.core_min_um,
        bg_min_um=args.bg_min_um,
        bg_max_um=args.bg_max_um,
        bright_tophat_um=args.bright_tophat_um,
        bright_threshold_mad=args.bright_threshold_mad,
        bright_threshold_q=args.bright_threshold_q,
        puncta_min_um2=args.puncta_min_um2,
        coated_neighborhood_um=args.coated_neighborhood_um,
        save_overlay_limit=args.max_overlay,
    )

    records = []
    overlay_count = 0

    for _, row in samples.iterrows():
        tif_path = Path(row["tif_path"])
        txt_path = Path(row["txt_path"])
        group = str(row["group"])
        sample_id = str(row.get("sample_id", tif_path.stem))
        try:
            rec = analyze_one_image(tif_path=tif_path, txt_path=txt_path, group=group, sample_id=sample_id, p=params)
            # copy any extra sample-sheet columns
            for c in samples.columns:
                if c not in rec and c not in {"tif_path", "txt_path"}:
                    rec[c] = row[c]
            records.append(rec)

            if overlay_count < args.max_overlay:
                out_png = overlay_dir / f"{sample_id}.overlay.png"
                save_overlay(rec["_gray_norm"], rec["_mask"], rec["_rim_mask"], rec["_bright_rim_mask"], out_png)
                overlay_count += 1

        except Exception as exc:
            eprint(f"[WARN] Failed on {tif_path.name}: {exc}")

    if not records:
        raise SystemExit("All images failed.")

    image_df = pd.DataFrame(records)

    # descriptive composite built from local rim-vs-background brightness features.
    # This is intentionally not called an "EPS index" because SEM brightness is indirect.
    composite_cols = [
        "rim_to_background_ratio",
        "rim_minus_background",
        "rim_tophat_to_bg_ratio",
    ]
    for c in composite_cols:
        image_df[f"z_{c}"] = robust_z(image_df[c].astype(float))
    image_df["salt_lining_brightness_index"] = image_df[[f"z_{c}" for c in composite_cols]].mean(axis=1)

    # strip internal arrays before saving
    internal_cols = [c for c in image_df.columns if c.startswith("_")]
    save_image_df = image_df.drop(columns=internal_cols)
    save_image_df.to_csv(outdir / "image_features.tsv", sep="\t", index=False)

    metadata_cols = [
        "sample_id", "file_name", "group", "magnification", "fov_width_um", "fov_height_um",
        "um_per_px", "detector", "date", "time", "image_height_px_cropped", "image_width_px",
    ]
    keep_meta = [c for c in metadata_cols if c in save_image_df.columns]
    save_image_df[keep_meta].to_csv(outdir / "image_metadata.tsv", sep="\t", index=False)

    group_order = [x.strip() for x in args.group_order.split(",") if x.strip()]
    # stats can be done at image level or aggregated level
    unit_df = aggregate_units(save_image_df, group_col="group", aggregate_col=args.aggregate_column)

    features = [f for f in PRIMARY_FEATURES if f in unit_df.columns]
    kr_df = kruskal_by_feature(unit_df, "group", features)
    mw_df = pairwise_mw(unit_df, "group", features)
    gs_df = group_summary(unit_df, "group", features)
    sp_df = spearman_trend(unit_df, "group", features, group_order=group_order)

    kr_df.to_csv(outdir / "kruskal_tests.tsv", sep="\t", index=False)
    mw_df.to_csv(outdir / "pairwise_mannwhitney.tsv", sep="\t", index=False)
    gs_df.to_csv(outdir / "group_summary.tsv", sep="\t", index=False)
    sp_df.to_csv(outdir / "spearman_trend.tsv", sep="\t", index=False)

    if args.aggregate_column and args.aggregate_column in save_image_df.columns:
        unit_df.to_csv(outdir / "analysis_units_aggregated.tsv", sep="\t", index=False)

    # plots
    for feature in features:
        plot_feature_scatter(unit_df, feature, outdir / f"feature_{feature}.png", group_order=group_order)

    # QC plots
    if "fov_width_um" in save_image_df.columns:
        plot_feature_scatter(save_image_df, "fov_width_um", outdir / "qc_fov_by_group.png", group_order=group_order)
    if "magnification" in save_image_df.columns:
        plot_feature_scatter(save_image_df, "magnification", outdir / "qc_magnification_by_group.png", group_order=group_order)

    # print a concise top-table to stdout for the user
    if not kr_df.empty:
        top = kr_df.sort_values(["p_fdr_bh", "p_value"]).head(10)
        print("[INFO] Top Kruskal results:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(top.to_string(index=False))
    else:
        print("[INFO] No valid Kruskal results were produced.")


if __name__ == "__main__":
    main()
