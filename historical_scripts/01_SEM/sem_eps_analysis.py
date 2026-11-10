#!/usr/bin/env python3
"""
SEM image analysis pipeline for EPS-like extracellular matrix proxy quantification.

Designed for JEOL/JCM-7000 style TIFF + sidecar TXT metadata files.
Key ideas:
1) Remove the SEM annotation strip automatically.
2) Use JEOL field-of-view metadata to convert pixels to micrometers.
3) Downsample high-magnification images to a common physical pixel size.
4) Quantify fixed-size physical tiles so images with different zoom remain comparable.
5) Use image-level medians (not tile-level p-values) for group statistics.
6) Treat SEM-derived output as an EPS morphology proxy, not direct biochemical EPS.

Outputs:
- image_metadata.tsv
- tile_features.tsv
- image_features.tsv
- group_summary.tsv
- kruskal_tests.tsv
- pairwise_mannwhitney.tsv
- spearman_trend.tsv
- optional assay_correlation.tsv
- QC / summary plots in the output directory
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, stats
from scipy.signal import convolve2d
from skimage import exposure, filters, morphology, transform
from skimage.feature import canny
from skimage.filters import sobel
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


@dataclass
class ImageRecord:
    image_path: Path
    metadata_path: Path
    image_id: str
    group: str
    salinity_numeric: float
    replicate: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify EPS-like SEM morphology across salinity conditions."
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with TIFF and TXT files")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--sample-sheet",
        type=Path,
        default=None,
        help=(
            "Optional TSV/CSV/XLSX with columns image_path, group, optional salinity_numeric, optional replicate. "
            "If omitted, groups are inferred from filenames."
        ),
    )
    p.add_argument(
        "--image-glob",
        default="*.tif",
        help="Glob for image discovery when --sample-sheet is not supplied (default: *.tif)",
    )
    p.add_argument(
        "--group-regex",
        default=r"^(?P<group>\d+)%",
        help=(
            "Regex applied to filename stem when inferring groups. Must define a named group 'group'. "
            "Default captures 0, 10, 20 from names like 10%%DSW_1.tif."
        ),
    )
    p.add_argument(
        "--target-um-per-px",
        type=float,
        default=0.10,
        help=(
            "Target physical pixel size in um/px. Images finer than this are downsampled; coarser images are not upsampled."
        ),
    )
    p.add_argument(
        "--tile-um",
        type=float,
        default=5.0,
        help="Tile size in micrometers for per-image aggregation (default: 5 um)",
    )
    p.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=0.03,
        help="CLAHE clip limit for contrast normalization (default: 0.03)",
    )
    p.add_argument(
        "--highpass-sigma-um",
        type=float,
        default=0.5,
        help="Gaussian sigma in um for the high-pass texture feature (default: 0.5)",
    )
    p.add_argument(
        "--entropy-radius-um",
        type=float,
        default=0.4,
        help="Neighborhood radius in um for local entropy (default: 0.4)",
    )
    p.add_argument(
        "--small-objects-px",
        type=int,
        default=16,
        help="Remove binary objects <= this size during tile segmentation (default: 16 px)",
    )
    p.add_argument(
        "--small-holes-px",
        type=int,
        default=16,
        help="Fill binary holes <= this size during tile segmentation (default: 16 px)",
    )
    p.add_argument(
        "--min-tiles-per-image",
        type=int,
        default=4,
        help="Warn when an image yields fewer than this many full tiles (default: 4)",
    )
    p.add_argument(
        "--assay-table",
        type=Path,
        default=None,
        help=(
            "Optional TSV/CSV/XLSX with biochemical EPS measurements. Accepted schemas: "
            "(1) image_id + eps_value, or (2) group + eps_value."
        ),
    )
    p.add_argument(
        "--qc-images-per-group",
        type=int,
        default=2,
        help="How many example overlays to save per group (default: 2)",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def infer_group_from_name(stem: str, regex: str) -> Tuple[str, float]:
    m = re.search(regex, stem)
    if not m:
        raise ValueError(
            f"Could not infer a group from filename stem '{stem}'. Use --sample-sheet or adjust --group-regex."
        )
    group = str(m.group("group"))
    nums = re.findall(r"[-+]?\d*\.?\d+", group)
    if not nums:
        raise ValueError(f"Could not parse numeric salinity from inferred group '{group}'.")
    return group, float(nums[0])


def discover_records(args: argparse.Namespace) -> List[ImageRecord]:
    records: List[ImageRecord] = []

    if args.sample_sheet is not None:
        df = read_table(args.sample_sheet)
        required = {"image_path", "group"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Sample sheet is missing required columns: {sorted(missing)}")
        for _, row in df.iterrows():
            image_path = Path(row["image_path"]).expanduser().resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            metadata_path = image_path.with_suffix(".txt")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing sidecar TXT metadata for {image_path.name}")
            group = str(row["group"])
            salinity_numeric = (
                float(row["salinity_numeric"])
                if "salinity_numeric" in df.columns and pd.notna(row.get("salinity_numeric"))
                else float(re.findall(r"[-+]?\d*\.?\d+", group)[0])
            )
            replicate = None
            if "replicate" in df.columns and pd.notna(row.get("replicate")):
                replicate = str(row["replicate"])
            records.append(
                ImageRecord(
                    image_path=image_path,
                    metadata_path=metadata_path,
                    image_id=image_path.stem,
                    group=group,
                    salinity_numeric=salinity_numeric,
                    replicate=replicate,
                )
            )
        return records

    for image_path in sorted(args.input_dir.glob(args.image_glob)):
        if image_path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        metadata_path = image_path.with_suffix(".txt")
        if not metadata_path.exists():
            print(f"[WARN] Skipping {image_path.name}: no matching .txt sidecar found", file=sys.stderr)
            continue
        try:
            group, salinity_numeric = infer_group_from_name(image_path.stem, args.group_regex)
        except ValueError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            continue
        records.append(
            ImageRecord(
                image_path=image_path.resolve(),
                metadata_path=metadata_path.resolve(),
                image_id=image_path.stem,
                group=group,
                salinity_numeric=salinity_numeric,
            )
        )
    if not records:
        raise FileNotFoundError(
            "No analyzable TIFF + TXT pairs were found. Check --input-dir, --image-glob, or use --sample-sheet."
        )
    return records


def parse_jeol_metadata(path: Path) -> Dict[str, str | float]:
    info: Dict[str, str | float] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith("$"):
                continue
            parts = line[1:].split(maxsplit=1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            info[key] = value

    fov = str(info.get("CM_FIELD_OF_VIEW", ""))
    size = str(info.get("CM_IMAGE_SIZE", ""))
    mag = float(info.get("CM_MAG", np.nan))
    wd = info.get("SM_WD", np.nan)

    m_fov = re.findall(r"([0-9.]+)µm", fov)
    if len(m_fov) < 2:
        raise ValueError(f"Could not parse CM_FIELD_OF_VIEW from {path}")
    width_um = float(m_fov[0])
    height_um = float(m_fov[1])

    m_size = re.findall(r"([0-9]+)", size)
    if len(m_size) < 2:
        raise ValueError(f"Could not parse CM_IMAGE_SIZE from {path}")
    width_px = int(m_size[0])
    height_px = int(m_size[1])

    return {
        "fov_um_x": width_um,
        "fov_um_y": height_um,
        "meta_width_px": width_px,
        "meta_height_px": height_px,
        "magnification": mag,
        "working_distance_mm": float(wd) if wd not in {np.nan, ""} else np.nan,
    }


def load_gray_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = arr[..., :3].astype(np.float32).mean(axis=2)
        else:
            arr = arr[..., 0].astype(np.float32)
    elif arr.ndim == 2:
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.astype(np.float32)


def crop_annotation_strip(
    gray: np.ndarray,
    dark_threshold: float = 0.05,
    dark_fraction_threshold: float = 0.70,
    confirm_rows: int = 12,
) -> Tuple[np.ndarray, int]:
    """
    Remove the dark annotation strip typically found at the bottom of SEM exports.
    This is more robust than looking only for a fully black run, because JEOL labels/
    scale bars introduce bright text inside the otherwise dark footer.
    Returns the cropped image and number of rows removed.
    """
    h = gray.shape[0]
    row_dark_fraction = (gray < dark_threshold).mean(axis=1)
    search_start = int(h * 0.70)
    start = None
    for i in range(search_start, h - confirm_rows + 1):
        window = row_dark_fraction[i : i + confirm_rows]
        if np.median(window) >= dark_fraction_threshold:
            start = i
            break
    if start is not None:
        cropped = gray[:start, :]
        return cropped, gray.shape[0] - cropped.shape[0]
    return gray, 0


def downsample_to_target(gray: np.ndarray, um_per_px: float, target_um_per_px: float) -> Tuple[np.ndarray, float, bool]:
    """
    Downsample only if the image is finer than the target resolution.
    Never upsample a coarser image, because that would create fake texture detail.
    """
    if um_per_px >= target_um_per_px * 0.98:
        return gray, um_per_px, False

    scale = um_per_px / target_um_per_px
    new_shape = (
        max(16, int(round(gray.shape[0] * scale))),
        max(16, int(round(gray.shape[1] * scale))),
    )
    resized = transform.resize(
        gray,
        new_shape,
        anti_aliasing=True,
        preserve_range=True,
        order=1,
    ).astype(np.float32)
    return resized, target_um_per_px, True


def iter_tiles(img: np.ndarray, tile_px: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    h, w = img.shape
    for y in range(0, h - tile_px + 1, tile_px):
        for x in range(0, w - tile_px + 1, tile_px):
            yield y, x, img[y : y + tile_px, x : x + tile_px]


def robust_z(series: pd.Series) -> pd.Series:
    med = float(series.median())
    mad = float(np.median(np.abs(series - med)))
    if mad == 0:
        sd = float(series.std(ddof=0))
        if sd == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - med) / sd
    return (series - med) / (1.4826 * mad)


def bootstrap_median_ci(values: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 1) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)


def mannwhitney_effect_size(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    u = stats.mannwhitneyu(x_arr, y_arr, alternative="two-sided").statistic
    return float((2 * u) / (len(x_arr) * len(y_arr)) - 1)


def preprocess_image(gray: np.ndarray, args: argparse.Namespace, um_per_px: float) -> Tuple[np.ndarray, float, int, bool]:
    cropped, crop_rows = crop_annotation_strip(gray)
    resampled, work_um_per_px, was_downsampled = downsample_to_target(
        cropped, um_per_px=um_per_px, target_um_per_px=args.target_um_per_px
    )
    eq = exposure.equalize_adapthist(resampled, clip_limit=args.clahe_clip_limit).astype(np.float32)
    return eq, work_um_per_px, crop_rows, was_downsampled


def compute_tile_features(tile: np.ndarray, work_um_per_px: float, args: argparse.Namespace) -> Dict[str, float]:
    if np.allclose(tile.std(), 0.0):
        raise ValueError("Flat tile")

    thr = filters.threshold_otsu(tile)
    material = tile > thr
    material = morphology.remove_small_objects(material, max_size=args.small_objects_px)
    material = morphology.remove_small_holes(material, max_size=args.small_holes_px)

    skel = morphology.skeletonize(material)
    neighbors = convolve2d(
        skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="same", boundary="fill", fillvalue=0
    ) - skel.astype(np.uint8)
    branchpoints = np.count_nonzero(skel & (neighbors >= 3))

    tile_area_um2 = tile.shape[0] * tile.shape[1] * (work_um_per_px ** 2)
    entropy_radius_px = max(1, int(round(args.entropy_radius_um / work_um_per_px)))
    entropy_map = entropy(img_as_ubyte(tile), morphology.disk(entropy_radius_px))
    hp_sigma_px = max(1.0, args.highpass_sigma_um / work_um_per_px)
    highpass = tile - filters.gaussian(tile, sigma=hp_sigma_px)

    return {
        "material_fraction": float(material.mean()),
        "pore_fraction": float(1.0 - material.mean()),
        "edge_mean": float(sobel(tile).mean()),
        "canny_fraction": float(canny(tile, sigma=1.0).mean()),
        "entropy_mean": float(entropy_map.mean()),
        "highpass_std": float(highpass.std()),
        "skel_length_um_per_100um2": float((skel.sum() * work_um_per_px) * 100.0 / tile_area_um2),
        "branchpoints_per_100um2": float(branchpoints * 100.0 / tile_area_um2),
        "tile_area_um2": float(tile_area_um2),
    }


def analyze_records(records: Sequence[ImageRecord], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_rows: List[Dict[str, object]] = []
    tile_rows: List[Dict[str, object]] = []
    metadata_rows: List[Dict[str, object]] = []

    for record in tqdm(records, desc="Analyzing images"):
        meta = parse_jeol_metadata(record.metadata_path)
        gray = load_gray_image(record.image_path)

        if gray.shape[1] != meta["meta_width_px"]:
            # Use actual loaded width for pixel calibration if image export adds alpha or annotation rows.
            pass
        um_per_px = float(meta["fov_um_x"]) / float(gray.shape[1])

        eq, work_um_per_px, crop_rows, was_downsampled = preprocess_image(gray, args, um_per_px)
        tile_px = max(16, int(round(args.tile_um / work_um_per_px)))

        metadata_rows.append(
            {
                "image_id": record.image_id,
                "image_path": str(record.image_path),
                "metadata_path": str(record.metadata_path),
                "group": record.group,
                "salinity_numeric": record.salinity_numeric,
                "replicate": record.replicate,
                "magnification": float(meta["magnification"]),
                "fov_um_x": float(meta["fov_um_x"]),
                "fov_um_y": float(meta["fov_um_y"]),
                "raw_um_per_px": um_per_px,
                "working_um_per_px": work_um_per_px,
                "tile_um": float(args.tile_um),
                "tile_px": int(tile_px),
                "annotation_rows_removed": int(crop_rows),
                "downsampled_to_target": bool(was_downsampled),
                "cropped_height_px": int(eq.shape[0]),
                "cropped_width_px": int(eq.shape[1]),
                "analyzed_area_um2": float(eq.shape[0] * eq.shape[1] * (work_um_per_px ** 2)),
            }
        )

        per_image_tiles: List[Dict[str, float]] = []
        for tile_index, (y, x, tile) in enumerate(iter_tiles(eq, tile_px), start=1):
            try:
                feats = compute_tile_features(tile, work_um_per_px=work_um_per_px, args=args)
            except Exception:
                continue
            feats.update(
                {
                    "image_id": record.image_id,
                    "group": record.group,
                    "salinity_numeric": record.salinity_numeric,
                    "replicate": record.replicate,
                    "tile_index": tile_index,
                    "tile_y_px": y,
                    "tile_x_px": x,
                }
            )
            tile_rows.append(feats)
            per_image_tiles.append(feats)

        if not per_image_tiles:
            print(f"[WARN] No analyzable tiles for {record.image_path.name}", file=sys.stderr)
            continue

        tile_df = pd.DataFrame(per_image_tiles)
        image_row: Dict[str, object] = {
            "image_id": record.image_id,
            "image_path": str(record.image_path),
            "group": record.group,
            "salinity_numeric": record.salinity_numeric,
            "replicate": record.replicate,
            "n_tiles": int(len(tile_df)),
        }
        for col in [
            "material_fraction",
            "pore_fraction",
            "edge_mean",
            "canny_fraction",
            "entropy_mean",
            "highpass_std",
            "skel_length_um_per_100um2",
            "branchpoints_per_100um2",
        ]:
            image_row[col] = float(tile_df[col].median())
            image_row[f"{col}_iqr"] = float(tile_df[col].quantile(0.75) - tile_df[col].quantile(0.25))
        image_rows.append(image_row)

    meta_df = pd.DataFrame(metadata_rows)
    tile_df_all = pd.DataFrame(tile_rows)
    image_df = pd.DataFrame(image_rows)

    if image_df.empty:
        raise RuntimeError("No image-level features were produced.")

    image_df = image_df.merge(
        meta_df[
            [
                "image_id",
                "magnification",
                "fov_um_x",
                "fov_um_y",
                "raw_um_per_px",
                "working_um_per_px",
                "tile_um",
                "tile_px",
                "annotation_rows_removed",
                "downsampled_to_target",
                "analyzed_area_um2",
            ]
        ],
        on="image_id",
        how="left",
    )

    eps_features = [
        "entropy_mean",
        "highpass_std",
        "skel_length_um_per_100um2",
        "branchpoints_per_100um2",
        "canny_fraction",
    ]
    for feat in eps_features:
        image_df[f"{feat}_rz"] = robust_z(image_df[feat])
    image_df["eps_proxy_index"] = image_df[[f"{feat}_rz" for feat in eps_features]].mean(axis=1)

    return meta_df, tile_df_all, image_df


def summarize_groups(image_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group, gdf in image_df.groupby("group", sort=False):
        for feature in features:
            values = gdf[feature].dropna().astype(float).values
            ci_low, ci_high = bootstrap_median_ci(values)
            rows.append(
                {
                    "group": group,
                    "n_images": int(len(gdf)),
                    "feature": feature,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "bootstrap_median_ci_low": ci_low,
                    "bootstrap_median_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def kruskal_and_pairwise(image_df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered_groups = (
        image_df[["group", "salinity_numeric"]]
        .drop_duplicates()
        .sort_values(["salinity_numeric", "group"])["group"]
        .tolist()
    )

    kruskal_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for feature in features:
        grouped_values = []
        group_sizes = []
        present_groups = []
        for group in ordered_groups:
            vals = image_df.loc[image_df["group"] == group, feature].dropna().astype(float).values
            if len(vals) > 0:
                grouped_values.append(vals)
                group_sizes.append(len(vals))
                present_groups.append(group)
        if len(grouped_values) < 2:
            continue

        h_stat, p_val = stats.kruskal(*grouped_values)
        n_total = sum(group_sizes)
        k = len(grouped_values)
        eps2 = (h_stat - k + 1) / (n_total - k) if n_total > k else np.nan
        kruskal_rows.append(
            {
                "feature": feature,
                "n_total": n_total,
                "n_groups": k,
                "group_sizes": ",".join(map(str, group_sizes)),
                "kruskal_H": float(h_stat),
                "p_value": float(p_val),
                "epsilon_squared": float(eps2),
            }
        )

        for g1, g2 in itertools.combinations(present_groups, 2):
            x = image_df.loc[image_df["group"] == g1, feature].dropna().astype(float).values
            y = image_df.loc[image_df["group"] == g2, feature].dropna().astype(float).values
            mw = stats.mannwhitneyu(x, y, alternative="two-sided")
            pair_rows.append(
                {
                    "feature": feature,
                    "group1": g1,
                    "group2": g2,
                    "n1": int(len(x)),
                    "n2": int(len(y)),
                    "u_stat": float(mw.statistic),
                    "p_value": float(mw.pvalue),
                    "rank_biserial": mannwhitney_effect_size(x, y),
                    "median_group1": float(np.median(x)),
                    "median_group2": float(np.median(y)),
                    "median_difference_g1_minus_g2": float(np.median(x) - np.median(y)),
                }
            )

    kruskal_df = pd.DataFrame(kruskal_rows)
    if not kruskal_df.empty:
        kruskal_df["p_fdr_bh"] = multipletests(kruskal_df["p_value"], method="fdr_bh")[1]
        kruskal_df = kruskal_df.sort_values(["p_fdr_bh", "p_value", "feature"])

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["p_fdr_bh"] = multipletests(pair_df["p_value"], method="fdr_bh")[1]
        pair_df = pair_df.sort_values(["p_fdr_bh", "p_value", "feature", "group1", "group2"])

    return kruskal_df, pair_df


def spearman_trends(image_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    x = image_df["salinity_numeric"].astype(float).values
    for feature in features:
        y = image_df[feature].astype(float).values
        rho, p_val = stats.spearmanr(x, y, nan_policy="omit")
        rows.append({"feature": feature, "spearman_rho": float(rho), "p_value": float(p_val)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_fdr_bh"] = multipletests(df["p_value"], method="fdr_bh")[1]
        df = df.sort_values(["p_fdr_bh", "p_value", "feature"])
    return df


def correlate_with_assay(image_df: pd.DataFrame, assay_path: Path) -> pd.DataFrame:
    assay = read_table(assay_path)
    if {"image_id", "eps_value"}.issubset(assay.columns):
        merged = image_df.merge(assay[["image_id", "eps_value"]], on="image_id", how="inner")
        unit = "image"
    elif {"group", "eps_value"}.issubset(assay.columns):
        grp = image_df.groupby("group", as_index=False)["eps_proxy_index"].median()
        merged = grp.merge(assay[["group", "eps_value"]], on="group", how="inner")
        unit = "group"
    else:
        raise ValueError("Assay table must contain either image_id+eps_value or group+eps_value.")

    if len(merged) < 3:
        raise ValueError("Need at least 3 matched observations for assay correlation.")

    rho, p_val = stats.spearmanr(merged["eps_proxy_index"], merged["eps_value"], nan_policy="omit")
    return pd.DataFrame(
        [
            {
                "match_unit": unit,
                "n_matched": int(len(merged)),
                "spearman_rho": float(rho),
                "p_value": float(p_val),
            }
        ]
    )


def make_qc_plots(meta_df: pd.DataFrame, image_df: pd.DataFrame, outdir: Path) -> None:
    ordered_groups = (
        meta_df[["group", "salinity_numeric"]]
        .drop_duplicates()
        .sort_values(["salinity_numeric", "group"])["group"]
        .tolist()
    )

    group_to_x = {g: i for i, g in enumerate(ordered_groups)}

    # FOV QC
    plt.figure(figsize=(8, 5))
    xs = [group_to_x[g] for g in meta_df["group"]]
    plt.scatter(xs, meta_df["fov_um_x"], alpha=0.8)
    plt.xticks(range(len(ordered_groups)), ordered_groups)
    plt.ylabel("Field of view width (um)")
    plt.xlabel("Group")
    plt.title("QC: field of view by group")
    plt.tight_layout()
    plt.savefig(outdir / "qc_fov_by_group.png", dpi=200)
    plt.close()

    # Magnification QC
    plt.figure(figsize=(8, 5))
    xs = [group_to_x[g] for g in meta_df["group"]]
    plt.scatter(xs, meta_df["magnification"], alpha=0.8)
    plt.xticks(range(len(ordered_groups)), ordered_groups)
    plt.ylabel("Magnification")
    plt.xlabel("Group")
    plt.title("QC: magnification by group")
    plt.tight_layout()
    plt.savefig(outdir / "qc_magnification_by_group.png", dpi=200)
    plt.close()

    # EPS proxy summary
    plt.figure(figsize=(8, 5))
    for i, group in enumerate(ordered_groups):
        vals = image_df.loc[image_df["group"] == group, "eps_proxy_index"].astype(float).values
        x = np.full_like(vals, i, dtype=float) + np.linspace(-0.08, 0.08, len(vals))
        plt.scatter(x, vals, alpha=0.8)
        if len(vals):
            plt.plot([i - 0.2, i + 0.2], [np.median(vals), np.median(vals)], linewidth=2)
    plt.xticks(range(len(ordered_groups)), ordered_groups)
    plt.ylabel("EPS proxy index (robust z composite)")
    plt.xlabel("Group")
    plt.title("Image-level EPS proxy index by group")
    plt.tight_layout()
    plt.savefig(outdir / "eps_proxy_by_group.png", dpi=200)
    plt.close()


def make_feature_plots(image_df: pd.DataFrame, outdir: Path, features: Sequence[str]) -> None:
    ordered_groups = (
        image_df[["group", "salinity_numeric"]]
        .drop_duplicates()
        .sort_values(["salinity_numeric", "group"])["group"]
        .tolist()
    )

    for feature in features:
        plt.figure(figsize=(8, 5))
        for i, group in enumerate(ordered_groups):
            vals = image_df.loc[image_df["group"] == group, feature].astype(float).values
            x = np.full_like(vals, i, dtype=float) + np.linspace(-0.08, 0.08, len(vals))
            plt.scatter(x, vals, alpha=0.8)
            if len(vals):
                q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                plt.plot([i - 0.18, i + 0.18], [med, med], linewidth=2)
                plt.vlines(i, q1, q3, linewidth=2)
        plt.xticks(range(len(ordered_groups)), ordered_groups)
        plt.xlabel("Group")
        plt.ylabel(feature)
        plt.title(feature)
        plt.tight_layout()
        plt.savefig(outdir / f"feature_{feature}.png", dpi=200)
        plt.close()


def save_qc_overlays(records: Sequence[ImageRecord], args: argparse.Namespace, outdir: Path) -> None:
    overlay_dir = outdir / "qc_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    by_group: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        by_group.setdefault(rec.group, []).append(rec)

    for group, items in by_group.items():
        for rec in items[: args.qc_images_per_group]:
            meta = parse_jeol_metadata(rec.metadata_path)
            gray = load_gray_image(rec.image_path)
            um_per_px = float(meta["fov_um_x"]) / float(gray.shape[1])
            eq, work_um_per_px, _, _ = preprocess_image(gray, args, um_per_px)
            tile_px = max(16, int(round(args.tile_um / work_um_per_px)))
            first_tile = None
            for _, _, tile in iter_tiles(eq, tile_px):
                first_tile = tile
                break
            if first_tile is None:
                continue
            thr = filters.threshold_otsu(first_tile)
            material = first_tile > thr
            material = morphology.remove_small_objects(material, max_size=args.small_objects_px)
            material = morphology.remove_small_holes(material, max_size=args.small_holes_px)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(eq, cmap="gray")
            plt.title(f"{rec.image_id}\nfull preprocessed frame")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(first_tile, cmap="gray")
            plt.contour(material.astype(float), levels=[0.5], linewidths=1)
            plt.title(f"first {args.tile_um:g} um tile + mask")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(overlay_dir / f"{rec.image_id}_overlay.png", dpi=200)
            plt.close()


def write_readme(args: argparse.Namespace, outdir: Path, image_df: pd.DataFrame) -> None:
    text = f"""SEM EPS proxy analysis
======================

What this pipeline does
-----------------------
- Reads TIFF images and JEOL sidecar TXT metadata.
- Removes the bottom SEM annotation strip.
- Converts pixels to micrometers using CM_FIELD_OF_VIEW.
- Downsamples only the higher-resolution images to {args.target_um_per_px:g} um/px.
- Computes features from {args.tile_um:g} um x {args.tile_um:g} um physical tiles.
- Aggregates tile values into one image-level median per feature.
- Tests group differences with Kruskal-Wallis + pairwise Mann-Whitney + BH FDR.
- Computes a trend test against numeric salinity with Spearman correlation.

Important interpretation note
-----------------------------
The output is an EPS-like morphology proxy. SEM can support an EPS hypothesis, but it does not directly measure EPS chemistry or neutralization capacity by itself.

Image counts analyzed
---------------------
{image_df.groupby('group').size().to_string()}

When to be careful
------------------
- If one condition was imaged mostly at much higher magnification, keep the tile size small.
- If several fields come from the same biological culture, image-level p-values are field-level evidence, not fully independent biological replication.
- For the strongest claim, correlate eps_proxy_index with an external biochemical EPS assay.
"""
    (outdir / "README.txt").write_text(text)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args)
    meta_df, tile_df, image_df = analyze_records(records, args)

    features = [
        "eps_proxy_index",
        "entropy_mean",
        "highpass_std",
        "skel_length_um_per_100um2",
        "branchpoints_per_100um2",
        "canny_fraction",
        "material_fraction",
        "pore_fraction",
        "edge_mean",
    ]

    group_summary = summarize_groups(image_df, features)
    kruskal_df, pairwise_df = kruskal_and_pairwise(image_df, features)
    trend_df = spearman_trends(image_df, features)

    meta_df.to_csv(args.outdir / "image_metadata.tsv", sep="\t", index=False)
    tile_df.to_csv(args.outdir / "tile_features.tsv", sep="\t", index=False)
    image_df.to_csv(args.outdir / "image_features.tsv", sep="\t", index=False)
    group_summary.to_csv(args.outdir / "group_summary.tsv", sep="\t", index=False)
    kruskal_df.to_csv(args.outdir / "kruskal_tests.tsv", sep="\t", index=False)
    pairwise_df.to_csv(args.outdir / "pairwise_mannwhitney.tsv", sep="\t", index=False)
    trend_df.to_csv(args.outdir / "spearman_trend.tsv", sep="\t", index=False)

    if args.assay_table is not None:
        assay_corr = correlate_with_assay(image_df, args.assay_table)
        assay_corr.to_csv(args.outdir / "assay_correlation.tsv", sep="\t", index=False)

    make_qc_plots(meta_df, image_df, args.outdir)
    make_feature_plots(
        image_df,
        args.outdir,
        [
            "eps_proxy_index",
            "entropy_mean",
            "highpass_std",
            "skel_length_um_per_100um2",
            "branchpoints_per_100um2",
            "material_fraction",
        ],
    )
    save_qc_overlays(records, args, args.outdir)
    write_readme(args, args.outdir, image_df)

    # Console summary for quick inspection.
    counts = image_df.groupby("group").size().to_dict()
    print("[INFO] Image counts by group:", counts)
    low_tile = image_df.loc[image_df["n_tiles"] < args.min_tiles_per_image, ["image_id", "group", "n_tiles", "fov_um_x"]]
    if not low_tile.empty:
        print("[WARN] Some images yielded few tiles. Consider reducing --tile-um for very high-magnification images:")
        print(low_tile.to_string(index=False))

    if not kruskal_df.empty:
        print("\n[INFO] Top Kruskal results:")
        print(kruskal_df.head(10).to_string(index=False))

    print(f"\n[INFO] Results written to: {args.outdir}")


if __name__ == "__main__":
    main()
