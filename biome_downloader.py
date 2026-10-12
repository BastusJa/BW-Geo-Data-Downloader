"""
============================================================
TF3 MAP GENERATOR
============================================================

Creates 8-bit grayscale PNG masks for Transport Fever 3
using:

    - Existing DGM GeoTIFF
    - OpenStreetMap / Overpass
    - Same center / area / resolution as the DGM config

Generated masks:

    river_mask.png
    water_mask.png
    mountain_mask.png

    forest_mask.png
    meadow_mask.png
    farmland_mask.png
    settlement_mask.png
    heath_mask.png
    wetland_mask.png
    rock_mask.png

    biome_mask.png

All masks use the EXACT same raster:

    CRS
    extent
    resolution
    width
    height

as the configured DGM area.

Default mask convention:

    0   = false / background
    255 = true / feature

============================================================
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml

from PIL import Image
from pyproj import CRS, Transformer

from shapely.geometry import LineString, Polygon, box
from shapely.ops import transform as shapely_transform

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from config_loader import load_config


# ============================================================
# CONFIG
# ============================================================



# ============================================================
# LOGGING
# ============================================================


def log(
    message: str,
    verbose: bool = True,
):

    if verbose:
        print(
            message,
            flush=True,
        )


# ============================================================
# NUMERIC HELPERS
# ============================================================


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def normalize(
    value,
    start: float,
    end: float,
):
    """
    Normalize scalar or NumPy array to 0..1.

    Works for both:

        normalize(400.0, 300, 600)

    and:

        normalize(dem_array, 300, 600)
    """

    if end <= start:
        raise ValueError(
            "Normalization end must be greater than start."
        )

    result = (
        np.asarray(value, dtype=np.float32) - start
    ) / (
        end - start
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# EXTENT
# ============================================================


def calculate_extent(
    config: dict,
):

    center = config["center"]
    area = config["area"]
    crs_config = config["crs"]

    latitude = float(
        center["latitude"]
    )

    longitude = float(
        center["longitude"]
    )

    width_m = float(
        area["width_m"]
    )

    height_m = float(
        area["height_m"]
    )

    input_crs = CRS.from_user_input(
        crs_config["input"]
    )

    processing_crs = CRS.from_user_input(
        crs_config["processing"]
    )

    to_processing = Transformer.from_crs(
        input_crs,
        processing_crs,
        always_xy=True,
    )

    to_wgs84 = Transformer.from_crs(
        processing_crs,
        input_crs,
        always_xy=True,
    )

    center_x, center_y = (
        to_processing.transform(
            longitude,
            latitude,
        )
    )

    half_width = width_m / 2.0
    half_height = height_m / 2.0

    min_x = center_x - half_width
    max_x = center_x + half_width

    min_y = center_y - half_height
    max_y = center_y + half_height

    bounds = (
        min_x,
        min_y,
        max_x,
        max_y,
    )

    corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ]

    wgs84_corners = [
        to_wgs84.transform(
            x,
            y,
        )
        for x, y in corners
    ]

    west = min(
        p[0]
        for p in wgs84_corners
    )

    east = max(
        p[0]
        for p in wgs84_corners
    )

    south = min(
        p[1]
        for p in wgs84_corners
    )

    north = max(
        p[1]
        for p in wgs84_corners
    )

    return {
        "bounds": bounds,
        "wgs84_bounds": (
            west,
            south,
            east,
            north,
        ),
        "processing_crs": processing_crs,
        "to_processing": to_processing,
        "to_wgs84": to_wgs84,
        "width_m": width_m,
        "height_m": height_m,
    }


# ============================================================
# RASTER GRID
# ============================================================


def create_grid(
    bounds,
    resolution,
):

    min_x, min_y, max_x, max_y = bounds

    width_m = max_x - min_x
    height_m = max_y - min_y

    width_px = int(
        round(
            width_m / resolution
        )
    )

    height_px = int(
        round(
            height_m / resolution
        )
    )

    if not math.isclose(
        width_px * resolution,
        width_m,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Area width is not divisible by resolution."
        )

    if not math.isclose(
        height_px * resolution,
        height_m,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Area height is not divisible by resolution."
        )

    transform = from_origin(
        min_x,
        max_y,
        resolution,
        resolution,
    )

    return {
        "width": width_px,
        "height": height_px,
        "transform": transform,
        "bounds": bounds,
        "resolution": resolution,
    }


# ============================================================
# PNG
# ============================================================


def save_mask(
    array: np.ndarray,
    filename: Path,
):

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = np.asarray(
        array,
        dtype=np.uint8,
    )

    image = Image.fromarray(
        array,
        mode="L",
    )

    image.save(
        filename,
        format="PNG",
        optimize=True,
    )


# ============================================================
# WORLD FILE
# ============================================================


def save_world_file(
    filename: Path,
    transform,
):

    pgw = filename.with_suffix(
        ".pgw"
    )

    with pgw.open(
        "w",
        encoding="ascii",
    ) as f:

        f.write(
            f"{transform.a:.12f}\n"
        )

        f.write(
            f"{transform.b:.12f}\n"
        )

        f.write(
            f"{transform.d:.12f}\n"
        )

        f.write(
            f"{transform.e:.12f}\n"
        )

        f.write(
            f"{transform.c:.12f}\n"
        )

        f.write(
            f"{transform.f:.12f}\n"
        )


# ============================================================
# OSM WIDTH
# ============================================================


def parse_width(
    value,
) -> Optional[float]:

    if value is None:
        return None

    text = str(value).strip().lower()

    match = re.match(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*$",
        text,
    )

    if not match:
        return None

    try:
        width = float(
            match.group(1)
        )

        if width > 0:
            return width

    except ValueError:
        pass

    return None


# ============================================================
# OVERPASS QUERY
# ============================================================


def build_overpass_query(
    wgs84_bounds,
    config,
):

    west, south, east, north = (
        wgs84_bounds
    )

    bbox = (
        f"{south:.8f},"
        f"{west:.8f},"
        f"{north:.8f},"
        f"{east:.8f}"
    )

    masks = config.get(
        "tf3",
        {}
    ).get(
        "masks",
        {}
    )

    waterways = masks.get(
        "rivers",
        {}
    ).get(
        "waterways",
        {}
    )

    waterway_types = list(
        waterways.keys()
    )

    if not waterway_types:
        waterway_types = [
            "river",
            "stream",
            "canal",
            "ditch",
            "drain",
        ]

    waterway_regex = "|".join(
        re.escape(
            str(x)
        )
        for x in waterway_types
    )

    osm = config.get(
        "tf3",
        {}
    ).get(
        "osm",
        {}
    )

    clauses = []

    # --------------------------------------------------------
    # Waterways
    # --------------------------------------------------------

    clauses.append(
        f'way["waterway"~"^({waterway_regex})$"]({bbox});'
    )

    # --------------------------------------------------------
    # Forest
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "forests",
        True,
    ):

        clauses.extend([
            f'way["natural"="wood"]({bbox});',
            f'way["landuse"="forest"]({bbox});',
        ])

    # --------------------------------------------------------
    # Meadows / grassland
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "meadows",
        True,
    ):

        clauses.extend([
            f'way["landuse"="meadow"]({bbox});',
            f'way["natural"="grassland"]({bbox});',
            f'way["landuse"="grass"]({bbox});',
        ])

    # --------------------------------------------------------
    # Farmland
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "farmland",
        True,
    ):

        clauses.extend([
            f'way["landuse"="farmland"]({bbox});',
            f'way["landuse"="farm"]({bbox});',
            f'way["landuse"="orchard"]({bbox});',
            f'way["landuse"="vineyard"]({bbox});',
        ])

    # --------------------------------------------------------
    # Settlements
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "settlements",
        True,
    ):

        settlement_types = [
            "residential",
            "industrial",
            "commercial",
            "retail",
            "construction",
        ]

        for landuse in settlement_types:

            clauses.append(
                f'way["landuse"="{landuse}"]({bbox});'
            )

    # --------------------------------------------------------
    # Heath
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "heath",
        True,
    ):

        clauses.append(
            f'way["natural"="heath"]({bbox});'
        )

    # --------------------------------------------------------
    # Wetlands
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "wetlands",
        True,
    ):

        clauses.append(
            f'way["natural"="wetland"]({bbox});'
        )

    # --------------------------------------------------------
    # Rocks
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "rocks",
        True,
    ):

        clauses.extend([
            f'way["natural"="bare_rock"]({bbox});',
            f'way["natural"="rock"]({bbox});',
        ])

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    if osm.get(
        "include",
        {}
    ).get(
        "water",
        True,
    ):

        clauses.extend([
            f'way["natural"="water"]({bbox});',
            f'way["landuse"="reservoir"]({bbox});',
        ])

    query = (
        "[out:json][timeout:180];\n"
        "(\n"
        + "\n".join(
            clauses
        )
        + "\n);\n"
        "out body geom;"
    )

    return query


# ============================================================
# OVERPASS DOWNLOAD
# ============================================================


def download_osm(
    url,
    query,
    timeout,
    retries,
    verbose,
):

    headers = {
        "User-Agent": (
            "TF3-Map-Generator/1.0 "
            "(OpenStreetMap data processing)"
        ),
    }

    for attempt in range(
        1,
        retries + 1,
    ):

        log(
            f"Downloading OSM data "
            f"({attempt}/{retries})...",
            verbose,
        )

        try:

            response = requests.post(
                url,
                data=query.encode(
                    "utf-8"
                ),
                headers=headers,
                timeout=timeout,
            )

            response.raise_for_status()

            data = response.json()

            elements = data.get(
                "elements"
            )

            if elements is None:
                raise RuntimeError(
                    "Invalid Overpass response."
                )

            log(
                f"OSM elements: "
                f"{len(elements)}",
                verbose,
            )

            return data

        except Exception as exc:

            log(
                f"Overpass error: {exc}",
                verbose,
            )

            if attempt < retries:

                wait = min(
                    5 * attempt,
                    30,
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        "Could not download OSM data."
    )


# ============================================================
# OSM GEOMETRY
# ============================================================


def element_geometry(
    element,
):

    geometry = element.get(
        "geometry"
    )

    if not geometry:
        return None

    coordinates = []

    for point in geometry:

        lon = point.get(
            "lon"
        )

        lat = point.get(
            "lat"
        )

        if lon is None or lat is None:
            continue

        coordinates.append(
            (
                float(lon),
                float(lat),
            )
        )

    if len(coordinates) < 2:
        return None

    # Closed ways become polygons.
    if (
        len(coordinates) >= 4
        and coordinates[0]
        == coordinates[-1]
    ):

        try:
            polygon = Polygon(
                coordinates
            )

            if polygon.is_valid:
                return polygon

        except Exception:
            pass

    return LineString(
        coordinates
    )


def transform_geometry(
    geometry,
    transformer,
):

    return shapely_transform(
        transformer.transform,
        geometry,
    )


# ============================================================
# OSM CLASSIFICATION
# ============================================================


BIOME_CLASSES = [
    "water",
    "forest",
    "meadow",
    "farmland",
    "settlement",
    "heath",
    "wetland",
    "rock",
]


def classify_element(
    tags: dict,
) -> Optional[str]:

    waterway = tags.get(
        "waterway"
    )

    if waterway in {
        "river",
        "stream",
        "canal",
        "ditch",
        "drain",
    }:

        return "water"

    natural = tags.get(
        "natural"
    )

    landuse = tags.get(
        "landuse"
    )

    if natural in {
        "wood",
    }:

        return "forest"

    if landuse in {
        "forest",
    }:

        return "forest"

    if natural in {
        "grassland",
    }:

        return "meadow"

    if landuse in {
        "meadow",
        "grass",
    }:

        return "meadow"

    if landuse in {
        "farmland",
        "farm",
        "orchard",
        "vineyard",
    }:

        return "farmland"

    if landuse in {
        "residential",
        "industrial",
        "commercial",
        "retail",
        "construction",
    }:

        return "settlement"

    if natural == "heath":

        return "heath"

    if natural == "wetland":

        return "wetland"

    if natural in {
        "bare_rock",
        "rock",
    }:

        return "rock"

    if natural == "water":

        return "water"

    if landuse == "reservoir":

        return "water"

    return None


# ============================================================
# COLLECT OSM FEATURES
# ============================================================


def collect_features(
    osm_data,
    extent,
    verbose,
):

    transformer = extent[
        "to_processing"
    ]

    processing_bounds = extent[
        "bounds"
    ]

    clip = box(
        *processing_bounds
    )

    features = {
        name: []
        for name in BIOME_CLASSES
    }

    river_lines = []

    count = 0

    for element in osm_data.get(
        "elements",
        [],
    ):

        if element.get(
            "type"
        ) != "way":

            continue

        tags = element.get(
            "tags",
            {},
        )

        classification = (
            classify_element(
                tags
            )
        )

        if classification is None:
            continue

        geometry = (
            element_geometry(
                element
            )
        )

        if geometry is None:
            continue

        geometry = (
            transform_geometry(
                geometry,
                transformer,
            )
        )

        geometry = geometry.intersection(
            clip
        )

        if geometry.is_empty:
            continue

        features[
            classification
        ].append(
            geometry
        )

        # Keep river geometry separately.
        if tags.get(
            "waterway"
        ) in {
            "river",
            "stream",
            "canal",
            "ditch",
            "drain",
        }:

            width = parse_width(
                tags.get(
                    "width"
                )
            )

            river_lines.append(
                (
                    geometry,
                    tags.get(
                        "waterway"
                    ),
                    width,
                )
            )

        count += 1

    log(
        f"Classified OSM features: {count}",
        verbose,
    )

    for name in BIOME_CLASSES:

        log(
            f"  {name:12s}: "
            f"{len(features[name])}",
            verbose,
        )

    return (
        features,
        river_lines,
    )


# ============================================================
# RASTERIZE POLYGONS
# ============================================================


def rasterize_geometries(
    geometries,
    grid,
):

    if not geometries:

        return np.zeros(
            (
                grid["height"],
                grid["width"],
            ),
            dtype=np.uint8,
        )

    shapes = [
        (
            geometry,
            255,
        )
        for geometry in geometries
    ]

    return rasterize(
        shapes,
        out_shape=(
            grid["height"],
            grid["width"],
        ),
        transform=grid["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


# ============================================================
# RIVER MASK
# ============================================================


def create_river_mask(
    river_lines,
    grid,
    config,
):

    river_config = (
        config
        .get("tf3", {})
        .get("masks", {})
        .get("rivers", {})
    )

    waterways = river_config.get(
        "waterways",
        {},
    )

    use_osm_width = bool(
        river_config.get(
            "use_osm_width",
            True,
        )
    )

    minimum_width = float(
        river_config.get(
            "minimum_width_m",
            2.0,
        )
    )

    shapes = []

    for geometry, waterway, osm_width in (
        river_lines
    ):

        settings = waterways.get(
            waterway,
            {},
        )

        default_width = float(
            settings.get(
                "width_m",
                minimum_width,
            )
        )

        if (
            use_osm_width
            and osm_width is not None
        ):

            width = osm_width

        else:

            width = default_width

        width = max(
            width,
            minimum_width,
        )

        buffered = geometry.buffer(
            width / 2.0,
            cap_style=2,
            join_style=2,
            resolution=4,
        )

        if not buffered.is_empty:

            shapes.append(
                (
                    buffered,
                    255,
                )
            )

    if not shapes:

        return np.zeros(
            (
                grid["height"],
                grid["width"],
            ),
            dtype=np.uint8,
        )

    return rasterize(
        shapes,
        out_shape=(
            grid["height"],
            grid["width"],
        ),
        transform=grid["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


# ============================================================
# MOUNTAIN MASK
# ============================================================


def calculate_slope(
    dem,
    resolution,
):

    # np.gradient returns elevation change per pixel.
    gy, gx = np.gradient(
        dem.astype(
            np.float32
        ),
        resolution,
        resolution,
    )

    slope_rad = np.arctan(
        np.sqrt(
            gx * gx
            +
            gy * gy
        )
    )

    slope_deg = (
        slope_rad
        *
        180.0
        /
        math.pi
    )

    return slope_deg


def create_mountain_mask(
    dem,
    resolution,
    config,
    verbose,
):

    mountain_config = (
        config
        .get("tf3", {})
        .get("masks", {})
        .get("mountains", {})
    )

    elevation_config = (
        mountain_config.get(
            "elevation",
            {},
        )
    )

    slope_config = (
        mountain_config.get(
            "slope",
            {},
        )
    )

    elevation_enabled = bool(
        elevation_config.get(
            "enabled",
            True,
        )
    )

    slope_enabled = bool(
        slope_config.get(
            "enabled",
            True,
        )
    )

    elevation_weight = float(
        mountain_config.get(
            "elevation_weight",
            0.55,
        )
    )

    slope_weight = float(
        mountain_config.get(
            "slope_weight",
            0.45,
        )
    )

    # --------------------------------------------------------
    # Valid DEM
    # --------------------------------------------------------

    valid = np.isfinite(
        dem
    )

    if not np.any(valid):

        raise RuntimeError(
            "DEM contains no valid elevation data."
        )

    result = np.zeros(
        dem.shape,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    elevation_component = (
        np.zeros_like(
            dem,
            dtype=np.float32,
        )
    )

    if elevation_enabled:

        start = float(
            elevation_config.get(
                "start_m",
                300.0,
            )
        )

        full = float(
            elevation_config.get(
                "full_m",
                600.0,
            )
        )

        elevation_component = normalize(
            dem,
            start,
            full,
        )

        elevation_component[
            ~valid
        ] = 0.0

    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    slope_component = (
        np.zeros_like(
            dem,
            dtype=np.float32,
        )
    )

    if slope_enabled:

        slope = calculate_slope(
            np.where(
                valid,
                dem,
                0.0,
            ),
            resolution,
        )

        slope_start = float(
            slope_config.get(
                "start_deg",
                12.0,
            )
        )

        slope_full = float(
            slope_config.get(
                "full_deg",
                35.0,
            )
        )

        slope_component = normalize(
            slope,
            slope_start,
            slope_full,
        )

        slope_component[
            ~valid
        ] = 0.0

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    weight_sum = 0.0

    if elevation_enabled:
        weight_sum += elevation_weight

    if slope_enabled:
        weight_sum += slope_weight

    if weight_sum <= 0:

        raise ValueError(
            "Mountain elevation/slope weights "
            "cannot both be zero."
        )

    result = (
        elevation_component
        *
        elevation_weight
        +
        slope_component
        *
        slope_weight
    ) / weight_sum

    result[
        ~valid
    ] = 0.0

    result = (
        np.clip(
            result,
            0.0,
            1.0,
        )
        *
        255.0
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# BIOME MASK
# ============================================================


def create_biome_mask(
    individual_masks,
    config,
):

    values = (
        config
        .get("tf3", {})
        .get("masks", {})
        .get("biomes", {})
        .get("values", {})
    )

    output = np.zeros_like(
        next(
            iter(
                individual_masks.values()
            )
        ),
        dtype=np.uint8,
    )

    # Priority order.
    #
    # Water gets highest priority.
    # Then settlement, farmland, etc.

    priority = [
        "rock",
        "heath",
        "wetland",
        "meadow",
        "forest",
        "farmland",
        "settlement",
        "water",
    ]

    for biome in priority:

        mask = individual_masks.get(
            biome
        )

        if mask is None:
            continue

        value = int(
            values.get(
                biome,
                0,
            )
        )

        output[
            mask > 0
        ] = value

    return output


# ============================================================
# LOAD DEM
# ============================================================


def find_dem(
    config,
):

    output = config.get(
        "output",
        {}
    )

    directory = Path(
        output.get(
            "directory",
            "./output",
        )
    )

    filename = output.get(
        "filename"
    )

    if filename:

        candidate = (
            directory / filename
        )

        if candidate.exists():

            return candidate

    candidates = [
        directory / "dem.tif",
        directory / "dgm.tif",
        directory / "dgm_4m.tif",
        directory / "processed_dem.tif",
    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    # Search for GeoTIFFs.
    tif_files = list(
        directory.glob(
            "*.tif"
        )
    )

    if len(tif_files) == 1:

        return tif_files[0]

    if not tif_files:

        raise FileNotFoundError(
            "Could not find DEM GeoTIFF in "
            f"{directory}"
        )

    raise FileNotFoundError(
        "Multiple GeoTIFFs found. "
        "Set output.filename to the DEM "
        "you want to use."
    )


def load_dem(
    filename,
    bounds,
    grid,
    processing_crs,
    verbose,
):

    log(
        f"Loading DEM: {filename}",
        verbose,
    )

    with rasterio.open(
        filename
    ) as src:

        log(
            f"DEM CRS: {src.crs}",
            verbose,
        )

        # ----------------------------------------------------
        # Reproject/resample DEM directly onto our exact grid.
        # ----------------------------------------------------

        from rasterio.warp import (
            reproject,
            Resampling,
        )

        destination = np.full(
            (
                grid["height"],
                grid["width"],
            ),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(
                src,
                1,
            ),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=processing_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

    return destination


# ============================================================
# WRITE CRS
# ============================================================


def save_prj(
    filename,
    crs,
):

    prj = filename.with_suffix(
        ".prj"
    )

    prj.write_text(
        crs.to_wkt(),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate TF3 8-bit grayscale "
            "terrain masks."
        )
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    runtime = config.get(
        "runtime",
        {}
    )

    verbose = bool(
        runtime.get(
            "verbose",
            True,
        )
    )

    tf3 = config.get(
        "tf3",
        {}
    )

    if not tf3.get(
        "enabled",
        True,
    ):

        print(
            "TF3 generation disabled."
        )

        return 0

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    resolution = float(
        config[
            "resolution_m"
        ]
    )

    if resolution <= 0:

        raise ValueError(
            "resolution_m must be > 0."
        )

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    extent = calculate_extent(
        config
    )

    grid = create_grid(
        extent["bounds"],
        resolution,
    )

    output_directory = Path(
        tf3.get(
            "output_directory",
            "./output/tf3",
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        "TF3 MAP GENERATOR",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        f"CRS:        "
        f"{extent['processing_crs'].to_string()}",
        verbose,
    )

    log(
        f"Resolution: {resolution} m",
        verbose,
    )

    log(
        f"Size:       "
        f"{grid['width']} x "
        f"{grid['height']} px",
        verbose,
    )

    log(
        f"Output:     {output_directory}",
        verbose,
    )

    # --------------------------------------------------------
    # DEM
    # --------------------------------------------------------

    dem_filename = find_dem(
        config
    )

    dem = load_dem(
        dem_filename,
        extent["bounds"],
        grid,
        extent["processing_crs"],
        verbose,
    )

    # --------------------------------------------------------
    # Mountain mask
    # --------------------------------------------------------

    mountains_config = (
        tf3
        .get("masks", {})
        .get("mountains", {})
    )

    if mountains_config.get(
        "enabled",
        True,
    ):

        log(
            "",
            verbose,
        )

        log(
            "Creating mountain mask...",
            verbose,
        )

        mountain_mask = (
            create_mountain_mask(
                dem,
                resolution,
                config,
                verbose,
            )
        )

        filename = (
            output_directory
            /
            mountains_config.get(
                "filename",
                "mountain_mask.png",
            )
        )

        save_mask(
            mountain_mask,
            filename,
        )

        save_world_file(
            filename,
            grid["transform"],
        )

        save_prj(
            filename,
            extent["processing_crs"],
        )

    # --------------------------------------------------------
    # OSM
    # --------------------------------------------------------

    osm_config = tf3.get(
        "osm",
        {}
    )

    overpass_url = osm_config.get(
        "overpass_url",
        "https://overpass-api.de/api/interpreter",
    )

    timeout = int(
        osm_config.get(
            "timeout_seconds",
            180,
        )
    )

    retries = int(
        osm_config.get(
            "retries",
            5,
        )
    )

    query = build_overpass_query(
        extent["wgs84_bounds"],
        config,
    )

    log(
        "",
        verbose,
    )

    log(
        "Downloading OSM data...",
        verbose,
    )

    osm_data = download_osm(
        overpass_url,
        query,
        timeout,
        retries,
        verbose,
    )

    # --------------------------------------------------------
    # Collect OSM features
    # --------------------------------------------------------

    features, river_lines = (
        collect_features(
            osm_data,
            extent,
            verbose,
        )
    )

    # --------------------------------------------------------
    # River mask
    # --------------------------------------------------------

    rivers_config = (
        tf3
        .get("masks", {})
        .get("rivers", {})
    )

    if rivers_config.get(
        "enabled",
        True,
    ):

        log(
            "Creating river mask...",
            verbose,
        )

        river_mask = (
            create_river_mask(
                river_lines,
                grid,
                config,
            )
        )

        filename = (
            output_directory
            /
            rivers_config.get(
                "filename",
                "river_mask.png",
            )
        )

        save_mask(
            river_mask,
            filename,
        )

        save_world_file(
            filename,
            grid["transform"],
        )

        save_prj(
            filename,
            extent["processing_crs"],
        )

    # --------------------------------------------------------
    # Water mask
    # --------------------------------------------------------

    water_config = (
        tf3
        .get("masks", {})
        .get("water", {})
    )

    individual_masks = {}

    if water_config.get(
        "enabled",
        True,
    ):

        log(
            "Creating water mask...",
            verbose,
        )

        water_mask = (
            rasterize_geometries(
                features["water"],
                grid,
            )
        )

        individual_masks[
            "water"
        ] = water_mask

        filename = (
            output_directory
            /
            water_config.get(
                "filename",
                "water_mask.png",
            )
        )

        save_mask(
            water_mask,
            filename,
        )

        save_world_file(
            filename,
            grid["transform"],
        )

        save_prj(
            filename,
            extent["processing_crs"],
        )

    # --------------------------------------------------------
    # Individual biome masks
    # --------------------------------------------------------

    individual_config = (
        tf3
        .get("masks", {})
        .get("individual_masks", {})
    )

    if individual_config.get(
        "enabled",
        True,
    ):

        for biome in [
            "forest",
            "meadow",
            "farmland",
            "settlement",
            "heath",
            "wetland",
            "rock",
        ]:

            settings = (
                individual_config.get(
                    biome,
                    {},
                )
            )

            if not settings.get(
                "enabled",
                True,
            ):
                continue

            log(
                f"Creating "
                f"{biome} mask...",
                verbose,
            )

            mask = (
                rasterize_geometries(
                    features[biome],
                    grid,
                )
            )

            value = int(
                settings.get(
                    "value",
                    255,
                )
            )

            if value != 255:

                mask = np.where(
                    mask > 0,
                    value,
                    0,
                ).astype(
                    np.uint8
                )

            individual_masks[
                biome
            ] = mask

            filename = (
                output_directory
                /
                settings.get(
                    "filename",
                    f"{biome}_mask.png",
                )
            )

            save_mask(
                mask,
                filename,
            )

            save_world_file(
                filename,
                grid["transform"],
            )

            save_prj(
                filename,
                extent["processing_crs"],
            )

    # --------------------------------------------------------
    # Combined biome mask
    # --------------------------------------------------------

    biome_config = (
        tf3
        .get("masks", {})
        .get("biomes", {})
    )

    if biome_config.get(
        "enabled",
        True,
    ) and individual_masks:

        log(
            "Creating combined biome mask...",
            verbose,
        )

        biome_mask = (
            create_biome_mask(
                individual_masks,
                config,
            )
        )

        filename = (
            output_directory
            /
            biome_config.get(
                "filename",
                "biome_mask.png",
            )
        )

        save_mask(
            biome_mask,
            filename,
        )

        save_world_file(
            filename,
            grid["transform"],
        )

        save_prj(
            filename,
            extent["processing_crs"],
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    log(
        "",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        "TF3 GENERATION COMPLETE",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        f"Raster: "
        f"{grid['width']} x "
        f"{grid['height']}",
        verbose,
    )

    log(
        f"Resolution: "
        f"{grid['resolution']} m/pixel",
        verbose,
    )

    log(
        f"CRS: "
        f"{extent['processing_crs'].to_string()}",
        verbose,
    )

    log(
        f"Output: "
        f"{output_directory}",
        verbose,
    )

    log(
        "",
        verbose,
    )

    for file in sorted(
        output_directory.glob(
            "*.png"
        )
    ):

        log(
            f"  {file.name}",
            verbose,
        )

    log(
        "",
        verbose,
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

        sys.exit(
            130
        )

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(
            1
        )