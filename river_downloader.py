#!/usr/bin/env python3

"""
LGL / OpenStreetMap River Mask Generator

Reads the same YAML configuration used by the DGM downloader and creates
an 8-bit grayscale PNG terrain mask suitable for TF3.

Coordinate workflow:

    WGS84 center
        |
        v
    EPSG:25832
        |
        +---- exact DEM/mask extent
        |
        v
    WGS84 bounding box
        |
        v
    OpenStreetMap / Overpass
        |
        v
    EPSG:25832 geometries
        |
        v
    rasterized mask
        |
        v
    8-bit grayscale PNG

Output:

    0   = background
    255 = river/waterway

For example:

    28,000 m / 4 m = 7,000 pixels

Therefore a 28 km x 28 km area produces:

    7000 x 7000 PNG
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import yaml

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box
from shapely.ops import transform as shapely_transform

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from PIL import Image

from config_loader import load_config


# ============================================================
# CONFIG
# ============================================================


# ============================================================
# HELPERS
# ============================================================


def log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def parse_osm_width(value: Optional[str]) -> Optional[float]:
    """
    Parse common OSM width formats.

    Examples:

        15
        15.0
        15 m
        15m

    Returns meters or None.
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    # Simple numeric value
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*$", text)

    if match:
        try:
            width = float(match.group(1))
            if width > 0:
                return width
        except ValueError:
            pass

    return None


def meters_to_degrees_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
) -> str:
    """
    Format an Overpass bbox.

    Overpass expects:

        south,west,north,east
    """

    return f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"


# ============================================================
# AREA CALCULATION
# ============================================================


def calculate_processing_extent(config: dict):
    center = config["center"]
    area = config["area"]
    crs_config = config["crs"]

    latitude = float(center["latitude"])
    longitude = float(center["longitude"])

    width_m = float(area["width_m"])
    height_m = float(area["height_m"])

    input_crs = CRS.from_user_input(crs_config["input"])
    processing_crs = CRS.from_user_input(crs_config["processing"])

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

    center_x, center_y = to_processing.transform(
        longitude,
        latitude,
    )

    half_width = width_m / 2.0
    half_height = height_m / 2.0

    min_x = center_x - half_width
    max_x = center_x + half_width

    min_y = center_y - half_height
    max_y = center_y + half_height

    processing_bounds = (
        min_x,
        min_y,
        max_x,
        max_y,
    )

    # Transform all four corners to WGS84.
    corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ]

    wgs84_corners = [
        to_wgs84.transform(x, y)
        for x, y in corners
    ]

    longitudes = [p[0] for p in wgs84_corners]
    latitudes = [p[1] for p in wgs84_corners]

    west = min(longitudes)
    east = max(longitudes)
    south = min(latitudes)
    north = max(latitudes)

    wgs84_bounds = (
        west,
        south,
        east,
        north,
    )

    return {
        "center_processing": (center_x, center_y),
        "processing_bounds": processing_bounds,
        "wgs84_bounds": wgs84_bounds,
        "input_crs": input_crs,
        "processing_crs": processing_crs,
        "to_processing": to_processing,
        "to_wgs84": to_wgs84,
        "width_m": width_m,
        "height_m": height_m,
    }


# ============================================================
# OVERPASS
# ============================================================


def build_overpass_query(
    wgs84_bounds: Tuple[float, float, float, float],
    waterways: List[str],
    include_river_areas: bool,
) -> str:

    west, south, east, north = wgs84_bounds

    bbox = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"

    waterway_regex = "|".join(
        re.escape(str(x))
        for x in waterways
    )

    parts = []

    # Linear waterways
    parts.append(
        f'way["waterway"~"^({waterway_regex})$"]({bbox});'
    )

    # Optional river polygons
    if include_river_areas:
        parts.append(
            f'way["natural"="water"]["water"="river"]({bbox});'
        )

    query = f"""
[out:json][timeout:180];

(
    {''.join(parts)}
);

out body geom;
"""

    return query


def download_osm(
    url: str,
    query: str,
    timeout: int,
    retries: int,
    verbose: bool,
) -> dict:

    headers = {
        "User-Agent": (
            "LGL-DGM1-RiverMask/1.0 "
            "(OpenStreetMap/Overpass data processing)"
        ),
        "Content-Type": "application/x-www-form-urlencoded",
    }

    for attempt in range(1, retries + 1):

        log(
            f"Downloading OSM waterways "
            f"(attempt {attempt}/{retries})...",
            verbose,
        )

        try:
            response = requests.post(
                url,
                data=query.encode("utf-8"),
                headers=headers,
                timeout=timeout,
            )

            response.raise_for_status()

            data = response.json()

            if "elements" not in data:
                raise RuntimeError(
                    "Overpass response does not contain 'elements'."
                )

            log(
                f"Received {len(data['elements'])} OSM elements.",
                verbose,
            )

            return data

        except Exception as exc:

            log(
                f"Overpass request failed: {exc}",
                verbose,
            )

            if attempt < retries:
                wait = min(5 * attempt, 30)

                log(
                    f"Retrying in {wait} seconds...",
                    verbose,
                )

                time.sleep(wait)

    raise RuntimeError(
        "Unable to download OSM data from Overpass."
    )


# ============================================================
# OSM GEOMETRY
# ============================================================


def osm_way_to_geometry(element: dict):
    geometry = element.get("geometry")

    if not geometry:
        return None

    coordinates = []

    for point in geometry:

        lon = point.get("lon")
        lat = point.get("lat")

        if lon is None or lat is None:
            continue

        coordinates.append(
            (float(lon), float(lat))
        )

    if len(coordinates) < 2:
        return None

    return LineString(coordinates)


def transform_geometry(
    geometry,
    transformer: Transformer,
):
    return shapely_transform(
        transformer.transform,
        geometry,
    )


def clip_geometry(
    geometry,
    clip_box,
):
    clipped = geometry.intersection(clip_box)

    if clipped.is_empty:
        return None

    return clipped


# ============================================================
# WIDTH HANDLING
# ============================================================


def determine_width(
    tags: dict,
    default_widths: Dict[str, float],
    minimum_width: float,
) -> float:

    waterway = tags.get("waterway")

    osm_width = parse_osm_width(
        tags.get("width")
    )

    if osm_width is not None:
        return max(
            osm_width,
            minimum_width,
        )

    default = default_widths.get(
        waterway,
        minimum_width,
    )

    return max(
        float(default),
        minimum_width,
    )


# ============================================================
# GEOMETRY COLLECTION
# ============================================================


def collect_waterways(
    osm_data: dict,
    transformer: Transformer,
    processing_bounds: Tuple[float, float, float, float],
    waterways: List[str],
    default_widths: Dict[str, float],
    minimum_width: float,
    include_river_areas: bool,
    verbose: bool,
):

    clip_box = box(
        *processing_bounds
    )

    allowed = set(
        str(x)
        for x in waterways
    )

    line_features = []
    polygon_features = []

    total = 0
    accepted = 0

    for element in osm_data.get("elements", []):

        if element.get("type") != "way":
            continue

        tags = element.get("tags", {})

        natural = tags.get("natural")
        waterway = tags.get("waterway")

        is_linear_waterway = (
            waterway in allowed
        )

        is_river_area = (
            include_river_areas
            and natural == "water"
            and tags.get("water") == "river"
        )

        if not is_linear_waterway and not is_river_area:
            continue

        total += 1

        geometry = osm_way_to_geometry(
            element
        )

        if geometry is None:
            continue

        geometry = transform_geometry(
            geometry,
            transformer,
        )

        geometry = clip_geometry(
            geometry,
            clip_box,
        )

        if geometry is None:
            continue

        if is_linear_waterway:

            width = determine_width(
                tags,
                default_widths,
                minimum_width,
            )

            line_features.append(
                (geometry, width)
            )

        elif is_river_area:

            polygon_features.append(
                geometry
            )

        accepted += 1

    log(
        f"OSM water features found: {total}",
        verbose,
    )

    log(
        f"Water features inside area: {accepted}",
        verbose,
    )

    log(
        f"Linear waterways: {len(line_features)}",
        verbose,
    )

    log(
        f"River polygons: {len(polygon_features)}",
        verbose,
    )

    return line_features, polygon_features


# ============================================================
# RASTER CREATION
# ============================================================


def create_mask(
    line_features,
    polygon_features,
    processing_bounds,
    resolution,
    background_value,
    river_value,
    output_path,
    processing_crs,
    verbose,
):

    min_x, min_y, max_x, max_y = processing_bounds

    width_m = max_x - min_x
    height_m = max_y - min_y

    # Force exact integer dimensions.
    raster_width = int(
        round(width_m / resolution)
    )

    raster_height = int(
        round(height_m / resolution)
    )

    actual_width = raster_width * resolution
    actual_height = raster_height * resolution

    if abs(actual_width - width_m) > 1e-6:
        raise ValueError(
            "Area width is not divisible by resolution."
        )

    if abs(actual_height - height_m) > 1e-6:
        raise ValueError(
            "Area height is not divisible by resolution."
        )

    # IMPORTANT:
    #
    # Raster origin is upper-left:
    #
    #   west, north
    #
    transform = from_origin(
        min_x,
        max_y,
        resolution,
        resolution,
    )

    log(
        f"Raster size: {raster_width} x {raster_height}",
        verbose,
    )

    log(
        f"Resolution: {resolution} m/pixel",
        verbose,
    )

    log(
        f"Raster extent: "
        f"{min_x:.3f}, "
        f"{min_y:.3f}, "
        f"{max_x:.3f}, "
        f"{max_y:.3f}",
        verbose,
    )

    # --------------------------------------------------------
    # Prepare shapes
    # --------------------------------------------------------

    shapes = []

    # Linear waterways are buffered using their OSM/default width.
    for geometry, width in line_features:

        try:
            buffered = geometry.buffer(
                width / 2.0,
                cap_style=2,
                join_style=2,
                resolution=4,
            )

            if not buffered.is_empty:
                shapes.append(
                    (buffered, river_value)
                )

        except Exception as exc:

            log(
                f"Warning: could not buffer river: {exc}",
                verbose,
            )

    # River polygons
    for geometry in polygon_features:

        if not geometry.is_empty:
            shapes.append(
                (geometry, river_value)
            )

    log(
        f"Rasterizing {len(shapes)} water features...",
        verbose,
    )

    mask = rasterize(
        shapes=shapes,
        out_shape=(
            raster_height,
            raster_width,
        ),
        transform=transform,
        fill=background_value,
        dtype="uint8",
        all_touched=True,
    )

    # --------------------------------------------------------
    # Save GeoTIFF temporarily
    # --------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # PNG itself cannot reliably store the complete georeferencing
    # information required by all GIS software, so we also write
    # a world file below.
    #
    # The GeoTIFF is useful as an optional diagnostic.
    tif_path = output_path.with_suffix(".tif")

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=raster_width,
        height=raster_height,
        count=1,
        dtype="uint8",
        crs=processing_crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as dst:

        dst.write(
            mask,
            1,
        )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    image = Image.fromarray(
        mask,
        mode="L",
    )

    image.save(
        output_path,
        format="PNG",
        optimize=True,
    )

    # --------------------------------------------------------
    # World file
    # --------------------------------------------------------

    # PNG world file extension is .pgw
    pgw_path = output_path.with_suffix(".pgw")

    with pgw_path.open(
        "w",
        encoding="ascii",
    ) as f:

        # Pixel size X
        f.write(f"{resolution:.12f}\n")

        # Rotation X
        f.write("0.000000000000\n")

        # Rotation Y
        f.write("0.000000000000\n")

        # Pixel size Y is negative because image coordinates
        # increase downward.
        f.write(f"{-resolution:.12f}\n")

        # X coordinate of center of upper-left pixel
        f.write(
            f"{min_x + resolution / 2.0:.12f}\n"
        )

        # Y coordinate of center of upper-left pixel
        f.write(
            f"{max_y - resolution / 2.0:.12f}\n"
        )

    # --------------------------------------------------------
    # CRS sidecar
    # --------------------------------------------------------

    prj_path = output_path.with_suffix(".prj")

    prj_path.write_text(
        processing_crs.to_wkt(),
        encoding="utf-8",
    )

    log(
        f"PNG written: {output_path}",
        verbose,
    )

    log(
        f"GeoTIFF diagnostic written: {tif_path}",
        verbose,
    )

    log(
        f"World file written: {pgw_path}",
        verbose,
    )

    log(
        f"CRS file written: {prj_path}",
        verbose,
    )

    return raster_width, raster_height


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create an 8-bit grayscale OSM river mask "
            "using the DGM YAML configuration."
        )
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to YAML configuration.",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    rivers = config.get(
        "rivers",
        {},
    )

    enabled = rivers.get(
        "enabled",
        True,
    )

    if not enabled:
        print(
            "River mask generation is disabled in configuration."
        )
        return 0

    verbose = bool(
        rivers.get(
            "verbose",
            config.get(
                "runtime",
                {}
            ).get(
                "verbose",
                True,
            ),
        )
    )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    resolution = float(
        config["resolution_m"]
    )

    if resolution <= 0:
        raise ValueError(
            "resolution_m must be greater than zero."
        )

    output_config = config.get(
        "output",
        {}
    )

    output_directory = Path(
        output_config.get(
            "directory",
            "./output",
        )
    )

    river_output = rivers.get(
        "output",
        {}
    )

    filename = river_output.get(
        "filename",
        "river_mask.png",
    )

    output_path = (
        output_directory / filename
    )

    background_value = int(
        rivers.get(
            "background_value",
            0,
        )
    )

    river_value = int(
        rivers.get(
            "river_value",
            255,
        )
    )

    if not 0 <= background_value <= 255:
        raise ValueError(
            "background_value must be between 0 and 255."
        )

    if not 0 <= river_value <= 255:
        raise ValueError(
            "river_value must be between 0 and 255."
        )

    waterways = rivers.get(
        "waterways",
        [
            "river",
            "stream",
            "canal",
            "ditch",
            "drain",
        ],
    )

    include_river_areas = bool(
        rivers.get(
            "include_river_areas",
            True,
        )
    )

    minimum_width = float(
        rivers.get(
            "minimum_width_m",
            2.0,
        )
    )

    default_widths = {
        str(k): float(v)
        for k, v in rivers.get(
            "default_width_m",
            {},
        ).items()
    }

    overpass = rivers.get(
        "overpass",
        {},
    )

    overpass_url = overpass.get(
        "url",
        "https://overpass-api.de/api/interpreter",
    )

    overpass_timeout = int(
        overpass.get(
            "timeout_seconds",
            180,
        )
    )

    # --------------------------------------------------------
    # Calculate exact processing area
    # --------------------------------------------------------

    extent = calculate_processing_extent(
        config
    )

    processing_bounds = extent[
        "processing_bounds"
    ]

    wgs84_bounds = extent[
        "wgs84_bounds"
    ]

    processing_crs = extent[
        "processing_crs"
    ]

    to_processing = extent[
        "to_processing"
    ]

    log(
        "",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        "OSM RIVER MASK GENERATOR",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        f"Processing CRS: {processing_crs.to_string()}",
        verbose,
    )

    log(
        f"Resolution: {resolution} m",
        verbose,
    )

    log(
        f"Processing bounds: {processing_bounds}",
        verbose,
    )

    log(
        "WGS84 bounding box:",
        verbose,
    )

    log(
        f"  West : {wgs84_bounds[0]:.8f}",
        verbose,
    )

    log(
        f"  South: {wgs84_bounds[1]:.8f}",
        verbose,
    )

    log(
        f"  East : {wgs84_bounds[2]:.8f}",
        verbose,
    )

    log(
        f"  North: {wgs84_bounds[3]:.8f}",
        verbose,
    )

    # --------------------------------------------------------
    # Overpass
    # --------------------------------------------------------

    query = build_overpass_query(
        wgs84_bounds,
        waterways,
        include_river_areas,
    )

    osm_data = download_osm(
        overpass_url,
        query,
        overpass_timeout,
        retries=5,
        verbose=verbose,
    )

    # --------------------------------------------------------
    # Convert OSM geometries
    # --------------------------------------------------------

    line_features, polygon_features = (
        collect_waterways(
            osm_data,
            to_processing,
            processing_bounds,
            waterways,
            default_widths,
            minimum_width,
            include_river_areas,
            verbose,
        )
    )

    # --------------------------------------------------------
    # Rasterize
    # --------------------------------------------------------

    raster_width, raster_height = create_mask(
        line_features=line_features,
        polygon_features=polygon_features,
        processing_bounds=processing_bounds,
        resolution=resolution,
        background_value=background_value,
        river_value=river_value,
        output_path=output_path,
        processing_crs=processing_crs,
        verbose=verbose,
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
        "DONE",
        verbose,
    )

    log(
        "============================================================",
        verbose,
    )

    log(
        f"PNG:       {output_path}",
        verbose,
    )

    log(
        f"Dimensions: {raster_width} x {raster_height}",
        verbose,
    )

    log(
        f"Resolution: {resolution} m/pixel",
        verbose,
    )

    log(
        f"Background: {background_value}",
        verbose,
    )

    log(
        f"River:      {river_value}",
        verbose,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nInterrupted."
        )
        sys.exit(130)

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)