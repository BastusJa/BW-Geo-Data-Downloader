#!/usr/bin/env python3

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rasterio
import yaml

from pyproj import Transformer
from tqdm import tqdm

from config_loader import load_config


# ============================================================
# CONFIG
# ============================================================

CONFIG_FILE = "config.yaml"



# ============================================================
# VALIDATION
# ============================================================

def validate_config(config: dict[str, Any]) -> None:

    area = config["area"]

    lat = float(
        area["center"]["latitude"]
    )

    lon = float(
        area["center"]["longitude"]
    )

    width = float(
        area["width_m"]
    )

    height = float(
        area["height_m"]
    )

    resolution = float(
        area["resolution_m"]
    )

    if not -90 <= lat <= 90:
        raise ValueError(
            "center.latitude must be between -90 and 90"
        )

    if not -180 <= lon <= 180:
        raise ValueError(
            "center.longitude must be between -180 and 180"
        )

    if width <= 0:
        raise ValueError(
            "area.width_m must be > 0"
        )

    if height <= 0:
        raise ValueError(
            "area.height_m must be > 0"
        )

    if resolution <= 0:
        raise ValueError(
            "area.resolution_m must be > 0"
        )

    aggregation = config[
        "processing"
    ][
        "aggregation"
    ]

    if aggregation not in (
        "mean",
        "min",
        "max",
    ):
        raise ValueError(
            "processing.aggregation must be "
            "'mean', 'min' or 'max'"
        )


# ============================================================
# LOGGING
# ============================================================

def log(
    message: str,
    config: dict[str, Any],
) -> None:

    if config[
        "runtime"
    ].get(
        "verbose",
        True,
    ):

        print(message)


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

def transform_center(
    config: dict[str, Any],
) -> tuple[float, float]:

    center = config[
        "area"
    ][
        "center"
    ]

    input_crs = config[
        "coordinate_system"
    ][
        "input_crs"
    ]

    processing_crs = config[
        "coordinate_system"
    ][
        "processing_crs"
    ]

    transformer = Transformer.from_crs(
        input_crs,
        processing_crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        float(center["longitude"]),
        float(center["latitude"]),
    )

    return x, y


# ============================================================
# BOUNDING BOX
# ============================================================

def calculate_bbox(
    config: dict[str, Any],
) -> tuple[float, float, float, float]:

    center_x, center_y = transform_center(
        config
    )

    area = config["area"]

    width = float(
        area["width_m"]
    )

    height = float(
        area["height_m"]
    )

    resolution = float(
        area["resolution_m"]
    )

    # --------------------------------------------------------
    # Make dimensions compatible with target resolution.
    # --------------------------------------------------------

    width_cells = round(
        width / resolution
    )

    height_cells = round(
        height / resolution
    )

    actual_width = (
        width_cells
        * resolution
    )

    actual_height = (
        height_cells
        * resolution
    )

    min_x = (
        center_x
        - actual_width / 2
    )

    max_x = (
        center_x
        + actual_width / 2
    )

    min_y = (
        center_y
        - actual_height / 2
    )

    max_y = (
        center_y
        + actual_height / 2
    )

    # Align bounding box to resolution
    min_x = (
        math.floor(
            min_x / resolution
        )
        * resolution
    )

    min_y = (
        math.floor(
            min_y / resolution
        )
        * resolution
    )

    max_x = (
        min_x
        + width_cells * resolution
    )

    max_y = (
        min_y
        + height_cells * resolution
    )

    return (
        min_x,
        min_y,
        max_x,
        max_y,
    )


# ============================================================
# LGL TILE CALCULATION
# ============================================================

def calculate_lgl_tiles(
    bbox: tuple[float, float, float, float],
    config: dict[str, Any],
) -> list[tuple[int, int]]:

    min_x, min_y, max_x, max_y = bbox

    tile_size = int(
        config[
            "lgl"
        ][
            "tile_size_m"
        ]
    )

    first_x = (
        math.floor(
            min_x / tile_size
        )
        * tile_size
    )

    last_x = (
        math.ceil(
            max_x / tile_size
        )
        * tile_size
        - tile_size
    )

    first_y = (
        math.floor(
            min_y / tile_size
        )
        * tile_size
    )

    last_y = (
        math.ceil(
            max_y / tile_size
        )
        * tile_size
        - tile_size
    )

    tiles = []

    y = first_y

    while y <= last_y:

        x = first_x

        while x <= last_x:

            x_km = int(
                x / 1000
            )

            y_km = int(
                y / 1000
            )

            # LGL DGM1 naming scheme:
            #
            # Easting  = odd km
            # Northing = even km

            if x_km % 2 == 0:
                x_km += 1

            if y_km % 2 != 0:
                y_km += 1

            tiles.append(
                (
                    x_km,
                    y_km,
                )
            )

            x += tile_size

        y += tile_size

    return sorted(
        set(tiles)
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_tile(
    x: int,
    y: int,
    config: dict[str, Any],
) -> Path | None:

    download_config = config[
        "download"
    ]

    lgl_config = config[
        "lgl"
    ]

    directory = Path(
        download_config[
            "directory"
        ]
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"dgm1_32_{x}_{y}_2_bw.zip"
    )

    destination = (
        directory
        / filename
    )

    url = lgl_config[
        "url_template"
    ].format(
        x=x,
        y=y,
    )

    if destination.exists():

        log(
            f"[SKIP] {filename}",
            config,
        )

        return destination

    max_retries = int(
        download_config[
            "max_retries"
        ]
    )

    timeout = int(
        download_config[
            "timeout_seconds"
        ]
    )

    chunk_size = int(
        download_config[
            "chunk_size_bytes"
        ]
    )

    user_agent = (
        download_config[
            "user_agent"
        ]
    )

    for attempt in range(
        1,
        max_retries + 1,
    ):

        log(
            f"[DOWNLOAD] {filename} "
            f"(attempt {attempt}/{max_retries})",
            config,
        )

        try:

            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={
                    "User-Agent":
                        user_agent
                },
            ) as response:

                if response.status_code == 404:

                    log(
                        f"[404] {url}",
                        config,
                    )

                    return None

                response.raise_for_status()

                total = int(
                    response.headers.get(
                        "content-length",
                        0,
                    )
                )

                with destination.open(
                    "wb"
                ) as f:

                    if total:

                        with tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=filename,
                        ) as progress:

                            for chunk in response.iter_content(
                                chunk_size=chunk_size
                            ):

                                if chunk:

                                    f.write(
                                        chunk
                                    )

                                    progress.update(
                                        len(chunk)
                                    )

                    else:

                        for chunk in response.iter_content(
                            chunk_size=chunk_size
                        ):

                            if chunk:
                                f.write(
                                    chunk
                                )

            return destination

        except Exception as exc:

            log(
                f"[WARN] {exc}",
                config,
            )

            if destination.exists():
                destination.unlink()

    raise RuntimeError(
        f"Download failed after "
        f"{max_retries} attempts:\n"
        f"{url}"
    )


# ============================================================
# XYZ FILE
# ============================================================

def find_xyz_file(
    zip_path: Path,
) -> str:

    with zipfile.ZipFile(
        zip_path
    ) as archive:

        files = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]

        if not files:

            raise RuntimeError(
                f"Empty ZIP: {zip_path}"
            )

        preferred = [
            name
            for name in files
            if name.lower().endswith(
                (
                    ".xyz",
                    ".txt",
                    ".asc",
                )
            )
        ]

        if preferred:
            return preferred[0]

        return files[0]


# ============================================================
# PROCESS TILE
# ============================================================

def process_tile(
    zip_path: Path,
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
    min_x: float,
    min_y: float,
    max_y: float,
    resolution: float,
    aggregation: str,
) -> None:

    xyz_name = find_xyz_file(
        zip_path
    )

    print(
        f"[PROCESS] {zip_path.name}"
    )

    with zipfile.ZipFile(
        zip_path
    ) as archive:

        with archive.open(
            xyz_name,
            "r",
        ) as f:

            for raw_line in f:

                try:

                    line = raw_line.decode(
                        "ascii",
                        errors="ignore",
                    ).strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) < 3:
                        continue

                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])

                except (
                    ValueError,
                    UnicodeDecodeError,
                ):
                    continue

                # ------------------------------------------------
                # Check target bbox
                # ------------------------------------------------

                if (
                    x < min_x
                    or x >= min_x
                    + sum_grid.shape[1]
                    * resolution
                ):
                    continue

                if (
                    y >= max_y
                    or y < max_y
                    - sum_grid.shape[0]
                    * resolution
                ):
                    continue

                col = int(
                    (x - min_x)
                    // resolution
                )

                row = int(
                    (max_y - y)
                    // resolution
                )

                if (
                    row < 0
                    or row >= sum_grid.shape[0]
                    or col < 0
                    or col >= sum_grid.shape[1]
                ):
                    continue

                if not math.isfinite(z):
                    continue

                if aggregation == "mean":

                    sum_grid[
                        row,
                        col,
                    ] += z

                    count_grid[
                        row,
                        col,
                    ] += 1

                elif aggregation == "min":

                    if (
                        count_grid[
                            row,
                            col,
                        ] == 0
                        or z
                        < sum_grid[
                            row,
                            col,
                        ]
                    ):

                        sum_grid[
                            row,
                            col,
                        ] = z

                    count_grid[
                        row,
                        col,
                    ] = 1

                elif aggregation == "max":

                    if (
                        count_grid[
                            row,
                            col,
                        ] == 0
                        or z
                        > sum_grid[
                            row,
                            col,
                        ]
                    ):

                        sum_grid[
                            row,
                            col,
                        ] = z

                    count_grid[
                        row,
                        col,
                    ] = 1


# ============================================================
# FINAL DEM
# ============================================================

def create_dem(
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
    aggregation: str,
) -> np.ndarray:

    dem = np.full(
        sum_grid.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        count_grid > 0
    )

    if aggregation == "mean":

        dem[valid] = (
            sum_grid[valid]
            / count_grid[valid]
        )

    else:

        dem[valid] = (
            sum_grid[valid]
        )

    return dem


# ============================================================
# WRITE GEOTIFF
# ============================================================

def write_geotiff(
    dem: np.ndarray,
    bbox: tuple[float, float, float, float],
    config: dict[str, Any],
) -> None:

    min_x, min_y, max_x, max_y = bbox

    output_config = config[
        "output"
    ]

    output_dir = Path(
        output_config[
            "directory"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / output_config[
            "filename"
        ]
    )

    resolution = float(
        config[
            "area"
        ][
            "resolution_m"
        ]
    )

    nodata = float(
        output_config[
            "nodata"
        ]
    )

    transform = rasterio.transform.from_origin(
        min_x,
        max_y,
        resolution,
        resolution,
    )

    output = np.where(
        np.isfinite(dem),
        dem,
        nodata,
    ).astype(
        np.float32
    )

    print()
    print(
        f"[WRITE] {output_file}"
    )

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        width=output.shape[1],
        height=output.shape[0],
        count=1,
        dtype="float32",
        crs=config[
            "coordinate_system"
        ][
            "processing_crs"
        ],
        transform=transform,
        nodata=nodata,
        compress=output_config[
            "compression"
        ],
        predictor=int(
            output_config[
                "predictor"
            ]
        ),
        tiled=bool(
            output_config[
                "tiled"
            ]
        ),
        BIGTIFF=output_config[
            "bigtiff"
        ],
    ) as dst:

        dst.write(
            output,
            1,
        )

        dst.set_band_description(
            1,
            "LGL DGM1 aggregated DEM",
        )

        dst.update_tags(
            SOURCE=(
                "LGL Baden-Württemberg DGM1"
            ),
            PROCESSING=(
                f"{config['processing']['aggregation']} "
                f"aggregation to "
                f"{resolution} m"
            ),
            CRS=config[
                "coordinate_system"
            ][
                "processing_crs"
            ],
            HEIGHT_REFERENCE=(
                "DHHN2016 / NHN"
            ),
        )

    print(
        "[DONE]"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config(
        CONFIG_FILE
    )

    validate_config(
        config
    )

    print()
    print("=" * 72)
    print("LGL DGM1 DOWNLOADER")
    print("=" * 72)

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    bbox = calculate_bbox(
        config
    )

    min_x, min_y, max_x, max_y = bbox

    area = config["area"]

    resolution = float(
        area["resolution_m"]
    )

    width = int(
        round(
            (max_x - min_x)
            / resolution
        )
    )

    height = int(
        round(
            (max_y - min_y)
            / resolution
        )
    )

    print()
    print("[AREA]")

    print(
        f"Center: "
        f"{area['center']['latitude']}, "
        f"{area['center']['longitude']}"
    )

    print(
        f"Size: "
        f"{max_x - min_x:.0f} x "
        f"{max_y - min_y:.0f} m"
    )

    print(
        f"Resolution: "
        f"{resolution} m"
    )

    print(
        f"Output grid: "
        f"{width} x {height}"
    )

    print(
        f"Cells: "
        f"{width * height:,}"
    )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory_gb = (
        width
        * height
        * (
            8 + 2
        )
        / 1024**3
    )

    print(
        f"Accumulator memory: "
        f"~{memory_gb:.2f} GiB"
    )

    # --------------------------------------------------------
    # TILES
    # --------------------------------------------------------

    tiles = calculate_lgl_tiles(
        bbox,
        config,
    )

    print()
    print(
        f"[TILES] "
        f"{len(tiles)} required"
    )

    for x, y in tiles:

        print(
            f"  dgm1_32_{x}_{y}_2_bw.zip"
        )

    # --------------------------------------------------------
    # ACCUMULATION ARRAYS
    # --------------------------------------------------------

    aggregation = config[
        "processing"
    ][
        "aggregation"
    ]

    if aggregation == "mean":

        sum_dtype = np.float64

    else:

        sum_dtype = np.float32

    sum_grid = np.zeros(
        (
            height,
            width,
        ),
        dtype=sum_dtype,
    )

    count_grid = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint16,
    )

    # --------------------------------------------------------
    # DOWNLOAD + PROCESS
    # --------------------------------------------------------

    for index, (
        x,
        y,
    ) in enumerate(
        tiles,
        start=1,
    ):

        print()
        print(
            "=" * 72
        )

        print(
            f"TILE {index}/{len(tiles)}"
        )

        zip_path = download_tile(
            x,
            y,
            config,
        )

        if zip_path is None:

            if config[
                "runtime"
            ][
                "fail_on_missing_tile"
            ]:

                raise RuntimeError(
                    f"Missing LGL tile: "
                    f"{x}/{y}"
                )

            print(
                "[WARNING] Tile skipped."
            )

            continue

        process_tile(
            zip_path=zip_path,
            sum_grid=sum_grid,
            count_grid=count_grid,
            min_x=min_x,
            min_y=min_y,
            max_y=max_y,
            resolution=resolution,
            aggregation=aggregation,
        )

        # ----------------------------------------------------
        # Optional cleanup
        # ----------------------------------------------------

        if config[
            "download"
        ][
            "delete_after_processing"
        ]:

            print(
                f"[DELETE] {zip_path}"
            )

            zip_path.unlink()

    # --------------------------------------------------------
    # CREATE DEM
    # --------------------------------------------------------

    print()
    print(
        "[FINALIZE]"
    )

    dem = create_dem(
        sum_grid,
        count_grid,
        aggregation,
    )

    del sum_grid
    del count_grid

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    write_geotiff(
        dem,
        bbox,
        config,
    )

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()