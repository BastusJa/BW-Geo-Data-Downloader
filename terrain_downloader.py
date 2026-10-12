"""
LGL DGM1 Downloader / Processor
--------------------------------

Funktionen:

- Lädt aktuelle LGL DGM1 2x2-km ZIP-Kacheln
- Verarbeitet ALLE enthaltenen 1x1-km XYZ-Dateien
- Ignoriert Metadaten-Dateien
- Verarbeitet in EPSG:25832
- Schneidet auf die gewünschte Fläche zu
- Aggregiert 1-m-XYZ-Daten auf ein Zielraster per Mittelwert
- Optionales Glätten künstlicher Geländeeinschnitte
  durch Straßen, Schienen und Gebäude
- Schreibt komprimiertes Float32 GeoTIFF
- Schreibt zusätzlich 16-bit Graustufen-PNG als Heightmap

Glättung:
    Die Glättung ist optional.

    Verarbeitung:

        1. Medianfilter
        2. Gaussian-Filter

    Dadurch werden kleine künstliche Vertiefungen,
    harte Kanten und lokale Ausreißer reduziert.

    NoData-Zellen werden bei der Glättung nicht als
    Höhenwerte behandelt.

PNG:
    0      = NoData
    1      = minimale Höhe
    65535  = maximale Höhe

Die PNG-Datei ist eine normalisierte 16-bit Heightmap.
Das GeoTIFF enthält weiterhin die tatsächlichen Höhenwerte.

Benötigte Pakete:

    numpy
    requests
    rasterio
    pyyaml
    pyproj
    scipy

Installation:

    pip install numpy requests rasterio pyyaml pyproj scipy
"""


import logging
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import rasterio
import yaml

from pyproj import Transformer
from rasterio.transform import from_origin

from scipy.ndimage import (
    gaussian_filter,
    median_filter,
)

from config_loader import load_config


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("LGL-DGM1")


def setup_logging(verbose=True):
    level = (
        logging.INFO
        if verbose
        else logging.WARNING
    )

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ============================================================
# Config
# ============================================================


def validate_config(config):
    required_sections = [
        "center",
        "area",
        "resolution_m",
        "crs",
        "output",
        "processing",
        "download",
        "lgl",
        "runtime",
    ]

    for section in required_sections:

        if section not in config:
            raise ValueError(
                f"Fehlender YAML-Bereich: "
                f"{section}"
            )

    if config["area"]["width_m"] <= 0:
        raise ValueError(
            "area.width_m muss > 0 sein."
        )

    if config["area"]["height_m"] <= 0:
        raise ValueError(
            "area.height_m muss > 0 sein."
        )

    if config["resolution_m"] <= 0:
        raise ValueError(
            "resolution_m muss > 0 sein."
        )

    if config["lgl"]["tile_size_m"] != 2000:
        raise ValueError(
            "Für die aktuelle LGL-DGM1-Struktur muss "
            "lgl.tile_size_m = 2000 sein."
        )

    if (
        config["processing"]["aggregation"].lower()
        != "mean"
    ):
        raise ValueError(
            "Aktuell wird nur aggregation: mean unterstützt."
        )

    # --------------------------------------------------------
    # Glättung validieren
    # --------------------------------------------------------

    smoothing = config[
        "processing"
    ].get(
        "smoothing",
        {},
    )

    enabled = bool(
        smoothing.get(
            "enabled",
            False,
        )
    )

    if enabled:

        median_size = int(
            smoothing.get(
                "median_size",
                3,
            )
        )

        gaussian_sigma = float(
            smoothing.get(
                "gaussian_sigma",
                0.8,
            )
        )

        if median_size < 1:
            raise ValueError(
                "processing.smoothing.median_size "
                "muss >= 1 sein."
            )

        if (
            median_size % 2 == 0
        ):
            raise ValueError(
                "processing.smoothing.median_size "
                "muss ungerade sein, z.B. 3, 5 oder 7."
            )

        if gaussian_sigma < 0:
            raise ValueError(
                "processing.smoothing.gaussian_sigma "
                "muss >= 0 sein."
            )


# ============================================================
# CRS / BBox
# ============================================================

def create_transformer(config):
    input_crs = config[
        "crs"
    ]["input"]

    processing_crs = config[
        "crs"
    ]["processing"]

    return Transformer.from_crs(
        input_crs,
        processing_crs,
        always_xy=True,
    )


def calculate_bbox(
    config,
    transformer,
):
    latitude = float(
        config["center"]["latitude"]
    )

    longitude = float(
        config["center"]["longitude"]
    )

    width_m = float(
        config["area"]["width_m"]
    )

    height_m = float(
        config["area"]["height_m"]
    )

    resolution = float(
        config["resolution_m"]
    )

    center_x, center_y = (
        transformer.transform(
            longitude,
            latitude,
        )
    )

    logger.info(
        "WGS84 Zentrum: lat=%.8f lon=%.8f",
        latitude,
        longitude,
    )

    logger.info(
        "UTM32 Zentrum: X=%.3f Y=%.3f",
        center_x,
        center_y,
    )

    min_x = (
        center_x
        - width_m / 2.0
    )

    max_x = (
        center_x
        + width_m / 2.0
    )

    min_y = (
        center_y
        - height_m / 2.0
    )

    max_y = (
        center_y
        + height_m / 2.0
    )

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

    width_cells = int(
        round(
            width_m / resolution
        )
    )

    height_cells = int(
        round(
            height_m / resolution
        )
    )

    max_x = (
        min_x
        + width_cells
        * resolution
    )

    max_y = (
        min_y
        + height_cells
        * resolution
    )

    logger.info(
        "Ziel-BBox EPSG:25832:"
    )

    logger.info(
        "  min_x = %.3f",
        min_x,
    )

    logger.info(
        "  min_y = %.3f",
        min_y,
    )

    logger.info(
        "  max_x = %.3f",
        max_x,
    )

    logger.info(
        "  max_y = %.3f",
        max_y,
    )

    logger.info(
        "Zielraster: %d x %d Pixel bei %.2f m",
        width_cells,
        height_cells,
        resolution,
    )

    return (
        min_x,
        min_y,
        max_x,
        max_y,
        width_cells,
        height_cells,
    )


# ============================================================
# LGL Tile Grid
# ============================================================

def calculate_lgl_tiles(
    min_x,
    min_y,
    max_x,
    max_y,
    tile_size=2000,
):
    """
    Berechnet die benötigten LGL-DGM1-2x2-km-Kacheln.

    Aktuelle Struktur:

        dgm1_32_<EASTING_KM>_<NORTHING_KM>_2_bw.zip

    LGL-Kachelraster:

        EASTING  = ungerade km-Werte
        NORTHING = gerade km-Werte
    """

    if tile_size != 2000:
        raise ValueError(
            "Das aktuelle LGL-DGM1-Kachelraster "
            "wird mit 2000 m erwartet."
        )

    easting_km = math.floor(
        min_x / 1000
    )

    if easting_km % 2 == 0:
        easting_km -= 1

    northing_km = math.floor(
        min_y / 1000
    )

    if northing_km % 2 != 0:
        northing_km -= 1

    tiles = []

    while True:

        tile_min_x = (
            easting_km
            * 1000
        )

        tile_max_x = (
            tile_min_x
            + tile_size
        )

        if tile_min_x >= max_x:
            break

        current_northing_km = (
            northing_km
        )

        while True:

            tile_min_y = (
                current_northing_km
                * 1000
            )

            tile_max_y = (
                tile_min_y
                + tile_size
            )

            if tile_min_y >= max_y:
                break

            intersects = not (
                tile_max_x <= min_x
                or tile_min_x >= max_x
                or tile_max_y <= min_y
                or tile_min_y >= max_y
            )

            if intersects:

                filename = (
                    "dgm1_32_"
                    f"{easting_km}_"
                    f"{current_northing_km}"
                    "_2_bw.zip"
                )

                tiles.append(
                    {
                        "x": int(
                            tile_min_x
                        ),
                        "y": int(
                            tile_min_y
                        ),
                        "x_km": int(
                            easting_km
                        ),
                        "y_km": int(
                            current_northing_km
                        ),
                        "filename": filename,
                    }
                )

            current_northing_km += 2

        easting_km += 2

    return tiles


# ============================================================
# Download
# ============================================================

def download_tile(
    tile,
    config,
):
    download_dir = Path(
        config["download"]["directory"]
    )

    download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = tile[
        "filename"
    ]

    url_template = (
        config["lgl"]["url_template"]
    )

    url = url_template.format(
        x=tile["x_km"],
        y=tile["y_km"],
    )

    output_path = (
        download_dir
        / filename
    )

    max_retries = int(
        config["download"][
            "max_retries"
        ]
    )

    timeout = int(
        config["download"][
            "timeout_seconds"
        ]
    )

    chunk_size = int(
        config["download"][
            "chunk_size_bytes"
        ]
    )

    user_agent = config[
        "download"
    ]["user_agent"]

    if output_path.exists():

        logger.info(
            "ZIP bereits vorhanden: %s",
            output_path,
        )

        return output_path

    logger.info(
        "Download: %s",
        url,
    )

    headers = {
        "User-Agent": user_agent,
    }

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        temporary_path = (
            output_path.with_suffix(
                output_path.suffix
                + ".part"
            )
        )

        try:

            with requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
            ) as response:

                response.raise_for_status()

                with temporary_path.open(
                    "wb"
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=chunk_size
                    ):

                        if chunk:
                            f.write(chunk)

            temporary_path.replace(
                output_path
            )

            logger.info(
                "Download erfolgreich: %s",
                output_path.name,
            )

            return output_path

        except Exception as exc:

            last_error = exc

            logger.warning(
                "Downloadversuch %d/%d "
                "fehlgeschlagen: %s",
                attempt,
                max_retries,
                exc,
            )

            try:

                if temporary_path.exists():
                    temporary_path.unlink()

            except OSError:
                pass

            if attempt < max_retries:
                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        f"Download fehlgeschlagen: {url}"
    ) from last_error


# ============================================================
# XYZ-Dateien
# ============================================================

XYZ_FILENAME_RE = re.compile(
    r"^dgm1_32_(\d+)_(\d+)_1_bw_.*\.xyz$",
    re.IGNORECASE,
)


def find_xyz_files(zip_file):

    xyz_files = []

    for name in zip_file.namelist():

        if name.endswith("/"):
            continue

        basename = os.path.basename(
            name
        )

        match = (
            XYZ_FILENAME_RE.match(
                basename
            )
        )

        if not match:
            continue

        easting_km = int(
            match.group(1)
        )

        northing_km = int(
            match.group(2)
        )

        xyz_files.append(
            {
                "path": name,
                "filename": basename,
                "x_km": easting_km,
                "y_km": northing_km,
                "min_x": (
                    easting_km
                    * 1000.0
                ),
                "min_y": (
                    northing_km
                    * 1000.0
                ),
                "max_x": (
                    easting_km
                    * 1000.0
                    + 1000.0
                ),
                "max_y": (
                    northing_km
                    * 1000.0
                    + 1000.0
                ),
            }
        )

    xyz_files.sort(
        key=lambda item: (
            item["y_km"],
            item["x_km"],
        )
    )

    if not xyz_files:
        raise RuntimeError(
            "Keine gültige XYZ-Datei gefunden."
        )

    logger.info(
        "%d XYZ-Dateien gefunden",
        len(xyz_files),
    )

    for item in xyz_files:

        logger.info(
            "  XYZ: %s",
            item["filename"],
        )

    return xyz_files


# ============================================================
# XYZ Parsing
# ============================================================

def parse_xyz_line(line):

    line = line.strip()

    if not line:
        return None

    if line.startswith("#"):
        return None

    if ";" in line:
        parts = line.split(";")

    elif "," in line:
        parts = line.split(",")

    else:
        parts = line.split()

    if len(parts) < 3:
        return None

    try:

        x = float(
            parts[0]
            .strip()
            .replace(",", ".")
        )

        y = float(
            parts[1]
            .strip()
            .replace(",", ".")
        )

        z = float(
            parts[2]
            .strip()
            .replace(",", ".")
        )

        return (
            x,
            y,
            z,
        )

    except ValueError:
        return None


# ============================================================
# Schneller XYZ-Parser
# ============================================================

def parse_xyz_block_fast(
    raw_lines,
    delimiter_mode="whitespace",
):

    try:

        text = b"".join(
            raw_lines
        ).decode(
            "utf-8",
            errors="ignore",
        )

        if not text.strip():

            return np.empty(
                (0, 3),
                dtype=np.float64,
            )

        if (
            delimiter_mode
            == "semicolon"
        ):

            text = text.replace(
                ";",
                " ",
            )

        elif (
            delimiter_mode
            == "comma"
        ):

            text = text.replace(
                ",",
                " ",
            )

        values = np.fromstring(
            text,
            sep=" ",
            dtype=np.float64,
        )

        if values.size == 0:

            return np.empty(
                (0, 3),
                dtype=np.float64,
            )

        if values.size % 3 != 0:
            return None

        return values.reshape(
            (-1, 3)
        )

    except Exception:
        return None


# ============================================================
# XYZ Datei verarbeiten
# ============================================================

def process_xyz_file(
    zip_file,
    xyz_info,
    sum_grid,
    count_grid,
    target_min_x,
    target_min_y,
    target_max_x,
    target_max_y,
    target_resolution,
    chunk_lines=50000,
):

    xyz_path = xyz_info[
        "path"
    ]

    source_min_x = xyz_info[
        "min_x"
    ]

    source_max_x = xyz_info[
        "max_x"
    ]

    source_min_y = xyz_info[
        "min_y"
    ]

    source_max_y = xyz_info[
        "max_y"
    ]

    points = 0
    accepted = 0
    outside_source = 0
    outside_target = 0
    invalid = 0

    # --------------------------------------------------------
    # Format erkennen
    # --------------------------------------------------------

    sample_lines = []

    with zip_file.open(
        xyz_path,
        "r",
    ) as f:

        for _ in range(100):

            line = f.readline()

            if not line:
                break

            sample_lines.append(
                line
            )

    detected_mode = (
        "whitespace"
    )

    for raw in sample_lines:

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        if ";" in line:
            detected_mode = (
                "semicolon"
            )
            break

        if "," in line:
            detected_mode = "comma"
            break

    # --------------------------------------------------------
    # Block-Verarbeitung
    # --------------------------------------------------------

    with zip_file.open(
        xyz_path,
        "r",
    ) as f:

        buffer = []

        def process_block(
            lines
        ):
            nonlocal points
            nonlocal accepted
            nonlocal outside_source
            nonlocal outside_target
            nonlocal invalid

            if not lines:
                return

            array = (
                parse_xyz_block_fast(
                    lines,
                    detected_mode,
                )
            )

            # ------------------------------------------------
            # Fallback für ungewöhnliche Daten
            # ------------------------------------------------

            if array is None:

                for raw_line in lines:

                    points += 1

                    try:

                        line = (
                            raw_line.decode(
                                "utf-8",
                                errors="ignore",
                            )
                        )

                    except Exception:

                        invalid += 1
                        continue

                    parsed = (
                        parse_xyz_line(
                            line
                        )
                    )

                    if parsed is None:

                        invalid += 1
                        continue

                    x, y, z = parsed

                    if not (
                        math.isfinite(x)
                        and math.isfinite(y)
                        and math.isfinite(z)
                    ):

                        invalid += 1
                        continue

                    if (
                        x < source_min_x
                        or x >= source_max_x
                        or y < source_min_y
                        or y >= source_max_y
                    ):

                        outside_source += 1
                        continue

                    if (
                        x < target_min_x
                        or x >= target_max_x
                        or y < target_min_y
                        or y >= target_max_y
                    ):

                        outside_target += 1
                        continue

                    col = int(
                        (
                            x
                            - target_min_x
                        )
                        // target_resolution
                    )

                    row = int(
                        (
                            target_max_y
                            - y
                        )
                        // target_resolution
                    )

                    if (
                        row < 0
                        or row
                        >= sum_grid.shape[0]
                        or col < 0
                        or col
                        >= sum_grid.shape[1]
                    ):

                        outside_target += 1
                        continue

                    sum_grid[
                        row,
                        col,
                    ] += z

                    count_grid[
                        row,
                        col,
                    ] += 1

                    accepted += 1

                return

            # ------------------------------------------------
            # Vectorisiert
            # ------------------------------------------------

            n = array.shape[0]

            points += n

            if n == 0:
                return

            finite_mask = np.isfinite(
                array
            ).all(
                axis=1
            )

            invalid += int(
                np.count_nonzero(
                    ~finite_mask
                )
            )

            if not np.any(
                finite_mask
            ):
                return

            data = array[
                finite_mask
            ]

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]

            source_mask = (
                (x >= source_min_x)
                & (x < source_max_x)
                & (y >= source_min_y)
                & (y < source_max_y)
            )

            outside_source += int(
                np.count_nonzero(
                    ~source_mask
                )
            )

            if not np.any(
                source_mask
            ):
                return

            data = data[
                source_mask
            ]

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]

            target_mask = (
                (x >= target_min_x)
                & (x < target_max_x)
                & (y >= target_min_y)
                & (y < target_max_y)
            )

            outside_target += int(
                np.count_nonzero(
                    ~target_mask
                )
            )

            if not np.any(
                target_mask
            ):
                return

            x = x[
                target_mask
            ]

            y = y[
                target_mask
            ]

            z = z[
                target_mask
            ]

            cols = np.floor(
                (
                    x
                    - target_min_x
                )
                / target_resolution
            ).astype(
                np.int64
            )

            rows = np.floor(
                (
                    target_max_y
                    - y
                )
                / target_resolution
            ).astype(
                np.int64
            )

            bounds_mask = (
                (rows >= 0)
                & (
                    rows
                    < sum_grid.shape[0]
                )
                & (cols >= 0)
                & (
                    cols
                    < sum_grid.shape[1]
                )
            )

            outside_target += int(
                np.count_nonzero(
                    ~bounds_mask
                )
            )

            if not np.any(
                bounds_mask
            ):
                return

            rows = rows[
                bounds_mask
            ]

            cols = cols[
                bounds_mask
            ]

            z = z[
                bounds_mask
            ]

            np.add.at(
                sum_grid,
                (rows, cols),
                z,
            )

            np.add.at(
                count_grid,
                (rows, cols),
                1,
            )

            accepted += int(
                z.size
            )

        # ----------------------------------------------------
        # Lesen
        # ----------------------------------------------------

        for raw_line in f:

            buffer.append(
                raw_line
            )

            if (
                len(buffer)
                >= chunk_lines
            ):

                process_block(
                    buffer
                )

                buffer.clear()

        if buffer:

            process_block(
                buffer
            )

    logger.info(
        "%s: %d Punkte, %d übernommen, "
        "%d außerhalb XYZ-Kachel, "
        "%d außerhalb Ziel, %d ungültig",
        xyz_info["filename"],
        points,
        accepted,
        outside_source,
        outside_target,
        invalid,
    )

    return {
        "points": points,
        "accepted": accepted,
        "outside_source": outside_source,
        "outside_target": outside_target,
        "invalid": invalid,
    }


# ============================================================
# ZIP verarbeiten
# ============================================================

def process_tile(
    zip_path,
    tile,
    sum_grid,
    count_grid,
    target_min_x,
    target_min_y,
    target_max_x,
    target_max_y,
    target_resolution,
    chunk_lines,
):

    total_points = 0
    total_accepted = 0
    total_outside_source = 0
    total_outside_target = 0
    total_invalid = 0

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        xyz_files = (
            find_xyz_files(zf)
        )

        for xyz_info in xyz_files:

            logger.info(
                "Verarbeite %s",
                xyz_info["filename"],
            )

            result = process_xyz_file(
                zip_file=zf,
                xyz_info=xyz_info,
                sum_grid=sum_grid,
                count_grid=count_grid,
                target_min_x=target_min_x,
                target_min_y=target_min_y,
                target_max_x=target_max_x,
                target_max_y=target_max_y,
                target_resolution=target_resolution,
                chunk_lines=chunk_lines,
            )

            total_points += result[
                "points"
            ]

            total_accepted += result[
                "accepted"
            ]

            total_outside_source += result[
                "outside_source"
            ]

            total_outside_target += result[
                "outside_target"
            ]

            total_invalid += result[
                "invalid"
            ]

    logger.info(
        "%s: GESAMT %d Punkte, %d übernommen, "
        "%d außerhalb XYZ-Kachel, "
        "%d außerhalb Ziel, %d ungültig",
        zip_path.name,
        total_points,
        total_accepted,
        total_outside_source,
        total_outside_target,
        total_invalid,
    )

    return {
        "points": total_points,
        "accepted": total_accepted,
        "outside_source": total_outside_source,
        "outside_target": total_outside_target,
        "invalid": total_invalid,
    }


# ============================================================
# DEM erstellen
# ============================================================

def create_dem(
    sum_grid,
    count_grid,
    nodata,
):

    dem = np.full(
        sum_grid.shape,
        nodata,
        dtype=np.float32,
    )

    valid = (
        count_grid > 0
    )

    dem[valid] = (
        sum_grid[valid]
        / count_grid[valid]
    ).astype(
        np.float32
    )

    return dem


# ============================================================
# DEM Glättung
# ============================================================

def smooth_dem(
    dem,
    nodata,
    config,
):
    """
    Glättet künstliche Einschnitte im DGM.

    Die Funktion arbeitet nur auf gültigen Höhenwerten.

    Verfahren:

        1. Medianfilter
        2. Gaussian-Filter

    Der Medianfilter entfernt lokale harte Ausreißer.

    Der Gaussian-Filter erzeugt anschließend eine sanfte
    Oberfläche.

    Wichtig:
        NoData wird nicht als Höhenwert in die Filterung
        einbezogen.

    Rückgabe:
        neues float32 DEM
    """

    smoothing = config[
        "processing"
    ].get(
        "smoothing",
        {},
    )

    enabled = bool(
        smoothing.get(
            "enabled",
            False,
        )
    )

    if not enabled:

        logger.info(
            "DEM-Glättung deaktiviert."
        )

        return dem

    median_size = int(
        smoothing.get(
            "median_size",
            3,
        )
    )

    gaussian_sigma = float(
        smoothing.get(
            "gaussian_sigma",
            0.8,
        )
    )

    gaussian_passes = int(
        smoothing.get(
            "gaussian_passes",
            1,
        )
    )

    if median_size < 1:
        raise ValueError(
            "median_size muss >= 1 sein."
        )

    if (
        median_size % 2 == 0
    ):
        raise ValueError(
            "median_size muss ungerade sein."
        )

    if gaussian_sigma < 0:
        raise ValueError(
            "gaussian_sigma muss >= 0 sein."
        )

    if gaussian_passes < 0:
        raise ValueError(
            "gaussian_passes muss >= 0 sein."
        )

    logger.info(
        "============================================================"
    )

    logger.info(
        "STARTE DEM-GLÄTTUNG"
    )

    logger.info(
        "  Medianfilter: %d x %d",
        median_size,
        median_size,
    )

    logger.info(
        "  Gaussian Sigma: %.3f Pixel",
        gaussian_sigma,
    )

    logger.info(
        "  Gaussian Durchläufe: %d",
        gaussian_passes,
    )

    # --------------------------------------------------------
    # Gültigkeitsmaske
    # --------------------------------------------------------

    valid = (
        np.isfinite(dem)
        & (dem != nodata)
    )

    valid_count = int(
        np.count_nonzero(valid)
    )

    if valid_count == 0:

        logger.warning(
            "Keine gültigen DEM-Werte. "
            "Glättung wird übersprungen."
        )

        return dem

    # --------------------------------------------------------
    # Float64 für die Filterung.
    # --------------------------------------------------------

    data = np.where(
        valid,
        dem.astype(
            np.float64,
            copy=False,
        ),
        0.0,
    )

    weights = valid.astype(
        np.float64
    )

    # ========================================================
    # 1. Medianfilter
    # ========================================================

    if median_size > 1:

        logger.info(
            "Führe Medianfilter aus..."
        )

        # ----------------------------------------------------
        # Der Medianfilter kennt keine Gewichtung.
        #
        # Deshalb werden NoData-Werte vorübergehend mit dem
        # nächstgelegenen gültigen Wert behandelt.
        #
        # Für typische DGM-Flächen innerhalb einer vollständig
        # abgedeckten Ziel-BBox ist dies normalerweise irrelevant.
        # ----------------------------------------------------

        filled = data.copy()

        if not np.all(valid):

            # Nearest-neighbour-artiges Auffüllen der
            # NoData-Bereiche über wiederholte Raster-
            # Transformation ist unnötig teuer.
            #
            # Stattdessen wird ein neutraler lokaler Wert
            # aus einem gewichteten Gaussian-Fill erzeugt.

            fill_sum = gaussian_filter(
                data,
                sigma=1.0,
                mode="nearest",
            )

            fill_weight = gaussian_filter(
                weights,
                sigma=1.0,
                mode="nearest",
            )

            fill_weight = np.maximum(
                fill_weight,
                1e-12,
            )

            filled_values = (
                fill_sum
                / fill_weight
            )

            filled[
                ~valid
            ] = filled_values[
                ~valid
            ]

        filtered = median_filter(
            filled,
            size=median_size,
            mode="nearest",
        )

    else:

        logger.info(
            "Medianfilter übersprungen."
        )

        filtered = data

    # ========================================================
    # 2. Gaussian Filter
    # ========================================================

    if (
        gaussian_sigma > 0
        and gaussian_passes > 0
    ):

        logger.info(
            "Führe Gaussian-Glättung aus..."
        )

        # ----------------------------------------------------
        # Wichtig:
        # NoData darf nicht als Höhenwert 0 in die Glättung
        # einfließen.
        #
        # Deshalb wird die gewichtete Gaussian-Methode
        # verwendet:
        #
        #   filtered * weight
        #   ------------------
        #          weight
        # ----------------------------------------------------

        current = filtered

        current_valid = valid.astype(
            np.float64
        )

        for pass_number in range(
            1,
            gaussian_passes + 1,
        ):

            weighted = (
                current
                * current_valid
            )

            blurred_values = (
                gaussian_filter(
                    weighted,
                    sigma=gaussian_sigma,
                    mode="nearest",
                )
            )

            blurred_weights = (
                gaussian_filter(
                    current_valid,
                    sigma=gaussian_sigma,
                    mode="nearest",
                )
            )

            blurred_weights = np.maximum(
                blurred_weights,
                1e-12,
            )

            current = (
                blurred_values
                / blurred_weights
            )

            logger.info(
                "  Gaussian-Durchlauf "
                "%d/%d",
                pass_number,
                gaussian_passes,
            )

        filtered = current

    else:

        logger.info(
            "Gaussian-Glättung übersprungen."
        )

    # --------------------------------------------------------
    # Original NoData wiederherstellen
    # --------------------------------------------------------

    result = filtered.astype(
        np.float32
    )

    result[
        ~valid
    ] = nodata

    logger.info(
        "DEM-Glättung abgeschlossen."
    )

    logger.info(
        "============================================================"
    )

    return result


# ============================================================
# Coverage
# ============================================================

def report_coverage(
    count_grid,
    resolution,
):

    total_cells = (
        count_grid.size
    )

    valid_cells = int(
        np.count_nonzero(
            count_grid
        )
    )

    empty_cells = (
        total_cells
        - valid_cells
    )

    coverage = (
        valid_cells
        / total_cells
        * 100.0
        if total_cells
        else 0.0
    )

    logger.info(
        "Zielraster:"
    )

    logger.info(
        "  Gesamtzellen: %d",
        total_cells,
    )

    logger.info(
        "  Gefüllte Zellen: %d",
        valid_cells,
    )

    logger.info(
        "  Leere Zellen: %d",
        empty_cells,
    )

    logger.info(
        "  Abdeckung: %.2f %%",
        coverage,
    )

    if empty_cells > 0:

        logger.warning(
            "Es existieren %d leere Zielzellen.",
            empty_cells,
        )

    return coverage


# ============================================================
# GeoTIFF
# ============================================================

def write_geotiff(
    output_path,
    dem,
    config,
    min_x,
    max_y,
    resolution,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = dem.shape

    output_config = (
        config["output"]
    )

    compression = (
        output_config.get(
            "compression",
            "deflate",
        )
    )

    predictor = (
        output_config.get(
            "predictor",
            3,
        )
    )

    tiled = (
        output_config.get(
            "tiled",
            True,
        )
    )

    bigtiff = (
        output_config.get(
            "bigtiff",
            "IF_SAFER",
        )
    )

    nodata = float(
        output_config.get(
            "nodata",
            -9999.0,
        )
    )

    transform = from_origin(
        min_x,
        max_y,
        resolution,
        resolution,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": config["crs"][
            "processing"
        ],
        "transform": transform,
        "nodata": nodata,
        "compress": compression,
        "predictor": predictor,
        "tiled": tiled,
        "bigtiff": bigtiff,
    }

    logger.info(
        "Schreibe GeoTIFF: %s",
        output_path,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            dem,
            1,
        )

        dst.set_band_description(
            1,
            "DGM1 aggregiert auf "
            f"{resolution:g} m",
        )

    logger.info(
        "GeoTIFF erfolgreich geschrieben."
    )


# ============================================================
# PNG Scaling
# ============================================================

def calculate_png_range(
    dem,
    nodata,
    png_config,
):

    valid = (
        np.isfinite(dem)
        & (dem != nodata)
    )

    if not np.any(valid):

        raise RuntimeError(
            "Keine gültigen Höhenwerte "
            "für PNG vorhanden."
        )

    configured_min = png_config.get(
        "min_elevation_m"
    )

    configured_max = png_config.get(
        "max_elevation_m"
    )

    if (
        configured_min is not None
        and configured_max is not None
    ):

        min_elevation = float(
            configured_min
        )

        max_elevation = float(
            configured_max
        )

    else:

        min_elevation = float(
            np.min(
                dem[valid]
            )
        )

        max_elevation = float(
            np.max(
                dem[valid]
            )
        )

    if not (
        math.isfinite(
            min_elevation
        )
        and math.isfinite(
            max_elevation
        )
    ):

        raise ValueError(
            "Ungültiger PNG-Höhenbereich."
        )

    if (
        max_elevation
        <= min_elevation
    ):

        raise ValueError(
            "PNG max_elevation_m muss "
            "größer als min_elevation_m sein."
        )

    return (
        min_elevation,
        max_elevation,
    )


# ============================================================
# 16-bit PNG
# ============================================================

def write_png16(
    output_path,
    dem,
    config,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_config = (
        config["output"]
        .get(
            "png",
            {},
        )
    )

    nodata = float(
        config["output"].get(
            "nodata",
            -9999.0,
        )
    )

    (
        min_elevation,
        max_elevation,
    ) = calculate_png_range(
        dem,
        nodata,
        png_config,
    )

    logger.info(
        "PNG-Höhenbereich:"
    )

    logger.info(
        "  Minimum: %.3f m",
        min_elevation,
    )

    logger.info(
        "  Maximum: %.3f m",
        max_elevation,
    )

    valid = (
        np.isfinite(dem)
        & (dem != nodata)
    )

    png = np.zeros(
        dem.shape,
        dtype=np.uint16,
    )

    scale = (
        65534.0
        / (
            max_elevation
            - min_elevation
        )
    )

    values = dem[
        valid
    ].astype(
        np.float64,
        copy=False,
    )

    scaled = (
        1.0
        + (
            values
            - min_elevation
        )
        * scale
    )

    scaled = np.clip(
        scaled,
        1.0,
        65535.0,
    )

    png[valid] = (
        np.rint(
            scaled
        ).astype(
            np.uint16
        )
    )

    height, width = (
        png.shape
    )

    profile = {
        "driver": "PNG",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint16",
    }

    logger.info(
        "Schreibe 16-bit PNG: %s",
        output_path,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            png,
            1,
        )

        dst.update_tags(
            1,
            elevation_min_m=(
                f"{min_elevation:.6f}"
            ),
            elevation_max_m=(
                f"{max_elevation:.6f}"
            ),
            nodata_value="0",
            encoding=(
                "linear_uint16_heightmap"
            ),
            description=(
                "16-bit grayscale "
                "elevation heightmap; "
                "0=NoData; "
                "1=min; "
                "65535=max"
            ),
        )

    logger.info(
        "16-bit PNG erfolgreich geschrieben."
    )

    return (
        min_elevation,
        max_elevation,
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Verwendung:\n"
            "  python download_new.py config.yaml"
        )

        sys.exit(1)

    config_path = sys.argv[1]

    config = load_config(
        config_path
    )

    validate_config(
        config
    )

    setup_logging(
        config["runtime"].get(
            "verbose",
            True,
        )
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "LGL DGM1 Downloader / Processor"
    )

    logger.info(
        "============================================================"
    )

    transformer = create_transformer(
        config
    )

    (
        target_min_x,
        target_min_y,
        target_max_x,
        target_max_y,
        width,
        height,
    ) = calculate_bbox(
        config,
        transformer,
    )

    resolution = float(
        config["resolution_m"]
    )

    # --------------------------------------------------------
    # LGL Kacheln
    # --------------------------------------------------------

    tiles = calculate_lgl_tiles(
        min_x=target_min_x,
        min_y=target_min_y,
        max_x=target_max_x,
        max_y=target_max_y,
        tile_size=int(
            config["lgl"][
                "tile_size_m"
            ]
        ),
    )

    if not tiles:

        raise RuntimeError(
            "Keine LGL-Kacheln für "
            "die Ziel-BBox gefunden."
        )

    logger.info(
        "Benötigte LGL-2x2-km-Kacheln: %d",
        len(tiles),
    )

    for index, tile in enumerate(
        tiles,
        start=1,
    ):

        logger.info(
            "  %3d: %s",
            index,
            tile["filename"],
        )

    # --------------------------------------------------------
    # Speicher
    # --------------------------------------------------------

    logger.info(
        "Allokiere Zielraster..."
    )

    sum_grid = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    count_grid = np.zeros(
        (height, width),
        dtype=np.uint32,
    )

    logger.info(
        "Speicher Summenraster: %.1f MB",
        sum_grid.nbytes
        / 1024
        / 1024,
    )

    logger.info(
        "Speicher Zählraster: %.1f MB",
        count_grid.nbytes
        / 1024
        / 1024,
    )

    # --------------------------------------------------------
    # Processing Optionen
    # --------------------------------------------------------

    chunk_lines = int(
        config["processing"].get(
            "chunk_lines",
            50000,
        )
    )

    if chunk_lines < 1000:

        raise ValueError(
            "processing.chunk_lines "
            "sollte >= 1000 sein."
        )

    # --------------------------------------------------------
    # Glättungsstatus anzeigen
    # --------------------------------------------------------

    smoothing_config = (
        config["processing"]
        .get(
            "smoothing",
            {},
        )
    )

    smoothing_enabled = bool(
        smoothing_config.get(
            "enabled",
            False,
        )
    )

    if smoothing_enabled:

        logger.info(
            "DEM-Glättung: AKTIV"
        )

        logger.info(
            "  Median: %d",
            int(
                smoothing_config.get(
                    "median_size",
                    3,
                )
            ),
        )

        logger.info(
            "  Gaussian Sigma: %.3f",
            float(
                smoothing_config.get(
                    "gaussian_sigma",
                    0.8,
                )
            ),
        )

    else:

        logger.info(
            "DEM-Glättung: DEAKTIVIERT"
        )

    # --------------------------------------------------------
    # Download + Verarbeitung
    # --------------------------------------------------------

    delete_after_processing = bool(
        config["download"].get(
            "delete_after_processing",
            False,
        )
    )

    total_stats = {
        "points": 0,
        "accepted": 0,
        "outside_source": 0,
        "outside_target": 0,
        "invalid": 0,
    }

    for index, tile in enumerate(
        tiles,
        start=1,
    ):

        logger.info(
            "============================================================"
        )

        logger.info(
            "========== Kachel %d / %d ==========",
            index,
            len(tiles),
        )

        logger.info(
            "LGL-Kachel: %s",
            tile["filename"],
        )

        zip_path = download_tile(
            tile,
            config,
        )

        try:

            stats = process_tile(
                zip_path=zip_path,
                tile=tile,
                sum_grid=sum_grid,
                count_grid=count_grid,
                target_min_x=target_min_x,
                target_min_y=target_min_y,
                target_max_x=target_max_x,
                target_max_y=target_max_y,
                target_resolution=resolution,
                chunk_lines=chunk_lines,
            )

            for key in total_stats:

                total_stats[key] += (
                    stats[key]
                )

        finally:

            if delete_after_processing:

                try:

                    zip_path.unlink()

                    logger.info(
                        "ZIP gelöscht: %s",
                        zip_path,
                    )

                except Exception as exc:

                    logger.warning(
                        "ZIP konnte nicht gelöscht "
                        "werden: %s",
                        exc,
                    )

    # --------------------------------------------------------
    # Statistik
    # --------------------------------------------------------

    logger.info(
        "============================================================"
    )

    logger.info(
        "GESAMTSTATISTIK"
    )

    logger.info(
        "  Punkte: %d",
        total_stats["points"],
    )

    logger.info(
        "  Übernommen: %d",
        total_stats["accepted"],
    )

    logger.info(
        "  Außerhalb XYZ-Kachel: %d",
        total_stats["outside_source"],
    )

    logger.info(
        "  Außerhalb Ziel: %d",
        total_stats["outside_target"],
    )

    logger.info(
        "  Ungültig: %d",
        total_stats["invalid"],
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = report_coverage(
        count_grid,
        resolution,
    )

    fail_on_missing_tile = bool(
        config["runtime"].get(
            "fail_on_missing_tile",
            True,
        )
    )

    if (
        fail_on_missing_tile
        and coverage <= 0
    ):

        raise RuntimeError(
            "Es wurde kein einziger "
            "Zielpixel befüllt."
        )

    # --------------------------------------------------------
    # Original DEM erzeugen
    # --------------------------------------------------------

    nodata = float(
        config["output"].get(
            "nodata",
            -9999.0,
        )
    )

    logger.info(
        "Erzeuge aggregiertes DEM..."
    )

    dem = create_dem(
        sum_grid=sum_grid,
        count_grid=count_grid,
        nodata=nodata,
    )

    # --------------------------------------------------------
    # Glättung
    #
    # Das Original-DGM bleibt erhalten.
    #
    # Für die Ausgabe wird bei aktivierter Glättung eine
    # separate Kopie erzeugt.
    # --------------------------------------------------------

    if smoothing_enabled:

        logger.info(
            "Erzeuge geglättete DEM-Ausgabe..."
        )

        output_dem = smooth_dem(
            dem=dem,
            nodata=nodata,
            config=config,
        )

    else:

        output_dem = dem

    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    output_directory = Path(
        config["output"][
            "directory"
        ]
    )

    output_filename = config[
        "output"
    ]["filename"]

    output_path = (
        output_directory
        / output_filename
    )

    write_geotiff(
        output_path=output_path,
        dem=output_dem,
        config=config,
        min_x=target_min_x,
        max_y=target_max_y,
        resolution=resolution,
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    png_config = (
        config["output"].get(
            "png",
            {},
        )
    )

    png_enabled = bool(
        png_config.get(
            "enabled",
            True,
        )
    )

    if png_enabled:

        png_filename = (
            png_config.get(
                "filename",
                Path(
                    output_filename
                ).with_suffix(
                    ".png"
                ).name,
            )
        )

        png_path = (
            output_directory
            / png_filename
        )

        (
            png_min,
            png_max,
        ) = write_png16(
            output_path=png_path,
            dem=output_dem,
            config=config,
        )

        logger.info(
            "PNG Minimum: %.3f m",
            png_min,
        )

        logger.info(
            "PNG Maximum: %.3f m",
            png_max,
        )

    # --------------------------------------------------------
    # Fertig
    # --------------------------------------------------------

    logger.info(
        "============================================================"
    )

    logger.info(
        "FERTIG"
    )

    logger.info(
        "GeoTIFF: %s",
        output_path.resolve(),
    )

    if png_enabled:

        logger.info(
            "PNG:     %s",
            png_path.resolve(),
        )

    logger.info(
        "Abdeckung: %.2f %%",
        coverage,
    )

    logger.info(
        "============================================================"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()