#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LGL DGM1 Downloader / Processor
--------------------------------

- Lädt aktuelle LGL DGM1 2x2-km ZIP-Kacheln
- Verarbeitet ALLE enthaltenen 1x1-km XYZ-Dateien
- Ignoriert Metadaten-Dateien
- Transformiert WGS84 -> ETRS89 / UTM32N (EPSG:25832)
- Schneidet auf die gewünschte Fläche zu
- Aggregiert 1-m-XYZ-Daten auf 4-m-Raster per Mittelwert
- Schreibt ein komprimiertes GeoTIFF

Aktuelle LGL-Struktur:

ZIP:
    dgm1_32_481_5404_2_bw.zip

Inhalt z.B.:
    dgm1_32_481_5404_2_bw/
        dgm1_32_481_5404_1_bw_2019.xyz
        dgm1_32_481_5405_1_bw_2019.xyz
        dgm1_32_482_5404_1_bw_2019.xyz
        dgm1_32_482_5405_1_bw_2019.xyz
        Meta-ATKIS_DGM1_ALS2.txt

Wichtig:
    Die ZIP-Datei ist 2x2 km groß.
    Die enthaltenen XYZ-Dateien sind jeweils 1x1 km groß.
    Es müssen ALLE passenden XYZ-Dateien verarbeitet werden.
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

from config_loader import load_config


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("LGL-DGM1")


def setup_logging(verbose=True):
    level = logging.INFO if verbose else logging.WARNING

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
                f"Fehlender YAML-Bereich: {section}"
            )

    if config["area"]["width_m"] <= 0:
        raise ValueError("area.width_m muss > 0 sein.")

    if config["area"]["height_m"] <= 0:
        raise ValueError("area.height_m muss > 0 sein.")

    if config["resolution_m"] <= 0:
        raise ValueError("resolution_m muss > 0 sein.")

    if config["lgl"]["tile_size_m"] != 2000:
        raise ValueError(
            "Für die aktuelle LGL-DGM1-Struktur muss "
            "lgl.tile_size_m = 2000 sein."
        )

    if config["processing"]["aggregation"].lower() != "mean":
        raise ValueError(
            "Aktuell wird nur aggregation: mean unterstützt."
        )


# ============================================================
# CRS / BBox
# ============================================================

def create_transformer(config):
    input_crs = config["crs"]["input"]
    processing_crs = config["crs"]["processing"]

    return Transformer.from_crs(
        input_crs,
        processing_crs,
        always_xy=True,
    )


def calculate_bbox(config, transformer):
    latitude = float(config["center"]["latitude"])
    longitude = float(config["center"]["longitude"])

    width_m = float(config["area"]["width_m"])
    height_m = float(config["area"]["height_m"])
    resolution = float(config["resolution_m"])

    center_x, center_y = transformer.transform(
        longitude,
        latitude,
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

    min_x = center_x - width_m / 2.0
    max_x = center_x + width_m / 2.0

    min_y = center_y - height_m / 2.0
    max_y = center_y + height_m / 2.0

    # Auf Zielauflösung ausrichten.
    min_x = math.floor(min_x / resolution) * resolution
    min_y = math.floor(min_y / resolution) * resolution

    width_cells = int(round(width_m / resolution))
    height_cells = int(round(height_m / resolution))

    max_x = min_x + width_cells * resolution
    max_y = min_y + height_cells * resolution

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
    Berechnet die aktuellen LGL-DGM1-2x2-km-Kacheln.

    Aktuelle Struktur:

        dgm1_32_<EASTING_KM>_<NORTHING_KM>_2_bw.zip

    LGL-Kachelraster:
        EASTING  = ungerade km-Werte
        NORTHING = gerade km-Werte

    Beispiel:

        dgm1_32_481_5404_2_bw.zip
        dgm1_32_483_5404_2_bw.zip
        ...
    """

    if tile_size != 2000:
        raise ValueError(
            "Das aktuelle LGL-DGM1-Kachelraster "
            "wird mit 2000 m erwartet."
        )

    easting_km = math.floor(min_x / 1000)

    if easting_km % 2 == 0:
        easting_km -= 1

    northing_km = math.floor(min_y / 1000)

    if northing_km % 2 != 0:
        northing_km -= 1

    tiles = []

    while True:
        tile_min_x = easting_km * 1000
        tile_max_x = tile_min_x + tile_size

        if tile_min_x >= max_x:
            break

        current_northing_km = northing_km

        while True:
            tile_min_y = current_northing_km * 1000
            tile_max_y = tile_min_y + tile_size

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
                    f"dgm1_32_"
                    f"{easting_km}_"
                    f"{current_northing_km}"
                    f"_2_bw.zip"
                )

                tiles.append(
                    {
                        "x": int(tile_min_x),
                        "y": int(tile_min_y),
                        "x_km": int(easting_km),
                        "y_km": int(current_northing_km),
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

    filename = tile["filename"]

    url_template = config["lgl"]["url_template"]

    url = url_template.format(
        x=tile["x_km"],
        y=tile["y_km"],
    )

    output_path = download_dir / filename

    max_retries = int(
        config["download"]["max_retries"]
    )

    timeout = int(
        config["download"]["timeout_seconds"]
    )

    chunk_size = int(
        config["download"]["chunk_size_bytes"]
    )

    user_agent = config["download"]["user_agent"]

    logger.info(
        "Download: %s",
        url,
    )

    headers = {
        "User-Agent": user_agent,
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
            ) as response:

                response.raise_for_status()

                temporary_path = output_path.with_suffix(
                    output_path.suffix + ".part"
                )

                with temporary_path.open("wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=chunk_size
                    ):
                        if chunk:
                            f.write(chunk)

                temporary_path.replace(output_path)

            logger.info(
                "Download erfolgreich: %s",
                output_path.name,
            )

            return output_path

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Downloadversuch %d/%d fehlgeschlagen: %s",
                attempt,
                max_retries,
                exc,
            )

            if attempt < max_retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Download fehlgeschlagen: {url}"
    ) from last_error


# ============================================================
# XYZ-Dateien im ZIP
# ============================================================

XYZ_FILENAME_RE = re.compile(
    r"^dgm1_32_(\d+)_(\d+)_1_bw_.*\.xyz$",
    re.IGNORECASE,
)


def find_xyz_files(zip_path):
    """
    Findet ALLE 1x1-km XYZ-Dateien im aktuellen 2x2-km ZIP.

    Beispiel:
        dgm1_32_481_5404_1_bw_2019.xyz
        dgm1_32_481_5405_1_bw_2019.xyz
        dgm1_32_482_5404_1_bw_2019.xyz
        dgm1_32_482_5405_1_bw_2019.xyz

    Meta-Dateien werden ignoriert.
    """

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    xyz_files = []

    for name in names:
        if name.endswith("/"):
            continue

        basename = os.path.basename(name)

        match = XYZ_FILENAME_RE.match(basename)

        if not match:
            continue

        easting_km = int(match.group(1))
        northing_km = int(match.group(2))

        xyz_files.append(
            {
                "path": name,
                "filename": basename,
                "x_km": easting_km,
                "y_km": northing_km,
                "min_x": easting_km * 1000.0,
                "min_y": northing_km * 1000.0,
                "max_x": easting_km * 1000.0 + 1000.0,
                "max_y": northing_km * 1000.0 + 1000.0,
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
            f"Keine gültige XYZ-Datei in {zip_path.name} gefunden."
        )

    logger.info(
        "%s: %d XYZ-Dateien gefunden",
        zip_path.name,
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
    """
    Unterstützt:

        X Y Z
        X;Y;Z
        X,Y,Z

    und Dezimalpunkte/-kommas.
    """

    line = line.strip()

    if not line:
        return None

    if line.startswith("#"):
        return None

    if ";" in line:
        parts = line.split(";")
    else:
        parts = line.split()

    if len(parts) < 3:
        return None

    try:
        x = float(parts[0].strip().replace(",", "."))
        y = float(parts[1].strip().replace(",", "."))
        z = float(parts[2].strip().replace(",", "."))

        return x, y, z

    except ValueError:
        return None


# ============================================================
# XYZ verarbeiten
# ============================================================

def process_xyz_file(
    zip_path,
    xyz_info,
    sum_grid,
    count_grid,
    target_min_x,
    target_min_y,
    target_max_x,
    target_max_y,
    target_resolution,
):
    """
    Verarbeitet genau EINE 1x1-km XYZ-Datei.

    Die Koordinaten der XYZ-Datei sind die Rasterpunktzentren.

    Die XYZ-Datei selbst deckt ab:

        [min_x, max_x)
        [min_y, max_y)

    Die Zielzellen werden über floor() zugeordnet.
    """

    xyz_path = xyz_info["path"]

    source_min_x = xyz_info["min_x"]
    source_max_x = xyz_info["max_x"]
    source_min_y = xyz_info["min_y"]
    source_max_y = xyz_info["max_y"]

    points = 0
    accepted = 0
    outside_source = 0
    outside_target = 0
    invalid = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(xyz_path, "r") as f:

            for raw_line in f:
                points += 1

                try:
                    line = raw_line.decode(
                        "utf-8",
                        errors="ignore",
                    )
                except Exception:
                    invalid += 1
                    continue

                parsed = parse_xyz_line(line)

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

                # ------------------------------------------------
                # Quelle: 1x1-km XYZ-Kachel
                # ------------------------------------------------

                if (
                    x < source_min_x
                    or x >= source_max_x
                    or y < source_min_y
                    or y >= source_max_y
                ):
                    outside_source += 1
                    continue

                # ------------------------------------------------
                # Ziel-BBox
                # ------------------------------------------------

                if (
                    x < target_min_x
                    or x >= target_max_x
                    or y < target_min_y
                    or y >= target_max_y
                ):
                    outside_target += 1
                    continue

                # ------------------------------------------------
                # Zielpixel bestimmen
                # ------------------------------------------------

                col = int(
                    math.floor(
                        (x - target_min_x)
                        / target_resolution
                    )
                )

                row = int(
                    math.floor(
                        (target_max_y - y)
                        / target_resolution
                    )
                )

                if (
                    row < 0
                    or row >= sum_grid.shape[0]
                    or col < 0
                    or col >= sum_grid.shape[1]
                ):
                    outside_target += 1
                    continue

                sum_grid[row, col] += z
                count_grid[row, col] += 1

                accepted += 1

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
):
    """
    Verarbeitet ALLE XYZ-Dateien eines 2x2-km ZIP-Archivs.

    Das ist der entscheidende Unterschied zur alten Version:
    Eine ZIP-Datei enthält vier 1x1-km XYZ-Dateien.
    """

    xyz_files = find_xyz_files(zip_path)

    total_points = 0
    total_accepted = 0
    total_outside_source = 0
    total_outside_target = 0
    total_invalid = 0

    for xyz_info in xyz_files:

        logger.info(
            "Verarbeite %s -> %s",
            zip_path.name,
            xyz_info["filename"],
        )

        result = process_xyz_file(
            zip_path=zip_path,
            xyz_info=xyz_info,
            sum_grid=sum_grid,
            count_grid=count_grid,
            target_min_x=target_min_x,
            target_min_y=target_min_y,
            target_max_x=target_max_x,
            target_max_y=target_max_y,
            target_resolution=target_resolution,
        )

        total_points += result["points"]
        total_accepted += result["accepted"]
        total_outside_source += result["outside_source"]
        total_outside_target += result["outside_target"]
        total_invalid += result["invalid"]

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
    """
    Erzeugt aus Summe / Anzahl das mittlere DGM.

    Ausgabe als float32.
    """

    dem = np.full(
        sum_grid.shape,
        nodata,
        dtype=np.float32,
    )

    valid = count_grid > 0

    dem[valid] = (
        sum_grid[valid]
        / count_grid[valid]
    ).astype(np.float32)

    return dem


# ============================================================
# Coverage Report
# ============================================================

def report_coverage(
    count_grid,
    resolution,
):
    total_cells = count_grid.size
    valid_cells = int(
        np.count_nonzero(count_grid)
    )
    empty_cells = total_cells - valid_cells

    coverage = (
        valid_cells / total_cells * 100.0
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
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = dem.shape

    output_config = config["output"]

    compression = output_config.get(
        "compression",
        "deflate",
    )

    predictor = output_config.get(
        "predictor",
        3,
    )

    tiled = output_config.get(
        "tiled",
        True,
    )

    bigtiff = output_config.get(
        "bigtiff",
        "IF_SAFER",
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
        "crs": config["crs"]["processing"],
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

    logger.info(
        "  Größe: %d x %d",
        width,
        height,
    )

    logger.info(
        "  CRS: %s",
        config["crs"]["processing"],
    )

    logger.info(
        "  Auflösung: %.3f m",
        resolution,
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
        "GeoTIFF erfolgreich geschrieben: %s",
        output_path,
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
    # LGL 2x2-km Kacheln
    # --------------------------------------------------------

    tiles = calculate_lgl_tiles(
        min_x=target_min_x,
        min_y=target_min_y,
        max_x=target_max_x,
        max_y=target_max_y,
        tile_size=int(
            config["lgl"]["tile_size_m"]
        ),
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
        "Speicher für Summenraster: %.1f MB",
        sum_grid.nbytes / 1024 / 1024,
    )

    logger.info(
        "Speicher für Zählraster: %.1f MB",
        count_grid.nbytes / 1024 / 1024,
    )

    # --------------------------------------------------------
    # Download + Verarbeitung
    # --------------------------------------------------------

    download_dir = Path(
        config["download"]["directory"]
    )

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
            )

            for key in total_stats:
                total_stats[key] += stats[key]

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
                        "ZIP konnte nicht gelöscht werden: %s",
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

    if fail_on_missing_tile and coverage <= 0:
        raise RuntimeError(
            "Es wurde kein einziger Zielpixel befüllt."
        )

    # --------------------------------------------------------
    # DEM
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
    # GeoTIFF
    # --------------------------------------------------------

    output_directory = Path(
        config["output"]["directory"]
    )

    output_filename = config["output"]["filename"]

    output_path = (
        output_directory
        / output_filename
    )

    write_geotiff(
        output_path=output_path,
        dem=dem,
        config=config,
        min_x=target_min_x,
        max_y=target_max_y,
        resolution=resolution,
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "FERTIG"
    )

    logger.info(
        "Ausgabe: %s",
        output_path.resolve(),
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