# BW Geo Data Downloader

Downloads Baden-Württemberg LGL DGM1 elevation data, processes it into a GeoTIFF/heightmap (working good), and generates optional Transport Fever 3 terrain and biome masks from OpenStreetMap data (experemental/not working well).

## Features

- Downloads the required LGL DGM1 tiles for a configured area.
- Processes the source data into a configurable raster resolution.
- Writes a compressed GeoTIFF and optional PNG heightmap.
- Generates OSM-based masks for:
  - Rivers and waterways
  - Water bodies
  - Forests
  - Meadows
  - Farmland
  - Settlements
  - Heath
  - Wetlands
  - Rocks
  - Mountain areas
- Uses one user-facing configuration file and one advanced technical configuration file.
- Runs the complete workflow through a single entry point: `main.py`.

## Requirements

- Python 3.10 or newer
- Internet access for the LGL and Overpass API downloads
- Sufficient disk space for the downloaded ZIP tiles and output rasters

The following Python packages are required:

- `numpy`
- `Pillow`
- `pyproj`
- `PyYAML`
- `requests`
- `rasterio`
- `scipy`
- `shapely`
- `tqdm`

## Installation

### Linux/macOS

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy Pillow pyproj PyYAML requests rasterio scipy shapely tqdm
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy Pillow pyproj PyYAML requests rasterio scipy shapely tqdm
```

`rasterio` may require platform-specific binary dependencies on some systems. Installing it inside a current virtual environment using the official Python wheels is recommended.

## Configuration

The project uses two YAML files:

- [config.yaml](config.yaml) contains settings normally changed by the user.
- [advanced_config.yaml](advanced_config.yaml) contains technical defaults that usually should not be changed.

The loader merges the advanced configuration first and then applies the values from `config.yaml`. This means values in `config.yaml` take precedence.

### Common settings

Edit [config.yaml](config.yaml) to change the download area and output:

```yaml
center:
  latitude: 48.93278
  longitude: 8.95639

area:
  width_m: 28000
  height_m: 28000

resolution_m: 4.0
```

The area coordinates use WGS84. `width_m` and `height_m` are measured in metres. Smaller `resolution_m` values produce more detailed but larger and slower outputs.

The default output is written to:

- `output/Vaihingen_DGM_4m_2.tif`
- `output/dgm_4m_heightmap.png`
- `output/tf3/` for TF3 masks

The downloaded source ZIP files are stored in `dgm1_tiles/` by default.

## Usage

Activate the virtual environment first, then run the complete pipeline:

```bash
source .venv/bin/activate
python main.py
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python main.py
```

The pipeline runs these steps in order:

1. `download_newest.py` downloads and processes the DGM1 data.
2. `river_downloader.py` downloads OSM waterway data and creates the river mask.
3. `biome_downloader.py` downloads OSM land-use data and creates the biome and terrain masks.

The pipeline stops if a required step fails.

### Use another configuration

```bash
python main.py path/to/my-config.yaml
```

The corresponding `advanced_config.yaml` is searched for in the same directory as the selected user configuration.

### Skip optional mask generation

Skip the river mask:

```bash
python main.py --skip-rivers
```

Skip biome and terrain masks:

```bash
python main.py --skip-biomes
```

Skip both mask stages and only create the DGM output:

```bash
python main.py --skip-rivers --skip-biomes
```

## Running individual components

The components can also be run separately when troubleshooting:

```bash
python download_newest.py config.yaml
python river_downloader.py config.yaml
python biome_downloader.py config.yaml
```

Normally, use `main.py` so that the complete process runs consistently.

## Output files

The main elevation output is a Float32 GeoTIFF using the configured processing CRS, by default EPSG:25832. The optional PNG heightmap is a normalized 16-bit grayscale image.

TF3 masks are 8-bit grayscale PNG files. By default:

- `0` means background/absence.
- `255` means full presence.
- The biome classification mask uses the configured grayscale values in `config.yaml`.

## Troubleshooting

### Missing Python package

Activate the project virtual environment and reinstall the dependencies:

```bash
source .venv/bin/activate
python -m pip install numpy Pillow pyproj PyYAML requests rasterio scipy shapely tqdm
```

### Missing or unavailable LGL tile

Check the internet connection and the LGL URL in [advanced_config.yaml](advanced_config.yaml). The default configuration is set to stop when a required tile cannot be downloaded.

### Overpass API timeout or rate limiting

The mask generators use the public Overpass API. Retry later if the service is busy. The endpoint, timeout, and retry count are configured in [advanced_config.yaml](advanced_config.yaml).

### Large memory or disk usage

Reduce `area.width_m` and `area.height_m`, or increase `resolution_m`. A 28 km by 28 km area at 4 m resolution produces approximately 49 million output pixels before additional temporary processing data is considered.

## Project structure

```text
.
├── main.py                  # Complete pipeline entry point
├── config.yaml              # User-facing configuration
├── advanced_config.yaml     # Technical configuration
├── config_loader.py         # Configuration merge logic
├── download_newest.py       # Current DGM downloader/processor
├── river_downloader.py      # OSM river mask generator
├── biome_downloader.py      # OSM biome and terrain mask generator
├── dgm1_tiles/              # Downloaded source archives
└── output/                  # Generated GeoTIFFs, heightmaps, and masks
```

## Data sources and attribution

- Elevation data: Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL), DGM1.
- Feature data: OpenStreetMap contributors, queried through the Overpass API.

Follow the licensing and attribution requirements of both data providers when distributing generated files.
