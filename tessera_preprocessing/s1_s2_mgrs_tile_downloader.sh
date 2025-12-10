#!/usr/bin/env bash
#
# s1_s2_mgrs_tile_downloader.sh — MGRS Tile-based Sentinel-1 & Sentinel-2 Downloader
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.7, jq
# Usage: bash s1_s2_mgrs_tile_downloader.sh

set -u

#######################################
# USER CONFIGURABLE PARAMETERS
#######################################

# === Basic Configuration ===
GEOJSON_FILE="/home/morteza/usask/tessera/mgrs_grids/Upper_Assiniboine_25km_grids.geojson"
BASE_OUT_DIR="/mnt/d/USASK/Upper_Assiniboine_grid_25km"
PYTHON_ENV="/home/morteza/usask/tessera/.venv/bin/python"

# === Sentinel-1 & Sentinel-2 Processing Configuration ===
YEAR=2024 # Range [2017-2024]
RESOLUTION=10.0  # Resolution of the input TIFF, also the output resolution (meters)

# === Sentinel-1 Configuration ===
S1_ENABLED=true                    # Enable S1 processing
S1_PARTITIONS=12                   # Number of S1 parallel partitions
S1_TOTAL_WORKERS=12                # Total number of S1 Dask workers
S1_WORKER_MEMORY=4                 # Memory per S1 worker (GB)
S1_CHUNKSIZE=1024                  # S1 stackstac chunk size
S1_ORBIT_STATE="both"              # Orbit state: ascending/descending/both
S1_MIN_COVERAGE=0.01               # Minimum valid pixel coverage for S1 (%)
S1_RESOLUTION=$RESOLUTION          # S1 output resolution (meters)
S1_OVERWRITE=true                  # Overwrite existing S1 files

# === Sentinel-2 Configuration ===
S2_ENABLED=true                    # Enable S2 processing
S2_PARTITIONS=24                   # Number of S2 parallel partitions
S2_TOTAL_WORKERS=24                # Total number of S2 Dask workers
S2_WORKER_MEMORY=4                 # Memory per S2 worker (GB)
S2_CHUNKSIZE=1024                  # S2 stackstac chunk size
S2_MAX_CLOUD=100                   # Maximum cloud coverage for S2 (%)
S2_RESOLUTION=$RESOLUTION          # S2 output resolution (meters)
S2_MIN_COVERAGE=0.01               # Minimum valid pixel coverage for S2 (%)
S2_OVERWRITE=true                  # Overwrite existing S2 files

# === System Configuration ===
DEBUG=false                        # Enable debug mode
LOG_INTERVAL=10                    # Progress update interval (seconds)

#######################################
# Internal Variables
#######################################
START_TIME="${YEAR}-01-01"
END_TIME="${YEAR}-12-31"

SCRIPT_START_TIME=$(date +%s)
SCRIPT_NAME=$(basename "$0")

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#######################################
# Logging Function
#######################################
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)
            echo -e "${timestamp} ${BLUE}[INFO]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message"
            ;;
        WARNING)
            echo -e "${timestamp} ${YELLOW}[WARNING]${NC} $message"
            ;;
        ERROR)
            echo -e "${timestamp} ${RED}[ERROR]${NC} $message"
            ;;
        HEADER)
            echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
            echo -e "${timestamp} ${PURPLE}$message${NC}"
            echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
            ;;
    esac
}

#######################################
# Utility Functions
#######################################
time_to_seconds() {
    date -d "$1" +%s
}

seconds_to_date() {
    date -d "@$1" +"%Y-%m-%d"
}

format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

calculate_partitions() {
    local start_sec=$(time_to_seconds "$START_TIME")
    local end_sec=$(time_to_seconds "$END_TIME")
    local partitions=$1
    local total_seconds=$((end_sec - start_sec + 86400))
    local seconds_per_partition=$((total_seconds / partitions))

    for ((i=0; i<partitions; i++)); do
        local partition_start_sec
        local partition_end_sec

        if [[ $i -eq 0 ]]; then
            partition_start_sec=$start_sec
        else
            partition_start_sec=$((start_sec + i * seconds_per_partition))
        fi

        if [[ $i -eq $((partitions - 1)) ]]; then
            partition_end_sec=$end_sec
        else
            partition_end_sec=$((start_sec + (i + 1) * seconds_per_partition - 86400))
        fi

        local p_start=$(seconds_to_date $partition_start_sec)
        local p_end=$(seconds_to_date $partition_end_sec)

        echo "$p_start,$p_end"
    done
}

calculate_workers_per_partition() {
    local total_workers=$1
    local partitions=$2
    local workers_per_partition=$((total_workers / partitions))
    local remaining_workers=$((total_workers % partitions))

    for ((i=0; i<partitions; i++)); do
        if [[ $i -lt $remaining_workers ]]; then
            echo $((workers_per_partition + 1))
        else
            echo $workers_per_partition
        fi
    done
}

generate_partition_id() {
    local p_start=$1
    local p_end=$2
    local p_index=$3
    local prefix=$4

    local start_year=$(date -d "${p_start}" +%Y)
    local start_month=$(date -d "${p_start}" +%m)
    local start_day=$(date -d "${p_start}" +%d)
    local end_year=$(date -d "${p_end}" +%Y)
    local end_month=$(date -d "${p_end}" +%m)
    local end_day=$(date -d "${p_end}" +%d)

    if [[ "$start_year" == "$end_year" && "$start_month" == "$end_month" ]]; then
        if [[ "$start_day" == "$end_day" ]]; then
            echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}"
        else
            echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_day}"
        fi
    elif [[ "$start_year" == "$end_year" ]]; then
        echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_month}${end_day}"
    else
        echo "${prefix}_P${p_index}_${start_year}${start_month}${start_day}-${end_year}${end_month}${end_day}"
    fi
}

#######################################
# Monitoring Function
#######################################
monitor_processes() {
    local -n pids=$1
    local -n partition_ids=$2
    local -n start_times=$3
    local -n completed=$4
    local -n failed=$5
    local prefix=$6

    declare -A finished_pids

    while true; do
        local all_done=true
        local running_count=0

        for i in "${!pids[@]}"; do
            local pid=${pids[i]}
            if [[ -n "${finished_pids[$pid]:-}" ]]; then
                continue
            fi

            if kill -0 $pid 2>/dev/null; then
                all_done=false
                running_count=$((running_count + 1))
            else
                local end_time=$(date +%s)
                local duration=$((end_time - ${start_times[i]}))
                local partition_id=${partition_ids[i]}

                wait $pid
                local exit_code=$?

                finished_pids[$pid]=1

                if [ $exit_code -eq 0 ]; then
                    log SUCCESS "$prefix partition $partition_id completed ($(format_duration $duration))"
                    completed+=("$partition_id")
                else
                    log ERROR "$prefix partition $partition_id failed with exit code $exit_code ($(format_duration $duration))"
                    failed+=("$partition_id")
                fi
            fi
        done

        if ! $all_done; then
            echo -ne "\r${CYAN}[$(date '+%H:%M:%S')]${NC} $prefix: ${running_count}/${#pids[@]} partitions running...${NC}"
        fi

        if $all_done; then
            echo -ne "\r\033[K"
            break
        fi

        sleep $LOG_INTERVAL
    done
}

#######################################
# Convert GeoJSON Feature to TIFF
#######################################
convert_feature_to_tiff() {
    local mgrs_id=$1
    local feature_json=$2
    local output_tiff=$3

    # Create a temporary GeoJSON file with single feature
    local temp_geojson="${output_tiff%.tiff}_temp.geojson"

    # Wrap the feature in a FeatureCollection
    cat > "$temp_geojson" <<EOF
{
  "type": "FeatureCollection",
  "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
  "features": [
    $feature_json
  ]
}
EOF

    # Convert to TIFF using Python script
    log INFO "Converting MGRS tile $mgrs_id to TIFF..."

    # Use the convert_shp_to_tiff.py logic via Python directly
    $PYTHON_ENV -c "
import json
import re
import sys
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# Read the GeoJSON
with open('$temp_geojson', 'r') as f:
    data = json.load(f)

feature = data['features'][0]
geometry = feature['geometry']

# Source CRS: WGS84
src_crs = CRS.from_epsg(4326)

# Extract MGRS properties (new field name is MGRS)
props = feature['properties']
mgrs_code = (props.get('MGRS') or '').upper()

match = re.match(r\"(?P<zone>\\d{1,2})(?P<band>[C-HJ-NP-X])\", mgrs_code)
if not match:
    raise ValueError(f\"Invalid or missing MGRS code: '{mgrs_code}'\")

zone_number = int(match.group('zone'))
lat_band = match.group('band')

northern_letters = set(\"NPQRSTUVWX\")
is_northern = lat_band in northern_letters

# Determine target UTM CRS
epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
target_crs = CRS.from_epsg(epsg_code)

# Transform geometry
transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True).transform
geom_shape = shape(geometry)
reprojected_geom = shp_transform(transformer, geom_shape)
reprojected_geom_json = mapping(reprojected_geom)

# Get bounds
minx, miny, maxx, maxy = reprojected_geom.bounds

# Calculate dimensions with pixel size
pixel_size = $RESOLUTION
width = int(np.ceil((maxx - minx) / pixel_size))
height = int(np.ceil((maxy - miny) / pixel_size))

# Create affine transform
transform_affine = from_origin(minx, maxy, pixel_size, pixel_size)

# Rasterize
raster = rasterize(
    [(reprojected_geom_json, 255)],
    out_shape=(height, width),
    transform=transform_affine,
    fill=0,
    dtype='uint8'
)

# Write TIFF
with rasterio.open(
    '$output_tiff',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=target_crs,
    transform=transform_affine,
) as dst:
    dst.write(raster, 1)

print(f'Successfully created TIFF: $output_tiff')
"

    local exit_code=$?

    # Clean up temp file
    rm -f "$temp_geojson"

    return $exit_code
}

#######################################
# Processing Functions for Single Tile
#######################################
process_tile_sentinel1() {
    local mgrs_id=$1
    local input_tiff=$2
    local out_dir=$3
    local log_dir=$4

    log HEADER "[$mgrs_id] Starting Sentinel-1 Processing"

    local s1_output="${out_dir}/data_sar_raw"
    mkdir -p "$s1_output"

    # Generate partitions
    mapfile -t s1_partitions < <(calculate_partitions $S1_PARTITIONS)
    mapfile -t s1_workers_per_partition < <(calculate_workers_per_partition $S1_TOTAL_WORKERS $S1_PARTITIONS)

    log INFO "[$mgrs_id] S1 Configuration: $S1_PARTITIONS partitions, $S1_TOTAL_WORKERS workers"

    # Start parallel processing
    local s1_pids=()
    local s1_partition_ids=()
    local s1_start_times=()
    local s1_completed=()
    local s1_failed=()

    for i in "${!s1_partitions[@]}"; do
        p_range=${s1_partitions[i]}
        p_start=${p_range%,*}
        p_end=${p_range#*,}
        workers=${s1_workers_per_partition[i]}
        partition_id=$(generate_partition_id "$p_start" "$p_end" "$i" "S1")

        s1_partition_ids+=("$partition_id")

        local overwrite_flag=""
        [[ "$S1_OVERWRITE" == "true" ]] && overwrite_flag="--overwrite"

        local debug_flag=""
        [[ "$DEBUG" == "true" ]] && debug_flag="--debug"

        $PYTHON_ENV s1_fast_processor.py \
            --input_tiff "$input_tiff" \
            --start_date "$p_start" \
            --end_date "$p_end" \
            --output "$s1_output" \
            --orbit_state "$S1_ORBIT_STATE" \
            --dask_workers "$workers" \
            --worker_memory "$S1_WORKER_MEMORY" \
            --resolution "$S1_RESOLUTION" \
            --chunksize "$S1_CHUNKSIZE" \
            --min_coverage "$S1_MIN_COVERAGE" \
            --partition_id "$partition_id" \
            $overwrite_flag $debug_flag \
            > "$log_dir/${partition_id}.log" 2>&1 &

        s1_pids+=($!)
        s1_start_times+=("$(date +%s)")

        sleep 2
    done

    log INFO "[$mgrs_id] S1: Launched ${#s1_pids[@]} partition processes"

    # Monitor processes
    monitor_processes s1_pids s1_partition_ids s1_start_times s1_completed s1_failed "S1"

    # Summary
    log INFO "[$mgrs_id] S1 Summary: ${#s1_completed[@]}/${#s1_partitions[@]} successful"

    if [[ ${#s1_failed[@]} -gt 0 ]]; then
        log WARNING "[$mgrs_id] S1 failed partitions: ${s1_failed[@]}"
        return 1
    fi

    return 0
}

process_tile_sentinel2() {
    local mgrs_id=$1
    local input_tiff=$2
    local out_dir=$3
    local log_dir=$4

    log HEADER "[$mgrs_id] Starting Sentinel-2 Processing"

    local s2_output="${out_dir}/data_raw"
    mkdir -p "$s2_output"

    # Generate partitions
    mapfile -t s2_partitions < <(calculate_partitions $S2_PARTITIONS)
    mapfile -t s2_workers_per_partition < <(calculate_workers_per_partition $S2_TOTAL_WORKERS $S2_PARTITIONS)

    log INFO "[$mgrs_id] S2 Configuration: $S2_PARTITIONS partitions, $S2_TOTAL_WORKERS workers"

    # Start parallel processing
    local s2_pids=()
    local s2_partition_ids=()
    local s2_start_times=()
    local s2_completed=()
    local s2_failed=()

    for i in "${!s2_partitions[@]}"; do
        p_range=${s2_partitions[i]}
        p_start=${p_range%,*}
        p_end=${p_range#*,}
        workers=${s2_workers_per_partition[i]}
        partition_id=$(generate_partition_id "$p_start" "$p_end" "$i" "S2")

        s2_partition_ids+=("$partition_id")

        local overwrite_flag=""
        [[ "$S2_OVERWRITE" == "true" ]] && overwrite_flag="--overwrite"

        local debug_flag=""
        [[ "$DEBUG" == "true" ]] && debug_flag="--debug"

        # Add time to S2
        local s2_start="${p_start}T00:00:00"
        local s2_end="${p_end}T23:59:59"

        $PYTHON_ENV s2_fast_processor.py \
            --input_tiff "$input_tiff" \
            --start_date "$s2_start" \
            --end_date "$s2_end" \
            --output "$s2_output" \
            --max_cloud "$S2_MAX_CLOUD" \
            --dask_workers "$workers" \
            --worker_memory "$S2_WORKER_MEMORY" \
            --chunksize "$S2_CHUNKSIZE" \
            --resolution "$S2_RESOLUTION" \
            --min_coverage "$S2_MIN_COVERAGE" \
            --partition_id "$partition_id" \
            $overwrite_flag $debug_flag \
            > "$log_dir/${partition_id}.log" 2>&1 &

        s2_pids+=($!)
        s2_start_times+=("$(date +%s)")

        sleep 2
    done

    log INFO "[$mgrs_id] S2: Launched ${#s2_pids[@]} partition processes"

    # Monitor processes
    monitor_processes s2_pids s2_partition_ids s2_start_times s2_completed s2_failed "S2"

    # Summary
    log INFO "[$mgrs_id] S2 Summary: ${#s2_completed[@]}/${#s2_partitions[@]} successful"

    if [[ ${#s2_failed[@]} -gt 0 ]]; then
        log WARNING "[$mgrs_id] S2 failed partitions: ${s2_failed[@]}"
        return 1
    fi

    return 0
}

#######################################
# Process Single MGRS Tile
#######################################
process_mgrs_tile() {
    local mgrs_id=$1
    local feature_json=$2

    log HEADER "Processing MGRS Tile: $mgrs_id"

    local tile_start_time=$(date +%s)

    # Create tile directory
    local tile_dir="${BASE_OUT_DIR}/${mgrs_id}"
    mkdir -p "$tile_dir"

    # Set up directories
    local log_dir="${tile_dir}/logs"
    local temp_dir="${tile_dir}/temp_dir"
    mkdir -p "$log_dir" "$temp_dir"

    export TEMP_DIR="$temp_dir"

    # Create TIFF for this tile
    local input_tiff="${tile_dir}/${mgrs_id}.tiff"

    if [[ ! -f "$input_tiff" ]]; then
        log INFO "[$mgrs_id] Creating TIFF from GeoJSON feature..."
        if ! convert_feature_to_tiff "$mgrs_id" "$feature_json" "$input_tiff"; then
            log ERROR "[$mgrs_id] Failed to create TIFF"
            return 1
        fi
    else
        log INFO "[$mgrs_id] TIFF already exists: $input_tiff"
    fi

    # Processing flags
    local s1_success=true
    local s2_success=true

    # Start S1 and S2 processing in parallel
    if [[ "$S1_ENABLED" == "true" && "$S2_ENABLED" == "true" ]]; then
        log INFO "[$mgrs_id] Starting parallel S1 and S2 processing..."

        ( process_tile_sentinel1 "$mgrs_id" "$input_tiff" "$tile_dir" "$log_dir" ) &
        local s1_pid=$!

        ( process_tile_sentinel2 "$mgrs_id" "$input_tiff" "$tile_dir" "$log_dir" ) &
        local s2_pid=$!

        if ! wait $s1_pid; then
            s1_success=false
        fi

        if ! wait $s2_pid; then
            s2_success=false
        fi

    elif [[ "$S1_ENABLED" == "true" ]]; then
        if ! process_tile_sentinel1 "$mgrs_id" "$input_tiff" "$tile_dir" "$log_dir"; then
            s1_success=false
        fi

    elif [[ "$S2_ENABLED" == "true" ]]; then
        if ! process_tile_sentinel2 "$mgrs_id" "$input_tiff" "$tile_dir" "$log_dir"; then
            s2_success=false
        fi
    fi

    # Calculate duration
    local tile_duration=$(($(date +%s) - tile_start_time))

    # Log result
    if [[ "$s1_success" == "true" && "$s2_success" == "true" ]]; then
        log SUCCESS "[$mgrs_id] Tile processing completed ($(format_duration $tile_duration))"
        return 0
    else
        log ERROR "[$mgrs_id] Tile processing failed ($(format_duration $tile_duration))"
        return 1
    fi
}

#######################################
# Main Program
#######################################
main() {
    log HEADER "MGRS Tile-based Sentinel Downloader Started"
    log INFO "GeoJSON: $GEOJSON_FILE"
    log INFO "Output Base: $BASE_OUT_DIR"
    log INFO "Year: $YEAR"

    # Check dependencies
    if ! command -v jq &> /dev/null; then
        log ERROR "jq is not installed. Please install it: sudo apt-get install jq"
        exit 1
    fi

    if [[ ! -f "$GEOJSON_FILE" ]]; then
        log ERROR "GeoJSON file not found: $GEOJSON_FILE"
        exit 1
    fi

    if [[ ! -x "$PYTHON_ENV" ]]; then
        log ERROR "Python environment not found: $PYTHON_ENV"
        exit 1
    fi

    # Create base output directory
    mkdir -p "$BASE_OUT_DIR"

    # Get total number of tiles
    local total_tiles=$(jq '.features | length' "$GEOJSON_FILE")
    log INFO "Found $total_tiles MGRS tiles to process"

    # Track results
    local tiles_completed=()
    local tiles_failed=()

    # Process each tile
    local tile_index=0
    while IFS= read -r feature; do
        local mgrs_id=$(echo "$feature" | jq -r '.properties.MGRS')

        log INFO "Processing tile $((tile_index + 1))/$total_tiles: $mgrs_id"

        if process_mgrs_tile "$mgrs_id" "$feature"; then
            tiles_completed+=("$mgrs_id")
        else
            tiles_failed+=("$mgrs_id")
        fi

        tile_index=$((tile_index + 1))

    done < <(jq -c '.features[]' "$GEOJSON_FILE")

    # Final summary
    log HEADER "Processing Complete"

    local total_duration=$(($(date +%s) - SCRIPT_START_TIME))
    log INFO "Total duration: $(format_duration $total_duration)"
    log INFO "Tiles processed: $total_tiles"
    log INFO "Successful: ${#tiles_completed[@]}"
    log INFO "Failed: ${#tiles_failed[@]}"

    if [[ ${#tiles_failed[@]} -gt 0 ]]; then
        log WARNING "Failed tiles: ${tiles_failed[@]}"
        exit 1
    else
        log SUCCESS "All tiles processed successfully!"
        exit 0
    fi
}

# Capture interrupt signal
trap 'log ERROR "Process interrupted by user"; exit 130' INT TERM

# Run main program
main "$@"
