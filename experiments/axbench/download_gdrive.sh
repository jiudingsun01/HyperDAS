#!/bin/bash

# Function to extract file ID from Google Drive URL
get_file_id() {
    local url=$1
    local file_id=""
    
    if [[ $url =~ /file/d/([^/]+) ]]; then
        file_id="${BASH_REMATCH[1]}"
    elif [[ $url =~ id=([^&]+) ]]; then
        file_id="${BASH_REMATCH[1]}"
    else
        echo "Invalid Google Drive URL"
        exit 1
    fi
    echo "$file_id"
}

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing now..."
    pip install gdown
fi

# Check if URL is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <google_drive_url> [output_directory]"
    exit 1
fi

# Get the Google Drive URL
DRIVE_URL="$1"
FILE_ID=$(get_file_id "$DRIVE_URL")

# Set output directory (default to current directory if not specified)
OUTPUT_DIR="${2:-.}"
mkdir -p "$OUTPUT_DIR"

# Temporary file for the download
TEMP_FILE="$OUTPUT_DIR/download.zip"

echo "Downloading file from Google Drive..."

# Download using gdown
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$TEMP_FILE"

# Check if file was downloaded successfully
if [ ! -f "$TEMP_FILE" ]; then
    echo "Download failed!"
    exit 1
fi

echo "Download complete. Unzipping file..."

# Unzip the file
unzip -q "$TEMP_FILE" -d "$OUTPUT_DIR"

# Check if unzip was successful
if [ $? -eq 0 ]; then
    echo "File successfully unzipped to $OUTPUT_DIR"
    # Clean up
    rm "$TEMP_FILE"
else
    echo "Failed to unzip file"
    exit 1
fi