# FTP Video Compressor

A powerful tool for automatically downloading, compressing, and re-uploading videos from FTP servers with optimized AV1 encoding. The tool is specially designed to work with BunnyCDN storage but is compatible with any standard FTP server.

## Overview

FTP Video Compressor helps you reduce your storage costs by intelligently compressing video files while maintaining a target visual quality. The tool uses NVIDIA's AV1 hardware encoding to achieve high compression ratios with minimal quality loss, as measured by the industry-standard VMAF quality metric.

Key features:
- Automatic recursive scanning of FTP directories
- Intelligent sample-based analysis to determine optimal encoding settings
- VMAF-guided quality control
- Multi-threading for parallel processing
- Special handling for BunnyCDN storage
- Detailed compression statistics
- Subtitle and audio track handling

## Requirements

- **NVIDIA GPU** with AV1 encoding support (RTX 40 series or newer)
- **Python 3.7+**
- **FFmpeg** with NVIDIA hardware encoding support
- **VMAF** executable in the working directory

### Required Python Packages:
```
ftplib
subprocess
multiprocessing
concurrent.futures
json
```

## Installation

1. Clone or download this repository

2. Install the required Python packages:
   ```bash
   pip install ftplib
   ```

3. Install FFmpeg with NVIDIA hardware acceleration support:
   - Windows: Download from [BtbN's FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases) with NVIDIA support
   - Linux: Follow [FFmpeg NVIDIA GPU acceleration guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/)

4. Ensure you have the latest NVIDIA drivers installed that support AV1 encoding

## Usage

Run the script with the following command-line arguments:

```bash
python ftp_video_compressor.py --host <ftp_host> --user <username> --password <password> --output <local_output_dir> [options]
```

### Required Arguments:

- `--host`: FTP server hostname or IP address
- `--user`: FTP username
- `--password`: FTP password
- `--output`: Local output directory for working files

### Optional Arguments:

- `--directory`: FTP directory to scan for videos (default: `/`)
- `--storage-zone`: BunnyCDN storage zone name (if using BunnyCDN)
- `--vmaf-diff`: Target VMAF quality difference (default: `8`)
- `--concurrent`: Number of videos to process concurrently (default: `2`)
- `--audio-track`: Audio track to include (default: `0`)
- `--include-subtitles`: Include subtitles in output
- `--subtitle-track`: Subtitle track to include (default: `0`)
- `--skip-existing`: Skip processing if output file already exists
- `--replace-originals`: Replace original files on FTP with compressed versions (default: `True`)
- `--keep-originals`: Keep original files on FTP and save compressed versions locally
- `--scan-all`: Scan all directories including those that fail initially (useful for BunnyCDN)

## Examples

### Standard FTP Server:

```bash
python ftp_video_compressor.py --host ftp.example.com --user myusername --password mypassword --output /path/to/output --directory /videos --vmaf-diff 10 --concurrent 4
```

### BunnyCDN Storage:

```bash
python ftp_video_compressor.py --host storage.bunnycdn.com --user storagezonename --password api-access-key --output /path/to/output --storage-zone storagezonename --vmaf-diff 8 --concurrent 2
```

### Process Files in Multiple Folders with Different Quality Settings:

```bash
# Process HD content with higher quality retention
python ftp_video_compressor.py --host ftp.example.com --user myusername --password mypassword --output /path/to/output --directory /HD_content --vmaf-diff 5

# Process SD content with more aggressive compression
python ftp_video_compressor.py --host ftp.example.com --user myusername --password mypassword --output /path/to/output --directory /SD_content --vmaf-diff 12
```

## How It Works

1. The script connects to the FTP server and recursively scans for video files (.mp4, .mkv)
2. For each directory found, it downloads a representative video file
3. It analyzes the video by extracting short samples from the beginning, middle, and end
4. It tests various encoding parameters to find the optimal setting that maintains the target VMAF score difference
5. Once the optimal encoding setting is determined for a directory, it's applied to all videos in that directory
6. The videos are processed concurrently (limited by the `--concurrent` parameter)
7. Processed videos are either re-uploaded to the FTP server (replacing originals) or saved locally
8. Detailed statistics are provided about space savings

## VMAF Explained

VMAF (Video Multi-method Assessment Fusion) is a perceptual video quality metric developed by Netflix. The `--vmaf-diff` parameter controls the quality/compression tradeoff:

- **Lower values** (e.g., 5): Higher quality retention but less compression
- **Higher values** (e.g., 15): More aggressive compression but potentially more visible quality loss
- **Recommended range**: 6-12 for most content

The default value of 8 provides a good balance between quality and compression for most content.

## Troubleshooting

### "Directory Not Found" Errors
If you encounter "Directory Not Found" errors when the directories are clearly visible:
1. Double-check your FTP path structure
2. Ensure you're not using absolute paths when already in a subdirectory
3. Try running without a specific directory to auto-scan from the current location

### Connection Issues with BunnyCDN
For BunnyCDN connections:
1. Ensure you're using the storage hostname (storage.bunnycdn.com)
2. The username should be your storage zone name
3. The password should be your API access key
4. Use the `--storage-zone` parameter if your FTP username is different from your storage zone name

### Performance Issues
If encoding is too slow:
1. Reduce the number of concurrent processes with `--concurrent`
2. Ensure your NVIDIA drivers are up to date
3. Monitor GPU utilization to ensure it's being properly utilized

## Limitations

- The script requires an NVIDIA GPU with AV1 encoding support
- Processing very large files may require substantial local storage
- The VMAF analysis temporarily needs disk space for sample extraction
- FTP transfer speeds may limit overall performance for large libraries

## Additional Notes

- The script caches encoding settings for each directory, so subsequent runs will be faster
- For optimal results, organize your videos by quality/resolution in different directories
- AV1 encoding provides significant space savings over H.264, often 40-60% for the same visual quality
- Consider using a higher `--vmaf-diff` value for content where visual quality is less critical
