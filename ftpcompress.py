import os
import subprocess
import json
import re
import tempfile
import time
import math
import multiprocessing
import ftplib
import argparse
import glob
import concurrent.futures
from pathlib import Path

class FtpVideoCompressor:
    def __init__(self, ftp_host, ftp_user, ftp_password, ftp_directory, output_dir, 
                 target_vmaf_diff=8, max_concurrent=2, audio_track=0, include_subtitles=False, 
                 subtitle_track=None, skip_existing=True, replace_originals=True, storage_zone=None,
                 scan_all=False):
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_password = ftp_password
        self.ftp_directory = ftp_directory
        self.output_dir = output_dir
        self.target_vmaf_diff = target_vmaf_diff
        self.max_concurrent = max_concurrent
        self.audio_track = audio_track
        self.include_subtitles = include_subtitles
        self.subtitle_track = subtitle_track
        self.skip_existing = skip_existing
        self.replace_originals = replace_originals
        self.storage_zone = storage_zone
        self.scan_all = scan_all
        self.download_dir = os.path.join(output_dir, "downloads")
        self.temp_dir = os.path.join(output_dir, "temp_analysis")
        self.folder_settings_cache = {}  # Cache for folder-specific encoding settings
        
        # Statistics tracking
        self.original_size_total = 0  # Total size of original files in bytes
        self.encoded_size_total = 0   # Total size of encoded files in bytes
        self.processed_files_count = 0  # Count of successfully processed files
        
        # Create necessary directories
        for directory in [self.output_dir, self.download_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)

    def scan_storage(self):
        """Special handling for scanning storage with proper path handling"""
        print("Using improvedstorage scanning mode...")

        files = []

        try:
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(self.ftp_user, self.ftp_password)
                print("Successfully connected to storage")

                # Get the current working directory - important for relative path operations
                current_root = ftp.pwd()
                print(f"Current FTP directory: {current_root}")

                # Function to recursively scan a directory and all its subdirectories
                def scan_directory_recursive(directory_name):
                    dir_files = []
                    print(f"Scanning directory: {directory_name}")

                    try:
                        # Try to access the directory
                        ftp.cwd(directory_name)

                        # Get current path after changing directory
                        current_path = ftp.pwd()
                        print(f"Entered directory: {current_path}")

                        # List all items
                        dir_items = []
                        ftp.retrlines('LIST', dir_items.append)

                        # Process items
                        for item in dir_items:
                            # Parse the item
                            parts = item.split()
                            if len(parts) < 9:
                                continue

                            # The filename is everything after the 8th field
                            name = ' '.join(parts[8:])
                            if name in ['.', '..']:
                                continue

                            is_dir = item.startswith('d')

                            if is_dir:
                                # It's a subdirectory - recursively scan it
                                try:
                                    # Remember where we are
                                    before_subdir = ftp.pwd()

                                    # Enter and scan subdirectory
                                    sub_files = scan_directory_recursive(name)
                                    dir_files.extend(sub_files)

                                    # Return to the parent directory
                                    ftp.cwd(before_subdir)
                                except Exception as e:
                                    print(f"Error scanning subdirectory {name}: {e}")
                                    # Try to recover position
                                    try:
                                        ftp.cwd(current_path)
                                    except:
                                        try:
                                            ftp.cwd(current_root)
                                        except:
                                            pass
                            elif name.lower().endswith(('.mp4', '.mkv')):
                                # It's a video file
                                file_path = f"{current_path}/{name}"
                                dir_files.append(file_path)
                                print(f"Found video file: {file_path}")

                    except Exception as e:
                        print(f"Error accessing directory {directory_name}: {e}")
                        # Try to recover by going back to root
                        try:
                            ftp.cwd(current_root)
                        except:
                            pass
                        
                    return dir_files

                # Get list of items at the current directory level
                root_items = []
                ftp.retrlines('LIST', root_items.append)

                # Parse the items to extract directories
                directories = []
                for item in root_items:
                    parts = item.split()
                    if len(parts) < 9:
                        continue
                    name = ' '.join(parts[8:])
                    if item.startswith('d') and name not in ['.', '..']:
                        directories.append(name)

                print(f"Found {len(directories)} directories: {', '.join(directories)}")

                # If a specific directory is provided
                if self.ftp_directory and self.ftp_directory != '/':
                    # Remove leading slash if present
                    target_dir = self.ftp_directory
                    if target_dir.startswith('/'):
                        target_dir = target_dir[1:]

                    # Check if the target directory exists
                    if target_dir in directories:
                        print(f"Scanning specified directory: {target_dir}")
                        found_files = scan_directory_recursive(target_dir)
                        files.extend(found_files)
                    else:
                        print(f"Specified directory '{target_dir}' not found in the current location")
                        # Check if it might be a subdirectory
                        for directory in directories:
                            try:
                                ftp.cwd(directory)
                                sub_items = []
                                ftp.retrlines('LIST', sub_items.append)
                                ftp.cwd(current_root)  # Go back

                                for sub_item in sub_items:
                                    parts = sub_item.split()
                                    if len(parts) < 9:
                                        continue
                                    sub_name = ' '.join(parts[8:])
                                    if sub_item.startswith('d') and sub_name == target_dir:
                                        print(f"Found target directory as subdirectory of {directory}")
                                        ftp.cwd(directory)
                                        found_files = scan_directory_recursive(target_dir)
                                        files.extend(found_files)
                                        ftp.cwd(current_root)  # Go back
                                        break
                            except:
                                ftp.cwd(current_root)  # Ensure we get back to root
                else:
                    # No specific directory, scan all directories
                    for directory in directories:
                        print(f"Scanning directory: {directory}")
                        found_files = scan_directory_recursive(directory)
                        files.extend(found_files)
                        # Make sure we're back at the root
                        ftp.cwd(current_root)

                print(f"Total video files found: {len(files)}")

        except Exception as e:
            print(f"Error scanning storage: {e}")
            import traceback
            traceback.print_exc()

        return files

    def run_ffprobe_command(self, command):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return json.loads(result.stdout)

    def get_streams(self, file, stream_type):
        command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", stream_type, file]
        return self.run_ffprobe_command(command)['streams']

    def get_audio_tracks(self, video_file):
        return [stream for stream in self.get_streams(video_file, "a") if stream['codec_type'] == 'audio']

    def get_subtitle_tracks(self, video_file):
        return self.get_streams(video_file, "s")
        command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    def get_video_duration(self, file_path):
        """Get video duration in seconds"""
        command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
        numbers = re.findall(r'\d+', filename)
        return [int(number) for number in numbers] if numbers else [0]

    def extract_number(self, filename):
        """Extract numbers from filename for natural sorting"""
        numbers = re.findall(r'\d+', filename)
        return [int(number) for number in numbers] if numbers else [0]
        command = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(duration),
            "-c:v", "copy",
            "-an",  # No audio
            output_file
        ]
        subprocess.run(command, check=True)
        return output_file

    def extract_yuv_segment(self, input_file, output_yuv, width=None, height=None):
        """Extract YUV file for VMAF comparison"""
        # Get video dimensions if not provided
        if width is None or height is None:
            video_info = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
                "stream=width,height", "-of", "json", input_file],
                stdout=subprocess.PIPE, text=True, check=True
            )
            info = json.loads(video_info.stdout)
            width = info["streams"][0]["width"]
            height = info["streams"][0]["height"]
        
        # Ensure dimensions are even
        width = width - (width % 2)
        height = height - (height % 2)
        
        # Extract raw YUV 420p with 8-bit depth
        command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-c:v", "rawvideo",
            "-pix_fmt", "yuv420p",  # 8-bit 4:2:0 format
            "-s", f"{width}x{height}",
            "-f", "rawvideo",
            output_yuv
        ]
        
        subprocess.run(command, check=True)
        return output_yuv, width, height, 8  # Return bitdepth=8

    def run_vmaf_exe(self, ref_yuv, dist_yuv, width, height, bitdepth=8):
        """Run VMAF using the standalone vmaf.exe executable"""
        temp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        
        vmaf_command = [
            "./vmaf",
            "--reference", ref_yuv,
            "--distorted", dist_yuv,
            "--width", str(width),
            "--height", str(height),
            "--pixel_format", "420",
            "--bitdepth", str(bitdepth),
            "--output", temp_json,
            "--json",
            "--model", "version=vmaf_v0.6.1"
        ]
        
        try:
            subprocess.run(vmaf_command, check=True)
            with open(temp_json, 'r') as f:
                results = json.load(f)
            os.remove(temp_json)
            
            # Extract the mean VMAF score
            return results["pooled_metrics"]["vmaf"]["mean"]
        except Exception as e:
            print(f"Error running VMAF: {e}")
            if os.path.exists(temp_json):
                os.remove(temp_json)
            return None

    def get_cpu_threads(self):
        """Get optimal number of threads for encoding"""
        cpu_count = multiprocessing.cpu_count()
        # Using 75% of available threads for encoding is a good balance
        return max(1, int(cpu_count * 0.75))

    def get_video_bitrate(self, file_path):
        """Get video bitrate in kbps"""
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                "-show_entries", "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        try:
            return int(result.stdout.strip()) // 1000  # Convert to kbps
        except ValueError:
            # If bitrate cannot be determined, estimate based on file size and duration
            filesize = os.path.getsize(file_path)  # in bytes
            duration = self.get_video_duration(file_path)  # in seconds
            total_bitrate = (filesize * 8) / duration / 1000  # in kbps
            # Assume video is ~85% of total bitrate
            return int(total_bitrate * 0.85)

    def extract_subtitle_to_vtt(self, input_file, output_file, subtitle_track):
        """Extract subtitle from video file to VTT format"""
        ffmpeg_command = [
            "ffmpeg",
            "-i", input_file,
            "-map", f"0:s:{subtitle_track}",
            "-f", "webvtt",
            output_file
        ]
        
        subprocess.run(ffmpeg_command, check=True)
        print(f"Extracted subtitle to: {output_file}")

    def find_optimal_settings(self, input_file, temp_dir, target_vmaf_diff=8):
        """Find optimal encoding settings using short samples with precise target matching"""
        print(f"\nAnalyzing optimal settings for {os.path.basename(input_file)}...")
        
        # Always use AV1 with NVENC
        print("Using NVIDIA AV1 hardware encoding (NVENC)")
        is_using_av1 = True
        
        # Get video duration
        duration = self.get_video_duration(input_file)
        
        # Create sample points - beginning, middle, and end of the video
        sample_duration = 5  # 5 seconds per sample
        sample_points = [
            max(30, duration * 0.1),   # 10% in or at least 30s in
            duration * 0.5,             # Middle
            min(duration - 30, duration * 0.9)  # 90% in or at least 30s from end
        ]
        
        samples = []
        for i, start_time in enumerate(sample_points):
            if start_time + sample_duration > duration:
                continue
                
            # Extract sample clip
            sample_file = os.path.join(temp_dir, f"sample_{i}.mp4")
            self.extract_video_sample(input_file, sample_file, start_time, sample_duration)
            samples.append(sample_file)
        
        # If no valid samples were created, use just one from the beginning
        if not samples and duration > 10:
            sample_file = os.path.join(temp_dir, "sample_0.mp4")
            self.extract_video_sample(input_file, sample_file, 30, sample_duration)
            samples.append(sample_file)
        
        # Get the original video bitrate to calibrate our test range
        original_bitrate = self.get_video_bitrate(input_file)
        print(f"Original video bitrate: {original_bitrate} kbps")
        
        # Test a range of values on the samples to find optimal value
        print(f"Testing encoding settings on {len(samples)} video samples...")
        
        # Target VMAF score based on the difference from original
        target_vmaf = 100 - target_vmaf_diff  
        print(f"Target VMAF: {target_vmaf} (difference: {target_vmaf_diff} ± 2)")
        
        # First pass: Test a range of bitrates based on original bitrate
        # Start with a fraction of original bitrate and go higher
        bitrate_factors = [0.15, 0.25, 0.35, 0.45, 0.55]
        bitrate_values = [int(original_bitrate * factor) for factor in bitrate_factors]
        param_name = "Bitrate (kbps)"
        
        first_pass_values = bitrate_values
        
        quality_results = {}  # Store results for all tested values
        best_bitrate = None
        best_avg_vmaf = 0
        best_vmaf_diff = float('inf')
        
        # Extract YUV from the reference samples once
        ref_yuv_files = []
        video_dims = None
        bitdepth = 8
        
        for i, sample in enumerate(samples):
            yuv_file = os.path.join(temp_dir, f"ref_sample_{i}.yuv")
            ref_yuv, width, height, bitdepth = self.extract_yuv_segment(sample, yuv_file)
            ref_yuv_files.append(ref_yuv)
            if video_dims is None:
                video_dims = (width, height)
        
        # First pass
        print("\nFirst pass: Finding approximate optimal range...")
        for bitrate in first_pass_values:
            print(f"\nTesting {param_name} value: {bitrate}")
            
            vmaf_scores = []
            for i, sample in enumerate(samples):
                # AV1 NVENC encoding with VBR mode
                encoded_sample = os.path.join(temp_dir, f"encoded_sample_{i}_br{bitrate}.mp4")
                encode_command = [
                    "ffmpeg", "-y",
                    "-hwaccel", "cuda",
                    "-i", sample,
                    "-c:v", "av1_nvenc",   # NVIDIA AV1 encoder
                    "-rc", "vbr",          # Variable bitrate mode
                    "-b:v", f"{bitrate}k",  # Target bitrate in kbps
                    "-maxrate:v", f"{int(bitrate * 1.5)}k",  # 1.5x max bitrate for VBR flexibility
                    "-bufsize:v", f"{bitrate * 2}k",  # 2x buffer size for smoother bitrate transitions
                    encoded_sample
                ]
                
                try:
                    subprocess.run(encode_command, check=True)
                    
                    # Extract YUV from encoded sample
                    dist_yuv = os.path.join(temp_dir, f"dist_sample_{i}_br{bitrate}.yuv")
                    _, _, _, _ = self.extract_yuv_segment(encoded_sample, dist_yuv, video_dims[0], video_dims[1])
                    
                    # Compare using VMAF
                    vmaf_score = self.run_vmaf_exe(ref_yuv_files[i], dist_yuv, video_dims[0], video_dims[1], bitdepth)
                    if vmaf_score is not None:
                        print(f"  Sample {i+1}: VMAF score = {vmaf_score:.2f}")
                        vmaf_scores.append(vmaf_score)
                    
                    # Clean up
                    os.remove(dist_yuv)
                    os.remove(encoded_sample)
                except Exception as e:
                    print(f"Error testing {param_name} {bitrate} on sample {i}: {e}")
                    continue
            
            # Calculate average VMAF score
            if vmaf_scores:
                avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                vmaf_diff = 100 - avg_vmaf  # Difference from original
                
                print(f"  Average VMAF score for {param_name} {bitrate}: {avg_vmaf:.2f}")
                print(f"  VMAF difference from original: {vmaf_diff:.2f}")
                
                quality_results[bitrate] = (avg_vmaf, vmaf_diff)
                
                # Check if this is closest to our target
                if abs(vmaf_diff - target_vmaf_diff) < abs(best_vmaf_diff - target_vmaf_diff):
                    best_bitrate = bitrate
                    best_avg_vmaf = avg_vmaf
                    best_vmaf_diff = vmaf_diff
        
        # Check if we need a second pass for more precision
        # If the best result is already within tolerance (±2), no need for a second pass
        if abs(best_vmaf_diff - target_vmaf_diff) <= 2:
            print(f"\nFound result within tolerance: {param_name} {best_bitrate} with VMAF difference {best_vmaf_diff:.2f}")
        else:
            print("\nSecond pass: Fine-tuning for precise target...")
            # Find the two closest bitrate values that bracket our target
            sorted_results = sorted([(q, diff) for q, (_, diff) in quality_results.items()], 
                                key=lambda x: x[1])
            
            # Find bracketing values (one below target, one above)
            lower_bitrate = None
            upper_bitrate = None
            
            for bitrate, diff in sorted_results:
                if diff <= target_vmaf_diff:
                    lower_bitrate = bitrate
                else:
                    upper_bitrate = bitrate
                    break
            
            # If we couldn't find proper bracketing values, use the two closest
            if lower_bitrate is None or upper_bitrate is None:
                if target_vmaf_diff < sorted_results[0][1]:  # Target below all tested
                    # Try higher bitrates
                    second_pass_values = [sorted_results[0][0] * 1.2, sorted_results[0][0] * 1.4]
                elif target_vmaf_diff > sorted_results[-1][1]:  # Target above all tested
                    # Try lower bitrates
                    second_pass_values = [sorted_results[-1][0] * 0.8, sorted_results[-1][0] * 0.6]
                else:
                    # Use the two closest values we have
                    closest = sorted(sorted_results, key=lambda x: abs(x[1] - target_vmaf_diff))
                    second_pass_values = [closest[0][0], closest[1][0]]
            else:
                # Found bracketing values, try values in between
                # Use interpolation to estimate a value that might hit closer to target
                weight = (target_vmaf_diff - quality_results[lower_bitrate][1]) / (quality_results[upper_bitrate][1] - quality_results[lower_bitrate][1])
                interpolated_bitrate = lower_bitrate + weight * (upper_bitrate - lower_bitrate)
                
                # Always test at least 2 new values
                second_pass_values = [int(interpolated_bitrate)]
                
                # Add one more value on the other side of the interpolated value
                if abs(interpolated_bitrate - lower_bitrate) < abs(interpolated_bitrate - upper_bitrate):
                    second_pass_values.append(int(interpolated_bitrate + (upper_bitrate - interpolated_bitrate)/2))
                else:
                    second_pass_values.append(int(interpolated_bitrate - (interpolated_bitrate - lower_bitrate)/2))
            
            # Make sure we don't re-test values from the first pass
            second_pass_values = [q for q in second_pass_values if q not in first_pass_values]
            
            # Second pass with the new values
            if second_pass_values:
                for bitrate in second_pass_values:
                    print(f"\nFine-tuning: Testing {param_name} value: {bitrate}")
                    
                    vmaf_scores = []
                    for i, sample in enumerate(samples):
                        # Encode the sample with current bitrate
                        encoded_sample = os.path.join(temp_dir, f"encoded_sample_{i}_br{bitrate}.mp4")
                        encode_command = [
                            "ffmpeg", "-y",
                            "-hwaccel", "cuda",
                            "-i", sample,
                            "-c:v", "av1_nvenc",   # NVIDIA AV1 encoder
                            "-rc", "vbr",          # Variable bitrate mode
                            "-b:v", f"{bitrate}k", # Target bitrate in kbps
                            "-maxrate:v", f"{int(bitrate * 1.5)}k",  # 1.5x max bitrate for VBR flexibility
                            "-bufsize:v", f"{bitrate * 2}k",  # 2x buffer size
                            encoded_sample
                        ]
                        
                        try:
                            subprocess.run(encode_command, check=True)
                            
                            dist_yuv = os.path.join(temp_dir, f"dist_sample_{i}_br{bitrate}.yuv")
                            _, _, _, _ = self.extract_yuv_segment(encoded_sample, dist_yuv, video_dims[0], video_dims[1])
                            
                            vmaf_score = self.run_vmaf_exe(ref_yuv_files[i], dist_yuv, video_dims[0], video_dims[1], bitdepth)
                            if vmaf_score is not None:
                                print(f"  Sample {i+1}: VMAF score = {vmaf_score:.2f}")
                                vmaf_scores.append(vmaf_score)
                            
                            os.remove(dist_yuv)
                            os.remove(encoded_sample)
                        except Exception as e:
                            print(f"Error testing {param_name} {bitrate} on sample {i}: {e}")
                            continue
                    
                    if vmaf_scores:
                        avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                        vmaf_diff = 100 - avg_vmaf
                        
                        print(f"  Average VMAF score for {param_name} {bitrate}: {avg_vmaf:.2f}")
                        print(f"  VMAF difference from original: {vmaf_diff:.2f}")
                        
                        # Check if this is closer to our target
                        if abs(vmaf_diff - target_vmaf_diff) < abs(best_vmaf_diff - target_vmaf_diff):
                            best_bitrate = bitrate
                            best_avg_vmaf = avg_vmaf
                            best_vmaf_diff = vmaf_diff
        
        # Clean up reference YUV files and samples
        for yuv_file in ref_yuv_files:
            os.remove(yuv_file)
        for sample in samples:
            os.remove(sample)
        
        print(f"\nBest encoding setting found: {param_name} {best_bitrate} with average VMAF {best_avg_vmaf:.2f}")
        print(f"VMAF difference from original: {best_vmaf_diff:.2f}")
        print(f"Target difference was: {target_vmaf_diff} (tolerance: ±2)")
        
        # Check if we hit within tolerance
        if abs(best_vmaf_diff - target_vmaf_diff) <= 2:
            print(f"✓ Success! Found setting within tolerance range")
        else:
            print(f"! Note: Could not find setting exactly within ±2 tolerance")
            print(f"  Closest setting is {abs(best_vmaf_diff - target_vmaf_diff):.2f} away from target")
        
        # If no good setting found, use a safe default
        if best_bitrate is None:
            best_bitrate = int(original_bitrate * 0.35)  # Safe default for AV1
            print(f"Could not determine optimal settings, using default {param_name} {best_bitrate}")
        
        return best_bitrate, best_avg_vmaf, True  # Always return True for is_using_av1

    def encode_video(self, input_file, output_file, bitrate, audio_track, include_subtitles, subtitle_track):
        """Encode a video file using the optimal settings"""
        # Get the audio track info
        audio_command = None
        audio_tracks = self.get_audio_tracks(input_file)
        
        if len(audio_tracks) > audio_track:
            original_channels = audio_tracks[audio_track].get('channels', 2)
            
            audio_command = [
                "-map", "0:v:0",       # First video stream 
                "-map", f"0:a:{audio_track}"  # Selected audio stream
            ]
            
            # Check if we need to downmix audio
            if original_channels > 2:
                print(f"Original audio: {original_channels} channels - Converting to stereo (2 channels)")
                audio_command.extend([
                    "-c:a", "aac",      # Convert audio to AAC
                    "-ac", "2",         # Limit to 2 audio channels (stereo)
                    "-b:a", "192k"      # Use a good quality bitrate for audio
                ])
            else:
                print(f"Original audio already has {original_channels} channels - Copying without conversion")
                audio_command.extend(["-c:a", "copy"])  # Just copy the audio
        else:
            print("Warning: Selected audio track not found, using default settings")
            audio_command = [
                "-map", "0:v:0",  # First video stream
                "-map", "0:a",    # All audio
                "-c:a", "copy"    # Copy audio
            ]
        
        # Handle subtitles
        subtitle_command = []
        if include_subtitles and subtitle_track is not None:
            # Check if the subtitle track exists
            subtitle_tracks = self.get_subtitle_tracks(input_file)
            if len(subtitle_tracks) > subtitle_track:
                # Add subtitle to MP4 as well if format is compatible
                sub_codec = subtitle_tracks[subtitle_track].get('codec_name', '')
                if sub_codec.lower() in ['mov_text', 'text', 'tx3g']:
                    subtitle_command = ["-map", f"0:s:{subtitle_track}", "-c:s", "mov_text"]
                    print(f"Including subtitle track in MP4")
                else:
                    print(f"Subtitle format {sub_codec} not directly compatible with MP4, extracting to VTT only")
            else:
                print(f"Warning: Subtitle track {subtitle_track} not found in {os.path.basename(input_file)}")
        
        if subtitle_command:
            audio_command.extend(subtitle_command)
        
        # Build the final encoding command for AV1 with NVENC
        print(f"\nEncoding with NVIDIA AV1 hardware encoder (Bitrate {bitrate}k)")
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", input_file,
            "-c:v", "av1_nvenc",       # NVIDIA AV1 encoder
            "-rc", "vbr",              # Variable bitrate mode
            "-b:v", f"{bitrate}k",     # Target bitrate in kbps
            "-maxrate:v", f"{int(bitrate * 1.5)}k",  # 1.5x max bitrate for VBR flexibility
            "-bufsize:v", f"{bitrate * 2}k"  # 2x buffer size for smoother bitrate transitions
        ]
        
        if audio_command:
            ffmpeg_command.extend(audio_command)
        
        ffmpeg_command.append(output_file)
        
        print(f"Command: {' '.join(ffmpeg_command)}")
        
        start_time = time.time()
        subprocess.run(ffmpeg_command, check=True)
        elapsed_time = time.time() - start_time
        
        print(f"Completed encoding in {elapsed_time:.1f} seconds")
        print(f"Output file: {output_file}")
        
        # Extract subtitle to VTT if requested
        if include_subtitles and subtitle_track is not None:
            subtitle_tracks = self.get_subtitle_tracks(input_file)
            if len(subtitle_tracks) > subtitle_track:
                vtt_filename = os.path.splitext(os.path.basename(output_file))[0] + ".vtt"
                vtt_file = os.path.join(os.path.dirname(output_file), vtt_filename)
                try:
                    self.extract_subtitle_to_vtt(input_file, vtt_file, subtitle_track)
                except Exception as e:
                    print(f"Error extracting subtitle: {e}")
        
        return output_file

    def download_from_ftp(self, remote_path):
        """Download a file from FTP server"""
        try:
            # Create the local path
            local_filename = os.path.basename(remote_path)
            local_path = os.path.join(self.download_dir, local_filename)
            
            # Check if file already exists locally
            if os.path.exists(local_path) and self.skip_existing:
                print(f"File {local_filename} already exists locally. Skipping download.")
                return local_path
            
            print(f"Downloading {remote_path} to {local_path}...")
            
            # Connect to FTP server
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(self.ftp_user, self.ftp_password)
                
                # Create local directory if it doesn't exist
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download the file
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {remote_path}', f.write)
                
                print(f"Download complete: {local_path}")
                return local_path
                
        except Exception as e:
            print(f"Error downloading {remote_path}: {e}")
            return None

    def list_ftp_files(self, directory=None):
        """List all video files in the FTP directory and subdirectories with proper path handling"""
        if directory is None:
            directory = self.ftp_directory

        files = []

        try:
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(self.ftp_user, self.ftp_password)

                # Get the current working directory
                current_root = ftp.pwd()
                print(f"Current FTP directory: {current_root}")

                # Normalize directory path - remove leading slash if working from a non-root directory
                if directory.startswith('/') and current_root != '/':
                    # If we're not at root level, adjust the path
                    dir_to_use = directory[1:] if directory != '/' else ''
                else:
                    dir_to_use = directory

                if dir_to_use and dir_to_use != '/':
                    try:
                        ftp.cwd(dir_to_use)
                        print(f"Changed to directory: {dir_to_use}")
                    except Exception as e:
                        print(f"Error accessing directory {dir_to_use}: {e}")
                        return files

                # Function to recursively process directories
                def process_directory(current_dir):
                    current_path = ftp.pwd()
                    print(f"Scanning directory: {current_path}")

                    try:
                        # Get raw directory listing
                        dir_items = []
                        ftp.retrlines('LIST', dir_items.append)

                        file_list = []
                        dir_list = []

                        # Parse directory items
                        for item in dir_items:
                            parts = item.split()
                            if len(parts) < 9:  # Not enough parts for a valid listing
                                continue

                            # The filename is everything after the 8th field (which has the time)
                            filename = ' '.join(parts[8:])
                            is_dir = item.startswith('d')

                            if is_dir:
                                # Skip parent and current directory refs
                                if filename not in ['.', '..']:
                                    dir_list.append(filename)
                            elif filename.lower().endswith(('.mp4', '.mkv')):
                                file_list.append(filename)

                        # Add files from current directory
                        for file in file_list:
                            file_path = f"{current_path}/{file}"
                            files.append(file_path)
                            print(f"Found video file: {file_path}")

                        # Recursively process subdirectories
                        for subdir in dir_list:
                            try:
                                # Save current position
                                previous_dir = ftp.pwd()

                                # Navigate to subdirectory
                                ftp.cwd(subdir)

                                # Process the subdirectory
                                process_directory(f"{current_dir}/{subdir}")

                                # Go back to previous directory
                                ftp.cwd(previous_dir)
                            except Exception as e:
                                print(f"Error processing subdirectory {subdir}: {e}")
                                # Try to recover by returning to known position
                                try:
                                    ftp.cwd(previous_dir)
                                except:
                                    try:
                                        ftp.cwd(current_root)
                                    except:
                                        pass
                                    
                    except Exception as e:
                        print(f"Error scanning directory {current_path}: {e}")
                        # Try to recover
                        try:
                            ftp.cwd(current_root)
                        except:
                            pass
                        
                # Start processing from the current directory
                start_dir = ftp.pwd()
                process_directory(start_dir)

                print(f"Total video files found: {len(files)}")

            return files

        except Exception as e:
            print(f"Error listing FTP files: {e}")
            import traceback
            traceback.print_exc()

        return files

    def get_output_path(self, input_file):
        """Generate output path for a processed file"""
        # Extract filename and remove extension
        basename = os.path.splitext(os.path.basename(input_file))[0]
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # Create output path with MP4 extension
        return os.path.join(self.output_dir, f"{basename}.mp4")

    def upload_to_ftp(self, local_file, remote_path):
        """Upload a file to FTP server"""
        try:
            print(f"Uploading {os.path.basename(local_file)} to {remote_path}...")
            
            # Connect to FTP server
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(self.ftp_user, self.ftp_password)
                
                # Upload the file
                with open(local_file, 'rb') as f:
                    ftp.storbinary(f'STOR {remote_path}', f)
                
                print(f"Upload complete: {remote_path}")
                return True
                
        except Exception as e:
            print(f"Error uploading {local_file} to {remote_path}: {e}")
            return False
            
    def process_batch(self):
        """Main method to process all videos from the FTP server"""
        try:
            print(f"Starting FTP Video Compressor with VMAF diff target: {self.target_vmaf_diff}")
            print(f"Using parallel processing with {self.max_concurrent} concurrent jobs")
            
            is_bunnycdn = "bunnycdn" in self.ftp_host.lower()
            if is_bunnycdn:
                print("Detected BunnyCDN storage service")
                remote_files = self.scan_storage()
            else:
                # Use standard FTP scanning
                remote_files = self.list_ftp_files()
                
            total_files = len(remote_files)
            
            if total_files == 0:
                print("No video files found on the FTP server.")
                return
                
            print(f"Found {total_files} video files to process.")
            
            # Process files with parallel execution
            processed_count = 0
            failed_count = 0
            
            # Group files by folder for efficient processing
            files_by_folder = {}
            for file in remote_files:
                folder = self.get_folder_from_path(file)
                if folder not in files_by_folder:
                    files_by_folder[folder] = []
                files_by_folder[folder].append(file)
            
            print(f"Files grouped into {len(files_by_folder)} folders")
            
            # Initialize stats tracking
            start_time = time.time()
            
            # Process each folder's files
            for folder, folder_files in files_by_folder.items():
                print(f"\nProcessing folder: {folder} ({len(folder_files)} files)")
                
                # Process files in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    futures = {executor.submit(self.process_video, file): file for file in folder_files}
                    
                    for future in concurrent.futures.as_completed(futures):
                        file = futures[future]
                        try:
                            success = future.result()
                            if success:
                                processed_count += 1
                                print(f"Progress: {processed_count}/{total_files} files processed")
                            else:
                                failed_count += 1
                                print(f"Failed to process: {file}")
                        except Exception as e:
                            failed_count += 1
                            print(f"Error processing {file}: {e}")
                
                # Display storage savings after each folder
                if self.processed_files_count > 0:
                    total_saved = self.original_size_total - self.encoded_size_total
                    total_saved_gb = total_saved / (1024 ** 3)
                    reduction_percent = (total_saved / self.original_size_total) * 100 if self.original_size_total > 0 else 0
                    print(f"\n--- Storage savings after folder '{folder}' ---")
                    print(f"Original size: {self.format_size(self.original_size_total)}")
                    print(f"Encoded size:  {self.format_size(self.encoded_size_total)}")
                    print(f"Space saved:   {self.format_size(total_saved)} ({reduction_percent:.1f}%)")
                    print(f"               {total_saved_gb:.2f} GB")
            
            # Calculate execution time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Final summary
            print("\n" + "="*60)
            print(f"Processing complete!")
            print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"Total files: {total_files}")
            print(f"Successfully processed: {processed_count}")
            print(f"Failed: {failed_count}")
            
            # Display final storage statistics
            if self.processed_files_count > 0:
                total_saved = self.original_size_total - self.encoded_size_total
                total_saved_gb = total_saved / (1024 ** 3)
                reduction_percent = (total_saved / self.original_size_total) * 100 if self.original_size_total > 0 else 0
                
                print("\n--- FINAL STORAGE SAVINGS ---")
                print(f"Original total size: {self.format_size(self.original_size_total)}")
                print(f"Encoded total size:  {self.format_size(self.encoded_size_total)}")
                print(f"Total space saved:   {self.format_size(total_saved)} ({reduction_percent:.1f}%)")
                print(f"                     {total_saved_gb:.2f} GB")
                
                # Calculate average file size reduction
                if processed_count > 0:
                    avg_original = self.original_size_total / processed_count
                    avg_encoded = self.encoded_size_total / processed_count
                    avg_savings = (avg_original - avg_encoded) / avg_original * 100 if avg_original > 0 else 0
                    print(f"\nAverage file size reduction: {avg_savings:.1f}%")
                    print(f"Average original size: {self.format_size(avg_original)}")
                    print(f"Average encoded size:  {self.format_size(avg_encoded)}")
            
            print("="*60)
            
            # Clean up temporary directory
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise

    def extract_video_sample(self, input_file, output_file, start_time, duration):
        """Extract a short sample from a video file for analysis"""
        command = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(duration),
            "-c:v", "copy",
            "-an",  # No audio
            output_file
        ]
        subprocess.run(command, check=True)
        return output_file

    def get_folder_from_path(self, file_path):
        """Extract folder path from full file path"""
        dirname = os.path.dirname(file_path)
        if not dirname:
            return "/"
        return dirname

    def process_video(self, remote_file):
        """Process a single video file"""
        try:
            # Extract folder information
            folder = self.get_folder_from_path(remote_file)

            # Generate the output path - temporary local path for encoded file
            filename = os.path.basename(remote_file)
            temp_output_file = os.path.join(self.output_dir, f"temp_{filename}.mp4")

            # Skip if output file already exists and skip_existing is True
            final_output = os.path.join(self.output_dir, os.path.basename(filename).replace("temp_", ""))
            if os.path.exists(final_output) and self.skip_existing:
                print(f"Output file {final_output} already exists. Skipping.")
                return True

            # Download the file from FTP
            local_file = self.download_from_ftp(remote_file)
            if not local_file or not os.path.exists(local_file):
                print(f"Failed to download {remote_file}. Skipping processing.")
                return False

            # Get original file size
            original_size = os.path.getsize(local_file)

            # Check if we already have encoding settings for this folder
            if folder in self.folder_settings_cache:
                print(f"Using cached encoding settings for folder: {folder}")
                bitrate, avg_vmaf, is_using_av1 = self.folder_settings_cache[folder]
            else:
                # Find optimal encoding settings for this file (representative of the folder)
                print(f"Finding optimal encoding settings for folder: {folder}")
                bitrate, avg_vmaf, is_using_av1 = self.find_optimal_settings(local_file, self.temp_dir, self.target_vmaf_diff)
                # Cache the settings for this folder
                self.folder_settings_cache[folder] = (bitrate, avg_vmaf, is_using_av1)

            # Encode the video using the optimal settings
            encoded_file = self.encode_video(
                local_file, 
                temp_output_file, 
                bitrate, 
                self.audio_track, 
                self.include_subtitles, 
                self.subtitle_track
            )

            # If encoding was successful
            if os.path.exists(encoded_file):
                # Get encoded file size
                encoded_size = os.path.getsize(encoded_file)

                # Track the sizes
                self.original_size_total += original_size
                self.encoded_size_total += encoded_size
                self.processed_files_count += 1

                # Calculate size difference
                size_diff = original_size - encoded_size
                size_reduction_percent = (size_diff / original_size) * 100 if original_size > 0 else 0

                # Display size information
                print(f"\nFile size comparison:")
                print(f"  Original: {self.format_size(original_size)}")
                print(f"  Encoded:  {self.format_size(encoded_size)}")
                print(f"  Savings:  {self.format_size(size_diff)} ({size_reduction_percent:.1f}%)")

                # Running total
                total_saved = self.original_size_total - self.encoded_size_total
                total_saved_gb = total_saved / (1024 ** 3)
                print(f"\nRunning total - Space saved: {total_saved_gb:.2f} GB after {self.processed_files_count} files")

                # If we need to replace original
                if self.replace_originals:
                    # Upload encoded file to FTP (replacing the original)
                    upload_success = self.upload_to_ftp(encoded_file, remote_file)

                    # If upload successful, cleanup
                    if upload_success:
                        # Delete the temporary files
                        os.remove(local_file)
                        os.remove(encoded_file)
                        print(f"Successfully replaced {remote_file} with compressed version")
                        return True
                    else:
                        print(f"Failed to replace {remote_file} on FTP server")
                        return False
                # If we're not replacing originals, just keep the encoded file locally
                else:
                    # Move to final output location
                    final_output = os.path.join(self.output_dir, os.path.basename(encoded_file).replace("temp_", ""))
                    os.rename(encoded_file, final_output)
                    # Delete downloaded file
                    os.remove(local_file)
                    print(f"Encoded file saved locally: {final_output}")
                    return True
            else:
                print(f"Encoding failed for {remote_file}")
                return False

        except Exception as e:
            print(f"Error processing {remote_file}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_from_ftp(self, remote_path):
        """Delete a file from FTP server"""
        try:
            print(f"Deleting {remote_path} from FTP server...")
            
            # Connect to FTP server
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login(self.ftp_user, self.ftp_password)
                
                # Delete the file
                ftp.delete(remote_path)
                
                print(f"File deleted: {remote_path}")
                return True
                
        except Exception as e:
            print(f"Error deleting {remote_path} from FTP server: {e}")
            return False

    def format_size(self, size_bytes):
        """Format size in bytes to human-readable format"""
        if size_bytes < 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = 0
        while size_bytes >= 1024 and i < len(size_name) - 1:
            size_bytes /= 1024
            i += 1
        return f"{size_bytes:.2f} {size_name[i]}"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download and compress videos from FTP server using AV1 encoding')
    
    # FTP settings
    parser.add_argument('--host', required=True, help='FTP server hostname or IP address')
    parser.add_argument('--user', required=True, help='FTP username')
    parser.add_argument('--password', required=True, help='FTP password')
    parser.add_argument('--directory', default='/', help='FTP directory to scan for videos')
    parser.add_argument('--storage-zone', help='BunnyCDN storage zone name (if using BunnyCDN)')
    
    # Output settings
    parser.add_argument('--output', required=True, help='Local output directory for working files')
    parser.add_argument('--skip-existing', action='store_true', help='Skip processing if output file already exists')
    parser.add_argument('--replace-originals', action='store_true', default=True, 
                        help='Replace original files on FTP with compressed versions (default: True)')
    parser.add_argument('--keep-originals', action='store_false', dest='replace_originals',
                        help='Keep original files on FTP and save compressed versions locally')
    
    # Encoding settings
    parser.add_argument('--vmaf-diff', type=int, default=8, help='Target VMAF quality difference (default: 8)')
    parser.add_argument('--concurrent', type=int, default=2, help='Number of videos to process concurrently (default: 2)')
    parser.add_argument('--audio-track', type=int, default=0, help='Audio track to include (default: 0)')
    parser.add_argument('--include-subtitles', action='store_true', help='Include subtitles in output')
    parser.add_argument('--subtitle-track', type=int, default=0, help='Subtitle track to include (default: 0)')
    parser.add_argument('--scan-all', action='store_true', help='Scan all directories including those that fail initially (for BunnyCDN)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        print(f"Connecting to FTP server: {args.host}")
        print(f"Using directory: {args.directory}")
        print(f"Target VMAF difference: {args.vmaf_diff}")
        print(f"Concurrent processing: {args.concurrent}")
        print(f"Output directory: {args.output}")
        print(f"Replace originals: {args.replace_originals}")
        
        # Detect if we're using BunnyCDN
        is_bunnycdn = "bunnycdn" in args.host.lower()
        if is_bunnycdn:
            print("Detected BunnyCDN storage service")
            # For BunnyCDN, if storage zone is not specified, use the username as zone
            if not args.storage_zone:
                args.storage_zone = args.user
                print(f"Using username as storage zone: {args.storage_zone}")
        
        # Normalize FTP directory path
        if not args.directory.startswith('/'):
            args.directory = '/' + args.directory
            
        # Make sure output directory exists and is writable
        output_dir = args.output
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                alt_output = '/tmp/ftp_video_output'
                print(f"Warning: Output directory {output_dir} is not writable.")
                print(f"Using alternative output directory: {alt_output}")
                os.makedirs(alt_output, exist_ok=True)
                output_dir = alt_output
        except Exception as e:
            print(f"Warning: Could not access output directory {output_dir}: {e}")
            print(f"Using fallback directory: /tmp/ftp_video_output")
            os.makedirs('/tmp/ftp_video_output', exist_ok=True)
            output_dir = '/tmp/ftp_video_output'
        
        # Test FTP connection before proceeding
        try:
            with ftplib.FTP(args.host) as ftp:
                ftp.login(args.user, args.password)
                print("FTP connection successful!")
                
                # Try to list the root directory
                welcome = ftp.getwelcome()
                print(f"FTP Server Welcome: {welcome}")
                
                # Get current directory
                current_dir = ftp.pwd()
                print(f"Current FTP directory: {current_dir}")
                
                # List directories at root level to help diagnose structure
                print("Available directories at root level:")
                try:
                    root_items = []
                    ftp.retrlines('LIST', root_items.append)
                    for item in root_items:
                        print(f"  {item}")
                except Exception as e:
                    print(f"Could not list root directory: {e}")
        except Exception as e:
            print(f"Error testing FTP connection: {e}")
            return 1
        
        # Create compressor instance
        compressor = FtpVideoCompressor(
            ftp_host=args.host,
            ftp_user=args.user,
            ftp_password=args.password,
            ftp_directory=args.directory,
            output_dir=output_dir,
            target_vmaf_diff=args.vmaf_diff,
            max_concurrent=args.concurrent,
            audio_track=args.audio_track,
            include_subtitles=args.include_subtitles,
            subtitle_track=args.subtitle_track if args.include_subtitles else None,
            skip_existing=args.skip_existing,
            replace_originals=args.replace_originals,
            storage_zone=args.storage_zone,
            scan_all=args.scan_all if hasattr(args, 'scan_all') else False
        )
        
        # Process all videos
        compressor.process_batch()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
    try:
        print(f"Starting FTP Video Compressor with VMAF diff target: {self.target_vmaf_diff}")
        print(f"Using parallel processing with {self.max_concurrent} concurrent jobs")
        
        # List all video files from FTP
        remote_files = self.list_ftp_files()
        total_files = len(remote_files)
        
        if total_files == 0:
            print("No video files found on the FTP server.")
            
        print(f"Found {total_files} video files to process.")
        
        # Process files with parallel execution
        processed_count = 0
        failed_count = 0
        
        # Group files by folder for efficient processing
        files_by_folder = {}
        for file in remote_files:
            folder = self.get_folder_from_path(file)
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append(file)
        
        print(f"Files grouped into {len(files_by_folder)} folders")
        
        # Initialize stats tracking
        start_time = time.time()
        
        # Process each folder's files
        for folder, folder_files in files_by_folder.items():
            print(f"\nProcessing folder: {folder} ({len(folder_files)} files)")
            
            # Process files in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                futures = {executor.submit(self.process_video, file): file for file in folder_files}
                
                for future in concurrent.futures.as_completed(futures):
                    file = futures[future]
                    try:
                        success = future.result()
                        if success:
                            processed_count += 1
                            print(f"Progress: {processed_count}/{total_files} files processed")
                        else:
                            failed_count += 1
                            print(f"Failed to process: {file}")
                    except Exception as e:
                        failed_count += 1
                        print(f"Error processing {file}: {e}")
            
            # Display storage savings after each folder
            if self.processed_files_count > 0:
                total_saved = self.original_size_total - self.encoded_size_total
                total_saved_gb = total_saved / (1024 ** 3)
                reduction_percent = (total_saved / self.original_size_total) * 100 if self.original_size_total > 0 else 0
                print(f"\n--- Storage savings after folder '{folder}' ---")
                print(f"Original size: {self.format_size(self.original_size_total)}")
                print(f"Encoded size:  {self.format_size(self.encoded_size_total)}")
                print(f"Space saved:   {self.format_size(total_saved)} ({reduction_percent:.1f}%)")
                print(f"               {total_saved_gb:.2f} GB")
        
        # Calculate execution time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Final summary
        print("\n" + "="*60)
        print(f"Processing complete!")
        print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        
        # Display final storage statistics
        if self.processed_files_count > 0:
            total_saved = self.original_size_total - self.encoded_size_total
            total_saved_gb = total_saved / (1024 ** 3)
            reduction_percent = (total_saved / self.original_size_total) * 100 if self.original_size_total > 0 else 0
            
            print("\n--- FINAL STORAGE SAVINGS ---")
            print(f"Original total size: {self.format_size(self.original_size_total)}")
            print(f"Encoded total size:  {self.format_size(self.encoded_size_total)}")
            print(f"Total space saved:   {self.format_size(total_saved)} ({reduction_percent:.1f}%)")
            print(f"                     {total_saved_gb:.2f} GB")
            
            # Calculate average file size reduction
            if processed_count > 0:
                avg_original = self.original_size_total / processed_count
                avg_encoded = self.encoded_size_total / processed_count
                avg_savings = (avg_original - avg_encoded) / avg_original * 100 if avg_original > 0 else 0
                print(f"\nAverage file size reduction: {avg_savings:.1f}%")
                print(f"Average original size: {self.format_size(avg_original)}")
                print(f"Average encoded size:  {self.format_size(avg_encoded)}")
        
        print("="*60)
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise
    try:
        # Extract folder information
        folder = self.get_folder_from_path(remote_file)
        
        # Generate the output path - temporary local path for encoded file
        filename = os.path.basename(remote_file)
        temp_output_file = os.path.join(self.output_dir, f"temp_{filename}.mp4")
        
        # Download the file from FTP
        local_file = self.download_from_ftp(remote_file)
        if not local_file or not os.path.exists(local_file):
            print(f"Failed to download {remote_file}. Skipping processing.")
        
        # Get original file size
        original_size = os.path.getsize(local_file)
        
        # Check if we already have encoding settings for this folder
        if folder in self.folder_settings_cache:
            print(f"Using cached encoding settings for folder: {folder}")
            bitrate, avg_vmaf = self.folder_settings_cache[folder]
        else:
            # Find optimal encoding settings for this file (representative of the folder)
            print(f"Finding optimal encoding settings for folder: {folder}")
            bitrate, avg_vmaf, _ = self.find_optimal_settings(local_file, self.temp_dir, self.target_vmaf_diff)
            # Cache the settings for this folder
            self.folder_settings_cache[folder] = (bitrate, avg_vmaf)
        
        # Encode the video using the optimal settings
        encoded_file = self.encode_video(
            local_file, 
            temp_output_file, 
            bitrate, 
            self.audio_track, 
            self.include_subtitles, 
            self.subtitle_track
        )
        
        # If encoding was successful
        if os.path.exists(encoded_file):
            # Get encoded file size
            encoded_size = os.path.getsize(encoded_file)
            
            # Track the sizes
            self.original_size_total += original_size
            self.encoded_size_total += encoded_size
            self.processed_files_count += 1
            
            # Calculate size difference
            size_diff = original_size - encoded_size
            size_reduction_percent = (size_diff / original_size) * 100 if original_size > 0 else 0
            
            # Display size information
            print(f"\nFile size comparison:")
            print(f"  Original: {self.format_size(original_size)}")
            print(f"  Encoded:  {self.format_size(encoded_size)}")
            print(f"  Savings:  {self.format_size(size_diff)} ({size_reduction_percent:.1f}%)")
            
            # Running total
            total_saved = self.original_size_total - self.encoded_size_total
            total_saved_gb = total_saved / (1024 ** 3)
            print(f"\nRunning total - Space saved: {total_saved_gb:.2f} GB after {self.processed_files_count} files")
            
            # If we need to replace original
            if self.replace_originals:
                # Upload encoded file to FTP (replacing the original)
                upload_success = self.upload_to_ftp(encoded_file, remote_file)
                
                # If upload successful, cleanup
                if upload_success:
                    # Delete the temporary files
                    os.remove(local_file)
                    os.remove(encoded_file)
                    print(f"Successfully replaced {remote_file} with compressed version")
                else:
                    print(f"Failed to replace {remote_file} on FTP server")
            # If we're not replacing originals, just keep the encoded file locally
            else:
                # Move to final output location
                final_output = os.path.join(self.output_dir, os.path.basename(encoded_file).replace("temp_", ""))
                os.rename(encoded_file, final_output)
                # Delete downloaded file
                os.remove(local_file)
                print(f"Encoded file saved locally: {final_output}")
        else:
            print(f"Encoding failed for {remote_file}")
        
    except Exception as e:
        print(f"Error processing {remote_file}: {e}")