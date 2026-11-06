import os
import subprocess
import sys
from pathlib import Path

INPUT_FOLDER = "E:\\Павел\\НИРС\\action-marker\\data\\sources\\UCF Crimes\\Training-Normal-Videos-Part-2"
OUTPUT_FOLDER = "E:\\Павел\\НИРС\\action-marker\\data\\sources\\UCF Crimes\\Normal-Videos"
TARGET_SEGMENT_DURATION = 15
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}


def get_video_duration(video_path: str) -> float:
    """Return video duration."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def process_long_video(video_path: str, output_dir: str, target_duration: float = 15.0):
    """Cut video on the same clips."""
    duration = get_video_duration(video_path)

    if duration <= 0:
        print(f"  [!] Skip (Zero duration): {video_path}")
        return

    base_name = Path(video_path).stem

    if duration < target_duration:
        output_file = os.path.join(output_dir, f"{base_name}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                output_file,
            ],
            check=True,
        )
        print(f"    [+] {os.path.basename(output_file)}")
    elif duration < 35:
        start = (duration - target_duration) / 2

        output_file = os.path.join(output_dir, f"{base_name}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{target_duration:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                output_file,
            ],
            check=True,
        )
        print(f"    [+] {os.path.basename(output_file)}")
    elif duration < 60:
        start = (duration - 2 * target_duration) / 3
        start_positions = [start, 2 * start + target_duration]

        for i in range(2):
            output_file = os.path.join(output_dir, f"{base_name}_{i + 1:02d}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start_positions[i]:.3f}",
                    "-i", video_path,
                    "-t", f"{target_duration:.3f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    output_file,
                ],
                check=True,
            )
            print(f"    [+] {os.path.basename(output_file)}")
    else:
        start = (duration - 3 * target_duration) / 4
        start_positions = [start, 2 * start + target_duration, 3 * start + 2 * target_duration]

        for i in range(3):
            output_file = os.path.join(output_dir, f"{base_name}_{i + 1:02d}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start_positions[i]:.3f}",
                    "-i", video_path,
                    "-t", f"{target_duration:.3f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    output_file,
                ],
                check=True,
            )
            print(f"    [+] {os.path.basename(output_file)}")


def main():
    input_folder = Path(INPUT_FOLDER)

    if not input_folder.is_dir():
        print(f"Error: folder '{INPUT_FOLDER}' not found.")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    videos = [
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"Video files not found in folder '{INPUT_FOLDER}'.")
        return

    print(f"Found videos: {len(videos)}\n")

    for video in videos:
        print(f"Processing: {video.name}")
        try:
            process_long_video(str(video), OUTPUT_FOLDER, TARGET_SEGMENT_DURATION)
        except subprocess.CalledProcessError as e:
            print(f"  [!] Error ffmpeg: {e}")
        except Exception as e:
            print(f"  [!] Error: {e}")
        print()

    print("Processed!")


if __name__ == "__main__":
    main()