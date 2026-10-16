import os
import subprocess
import sys
from pathlib import Path

INPUT_FOLDER = "D:\\Video\\video\\working with electric panel\\split"
OUTPUT_FOLDER = "D:\\Video\\video\\working with electric panel\\processed"
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


def cut_video(video_path: str, output_dir: str, target_duration: float = 15.0):
    """Cut video on the same clips."""
    duration = get_video_duration(video_path)

    if duration <= 0:
        print(f"  [!] Skip (Zero duration): {video_path}")
        return

    num_segments = max(1, round(duration / target_duration))
    segment_length = duration / num_segments

    base_name = Path(video_path).stem

    print(f"  Duration: {duration:.2f} c -> {num_segments} clips at {segment_length:.2f} s")

    for i in range(num_segments):
        start = i * segment_length
        if i == num_segments - 1:
            length = duration - start
        else:
            length = segment_length

        output_file = os.path.join(
            output_dir,
            f"{base_name}_{i + 1:02d}.mp4",
        )

        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{length:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_file,
            ],
            check=True,
        )
        print(f"    [+] {os.path.basename(output_file)}  ({length:.2f} s)")


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
            cut_video(str(video), OUTPUT_FOLDER, TARGET_SEGMENT_DURATION)
        except subprocess.CalledProcessError as e:
            print(f"  [!] Error ffmpeg: {e}")
        except Exception as e:
            print(f"  [!] Error: {e}")
        print()

    print("Processed!")


if __name__ == "__main__":
    main()