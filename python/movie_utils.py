"""Movie creation helpers with an ``imageio-ffmpeg`` fallback."""

import shutil
import subprocess
from pathlib import Path

from pyathena.plt_tools.make_movie import make_movie as _pyathena_make_movie


def _imageio_ffmpeg_executable():
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FileNotFoundError(
            "ffmpeg was not found on PATH and imageio-ffmpeg is not installed"
        ) from exc

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not executable.is_file():
        raise FileNotFoundError(f"imageio-ffmpeg executable not found: {executable}")
    return str(executable)


def _make_movie_with_executable(executable, fname_glob, fname_out, fps_in, fps_out):
    command = [
        executable,
        "-y",
        "-r",
        str(fps_in),
        "-f",
        "image2",
        "-pattern_type",
        "glob",
        "-i",
        fname_glob,
        "-r",
        str(fps_out),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-f",
        "mp4",
        fname_out,
    ]

    print("[make_movie]: imageio-ffmpeg command:")
    print(" ".join(command))
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print("\x1b[31m[make_movie]: subprocess.check_output returned:\x1b[0m")
        print(exc.output.decode("utf-8", errors="replace"))
        return False

    print("[make_movie]: Successful execution.")
    print("[make_movie]: Movie:")
    print(fname_out)
    return True


def make_movie(fname_glob, fname_out, fps_in=15, fps_out=15):
    """Create an MP4, falling back to imageio's ffmpeg when needed."""
    if shutil.which("ffmpeg") is not None:
        return _pyathena_make_movie(
            fname_glob, fname_out, fps_in=fps_in, fps_out=fps_out
        )

    executable = _imageio_ffmpeg_executable()
    return _make_movie_with_executable(
        executable, fname_glob, fname_out, fps_in, fps_out
    )
