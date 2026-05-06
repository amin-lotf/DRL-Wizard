from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_VIDEO_FPS = 30


@dataclass(frozen=True)
class EncodedVideo:
    data: bytes
    mime_type: str


def normalize_video_frame(frame: object) -> np.ndarray:
    if frame is None:
        raise ValueError("Environment returned no frame data.")

    array = np.asarray(frame)
    if array.size == 0:
        raise ValueError("Environment returned an empty frame.")

    if array.dtype == object:
        if array.size != 1:
            raise ValueError(f"Expected a single rendered frame, got {array.size}.")
        array = np.asarray(array.flat[0])

    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one rendered frame batch, got shape {tuple(array.shape)}.")
        array = np.asarray(array[0])

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {tuple(array.shape)}.")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def encode_video(frames: Sequence[object], fps: int = DEFAULT_VIDEO_FPS) -> EncodedVideo:
    if not frames:
        raise ValueError("No frames were collected for rendering.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to encode rendered videos.") from exc

    normalized_frames = [normalize_video_frame(frame) for frame in frames]
    height, width = normalized_frames[0].shape[:2]
    for index, frame in enumerate(normalized_frames[1:], start=1):
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Frame {index} has size {frame.shape[:2]}, expected {(height, width)}."
            )

    errors: list[str] = []
    candidates = [
        ("avc1", ".mp4", "video/mp4"),
        ("VP90", ".webm", "video/webm"),
        ("VP80", ".webm", "video/webm"),
        ("mp4v", ".mp4", "video/mp4"),
    ]

    for fourcc_name, suffix, mime_type in candidates:
        try:
            return _encode_with_opencv(
                normalized_frames=normalized_frames,
                fps=fps,
                width=width,
                height=height,
                fourcc_name=fourcc_name,
                suffix=suffix,
                mime_type=mime_type,
            )
        except Exception as exc:
            errors.append(f"{fourcc_name}: {exc}")

    raise RuntimeError("Video encoding failed. " + " ".join(errors))


def _encode_with_opencv(
    normalized_frames: Sequence[np.ndarray],
    fps: int,
    width: int,
    height: int,
    fourcc_name: str,
    suffix: str,
    mime_type: str,
) -> EncodedVideo:
    import cv2

    tmp_path: Path | None = None
    writer = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(max(int(fps), 1)),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("writer initialization failed")

        for frame in normalized_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        writer = None

        video_bytes = tmp_path.read_bytes()
        if not video_bytes:
            raise RuntimeError("encoded video file is empty")
        return EncodedVideo(data=video_bytes, mime_type=mime_type)
    finally:
        if writer is not None:
            writer.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
