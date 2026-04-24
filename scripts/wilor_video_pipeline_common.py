#!/usr/bin/env python3
"""Shared helpers for the HoloAssist -> WiLoR video pipeline."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_DATASET_ROOT = Path(
    "/share/project/RoboBrain-World-dataset/HoloAssist/HoloAssist_extracted"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/share/project/RoboBrain-World-dataset/HoloAssist-wilor-video"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WILOR_VENV = Path("/share/project/husicheng/WiLoR/.venv")
WILOR_PYTHON = WILOR_VENV / "bin" / "python"
WILOR_CHECKPOINT = REPO_ROOT / "pretrained_models" / "wilor_final.ckpt"
WILOR_CONFIG = REPO_ROOT / "pretrained_models" / "model_config.yaml"
WILOR_DETECTOR = REPO_ROOT / "pretrained_models" / "detector.pt"
MANO_DATA_DIR = REPO_ROOT / "mano_data"

ANNOTATION_FILE = "data-annotation-trainval-v1_1.json"
REQUIRED_RELATIVE_FILES = (
    "Export_py/Video_compress.mp4",
    "Export_py/Video/Pose_sync.txt",
    "Export_py/Video/Intrinsics.txt",
    "Export_py/Hands/Left_sync.txt",
    "Export_py/Hands/Right_sync.txt",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "sample"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def load_annotations(dataset_root: Path) -> List[Dict[str, Any]]:
    path = dataset_root / ANNOTATION_FILE
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected annotation JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def annotation_by_name(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for item in load_annotations(dataset_root):
        name = item.get("video_name")
        if isinstance(name, str) and name not in records:
            records[name] = item
    return records


def annotation_order(dataset_root: Path) -> List[str]:
    names: List[str] = []
    seen = set()
    for item in load_annotations(dataset_root):
        name = item.get("video_name")
        if isinstance(name, str) and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def iter_video_dirs(dataset_root: Path) -> List[Path]:
    if not dataset_root.is_dir():
        return []
    return sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=lambda p: p.name)


def video_dir_map(dataset_root: Path) -> Dict[str, Path]:
    return {p.name: p for p in iter_video_dirs(dataset_root)}


def required_paths(video_dir: Path) -> Dict[str, Path]:
    return {rel: video_dir / rel for rel in REQUIRED_RELATIVE_FILES}


def missing_required_files(video_dir: Path) -> List[str]:
    return [rel for rel, path in required_paths(video_dir).items() if not path.is_file()]


def is_processable_video_dir(video_dir: Path) -> bool:
    return not missing_required_files(video_dir)


def read_video_metadata(video_path: Path) -> Dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "path": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": (frame_count / fps) if fps > 0 else None,
    }


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_sample_dir(sample_dir: Path, overwrite: bool) -> None:
    if overwrite and sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)


def clip_video(
    source_video: Path,
    output_video: Path,
    *,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    import cv2

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for clipping: {source_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {source_video}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create clipped video: {output_video}")

    written = 0
    while True:
        if max_frames is not None and written >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    if written == 0:
        remove_path(output_video)
        raise RuntimeError(
            f"Clip request produced zero frames: {source_video} "
            f"(start_frame={start_frame}, max_frames={max_frames})"
        )

    metadata = read_video_metadata(output_video)
    metadata["frame_count"] = written
    metadata["fps"] = fps
    metadata["width"] = width
    metadata["height"] = height
    metadata["duration_sec"] = written / fps if fps > 0 else None
    return metadata


def prepare_workspace_video(
    source_video: Path,
    workspace_video: Path,
    *,
    copy_video: bool = False,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    workspace_video.parent.mkdir(parents=True, exist_ok=True)
    start_frame = max(0, int(start_frame or 0))
    clip_requested = start_frame > 0 or max_frames is not None

    if workspace_video.exists() or workspace_video.is_symlink():
        if overwrite:
            remove_path(workspace_video)
        elif clip_requested:
            raise FileExistsError(f"Workspace video already exists: {workspace_video}")
        else:
            return read_video_metadata(workspace_video), False

    if clip_requested:
        metadata = clip_video(
            source_video,
            workspace_video,
            start_frame=start_frame,
            max_frames=max_frames,
        )
        return metadata, True

    if copy_video:
        shutil.copy2(source_video, workspace_video)
    else:
        os.symlink(source_video, workspace_video)
    return read_video_metadata(workspace_video), False


def sample_output_paths(output_root: Path, seq_name: str) -> Dict[str, Path]:
    sample_dir = output_root / seq_name
    return {
        "sample_dir": sample_dir,
        "workspace_dir": sample_dir / "wilor_workspace",
        "workspace_video": sample_dir / "wilor_workspace" / "videos" / f"{seq_name}.mp4",
        "results_dir": sample_dir / "wilor_results",
        "detections_npz": sample_dir / "wilor_results" / "detections.npz",
        "frames_jsonl": sample_dir / "wilor_results" / "frames.jsonl",
        "summary_json": sample_dir / "wilor_results" / "summary.json",
        "manifest_json": sample_dir / "manifest.json",
        "run_script": sample_dir / "run_wilor_video.sh",
        "overlay_video": sample_dir / "handpose_skeleton_overlay.mp4",
        "hands_qa_npz": sample_dir / "holoassist_hands_qa.npz",
    }


def annotation_summary(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    events = record.get("events")
    return {
        "batch": record.get("batch"),
        "taskId": record.get("taskId"),
        "taskType": record.get("taskType"),
        "num_events": len(events) if isinstance(events, list) else None,
    }


def _quote(value: Path | str | float | int) -> str:
    return shlex.quote(str(value))


def write_run_script(
    run_script: Path,
    sample_dir: Path,
    *,
    conf: float,
    iou: float,
    rescale_factor: float,
    batch_size: int,
    fast: bool,
    overwrite: bool,
) -> None:
    args = [
        _quote(WILOR_PYTHON),
        _quote(REPO_ROOT / "scripts" / "holoassist_wilor_video_batch.py"),
        "infer-one",
        "--sample_dir",
        _quote(sample_dir),
        "--conf",
        _quote(conf),
        "--iou",
        _quote(iou),
        "--rescale_factor",
        _quote(rescale_factor),
        "--batch_size",
        _quote(batch_size),
    ]
    if fast:
        args.append("--fast")
    if overwrite:
        args.append("--overwrite")

    content = f"""#!/usr/bin/env bash
set -euo pipefail

export VIRTUAL_ENV={_quote(WILOR_VENV)}
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"

cd {_quote(REPO_ROOT)}
exec {" ".join(args)}
"""
    run_script.parent.mkdir(parents=True, exist_ok=True)
    run_script.write_text(content, encoding="utf-8")
    run_script.chmod(0o755)


def load_hand_sync(path: Path, joint_count: int = 26) -> Dict[str, Any]:
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"Hand sync file has fewer than 3 columns: {path}")

    times = data[:, 0].astype(np.float64)
    ticks = data[:, 1].astype(np.float64)
    valid = data[:, 2].astype(np.int8)
    positions = np.full((data.shape[0], joint_count, 3), np.nan, dtype=np.float32)
    matrix_cols = joint_count * 16
    if data.shape[1] >= 3 + matrix_cols:
        matrices = data[:, 3 : 3 + matrix_cols].reshape(data.shape[0], joint_count, 4, 4)
        positions = matrices[:, :, :3, 3].astype(np.float32)

    return {
        "time_sec": times,
        "ticks": ticks,
        "valid": valid,
        "joint_positions": positions,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
    }


def nearest_indices(source_times: Any, target_times: Any) -> Any:
    import numpy as np

    if len(source_times) == 0:
        return np.full(target_times.shape, -1, dtype=np.int32)
    idx = np.searchsorted(source_times, target_times, side="left")
    idx = np.clip(idx, 0, len(source_times) - 1)
    left = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_left = np.abs(target_times - source_times[left]) < np.abs(
        target_times - source_times[idx]
    )
    idx[choose_left] = left[choose_left]
    return idx.astype(np.int32)


def export_holoassist_hands_qa(
    video_dir: Path,
    output_path: Path,
    *,
    source_start_frame: int,
    frame_count: int,
    fps: float,
) -> Dict[str, Any]:
    import numpy as np

    if frame_count <= 0:
        raise ValueError("frame_count must be positive for hands QA export")
    if fps <= 0:
        fps = 30.0

    left = load_hand_sync(video_dir / "Export_py" / "Hands" / "Left_sync.txt")
    right = load_hand_sync(video_dir / "Export_py" / "Hands" / "Right_sync.txt")
    target_frame_index = np.arange(frame_count, dtype=np.int32)
    source_frame_index = target_frame_index + int(source_start_frame)
    target_time_sec = source_frame_index.astype(np.float64) / float(fps)

    left_idx = nearest_indices(left["time_sec"], target_time_sec)
    right_idx = nearest_indices(right["time_sec"], target_time_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        target_frame_index=target_frame_index,
        source_frame_index=source_frame_index.astype(np.int32),
        target_time_sec=target_time_sec.astype(np.float64),
        left_nearest_index=left_idx,
        right_nearest_index=right_idx,
        left_time_sec=left["time_sec"][left_idx].astype(np.float64),
        right_time_sec=right["time_sec"][right_idx].astype(np.float64),
        left_valid=left["valid"][left_idx].astype(np.int8),
        right_valid=right["valid"][right_idx].astype(np.int8),
        left_joint_positions=left["joint_positions"][left_idx].astype(np.float32),
        right_joint_positions=right["joint_positions"][right_idx].astype(np.float32),
        joint_count=np.array(left["joint_positions"].shape[1], dtype=np.int16),
        fps=np.array(fps, dtype=np.float32),
    )
    return {
        "path": str(output_path),
        "frame_count": int(frame_count),
        "fps": float(fps),
        "left_rows": left["rows"],
        "right_rows": right["rows"],
        "joint_count": int(left["joint_positions"].shape[1]),
    }


def verify_runtime_paths() -> List[str]:
    missing = []
    for path in (WILOR_PYTHON, WILOR_CHECKPOINT, WILOR_CONFIG, WILOR_DETECTOR, MANO_DATA_DIR):
        if not path.exists():
            missing.append(str(path))
    return missing


def count_processable_dirs(video_dirs: Iterable[Path]) -> int:
    return sum(1 for p in video_dirs if is_processable_video_dir(p))


def missing_examples(video_dirs: Sequence[Path], limit: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for video_dir in video_dirs:
        missing = missing_required_files(video_dir)
        if missing:
            examples.append({"video_name": video_dir.name, "missing": missing})
            if len(examples) >= limit:
                break
    return examples
