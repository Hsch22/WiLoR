#!/usr/bin/env python3
"""Batch HoloAssist videos through WiLoR hand detection and reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import wilor_video_pipeline_common as common

if str(common.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(common.REPO_ROOT))

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
HAND_SKELETON = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


@contextmanager
def timed_stage(timings: Dict[str, Any], name: str, metadata: Optional[Dict[str, Any]] = None):
    start_utc = common.utc_now_iso()
    start_perf = time.perf_counter()
    try:
        yield
    finally:
        record: Dict[str, Any] = {
            "name": name,
            "start_utc": start_utc,
            "end_utc": common.utc_now_iso(),
            "elapsed_sec": time.perf_counter() - start_perf,
        }
        if metadata is not None:
            record["metadata"] = metadata
        timings.setdefault("stages", []).append(record)


def add_elapsed(accumulator: Dict[str, float], name: str, start_perf: float) -> None:
    accumulator[name] = float(accumulator.get(name, 0.0)) + time.perf_counter() - start_perf


def positive_rate(count: int | float, elapsed_sec: Optional[float]) -> Optional[float]:
    if elapsed_sec is None or elapsed_sec <= 0:
        return None
    return float(count) / float(elapsed_sec)


def ms_per_item(elapsed_sec: Optional[float], count: int | float) -> Optional[float]:
    if elapsed_sec is None or elapsed_sec <= 0 or count <= 0:
        return None
    return float(elapsed_sec) * 1000.0 / float(count)


def stage_elapsed_map(timings: Dict[str, Any]) -> Dict[str, float]:
    return {
        str(stage.get("name")): float(stage.get("elapsed_sec", 0.0))
        for stage in timings.get("stages", [])
        if isinstance(stage, dict) and stage.get("name")
    }


def add_perf_metric(
    metrics: Dict[str, Any],
    name: str,
    *,
    elapsed_sec: Optional[float],
    frames: int,
    detections: Optional[int] = None,
    source: str,
    notes: Optional[str] = None,
) -> None:
    if elapsed_sec is None or elapsed_sec < 0:
        return
    record: Dict[str, Any] = {
        "elapsed_sec": float(elapsed_sec),
        "frames_per_sec": positive_rate(frames, elapsed_sec),
        "ms_per_frame": ms_per_item(elapsed_sec, frames),
        "source": source,
    }
    if detections is not None:
        record["detections_per_sec"] = positive_rate(detections, elapsed_sec)
        record["ms_per_detection"] = ms_per_item(elapsed_sec, detections)
    if notes:
        record["notes"] = notes
    metrics[name] = record


def build_wilor_performance_metrics(
    timings: Dict[str, Any],
    *,
    frame_count: Optional[int] = None,
    detection_count: Optional[int] = None,
    input_fps: Optional[float] = None,
) -> Dict[str, Any]:
    breakdown = timings.get("frame_loop_breakdown") or {}
    stages = stage_elapsed_map(timings)
    frames = int(frame_count or breakdown.get("frames") or 0)
    detections = int(detection_count if detection_count is not None else breakdown.get("detections") or 0)
    metrics: Dict[str, Any] = {
        "format": "wilor_performance_metrics_v1",
        "frame_count": frames,
        "detection_count": detections,
        "input_video_fps": float(input_fps) if input_fps else None,
        "average_detections_per_frame": (float(detections) / frames) if frames > 0 else None,
        "camera_coords_enabled": bool(timings.get("camera_coords_enabled", True)),
        "metrics": {},
    }
    records = metrics["metrics"]
    add_perf_metric(
        records,
        "end_to_end_run_script",
        elapsed_sec=(timings.get("run_script") or {}).get("elapsed_sec"),
        frames=frames,
        detections=detections,
        source="run_script_wall_time",
        notes="Includes process startup, imports, model loading, inference, rendering, video encoding, and manifest updates.",
    )
    add_perf_metric(
        records,
        "run_inference_total",
        elapsed_sec=timings.get("total_elapsed_sec"),
        frames=frames,
        detections=detections,
        source="in_process_wall_time",
        notes="Includes imports, model loading, inference, rendering, and output serialization.",
    )
    process_loop_sec = stages.get("process_frames_loop")
    add_perf_metric(
        records,
        "loaded_online_loop_with_render",
        elapsed_sec=process_loop_sec,
        frames=frames,
        detections=detections,
        source="process_frames_loop",
        notes="Excludes model loading but includes detection, WiLoR forward, overlays, video writes, and JSON/NPZ accumulation.",
    )
    render_sec = float(breakdown.get("draw_public_overlay_frames_sec", 0.0))
    write_sec = float(breakdown.get("write_public_video_frames_sec", 0.0))
    if process_loop_sec is not None:
        no_render_sec = max(0.0, float(process_loop_sec) - render_sec - write_sec)
        add_perf_metric(
            records,
            "loaded_online_no_render_estimate",
            elapsed_sec=no_render_sec,
            frames=frames,
            detections=detections,
            source="derived_from_process_frames_loop",
            notes="Derived by subtracting overlay rendering and video writing from the measured loaded frame loop.",
        )
    add_perf_metric(
        records,
        "detector_only",
        elapsed_sec=breakdown.get("detect_hands_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.detect_hands_sec",
    )
    add_perf_metric(
        records,
        "wilor_forward_only",
        elapsed_sec=breakdown.get("wilor_model_forward_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.wilor_model_forward_sec",
    )
    add_perf_metric(
        records,
        "wilor_infer_and_camera_coords",
        elapsed_sec=breakdown.get("wilor_infer_and_camera_coords_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.wilor_infer_and_camera_coords_sec",
    )
    add_perf_metric(
        records,
        "camera_coordinate_projection",
        elapsed_sec=breakdown.get("wilor_camera_coordinate_projection_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.wilor_camera_coordinate_projection_sec",
    )
    add_perf_metric(
        records,
        "keypoint_2d_projection",
        elapsed_sec=breakdown.get("wilor_2d_projection_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.wilor_2d_projection_sec",
    )
    add_perf_metric(
        records,
        "overlay_rendering",
        elapsed_sec=breakdown.get("draw_public_overlay_frames_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.draw_public_overlay_frames_sec",
    )
    add_perf_metric(
        records,
        "video_write",
        elapsed_sec=breakdown.get("write_public_video_frames_sec"),
        frames=frames,
        detections=detections,
        source="frame_loop_breakdown.write_public_video_frames_sec",
    )
    return metrics


def refresh_wilor_performance_metrics(manifest: Dict[str, Any]) -> None:
    timings = dict(manifest.get("timings") or {})
    results = dict(manifest.get("results") or {})
    frame_count = results.get("frame_count")
    if frame_count is None:
        frame_count = (timings.get("frame_loop_breakdown") or {}).get("frames")
    detection_count = results.get("total_detections")
    if detection_count is None:
        detection_count = (timings.get("frame_loop_breakdown") or {}).get("detections")
    input_fps = results.get("fps")
    if input_fps is None:
        input_fps = (manifest.get("workspace_video_metadata") or {}).get("fps")
    perf = build_wilor_performance_metrics(
        timings,
        frame_count=frame_count,
        detection_count=detection_count,
        input_fps=input_fps,
    )
    timings["performance_metrics"] = perf
    manifest["timings"] = timings
    if results:
        results["performance_metrics"] = perf
        results["timings"] = timings
        manifest["results"] = results


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=common.DEFAULT_DATASET_ROOT,
        help="HoloAssist_extracted root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT,
        help="Output root for WiLoR video results.",
    )
    parser.add_argument(
        "--video_name",
        action="append",
        default=[],
        help="HoloAssist video directory name. May be repeated.",
    )
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--start_frame", type=positive_int, default=0)
    parser.add_argument("--max_frames", type=positive_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--copy_video", action="store_true")
    parser.add_argument("--export_hands", action="store_true")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--no_camera_coords",
        action="store_true",
        help="Skip depth-bearing camera-coordinate keypoint export and coordinate overlays.",
    )
    parser.add_argument(
        "--no_fast",
        action="store_true",
        help="Disable WiLoR fast mode. Fast mode is enabled by default.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WiLoR on HoloAssist_extracted videos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Inspect dataset availability.")
    scan_parser.add_argument(
        "--dataset_root",
        type=Path,
        default=common.DEFAULT_DATASET_ROOT,
        help="HoloAssist_extracted root.",
    )
    scan_parser.add_argument(
        "--limit",
        type=positive_int,
        default=10,
        help="Number of missing examples to print.",
    )
    scan_parser.add_argument("--video_name", action="append", default=[])
    scan_parser.set_defaults(func=cmd_scan)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Create per-sample workspace and run script."
    )
    add_common_args(prepare_parser)
    prepare_parser.set_defaults(func=cmd_prepare)

    run_parser = subparsers.add_parser("run", help="Prepare and run WiLoR inference.")
    add_common_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    infer_parser = subparsers.add_parser("infer-one", help=argparse.SUPPRESS)
    infer_parser.add_argument("--sample_dir", type=Path, required=True)
    infer_parser.add_argument("--conf", type=float, default=0.3)
    infer_parser.add_argument("--iou", type=float, default=0.5)
    infer_parser.add_argument("--rescale_factor", type=float, default=2.0)
    infer_parser.add_argument("--batch_size", type=int, default=16)
    infer_parser.add_argument("--fast", action="store_true")
    infer_parser.add_argument("--no_camera_coords", action="store_true")
    infer_parser.add_argument("--overwrite", action="store_true")
    infer_parser.set_defaults(func=cmd_infer_one)

    render_parser = subparsers.add_parser(
        "render-overlays",
        help="Rebuild WiLoR public overlay videos from existing detections.",
    )
    render_parser.add_argument("--sample_dir", type=Path, required=True)
    render_parser.add_argument("--overwrite", action="store_true")
    render_parser.set_defaults(func=cmd_render_overlays)
    return parser


def selected_video_names(args: argparse.Namespace) -> List[str]:
    dataset_root = args.dataset_root
    dir_map = common.video_dir_map(dataset_root)
    explicit = bool(args.video_name)

    if explicit:
        names = list(args.video_name)
        if args.limit is not None:
            names = names[: args.limit]
    else:
        names = common.annotation_order(dataset_root)
        if not names:
            names = [p.name for p in common.iter_video_dirs(dataset_root)]

    selected: List[str] = []
    skipped_missing: List[Dict[str, Any]] = []

    for name in names:
        video_dir = dir_map.get(name)
        missing: List[str]
        if video_dir is None:
            missing = ["<video_dir>"]
        else:
            missing = common.missing_required_files(video_dir)

        if missing:
            if explicit and not args.skip_missing:
                raise FileNotFoundError(f"{name}: missing {', '.join(missing)}")
            skipped_missing.append({"video_name": name, "missing": missing})
            continue

        selected.append(name)
        if not explicit and args.limit is not None and len(selected) >= args.limit:
            break

    if not selected:
        suffix = " after skipping missing inputs" if skipped_missing else ""
        raise RuntimeError(f"No processable videos selected{suffix}.")
    return selected


def cmd_scan(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root
    annotations = common.load_annotations(dataset_root)
    annotation_names = [
        item.get("video_name") for item in annotations if isinstance(item.get("video_name"), str)
    ]
    dirs = common.iter_video_dirs(dataset_root)
    dir_map = {p.name: p for p in dirs}
    processable_dirs = common.count_processable_dirs(dirs)

    if args.video_name:
        scan_dirs = [dir_map[name] for name in args.video_name if name in dir_map]
        missing_specific = [
            {
                "video_name": name,
                "missing": ["<video_dir>"],
            }
            for name in args.video_name
            if name not in dir_map
        ]
    else:
        scan_dirs = dirs
        missing_specific = []

    annotation_with_dir = sum(1 for name in annotation_names if name in dir_map)
    annotation_processable = sum(
        1
        for name in annotation_names
        if name in dir_map and common.is_processable_video_dir(dir_map[name])
    )
    missing = missing_specific + common.missing_examples(scan_dirs, args.limit)

    print("HoloAssist WiLoR scan")
    print(f"dataset_root: {dataset_root}")
    print(f"annotations: {len(annotations)}")
    print(f"video_dirs: {len(dirs)}")
    print(f"processable_dirs: {processable_dirs}")
    print(f"annotation_names_with_dirs: {annotation_with_dir}")
    print(f"annotation_names_processable: {annotation_processable}")
    if missing:
        print(f"missing_examples: {len(missing)}")
        for item in missing[: args.limit]:
            print(f"  {item['video_name']}: {', '.join(item['missing'])}")
    else:
        print("missing_examples: 0")
    runtime_missing = common.verify_runtime_paths()
    if runtime_missing:
        print("missing_runtime_paths:")
        for path in runtime_missing:
            print(f"  {path}")
    return 0


def build_manifest_for_sample(
    *,
    dataset_root: Path,
    output_root: Path,
    video_name: str,
    annotation_record: Optional[Dict[str, Any]],
    start_frame: int,
    max_frames: Optional[int],
    copy_video: bool,
    export_hands: bool,
    conf: float,
    iou: float,
    rescale_factor: float,
    batch_size: int,
    fast: bool,
    camera_coords: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    seq_name = common.safe_name(video_name)
    paths = common.sample_output_paths(output_root, seq_name)
    video_dir = dataset_root / video_name
    missing = common.missing_required_files(video_dir)
    if missing:
        raise FileNotFoundError(f"{video_name}: missing {', '.join(missing)}")

    common.ensure_sample_dir(paths["sample_dir"], overwrite=overwrite)
    paths["workspace_video"].parent.mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    source_video = video_dir / "Export_py" / "Video_compress.mp4"
    source_metadata = common.read_video_metadata(source_video)
    workspace_metadata, clipped = common.prepare_workspace_video(
        source_video,
        paths["workspace_video"],
        copy_video=copy_video,
        start_frame=start_frame,
        max_frames=max_frames,
        overwrite=overwrite,
    )

    hands_qa: Optional[Dict[str, Any]] = None
    if export_hands:
        hands_qa = common.export_holoassist_hands_qa(
            video_dir,
            paths["hands_qa_npz"],
            source_start_frame=start_frame,
            frame_count=int(workspace_metadata["frame_count"]),
            fps=float(source_metadata["fps"] or workspace_metadata["fps"] or 30.0),
        )

    output_video_keys = (
        tuple(common.WILOR_PUBLIC_VIDEO_OUTPUTS)
        if camera_coords
        else ("handmesh_overlay", "hand_skeleton_overlay")
    )
    output_video_path_keys = {f"{key}_video" for key in output_video_keys}
    output_paths = {
        key: str(value)
        for key, value in paths.items()
        if key
        in {
            "workspace_video",
            "detections_npz",
            "frames_jsonl",
            "summary_json",
            "manifest_json",
            "run_script",
            "overlay_video",
            "legacy_overlay_video",
            "hands_qa_npz",
        }
        or key in output_video_path_keys
    }
    manifest: Dict[str, Any] = {
        "status": "prepared",
        "prepared_at": common.utc_now_iso(),
        "video_name": video_name,
        "seq_name": seq_name,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "repo_root": str(common.REPO_ROOT),
        "source_dir": str(video_dir),
        "source_video": str(source_video),
        "required_files": {
            rel: str(path) for rel, path in common.required_paths(video_dir).items()
        },
        "annotation": common.annotation_summary(annotation_record),
        "source_video_metadata": source_metadata,
        "workspace_video_metadata": workspace_metadata,
        "debug_clip": {
            "enabled": clipped,
            "start_frame": int(start_frame),
            "max_frames": max_frames,
        },
        "copy_video": bool(copy_video),
        "export_hands": bool(export_hands),
        "holoassist_hands_qa": hands_qa,
        "wilor": {
            "python": str(common.WILOR_PYTHON),
            "checkpoint": str(common.WILOR_CHECKPOINT),
            "config": str(common.WILOR_CONFIG),
            "detector": str(common.WILOR_DETECTOR),
            "mano_data": str(common.MANO_DATA_DIR),
            "conf": float(conf),
            "iou": float(iou),
            "rescale_factor": float(rescale_factor),
            "batch_size": int(batch_size),
            "fast": bool(fast),
            "camera_coords": bool(camera_coords),
        },
        "outputs": output_paths,
        "output_videos": common.wilor_output_videos_manifest(
            paths["sample_dir"],
            output_keys=output_video_keys,
        ),
        "omitted_output_videos": common.wilor_omitted_output_videos_manifest(
            output_keys=output_video_keys,
        ),
    }
    common.write_json(paths["manifest_json"], manifest)
    common.write_run_script(
        paths["run_script"],
        paths["sample_dir"],
        conf=conf,
        iou=iou,
        rescale_factor=rescale_factor,
        batch_size=batch_size,
        fast=fast,
        camera_coords=camera_coords,
        overwrite=overwrite,
    )
    return manifest


def prepare_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_root = args.dataset_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    annotation_records = common.annotation_by_name(dataset_root)
    names = selected_video_names(args)
    fast = not args.no_fast
    manifests: List[Dict[str, Any]] = []
    for name in names:
        prepare_start_utc = common.utc_now_iso()
        prepare_t0 = time.perf_counter()
        manifest = build_manifest_for_sample(
            dataset_root=dataset_root,
            output_root=output_root,
            video_name=name,
            annotation_record=annotation_records.get(name),
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            copy_video=args.copy_video,
            export_hands=args.export_hands,
            conf=args.conf,
            iou=args.iou,
            rescale_factor=args.rescale_factor,
            batch_size=args.batch_size,
            fast=fast,
            camera_coords=not args.no_camera_coords,
            overwrite=args.overwrite,
        )
        prepare_timing = {
            "start_utc": prepare_start_utc,
            "end_utc": common.utc_now_iso(),
            "elapsed_sec": time.perf_counter() - prepare_t0,
            "copy_video": bool(args.copy_video),
            "export_hands": bool(args.export_hands),
            "camera_coords": bool(not args.no_camera_coords),
        }
        manifest_timings = dict(manifest.get("timings") or {})
        manifest_timings["prepare_sample"] = prepare_timing
        manifest["timings"] = manifest_timings
        common.write_json(Path(manifest["outputs"]["manifest_json"]), manifest)
        manifests.append(manifest)
    return manifests


def cmd_prepare(args: argparse.Namespace) -> int:
    manifests = prepare_samples(args)
    print(f"prepared_samples: {len(manifests)}")
    for manifest in manifests:
        print(
            f"  {manifest['video_name']} -> "
            f"{manifest['outputs']['run_script']}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifests = prepare_samples(args)
    batch_records: List[Dict[str, Any]] = []
    for manifest in manifests:
        run_script = Path(manifest["outputs"]["run_script"])
        sample_manifest_path = Path(manifest["outputs"]["manifest_json"])
        print(f"running: {manifest['video_name']}")
        run_start_utc = common.utc_now_iso()
        run_t0 = time.perf_counter()
        completed = subprocess.run([str(run_script)], cwd=str(common.REPO_ROOT))
        run_script_timing = {
            "start_utc": run_start_utc,
            "end_utc": common.utc_now_iso(),
            "elapsed_sec": time.perf_counter() - run_t0,
            "returncode": int(completed.returncode),
        }
        if completed.returncode != 0:
            current = common.read_json(sample_manifest_path)
            current_timings = dict(current.get("timings") or {})
            current_timings["run_script"] = run_script_timing
            current.update(
                {
                    "status": "failed",
                    "failed_at": common.utc_now_iso(),
                    "exit_code": int(completed.returncode),
                    "timings": current_timings,
                }
            )
            common.write_json(sample_manifest_path, current)
            batch_records.append(
                {
                    "video_name": manifest["video_name"],
                    "manifest": str(sample_manifest_path),
                    "status": "failed",
                    "exit_code": int(completed.returncode),
                    "elapsed_sec": run_script_timing["elapsed_sec"],
                }
            )
            write_batch_manifest(args.output_root, batch_records, "failed")
            return completed.returncode

        final_manifest = common.read_json(sample_manifest_path)
        final_timings = dict(final_manifest.get("timings") or {})
        final_timings["run_script"] = run_script_timing
        final_manifest["timings"] = final_timings
        refresh_wilor_performance_metrics(final_manifest)
        common.write_json(sample_manifest_path, final_manifest)
        batch_records.append(
            {
                "video_name": manifest["video_name"],
                "manifest": str(sample_manifest_path),
                "status": final_manifest.get("status", "complete"),
                "exit_code": int(final_manifest.get("exit_code", 0)),
                "elapsed_sec": run_script_timing["elapsed_sec"],
            }
        )

    write_batch_manifest(args.output_root, batch_records, "complete")
    print(f"completed_samples: {len(batch_records)}")
    return 0


def write_batch_manifest(
    output_root: Path, records: Sequence[Dict[str, Any]], status: str
) -> None:
    manifest = {
        "status": status,
        "updated_at": common.utc_now_iso(),
        "samples": list(records),
    }
    common.write_json(output_root / "last_batch_manifest.json", manifest)


def cmd_infer_one(args: argparse.Namespace) -> int:
    run_inference(
        sample_dir=args.sample_dir,
        conf=args.conf,
        iou=args.iou,
        rescale_factor=args.rescale_factor,
        batch_size=args.batch_size,
        fast=args.fast,
        camera_coords=not args.no_camera_coords,
        overwrite=args.overwrite,
    )
    return 0


def cmd_render_overlays(args: argparse.Namespace) -> int:
    render_overlays_from_results(sample_dir=args.sample_dir, overwrite=args.overwrite)
    return 0


def project_points_full_img(points, cam_trans, focal_length: float, img_res):
    import numpy as np

    points_cam = points.astype(np.float32) + cam_trans.astype(np.float32).reshape(1, 3)
    z = np.maximum(points_cam[:, 2:3], 1e-6)
    projected = points_cam[:, :2] / z
    projected[:, 0] = projected[:, 0] * focal_length + float(img_res[0]) / 2.0
    projected[:, 1] = projected[:, 1] * focal_length + float(img_res[1]) / 2.0
    return projected.astype(np.float32)


def detect_hands(detector, frame, conf: float, iou: float):
    import numpy as np

    detections = detector(frame, conf=conf, iou=iou, verbose=False)[0]
    if detections.boxes is None or len(detections.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    boxes = detections.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = detections.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = detections.boxes.cls.detach().cpu().numpy().astype(np.float32)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    right = (classes[valid] > 0.5).astype(np.float32)
    return boxes, scores, right


def infer_frame_hands(
    *,
    frame,
    boxes,
    scores,
    right,
    model,
    model_cfg,
    device,
    batch_size: int,
    rescale_factor: float,
    fast: bool,
    camera_coords: bool,
    stage_times: Optional[Dict[str, float]] = None,
):
    import torch

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    def sync_device() -> None:
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    stage_t0 = time.perf_counter()
    dataset = ViTDetDataset(
        model_cfg,
        frame,
        boxes,
        right,
        rescale_factor=rescale_factor,
        fp16=fast,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    if stage_times is not None:
        add_elapsed(stage_times, "wilor_crop_dataset_sec", stage_t0)

    hands: List[Dict[str, Any]] = []
    for batch in dataloader:
        stage_t0 = time.perf_counter()
        batch = recursive_to(batch, device)
        if stage_times is not None:
            add_elapsed(stage_times, "wilor_batch_to_device_sec", stage_t0)
        sync_device()
        stage_t0 = time.perf_counter()
        with torch.no_grad():
            out = model(batch)
        sync_device()
        if stage_times is not None:
            add_elapsed(stage_times, "wilor_model_forward_sec", stage_t0)

        stage_t0 = time.perf_counter()
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        focal_length = float(scaled_focal_length.detach().cpu().item())
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            )
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )
        if stage_times is not None:
            add_elapsed(stage_times, "wilor_camera_translation_sec", stage_t0)

        for n in range(batch["img"].shape[0]):
            stage_t0 = time.perf_counter()
            det_id = int(batch["personid"][n].detach().cpu().item())
            is_right = int(batch["right"][n].detach().cpu().item() >= 0.5)
            side = 2 * is_right - 1
            verts = out["pred_vertices"][n].float().detach().cpu().numpy().astype("float32")
            joints = (
                out["pred_keypoints_3d"][n].float().detach().cpu().numpy().astype("float32")
            )
            verts[:, 0] = side * verts[:, 0]
            joints[:, 0] = side * joints[:, 0]
            cam_t = pred_cam_t_full[n]
            img_res = img_size[n].detach().cpu().numpy().astype("float32")
            if stage_times is not None:
                add_elapsed(stage_times, "wilor_extract_predictions_sec", stage_t0)
            stage_t0 = time.perf_counter()
            joints_2d = project_points_full_img(joints, cam_t, focal_length, img_res)
            if stage_times is not None:
                add_elapsed(stage_times, "wilor_2d_projection_sec", stage_t0)
            hand = {
                "det_id": det_id,
                "bbox": boxes[det_id],
                "score": scores[det_id],
                "is_right": is_right,
                "cam_t": cam_t,
                "joints_3d": joints,
                "joints_2d": joints_2d,
                "vertices": verts,
                "focal_length": focal_length,
                "img_res": img_res,
            }
            if camera_coords:
                stage_t0 = time.perf_counter()
                hand["joints_cam"] = joints.astype("float32") + cam_t.astype("float32").reshape(1, 3)
                if stage_times is not None:
                    add_elapsed(stage_times, "wilor_camera_coordinate_projection_sec", stage_t0)
            hands.append(hand)
    hands.sort(key=lambda item: item["det_id"])
    return hands


def overlay_meshes(frame, hands, renderer):
    import numpy as np

    if not hands:
        return frame.copy(), 0

    render_res = [int(frame.shape[1]), int(frame.shape[0])]
    focal_length = float(hands[0]["focal_length"])
    cam_view = renderer.render_rgba_multiple(
        [hand["vertices"] for hand in hands],
        cam_t=[hand["cam_t"] for hand in hands],
        render_res=render_res,
        is_right=[hand["is_right"] for hand in hands],
        mesh_base_color=LIGHT_PURPLE,
        scene_bg_color=(1, 1, 1),
        focal_length=focal_length,
    )
    input_rgb = frame.astype(np.float32)[:, :, ::-1] / 255.0
    alpha = cam_view[:, :, 3:]
    overlay_rgb = input_rgb[:, :, :3] * (1.0 - alpha) + cam_view[:, :, :3] * alpha
    overlay_bgr = np.clip(overlay_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return overlay_bgr, 0


def hand_color(is_right: int) -> Tuple[int, int, int]:
    return (42, 180, 96) if is_right else (42, 130, 220)


def point_in_loose_frame(point, width: int, height: int) -> bool:
    return -width <= point[0] <= 2 * width and -height <= point[1] <= 2 * height


def draw_hand_skeleton(frame, hands) -> None:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    for hand in hands:
        color = hand_color(int(hand["is_right"]))
        points = hand["joints_2d"]
        finite = np.isfinite(points).all(axis=1)
        for start, end in HAND_SKELETON:
            if start >= len(points) or end >= len(points):
                continue
            if not (finite[start] and finite[end]):
                continue
            p1 = points[start]
            p2 = points[end]
            if point_in_loose_frame(p1, width, height) and point_in_loose_frame(
                p2, width, height
            ):
                cv2.line(
                    frame,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for point in points:
            if not np.isfinite(point).all():
                continue
            if point_in_loose_frame(point, width, height):
                cv2.circle(
                    frame,
                    (int(round(point[0])), int(round(point[1]))),
                    3,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    color,
                    -1,
                    cv2.LINE_AA,
                )


def draw_camera_coord_labels(frame, hands) -> None:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.34
    thickness = 1
    pad = 2
    for hand in hands:
        if "joints_cam" not in hand:
            continue
        color = hand_color(int(hand["is_right"]))
        points_2d = hand["joints_2d"]
        joints_cam = hand["joints_cam"]
        joint_count = min(len(points_2d), len(joints_cam), 21)
        for joint_idx in range(joint_count):
            point = points_2d[joint_idx]
            coord = joints_cam[joint_idx]
            if not np.isfinite(point).all() or not np.isfinite(coord).all():
                continue
            if float(coord[2]) <= 0.0:
                continue
            if not point_in_loose_frame(point, width, height):
                continue
            text = (
                f"{joint_idx}:("
                f"{float(coord[0]):.2f},{float(coord[1]):.2f},{float(coord[2]):.2f})"
            )
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            x = int(round(point[0])) + 4
            y = int(round(point[1])) - 4
            x = max(0, min(width - text_w - 2 * pad, x))
            y = max(text_h + 2 * pad, min(height - baseline - pad, y))
            cv2.rectangle(
                frame,
                (x - pad, y - text_h - pad),
                (x + text_w + pad, y + baseline + pad),
                (0, 0, 0),
                -1,
            )
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_public_overlay_frames(
    frame,
    hands,
    renderer,
    *,
    camera_coords: bool,
) -> Tuple[Dict[str, Any], Optional[Exception]]:
    if not hands:
        frames = {
            "handmesh_overlay": frame,
            "hand_skeleton_overlay": frame,
        }
        if camera_coords:
            frames.update(
                {
                    "handmesh_fitted_camera_coords_overlay": frame,
                    "hand_skeleton_fitted_camera_coords_overlay": frame,
                }
            )
        return frames, None

    skeleton = frame.copy()
    draw_hand_skeleton(skeleton, hands)

    skeleton_coords = None
    if camera_coords:
        skeleton_coords = skeleton.copy()
        draw_camera_coord_labels(skeleton_coords, hands)

    render_error: Optional[Exception] = None
    try:
        mesh, _ = overlay_meshes(frame, hands, renderer)
    except Exception as exc:
        render_error = exc
        mesh = frame.copy()

    frames = {
        "handmesh_overlay": mesh,
        "hand_skeleton_overlay": skeleton,
    }
    if camera_coords:
        if render_error is not None:
            mesh_coords = frame.copy()
        else:
            mesh_coords = mesh.copy()
            draw_hand_skeleton(mesh_coords, hands)
            draw_camera_coord_labels(mesh_coords, hands)
        frames.update(
            {
                "handmesh_fitted_camera_coords_overlay": mesh_coords,
                "hand_skeleton_fitted_camera_coords_overlay": skeleton_coords,
            }
        )
    return frames, render_error


def update_manifest_public_outputs(
    manifest: Dict[str, Any],
    sample_dir: Path,
    *,
    camera_coords: bool = True,
) -> None:
    outputs = manifest.setdefault("outputs", {})
    output_keys = (
        tuple(common.WILOR_PUBLIC_VIDEO_OUTPUTS)
        if camera_coords
        else ("handmesh_overlay", "hand_skeleton_overlay")
    )
    public_paths = common.wilor_public_video_paths(sample_dir, output_keys=output_keys)
    for key, path in public_paths.items():
        outputs[f"{key}_video"] = str(path)
    outputs["overlay_video"] = str(public_paths["hand_skeleton_overlay"])
    outputs.setdefault("legacy_overlay_video", str(sample_dir / "handpose_skeleton_overlay.mp4"))
    manifest["output_videos"] = common.wilor_output_videos_manifest(
        sample_dir,
        output_keys=output_keys,
    )
    manifest["omitted_output_videos"] = common.wilor_omitted_output_videos_manifest(
        output_keys=output_keys,
    )


def prepare_public_video_writers(
    sample_dir: Path,
    *,
    fps: float,
    width: int,
    height: int,
    camera_coords: bool,
    overwrite: bool,
):
    import cv2

    output_keys = (
        tuple(common.WILOR_PUBLIC_VIDEO_OUTPUTS)
        if camera_coords
        else ("handmesh_overlay", "hand_skeleton_overlay")
    )
    output_paths = common.wilor_public_video_paths(sample_dir, output_keys=output_keys)
    for path in output_paths.values():
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"Public overlay already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        key: cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for key, path in output_paths.items()
    }
    failed = [key for key, writer in writers.items() if not writer.isOpened()]
    if failed:
        for writer in writers.values():
            writer.release()
        raise RuntimeError(f"Could not create public overlay videos: {', '.join(failed)}")
    return writers


def close_video_writers(writers) -> None:
    for writer in writers.values():
        writer.release()


def write_public_video_frames(writers, frames: Dict[str, Any]) -> None:
    for key in writers:
        writers[key].write(frames[key])


def load_overlay_renderer():
    import numpy as np

    from wilor.configs import get_config
    from wilor.utils.renderer import Renderer

    model_cfg = get_config(str(common.WILOR_CONFIG), update_cachedir=True)
    mano_path = common.MANO_DATA_DIR / "MANO_RIGHT.pkl"
    with mano_path.open("rb") as f:
        mano_data = pickle.load(f, encoding="latin1")
    faces = np.asarray(mano_data["f"], dtype=np.int32)
    return Renderer(model_cfg, faces=faces), model_cfg


def scaled_focal_length(model_cfg, width: int, height: int) -> float:
    return float(model_cfg.EXTRA.FOCAL_LENGTH) / float(model_cfg.MODEL.IMAGE_SIZE) * float(
        max(width, height)
    )


def normalize_focal_length(values, det_count: int, default_focal_length: float):
    import numpy as np

    if values is None:
        return np.full((det_count,), default_focal_length, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return np.full((det_count,), float(values), dtype=np.float32)
    values = values.reshape(-1)
    if len(values) == det_count:
        return values.astype(np.float32)
    if len(values) == 1:
        return np.full((det_count,), float(values[0]), dtype=np.float32)
    return np.full((det_count,), default_focal_length, dtype=np.float32)


def load_detection_arrays(
    detections_npz: Path,
    *,
    default_focal_length: float,
) -> Tuple[Dict[str, Any], bool]:
    import numpy as np

    with np.load(detections_npz, allow_pickle=False) as data:
        arrays: Dict[str, Any] = {name: data[name] for name in data.files}
        original_files = set(data.files)

    det_count = int(len(arrays.get("frame_index", [])))
    needs_update = False

    if "joints_cam" not in arrays:
        joints_3d = np.asarray(arrays.get("joints_3d", np.zeros((det_count, 21, 3))))
        cam_t = np.asarray(arrays.get("cam_t", np.zeros((det_count, 3))))
        if det_count:
            arrays["joints_cam"] = (
                joints_3d.astype(np.float32) + cam_t.astype(np.float32).reshape(det_count, 1, 3)
            )
        else:
            arrays["joints_cam"] = np.zeros((0, 21, 3), dtype=np.float32)
        needs_update = True
    else:
        arrays["joints_cam"] = np.asarray(arrays["joints_cam"], dtype=np.float32)

    focal_values = arrays.get("focal_length")
    arrays["focal_length"] = normalize_focal_length(
        focal_values,
        det_count,
        default_focal_length,
    )
    if "focal_length" not in original_files:
        needs_update = True

    return arrays, needs_update


def rewrite_detection_npz_if_needed(detections_npz: Path, arrays: Dict[str, Any], needs_update: bool) -> None:
    if not needs_update:
        return
    import numpy as np

    np.savez_compressed(detections_npz, **arrays)


def read_frame_records(frames_jsonl: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with frames_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def scalar_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(value.reshape(()))
        except Exception:
            return default


def detection_indices_for_frame(
    arrays: Dict[str, Any],
    record: Dict[str, Any],
    frame_index: int,
) -> Iterable[int]:
    import numpy as np

    if "det_start" in record and "det_end" in record:
        return range(int(record["det_start"]), int(record["det_end"]))
    frame_indices = np.asarray(arrays["frame_index"])
    return np.flatnonzero(frame_indices == frame_index).tolist()


def hands_from_detection_arrays(
    arrays: Dict[str, Any],
    record: Dict[str, Any],
    frame_index: int,
) -> List[Dict[str, Any]]:
    hands: List[Dict[str, Any]] = []
    for det_idx in detection_indices_for_frame(arrays, record, frame_index):
        hands.append(
            {
                "is_right": int(arrays["is_right"][det_idx]),
                "cam_t": arrays["cam_t"][det_idx].astype("float32"),
                "joints_2d": arrays["joints_2d"][det_idx].astype("float32"),
                "joints_cam": arrays["joints_cam"][det_idx].astype("float32"),
                "vertices": arrays["vertices"][det_idx].astype("float32"),
                "focal_length": float(arrays["focal_length"][det_idx]),
            }
        )
    return hands


def run_inference(
    *,
    sample_dir: Path,
    conf: float,
    iou: float,
    rescale_factor: float,
    batch_size: int,
    fast: bool,
    camera_coords: bool,
    overwrite: bool,
) -> None:
    total_start_utc = common.utc_now_iso()
    total_t0 = time.perf_counter()
    timings: Dict[str, Any] = {
        "format": "wilor_stage_timings_v1",
        "started_at": total_start_utc,
        "stages": [],
    }
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.chdir(common.REPO_ROOT)

    with timed_stage(timings, "import_runtime_modules"):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        from wilor.models import load_wilor
        from wilor.utils.renderer import Renderer

    with timed_stage(timings, "prepare_manifest_and_output"):
        manifest_path = sample_dir / "manifest.json"
        manifest = common.read_json(manifest_path)
        update_manifest_public_outputs(manifest, sample_dir, camera_coords=camera_coords)
        workspace_video = Path(manifest["outputs"]["workspace_video"])
        results_dir = Path(manifest["outputs"]["summary_json"]).parent

        if results_dir.exists() and overwrite:
            shutil.rmtree(results_dir)
        elif (results_dir / "summary.json").exists() and not overwrite:
            raise FileExistsError(f"Results already exist: {results_dir}")
        results_dir.mkdir(parents=True, exist_ok=True)

    with timed_stage(timings, "open_video"):
        cap = cv2.VideoCapture(str(workspace_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open workspace video: {workspace_video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Could not read video dimensions: {workspace_video}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with timed_stage(
        timings,
        "load_wilor_model",
        metadata={
            "device": str(device),
            "checkpoint": str(common.WILOR_CHECKPOINT),
            "config": str(common.WILOR_CONFIG),
        },
    ):
        print(f"loading WiLoR on {device}")
        model, model_cfg = load_wilor(
            checkpoint_path=str(common.WILOR_CHECKPOINT),
            cfg_path=str(common.WILOR_CONFIG),
        )
    fast_metadata = {"requested_fast": bool(fast), "device": str(device)}
    with timed_stage(timings, "configure_fast_mode", metadata=fast_metadata):
        if fast and device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            model = model.half()
            try:
                model.backbone = torch.compile(model.backbone)
                fast_metadata["torch_compile_backbone"] = True
            except Exception as exc:
                fast_metadata["torch_compile_backbone"] = False
                fast_metadata["torch_compile_error"] = str(exc)
                print(f"warning: torch.compile failed, continuing without compile: {exc}", file=sys.stderr)
            try:
                model.backbone.skip_blocks = True
                fast_metadata["skip_blocks"] = True
            except Exception as exc:
                fast_metadata["skip_blocks"] = False
                fast_metadata["skip_blocks_error"] = str(exc)
                print(f"warning: could not enable backbone skip_blocks: {exc}", file=sys.stderr)
        elif fast:
            print("warning: fast mode requested without CUDA; using full precision", file=sys.stderr)
            fast = False
        fast_metadata["effective_fast"] = bool(fast)

    with timed_stage(
        timings,
        "load_detector",
        metadata={"device": str(device), "detector": str(common.WILOR_DETECTOR)},
    ):
        detector = YOLO(str(common.WILOR_DETECTOR))
    with timed_stage(timings, "move_models_to_device", metadata={"device": str(device)}):
        model = model.to(device)
        detector = detector.to(device)
        model.eval()
    with timed_stage(timings, "init_renderer"):
        renderer = Renderer(model_cfg, faces=model.mano.faces)

    with timed_stage(timings, "prepare_public_video_writers"):
        writers = prepare_public_video_writers(
            sample_dir,
            fps=fps,
            width=width,
            height=height,
            camera_coords=camera_coords,
            overwrite=overwrite,
        )

    frame_indices: List[int] = []
    det_indices: List[int] = []
    bbox_xyxy: List[Any] = []
    scores: List[float] = []
    is_right_values: List[int] = []
    cam_t_values: List[Any] = []
    joints_3d_values: List[Any] = []
    joints_cam_values: List[Any] = []
    joints_2d_values: List[Any] = []
    vertices_values: List[Any] = []
    focal_length_values: List[float] = []

    total_frames = 0
    total_detections = 0
    left_detections = 0
    right_detections = 0
    render_failures = 0
    frames_jsonl = results_dir / "frames.jsonl"
    frame_loop_breakdown: Dict[str, Any] = {
        "read_frame_sec": 0.0,
        "detect_hands_sec": 0.0,
        "wilor_infer_and_camera_coords_sec": 0.0,
        "wilor_crop_dataset_sec": 0.0,
        "wilor_batch_to_device_sec": 0.0,
        "wilor_model_forward_sec": 0.0,
        "wilor_camera_translation_sec": 0.0,
        "wilor_extract_predictions_sec": 0.0,
        "wilor_2d_projection_sec": 0.0,
        "wilor_camera_coordinate_projection_sec": 0.0,
        "draw_public_overlay_frames_sec": 0.0,
        "write_public_video_frames_sec": 0.0,
        "record_outputs_sec": 0.0,
    }

    with timed_stage(timings, "process_frames_loop"):
        with frames_jsonl.open("w", encoding="utf-8") as frames_out:
            while True:
                stage_t0 = time.perf_counter()
                ok, frame = cap.read()
                add_elapsed(frame_loop_breakdown, "read_frame_sec", stage_t0)
                if not ok:
                    break

                det_start = len(frame_indices)
                stage_t0 = time.perf_counter()
                boxes, det_scores, right = detect_hands(detector, frame, conf=conf, iou=iou)
                add_elapsed(frame_loop_breakdown, "detect_hands_sec", stage_t0)

                hands: List[Dict[str, Any]] = []
                stage_t0 = time.perf_counter()
                if len(boxes) > 0:
                    hands = infer_frame_hands(
                        frame=frame,
                        boxes=boxes,
                        scores=det_scores,
                        right=right,
                        model=model,
                        model_cfg=model_cfg,
                        device=device,
                        batch_size=batch_size,
                        rescale_factor=rescale_factor,
                        fast=fast,
                        camera_coords=camera_coords,
                        stage_times=frame_loop_breakdown,
                    )
                add_elapsed(frame_loop_breakdown, "wilor_infer_and_camera_coords_sec", stage_t0)

                stage_t0 = time.perf_counter()
                public_frames, render_error = draw_public_overlay_frames(
                    frame,
                    hands,
                    renderer,
                    camera_coords=camera_coords,
                )
                if render_error is not None:
                    if render_failures == 0:
                        print(
                            f"warning: mesh rendering failed, using original frame for mesh overlays: {render_error}",
                            file=sys.stderr,
                        )
                    render_failures += 1
                add_elapsed(frame_loop_breakdown, "draw_public_overlay_frames_sec", stage_t0)

                stage_t0 = time.perf_counter()
                write_public_video_frames(writers, public_frames)
                add_elapsed(frame_loop_breakdown, "write_public_video_frames_sec", stage_t0)

                stage_t0 = time.perf_counter()
                for det_index, hand in enumerate(hands):
                    frame_indices.append(total_frames)
                    det_indices.append(det_index)
                    bbox_xyxy.append(hand["bbox"])
                    scores.append(float(hand["score"]))
                    is_right_values.append(int(hand["is_right"]))
                    cam_t_values.append(hand["cam_t"])
                    joints_3d_values.append(hand["joints_3d"])
                    if camera_coords and "joints_cam" in hand:
                        joints_cam_values.append(hand["joints_cam"])
                    joints_2d_values.append(hand["joints_2d"])
                    vertices_values.append(hand["vertices"])
                    focal_length_values.append(float(hand["focal_length"]))
                    if hand["is_right"]:
                        right_detections += 1
                    else:
                        left_detections += 1

                det_end = len(frame_indices)
                num_detections = det_end - det_start
                total_detections += num_detections
                frames_out.write(
                    json.dumps(
                        {
                            "frame_index": total_frames,
                            "source_frame_index": int(
                                manifest.get("debug_clip", {}).get("start_frame", 0)
                                + total_frames
                            ),
                            "time_sec": total_frames / fps,
                            "num_detections": num_detections,
                            "det_start": det_start,
                            "det_end": det_end,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                add_elapsed(frame_loop_breakdown, "record_outputs_sec", stage_t0)

                total_frames += 1
                if total_frames % 100 == 0:
                    print(f"processed_frames: {total_frames}")

    with timed_stage(timings, "close_video_outputs"):
        close_video_writers(writers)
        cap.release()
    frame_loop_breakdown["frames"] = int(total_frames)
    frame_loop_breakdown["detections"] = int(total_detections)
    timings["frame_loop_breakdown"] = frame_loop_breakdown

    detections_npz = results_dir / "detections.npz"
    with timed_stage(timings, "save_detections_npz"):
        detection_arrays = {
            "frame_index": np.asarray(frame_indices, dtype=np.int32),
            "det_index": np.asarray(det_indices, dtype=np.int16),
            "bbox_xyxy": (
                np.stack(bbox_xyxy).astype(np.float32)
                if bbox_xyxy
                else np.zeros((0, 4), dtype=np.float32)
            ),
            "score": np.asarray(scores, dtype=np.float32),
            "is_right": np.asarray(is_right_values, dtype=np.int8),
            "cam_t": (
                np.stack(cam_t_values).astype(np.float32)
                if cam_t_values
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "joints_3d": (
                np.stack(joints_3d_values).astype(np.float16)
                if joints_3d_values
                else np.zeros((0, 21, 3), dtype=np.float16)
            ),
            "joints_2d": (
                np.stack(joints_2d_values).astype(np.float32)
                if joints_2d_values
                else np.zeros((0, 21, 2), dtype=np.float32)
            ),
            "vertices": (
                np.stack(vertices_values).astype(np.float16)
                if vertices_values
                else np.zeros((0, 778, 3), dtype=np.float16)
            ),
            "focal_length": np.asarray(focal_length_values, dtype=np.float32),
            "frame_count": np.array(total_frames, dtype=np.int32),
            "fps": np.array(fps, dtype=np.float32),
            "width": np.array(width, dtype=np.int32),
            "height": np.array(height, dtype=np.int32),
        }
        if camera_coords:
            detection_arrays["joints_cam"] = (
                np.stack(joints_cam_values).astype(np.float32)
                if joints_cam_values
                else np.zeros((0, 21, 3), dtype=np.float32)
            )
        np.savez_compressed(detections_npz, **detection_arrays)

    output_video_keys = (
        tuple(common.WILOR_PUBLIC_VIDEO_OUTPUTS)
        if camera_coords
        else ("handmesh_overlay", "hand_skeleton_overlay")
    )
    output_videos = common.wilor_output_videos_manifest(
        sample_dir,
        output_keys=output_video_keys,
    )
    omitted_output_videos = common.wilor_omitted_output_videos_manifest(
        output_keys=output_video_keys,
    )
    timings["camera_coords_enabled"] = bool(camera_coords)
    timings["completed_at"] = common.utc_now_iso()
    timings["total_elapsed_sec"] = time.perf_counter() - total_t0
    performance_metrics = build_wilor_performance_metrics(
        timings,
        frame_count=total_frames,
        detection_count=total_detections,
        input_fps=fps,
    )
    timings["performance_metrics"] = performance_metrics
    summary = {
        "status": "complete",
        "video_path": str(workspace_video),
        "overlay_video": output_videos["hand_skeleton_overlay"]["path"],
        "output_videos": output_videos,
        "omitted_output_videos": omitted_output_videos,
        "detections_npz": str(detections_npz),
        "frames_jsonl": str(frames_jsonl),
        "frame_count": int(total_frames),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "total_detections": int(total_detections),
        "left_detections": int(left_detections),
        "right_detections": int(right_detections),
        "render_failures": int(render_failures),
        "conf": float(conf),
        "iou": float(iou),
        "rescale_factor": float(rescale_factor),
        "batch_size": int(batch_size),
        "fast": bool(fast),
        "camera_coords_enabled": bool(camera_coords),
        "device": str(device),
        "performance_metrics": performance_metrics,
        "timings": timings,
        "completed_at": timings["completed_at"],
    }
    common.write_json(results_dir / "summary.json", summary)

    manifest.update(
        {
            "status": "complete",
            "completed_at": summary["completed_at"],
            "exit_code": 0,
            "results": summary,
            "output_videos": output_videos,
            "omitted_output_videos": omitted_output_videos,
            "wilor": {
                **manifest.get("wilor", {}),
                "camera_coords": bool(camera_coords),
            },
            "timings": timings,
        }
    )
    refresh_wilor_performance_metrics(manifest)
    update_manifest_public_outputs(manifest, sample_dir, camera_coords=camera_coords)
    common.write_json(manifest_path, manifest)
    print(f"complete: {manifest['video_name']} frames={total_frames} detections={total_detections}")


def render_overlays_from_results(*, sample_dir: Path, overwrite: bool) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.chdir(common.REPO_ROOT)

    import cv2

    manifest_path = sample_dir / "manifest.json"
    manifest = common.read_json(manifest_path)
    update_manifest_public_outputs(manifest, sample_dir)
    outputs = manifest["outputs"]
    workspace_video = Path(outputs["workspace_video"])
    detections_npz = Path(outputs.get("detections_npz", sample_dir / "wilor_results" / "detections.npz"))
    frames_jsonl = Path(outputs.get("frames_jsonl", sample_dir / "wilor_results" / "frames.jsonl"))

    if not workspace_video.is_file():
        raise FileNotFoundError(f"Workspace video is missing: {workspace_video}")
    if not detections_npz.is_file():
        raise FileNotFoundError(f"WiLoR detections are missing: {detections_npz}")
    if not frames_jsonl.is_file():
        raise FileNotFoundError(f"WiLoR frame records are missing: {frames_jsonl}")

    records = read_frame_records(frames_jsonl)
    if not records:
        raise RuntimeError(f"No frame records found in {frames_jsonl}")

    cap = cv2.VideoCapture(str(workspace_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open workspace video: {workspace_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {workspace_video}")

    renderer, model_cfg = load_overlay_renderer()
    detections, detections_need_update = load_detection_arrays(
        detections_npz,
        default_focal_length=scaled_focal_length(model_cfg, width, height),
    )
    npz_frame_count = scalar_int(detections.get("frame_count"), default=len(records))
    if npz_frame_count and npz_frame_count != len(records):
        cap.release()
        raise RuntimeError(
            f"Frame count mismatch: detections.npz frame_count={npz_frame_count}, "
            f"frames.jsonl lines={len(records)}"
        )
    rewrite_detection_npz_if_needed(detections_npz, detections, detections_need_update)

    writers = prepare_public_video_writers(
        sample_dir,
        fps=fps,
        width=width,
        height=height,
        overwrite=overwrite,
    )

    total_frames = 0
    total_detections = 0
    render_failures = 0
    try:
        for frame_number, record in enumerate(records):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Workspace video ended early at frame {frame_number}; "
                    f"expected {len(records)} frames"
                )
            frame_index = int(record.get("frame_index", frame_number))
            hands = hands_from_detection_arrays(detections, record, frame_index)
            total_detections += len(hands)
            public_frames, render_error = draw_public_overlay_frames(frame, hands, renderer)
            if render_error is not None:
                if render_failures == 0:
                    print(
                        f"warning: mesh rendering failed, using original frame for mesh overlays: {render_error}",
                        file=sys.stderr,
                    )
                render_failures += 1
            write_public_video_frames(writers, public_frames)
            total_frames += 1
            if total_frames % 100 == 0:
                print(f"rendered_frames: {total_frames}")
    finally:
        close_video_writers(writers)
        cap.release()

    output_videos = common.wilor_output_videos_manifest(sample_dir)
    rendered_at = common.utc_now_iso()
    summary_path = Path(outputs.get("summary_json", sample_dir / "wilor_results" / "summary.json"))
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = common.read_json(summary_path)
    summary.update(
        {
            "status": summary.get("status", "complete"),
            "video_path": str(workspace_video),
            "overlay_video": output_videos["hand_skeleton_overlay"]["path"],
            "output_videos": output_videos,
            "omitted_output_videos": common.wilor_omitted_output_videos_manifest(),
            "detections_npz": str(detections_npz),
            "frames_jsonl": str(frames_jsonl),
            "frame_count": int(total_frames),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "overlay_render_failures": int(render_failures),
            "overlays_rendered_at": rendered_at,
        }
    )
    common.write_json(summary_path, summary)

    manifest.update(
        {
            "status": manifest.get("status", "complete"),
            "exit_code": int(manifest.get("exit_code", 0)),
            "results": summary,
            "output_videos": output_videos,
            "omitted_output_videos": common.wilor_omitted_output_videos_manifest(),
            "overlays_rendered_at": rendered_at,
        }
    )
    update_manifest_public_outputs(manifest, sample_dir)
    common.write_json(manifest_path, manifest)
    print(
        f"rendered-overlays: {manifest.get('video_name', sample_dir.name)} "
        f"frames={total_frames} detections={total_detections}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
