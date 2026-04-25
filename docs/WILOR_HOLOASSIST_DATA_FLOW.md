# WiLoR HoloAssist Video Data Flow

This pipeline runs WiLoR on already-extracted HoloAssist videos and writes
WiLoR-fitted hand mesh, skeleton, and camera-coordinate QA videos plus
structured per-detection outputs. It is intentionally not a Dyn-HaMR or HaMeR
replica: it does not do ego-hand filtering, tracking, smoothing, root
optimization, identity locking, or multi-view rendering.

## Inputs

Default dataset root:

```bash
/share/project/RoboBrain-World-dataset/HoloAssist/HoloAssist_extracted
```

Each processable video directory must contain:

```text
<video_name>/Export_py/Video_compress.mp4
<video_name>/Export_py/Video/Pose_sync.txt
<video_name>/Export_py/Video/Intrinsics.txt
<video_name>/Export_py/Hands/Left_sync.txt
<video_name>/Export_py/Hands/Right_sync.txt
```

`Pose_sync.txt`, `Intrinsics.txt`, and HoloAssist hands are required for input
completeness and optional QA export only. WiLoR detections are not filtered by
HoloAssist hands.

## Commands

Scan availability:

```bash
cd /share/project/husicheng/WiLoR
./.venv/bin/python scripts/holoassist_wilor_video_batch.py scan --limit 20
```

Prepare one debug sample:

```bash
./.venv/bin/python scripts/holoassist_wilor_video_batch.py prepare \
  --video_name R007-7July-DSLR \
  --max_frames 30 \
  --overwrite \
  --export_hands
```

Run WiLoR on one sample:

```bash
./.venv/bin/python scripts/holoassist_wilor_video_batch.py run \
  --video_name R035-12July-Nespresso \
  --start_frame 0 \
  --max_frames 30 \
  --overwrite \
  --export_hands
```

Rebuild public videos from existing results without detector/model inference:

```bash
./.venv/bin/python scripts/holoassist_wilor_video_batch.py render-overlays \
  --sample_dir /share/project/RoboBrain-World-dataset/HoloAssist-wilor-video/R035-12July-Nespresso \
  --overwrite
```

Useful options:

```text
--dataset_root      Default HoloAssist_extracted path
--output_root       Default /share/project/RoboBrain-World-dataset/HoloAssist-wilor-video
--video_name        Repeatable HoloAssist video directory name
--limit             Limit selected processable videos
--start_frame       Debug clip start frame
--max_frames        Debug clip frame count
--overwrite         Remove and rebuild this sample output or public videos
--skip_missing      Skip explicit missing video names
--copy_video        Copy source MP4 instead of symlinking
--export_hands      Export resampled HoloAssist hands QA
--conf              YOLO hand detector confidence, default 0.3
--iou               YOLO hand detector IoU, default 0.5
--rescale_factor    WiLoR crop rescale factor, default 2.0
--batch_size        WiLoR hand crop batch size, default 16
--no_fast           Disable default WiLoR fast mode
```

## Runtime

`prepare` writes a per-sample `run_wilor_video.sh` that pins the WiLoR runtime:

```text
VIRTUAL_ENV=/share/project/husicheng/WiLoR/.venv
PYTHONNOUSERSITE=1
PYOPENGL_PLATFORM=egl
MESA_GL_VERSION_OVERRIDE=4.1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
```

The WiLoR checkpoint, config, detector, and MANO data are read from this repo:

```text
pretrained_models/wilor_final.ckpt
pretrained_models/model_config.yaml
pretrained_models/detector.pt
mano_data/
```

## Outputs

Default output root:

```bash
/share/project/RoboBrain-World-dataset/HoloAssist-wilor-video
```

Per sample:

```text
<output_root>/<seq_name>/
  wilor_workspace/videos/<seq_name>.mp4
  wilor_results/
    detections.npz
    frames.jsonl
    summary.json
  manifest.json
  run_wilor_video.sh
  handmesh_overlay.mp4
  hand_skeleton_overlay.mp4
  handmesh_fitted_camera_coords_overlay.mp4
  hand_skeleton_fitted_camera_coords_overlay.mp4
  holoassist_hands_qa.npz        # only with --export_hands
```

Four public videos are recorded in `manifest.output_videos`:

| Key | File | Content |
| --- | --- | --- |
| `handmesh_overlay` | `handmesh_overlay.mp4` | WiLoR fitted mesh only |
| `hand_skeleton_overlay` | `hand_skeleton_overlay.mp4` | WiLoR fitted 21-joint skeleton only |
| `handmesh_fitted_camera_coords_overlay` | `handmesh_fitted_camera_coords_overlay.mp4` | mesh + fitted skeleton + fitted camera-coordinate labels |
| `hand_skeleton_fitted_camera_coords_overlay` | `hand_skeleton_fitted_camera_coords_overlay.mp4` | fitted skeleton + fitted camera-coordinate labels |

`outputs.overlay_video` is kept only as a compatibility field and points to
`hand_skeleton_overlay.mp4`. The old `handpose_skeleton_overlay.mp4` is not a
main output.

WiLoR does not generate Dyn-HaMR-style observed videos. `manifest` records
`omitted_output_videos` for:

```text
handmesh_observed_camera_coords_overlay
hand_skeleton_observed_camera_coords_overlay
```

## Detection Arrays

`detections.npz` contains:

```text
frame_index[D] int32
det_index[D] int16
bbox_xyxy[D,4] float32
score[D] float32
is_right[D] int8                # 0=left, 1=right
cam_t[D,3] float32
joints_3d[D,21,3] float16
joints_cam[D,21,3] float32      # WiLoR fitted camera coords: joints_3d + cam_t
joints_2d[D,21,2] float32
vertices[D,778,3] float16
focal_length[D] float32
frame_count int32
fps float32
width int32
height int32
```

The camera-coordinate labels use WiLoR fitted camera coordinates:

```text
joints_cam = joints_3d + cam_t
```

These coordinates match the WiLoR mesh rendering and full-image 2D projection
used by this pipeline. They are not HoloAssist world coordinates, device-camera
coordinates, or HoloAssist hand matrices.

`frames.jsonl` has one line per processed frame:

```json
{"frame_index": 0, "source_frame_index": 0, "time_sec": 0.0, "num_detections": 2, "det_start": 0, "det_end": 2}
```

`holoassist_hands_qa.npz` stores nearest-neighbor resampled HoloAssist left and
right hand sidecars for the processed frame times. It is for QA only.

## Frame Processing

For each decoded frame in `infer-one`:

1. Run the WiLoR YOLO detector and keep every valid hand detection.
2. Build WiLoR crops with `ViTDetDataset`.
3. Batch all detected hands through WiLoR.
4. Restore left/right hand orientation, compute full-image camera translation,
   project the 21 predicted joints to image coordinates, and compute
   `joints_cam`.
5. Render the mesh once with `Renderer.render_rgba_multiple`.
6. Write four public videos. Frames without detections write the original
   frame to all four videos.
7. Write one `frames.jsonl` record.

If mesh rendering fails for a frame, mesh videos fall back to the original frame
and `render_failures` is incremented. Skeleton videos are still written from the
WiLoR fitted 2D joints. Coordinate labels are drawn only when both the 2D point
and `joints_cam` are finite and `z > 0`.

## Completed Sample Migration

For already completed full samples, run postprocess rendering only:

```bash
for sample in R007-7July-DSLR R029-12July-DSLR R035-12July-Nespresso; do
  ./.venv/bin/python scripts/holoassist_wilor_video_batch.py render-overlays \
    --sample_dir "/share/project/RoboBrain-World-dataset/HoloAssist-wilor-video/${sample}" \
    --overwrite
done
```

This reads existing `detections.npz`, `frames.jsonl`, and workspace video. It
does not run the detector or WiLoR model inference. Old `detections.npz` files
are extended with `joints_cam` and `focal_length` when those fields are absent.

## Verification

Static check:

```bash
./.venv/bin/python -m py_compile \
  scripts/holoassist_wilor_video_batch.py \
  scripts/wilor_video_pipeline_common.py
```

Video and manifest check:

```bash
./.venv/bin/python - <<'PY'
import cv2, json, numpy as np
from pathlib import Path

root = Path("/share/project/RoboBrain-World-dataset/HoloAssist-wilor-video/R035-12July-Nespresso")
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "complete"
assert set(manifest["output_videos"]) == {
    "handmesh_overlay",
    "hand_skeleton_overlay",
    "handmesh_fitted_camera_coords_overlay",
    "hand_skeleton_fitted_camera_coords_overlay",
}
assert not any("observed" in key for key in manifest["output_videos"])

for item in manifest["output_videos"].values():
    p = Path(item["path"])
    cap = cv2.VideoCapture(str(p))
    print(p.name, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

data = np.load(root / "wilor_results" / "detections.npz")
assert {"joints_3d", "joints_cam", "joints_2d", "cam_t", "vertices", "focal_length"} <= set(data.files)
PY
```

## Constraints

- Full videos are much slower than smoke clips because mesh rendering is per
  frame with detections.
- Current output is fitted-only. HoloAssist hands remain QA sidecars and are
  not observed labels.
- The pipeline does not lock identity to the recording wearer’s hands; all
  WiLoR hand detections are exported.
