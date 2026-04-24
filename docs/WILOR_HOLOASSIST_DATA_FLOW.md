# WiLoR HoloAssist Video Data Flow

This pipeline runs WiLoR on already-extracted HoloAssist videos and writes a
per-video overlay plus structured per-detection outputs. It is intentionally
not a Dyn-HaMR or HaMeR replica: it does not do ego-hand filtering, tracking,
smoothing, root optimization, or multi-view rendering.

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

The annotation file is read from:

```text
data-annotation-trainval-v1_1.json
```

When `--video_name` is omitted, batch selection follows annotation order and
skips missing inputs. Explicit `--video_name` requests fail on missing files
unless `--skip_missing` is set.

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

Useful options:

```text
--dataset_root      Default HoloAssist_extracted path
--output_root       Default /share/project/RoboBrain-World-dataset/HoloAssist-wilor-video
--video_name        Repeatable HoloAssist video directory name
--limit             Limit selected processable videos
--start_frame       Debug clip start frame
--max_frames        Debug clip frame count
--overwrite         Remove and rebuild this sample output
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
  handpose_skeleton_overlay.mp4
  holoassist_hands_qa.npz        # only with --export_hands
```

`detections.npz` contains:

```text
frame_index[D] int32
det_index[D] int16
bbox_xyxy[D,4] float32
score[D] float32
is_right[D] int8                # 0=left, 1=right
cam_t[D,3] float32
joints_3d[D,21,3] float16
joints_2d[D,21,2] float32
vertices[D,778,3] float16
frame_count int32
fps float32
width int32
height int32
```

`frames.jsonl` has one line per processed frame:

```json
{"frame_index": 0, "source_frame_index": 0, "time_sec": 0.0, "num_detections": 2, "det_start": 0, "det_end": 2}
```

`holoassist_hands_qa.npz` stores nearest-neighbor resampled HoloAssist left and
right hand sidecars for the processed frame times. It is for QA only and is not
used to filter WiLoR detections.

## Frame Processing

For each decoded frame:

1. Run the WiLoR YOLO detector and keep every hand detection.
2. Build WiLoR crops with `ViTDetDataset`.
3. Batch all detected hands through WiLoR.
4. Restore left/right hand orientation, compute full-image camera translation,
   and project the 21 predicted joints to image coordinates.
5. Render all meshes with `Renderer.render_rgba_multiple`.
6. Draw bbox, left/right label, confidence, and 21-joint skeleton.
7. Write one overlay video frame and one `frames.jsonl` record.

Frames with no detections are still written to the overlay video and recorded
with `num_detections=0`.

## Verified Locally

On this machine:

```text
annotations: 1758
video_dirs: 2221
processable_dirs: 2111
```

Smoke checks:

```text
R035-12July-Nespresso, frames=30, detections=0
R007-7July-DSLR, frames=5, detections=10
```

Both runs produced readable overlay MP4 files, loadable `detections.npz`, and
`frames.jsonl` line counts matching the processed frame count.
