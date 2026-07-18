# LabelMe 5.9.1-Yolo-DWPose

[Français](README.fr.md) · [Integration interfaces](INTERFACES.md)

LabelMe for pose annotation — automatic detection (YOLO11 + DWPose), optional rotation for non-standard poses, manual correction, ControlNet-ready export.

![Annotation example — standard pose](docs/images/annotation_standard.jpg)

---

## Use Cases

- **ControlNet OpenPose datasets for ComfyUI** — generate image/skeleton pairs to train or fine-tune a ControlNet, with manual correction on poses missed by generic models
- **Animation / retargeting pipelines** — extract the 2D skeleton to drive a 3D mannequin or avatar, alongside a ComfyUI + AnimateDiff workflow or other animation pipeline
- **Sport / dance / martial arts** — annotate sequences with non-standard poses (falls, throws, technical movements) that generic pose models miss, to enrich specialized ControlNet datasets

---

## Official LabelMe vs this fork

| Feature | LabelMe 5.9.1 (official) | LabelMe-Yolo-DWPose |
|---|---|---|
| Manual annotation (polygons, boxes, points) | ✅ | ✅ |
| Automatic detection (YOLO11) | ❌ | ✅ Boxes, segmentation, 17-pt skeleton |
| Full-body pose estimation (DWPose) | ❌ | ✅ 133/134 pts |
| Handles inverted/tilted subjects | ❌ | ✅ Pre-inference rotation |
| Targeted re-inference from a box | ❌ | ✅ |
| Bulk keypoint add/remove | ❌ | ✅ Rubber-band, right-click |
| Direct OpenPose export (PNG + JSON) | ❌ | ✅ Batch mode |
| Native video processing | ❌ | ✅ Kalman tracking + mp4 preview |
| Scriptable pipeline (mode, skeleton, rotation, fps…) | ❌ | ✅ |

---

## Features

- **Automatic detection** — YOLO11 for bounding boxes, segmentation masks (SAM2 compatible), and 17-pt COCO skeletons
- **DWPose integration** — 133-pt COCO-WholeBody or 134-pt OpenPose full-body pose estimation
- **Flexible skeleton groups** — body, feet, face, hands — enable only what you need
- **Pre-inference rotation** — 0° / 90°L / 180° / 90°R to handle inverted or tilted poses
- **OpenPose PNG export** — black background rainbow skeleton + silhouette mask, batch mode available
- **Re-inference from boxes** — redraw a box and re-run DWPose on a single image without re-running the full pipeline
- **Manual keypoint correction** — several tools available:
  - Right-click on a skeleton → **Add missing keypoint** — activates a keypoint that wasn't detected (was at [0,0]) and places it at the skeleton's center; you then drag it into position (can be tricky to spot amid a dense cluster of hand/face points)
  - Right-click directly on a keypoint → **Remove** — sets it back to [0,0]
  - **Ctrl+Alt+drag** (rubber-band selection) — clears every keypoint inside the selected area, regardless of group. Built to avoid deleting points one by one on dense groups (hands: 21 pts, face: ~70 pts); for feet (3 pts), correcting individually is fine
- **Presets** — save and restore your favourite inference settings

---

## Screenshots

### Settings dialog
![Settings dialog](docs/images/settings_dialog.jpg)

### Edit menu
![Edit menu](docs/images/edit_menu.jpg)

### Standard pose — 0° (Muhammad Numan / Unsplash)
![Standard pose](docs/images/annotation_standard.jpg)

### Inverted pose — 180° rotation (Pooja Shah / Unsplash)

The model expects an upright subject — head up, feet down. Pre-inference rotation temporarily presents the subject that way so DWPose can detect it correctly, then the pipeline maps the result back to the original image. The screenshots below show the raw output, with no manual correction applied.

![Inverted pose 180°](docs/images/annotation_180_yoga.jpg)

### Tilted pose — 90°L rotation (Devin Santiago / Unsplash)

The dancer's body leans heavily to the right. DWPose, trained on upright subjects, fails to correctly map the skeleton on the original frame. By rotating the image 90°L (counterclockwise), the body is presented roughly upright to the model — compensating for the lean and allowing correct keypoint placement. The pipeline then rotates the result back to the original coordinates.

![Tilted pose 90°L](docs/images/annotation_90L_dance.jpg)

### Extreme pose — 180° (Lorenzo Fatto Offidani / Unsplash)
> ⚠️ Complex one-arm handstand: right arm misplaced — illustrates model limits on extreme poses.

![Extreme pose](docs/images/annotation_180_break.jpg)

### Aerial pose — 180° (Seyi Ariyo / Unsplash)
![Aerial pose](docs/images/annotation_180_aerial.jpg)

---

## Requirements

| Platform | GPU | Notes |
|----------|-----|-------|
| Windows / Linux | NVIDIA GPU (recommended) | Auto-detected during install |
| Windows / Linux | No GPU | CPU fallback — inference will be slow |

> Driver requirements: CUDA 11.8 (driver ≥ 452), CUDA 12.1 (driver ≥ 526), CUDA 12.6 (driver ≥ 536).
> The installer detects your driver automatically and installs the correct PyTorch version.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/labelme-yolo-dwpose
cd labelme-yolo-dwpose
```

**Windows:**
```bat
install.bat
```

**Linux:**
```bash
bash install.sh
```

The installer will:
1. Download and install Miniconda (isolated — does not affect your system Python)
2. Create a dedicated Python 3.10 environment
3. Detect your GPU and install the correct PyTorch/CUDA version
4. Install all dependencies
5. Download YOLO11 and DWPose models

☕ This takes a few minutes — grab a coffee.

---

## Launch

**Windows:**
```bat
launch.bat
```

**Linux:**
```bash
bash launch.sh
```

---

## Workflow

### Why use rotation before inference?

DWPose was trained on upright subjects. When a person is upside-down (handstand, headstand) or significantly tilted, the model struggles to correctly identify body parts — it always starts from the head and works downward.

By rotating the image before inference (e.g. 180° for a headstand), you present the subject upright to DWPose, which dramatically improves keypoint placement. The pipeline automatically rotates the result back to the original image coordinates.

> **Rule of thumb:** if the head is at the bottom of the frame, use 180°. If the subject is leaning heavily to one side, try 90°L or 90°R.

### Video processing

The pipeline supports video files in addition to images and folders:

`Edit → YOLO11 — Fichier` — works on `.mp4`, `.avi`, `.mov` and other common formats

The **FPS** setting controls the **export sampling rate** — not the processing speed. Every frame is processed internally (so the Kalman tracker maintains consistent person IDs throughout), but only 1 frame out of every N is saved to disk, targeting the chosen FPS. For example:
- `1.0` = 1 annotated frame saved per second of video
- `12.0` = 12 annotated frames saved per second of video

Frames are extracted, annotated and saved as individual images + JSON files in the `processed/` folder.

A **preview video** (`_annotated.mp4`) is also generated alongside the processed frames — so you can quickly scrub through the full video to visually check what was captured before opening individual frames in LabelMe.

---

## Command Line Interface

The pipeline can also be run directly from the command line (inside the conda environment):

```bash
# Activate the environment first
# Windows:
miniconda\envs\labelme-env\Scripts\activate
# Linux:
source miniconda/envs/labelme-env/bin/activate

# Basic usage
python -m yolo11.main path/to/video.mp4
python -m yolo11.main path/to/image.jpg
python -m yolo11.main path/to/folder/

# DWPose with OpenPose format, body + hands
python -m yolo11.main video.mp4 --mode dwpose --skeleton openpose --groups body,hands

# Full body with segmentation, 12 fps sampling
python -m yolo11.main video.mp4 --mode dwpose --groups body,feet,hands,face --tasks boxes,segmentation,skeletons --fps 12

# With rotation (useful for inverted subjects)
python -m yolo11.main video.mp4 --mode dwpose --groups body,hands --rotation 180
```

| Argument | Description |
|----------|-------------|
| `--tasks` | `boxes`, `segmentation`, `skeletons` (comma-separated) |
| `--mode` | `yolo` (17 pts) or `dwpose` (133/134 pts) |
| `--skeleton` | `openpose` (134 pts) or `coco` (133 pts) — DWPose only |
| `--groups` | `body`, `feet`, `face`, `hands` (comma-separated) — DWPose only |
| `--fps` | Export sampling rate (default: from config.yaml) |
| `--min-conf` | Keypoint confidence threshold |
| `--nms-iou` | NMS IoU threshold for person box deduplication |
| `--rotation` | Pre-inference rotation: `0`, `90`, `180`, `270` |
| `--output` | Save a JSON summary to this file (optional) |

### 1. Run the pipeline on a folder or image

`Edit → YOLO11 — Fichier` or `Edit → YOLO11 — Dossier`

This generates annotated images and JSON files in a `processed/` subfolder.

### 2. Open the processed folder in LabelMe

`File → Open Dir → processed/`

### 3. Review and correct annotations

- **Right-click on a skeleton** → **Add missing keypoint** — adds a body or foot keypoint that was not detected (was at [0,0]); it is placed at the skeleton center so you can drag it to the correct position
- **Right-click directly on a keypoint** → **Remove** — sets it back to [0,0]; useful for badly placed face or hand points
- **Ctrl+Alt+drag** — rubber-band selection to select and delete a group of keypoints at once (e.g. wipe out all unreliable face points in one move)
- **Re-inference** → `Edit → Réinférer DWPose depuis les boxes…` — useful when the auto result is wrong; respects the current rotation setting
- Use **rotation** (90°L / 180° / 90°R) before re-inferring on inverted or tilted subjects

### 4. Export OpenPose PNG

`Edit → Exporter PNG OpenPose…` — single image
`Edit → Exporter dataset OpenPose — dossier…` — batch mode

Generates a `PNG-{stem}.png` (black background, rainbow skeleton + silhouette) and a matching `PNG-{stem}.json` for each image.

---

## Settings

Open `Edit → Réglages YOLO11…` to configure:

| Setting | Description |
|---------|-------------|
| **Tasks** | Boxes / Segmentation / Skeletons |
| **Mode** | YOLO (17 pts) or DWPose (133/134 pts) |
| **Format** | OpenPose (134 pts) or COCO (133 pts) |
| **Groups** | body / feet / face / hands |
| **Min confidence** | Keypoint confidence threshold |
| **FPS** | Sampling rate for video processing |
| **Rotation** | Pre-inference rotation for non-upright subjects |

---

## Known Limitations

- **Extreme poses** (handstands, deep backbends, crossed limbs) — DWPose may misplace keypoints; compensate with rotation + re-inference to improve results (e.g. subject leaning right → rotate 90°L to present them upright to the model)
- **Isolated body parts** (legs/feet only, hands only) — DWPose requires a full-body crop; partial crops yield poor results. Place keypoints manually via right-click.
- **Face keypoints** — unreliable on small faces, profile views, or low resolution crops
- **GTX 1060 / Pascal** — tested and working on Windows (author's setup); not tested on Linux yet. Newer GPUs (RTX, etc.) should work but may need CUDA config adjustments for their architecture
- **Multi-person tracking** — ID swaps theoretically possible below ~20 FPS of the source video (threshold untested empirically); best-effort only
- **No dedicated CLI launcher** — unlike `launch.bat`/`launch.sh` for the GUI, the YOLO pipeline (deliberately detachable from LabelMe so it can be driven standalone via CLI) requires manually activating the conda environment before each use

---

## Recommended Workflow

The intended workflow is: **YOLO pipeline first → processed folder → LabelMe for correction**.

Opening an image directly in LabelMe without going through the pipeline works for drawing boxes and re-inferring, but LabelMe may prompt for a save path if it's the first annotation.

---

## Credits

- [LabelMe](https://github.com/labelmeai/labelme) — base annotation tool
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — detection, segmentation, pose
- [controlnet-dwpose](https://github.com/nateraw/controlnet-dwpose) — DWPose ONNX backend
- [DWPose](https://github.com/IDEA-Research/DWPose) — whole-body pose estimation

Sample images: [Unsplash](https://unsplash.com) — Muhammad Numan, Pooja Shah, Devin Santiago, Lorenzo Fatto Offidani, Seyi Ariyo (Unsplash License)

---

## License

This project is a fork of LabelMe and integrates YOLO11 (Ultralytics) and PyQt5. Due to the AGPL-3.0 license of these dependencies, this project is distributed under **GNU GPL v3**.

See [LICENSE](LICENSE) for details.

> If you intend to use this project as part of a closed-source or commercial product, you will need a commercial license from [Ultralytics](https://ultralytics.com/license).
