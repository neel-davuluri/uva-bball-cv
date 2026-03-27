# uva-bball-cv
# UVA Basketball — Defensive Tracking from Broadcast Film

Computer vision pipeline that extracts player positions from broadcast game footage and generates real-time defensive analytics. Built as a proof of concept for automating the kind of possession-level spatial analysis that currently requires manual film coding.

## Demo
The demo shows a UVA defensive possession against Miami with player tracking and defensive gap metrics updating in real time alongside the broadcast footage.

## Pipeline
1. **Frame extraction** — clips broadcast footage into individual frames
2. **Court homography** — maps camera perspective to real-world court coordinates using manually annotated paint box corners
3. **Player detection** — YOLOv8 + ByteTrack for detection and tracking across frames
4. **Team classification** — SigLIP embeddings + UMAP + K-means for jersey-based team assignment
5. **Analytics + visualization** — nearest defender gap index, defensive coverage map, paint presence tracking

## Notebooks
| Notebook | Description |
|---|---|
| `01_frame_extraction.ipynb` | Extract and audit frames from broadcast clips |
| `02_homography_final.ipynb` | Annotate court keypoints and compute homography matrix |
| `03_player_detection.ipynb` | Run YOLO + ByteTrack, map positions to court coordinates |
| `04_analytics_demo_v2.ipynb` | Team classification, metrics, animated demo render |

## Stack
Python, OpenCV, YOLOv8, ByteTrack, Supervision, SigLIP, UMAP, Matplotlib, scipy

## Limitations & Next Steps
- Homography accuracy is limited by the 4-point annotation approach — overhead or fixed cameras would significantly improve coordinate precision
- Team classification works best with high-contrast jersey colors; darker uniforms require tuning
- With access to practice film or tagged Synergy clips, this pipeline could scale to full-game automated analysis
