# Celeb-DF VGG16 DeepFake Detection System

An academic, end-to-end video deepfake detection system powered by **VGG16 Transfer Learning** on the **Celeb-DF-v1 dataset**, featuring frame-level face extraction, video-level aggregation, standalone CLI inference, a RESTful **FastAPI** backend, and an interactive modern web dashboard.

---

## 1. Project Overview

The **Celeb-DF VGG16 DeepFake Detection System** analyzes user-submitted video footage to detect facial manipulation and synthesis artifacts. Videos are sampled across time, processed through an automated face detection and normalization pipeline, passed through an ImageNet-pretrained VGG16 feature extractor with a custom classification head, and aggregated to output a single, confident **REAL** or **FAKE** classification along with confidence percentages and per-frame probability diagnostics.

### Key Highlights
- **Dataset:** Celeb-DF-v1 (1,203 MP4 videos; 408 Real, 795 Fake).
- **Zero Data Leakage:** Partitioning is strictly enforced at the **VIDEO** level (never leaking frames between train, validation, or test sets).
- **Official Benchmark Evaluation:** Direct integration with official `List_of_testing_videos.txt` (100 benchmark test videos).
- **Model:** VGG16 Base + GlobalAveragePooling2D + Dense(256) + Dropout(0.5) + Sigmoid binary output with fine-tuning support.
- **Production-Ready Backend:** High-performance asynchronous FastAPI server (`GET /health`, `POST /predict`) with automatic upload cleanup.
- **Interactive UI:** Built-in dashboard (`frontend/index.html`) with video preview, verdict badge, confidence gauge, and per-frame probability badges.

---

## 2. System Architecture

```
                 User Upload / Input Video
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │     Temporal Frame Extractor         │  ── Samples N frames (default: 10)
         │       (OpenCV VideoCapture)          │     uniformly across timeline
         └──────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │      Modular Face Detector           │  ── Detects faces, selects primary face,
         │    (Haar Cascade + Margin Pad)       │     pads margin & resizes to 224x224 RGB
         └──────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │      VGG16 Zero-Centering            │  ── ImageNet BGR mean subtraction
         │       (tf.keras.applications)        │
         └──────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │     VGG16 Transfer Model             │  ── Pretrained VGG16 (frozen/fine-tuned)
         │   (GAP + Dense + Dropout + Sigmoid)  │     outputs frame probability p in [0, 1]
         └──────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │    Frame-to-Video Pooling            │  ── Mean probability aggregation
         │   p_video = (1/N) * sum(p_i)         │     over valid face frames
         └──────────────────────────────────────┘
                            │
                            ▼
          Final Verdict: REAL / FAKE, Confidence %,
          Valid Face Count, Frame Probabilities
```

---

## 3. Dataset Information

The system is configured to work directly with the local **Celeb-DF-v1** dataset located in `Celeb-DF/`:

- **Real Videos:**
  - `Celeb-DF/Celeb-real/`: 158 videos (257.66 MB, avg 1.63 MB/video)
  - `Celeb-DF/YouTube-real/`: 250 videos (572.95 MB, avg 2.29 MB/video)
  - **Total Real:** 408 videos
- **Fake Videos:**
  - `Celeb-DF/Celeb-synthesis/`: 795 videos (1,234.69 MB, avg 1.55 MB/video)
  - **Total Fake:** 795 videos
- **Total Dataset Size:** 1,203 MP4 videos (~2.02 GB)
- **Official Test Set (`List_of_testing_videos.txt`):**
  - Exactly 100 benchmark test videos (38 Real, 62 Fake).
  - All 100 videos verified physically present in workspace.

### Dataset Splitting Strategy
- **Video-Level Separation:** All splits partition entire video files; frames from the same video NEVER appear across multiple splits.
- **Test Set:** 100 videos (38 Real, 62 Fake) directly matched to `List_of_testing_videos.txt`.
- **Validation Set:** 181 videos (61 Real, 120 Fake), representing 15% of the total dataset.
- **Training Set:** 922 videos (309 Real, 613 Fake), representing remaining videos with balanced class weights applied during training.
- **Split Artifacts:** Persisted to `data/splits/train_videos.csv`, `val_videos.csv`, and `test_videos.csv`.

---

## 4. Project Structure

```
DeepFake Detection Project/
├── AGENTS.md                  # Persistent AI development log & handoff record
├── README.md                  # Complete system documentation
├── requirements.txt           # Python dependency specification
├── .gitignore                 # Cache, model checkpoints, and venv ignores
│
├── config/
│   └── config.yaml            # Central configuration (paths, hyperparameters)
│
├── data/
│   ├── processed/faces/       # Cached extracted face arrays (.npz)
│   └── splits/                # Video-level split CSVs (train, val, test)
│
├── models/
│   ├── pretrained/            # Cached VGG16 ImageNet weights (offline mode)
│   └── trained_model/         # Best trained Keras model checkpoints
│
├── outputs/
│   ├── plots/                 # Training curves and confusion matrix heatmaps
│   └── reports/               # Evaluation metrics JSON and training history
│
├── src/
│   ├── preprocessing/
│   │   ├── face_detector.py   # Modular face detection & margin cropping
│   │   ├── frame_extractor.py # Video sampling & OpenCV extraction
│   │   └── video_processor.py # High-level video pipeline orchestrator
│   ├── dataset/
│   │   └── dataset_loader.py  # Split generator & tf.keras data sequence
│   ├── models/
│   │   ├── vgg16_model.py     # VGG16 architecture & fine-tuning layer manager
│   │   └── model_utils.py     # Serialization, saving, and loading
│   ├── training/
│   │   ├── train.py           # Training loop with callbacks & history plotting
│   │   └── evaluate.py        # Video-level and frame-level test evaluation
│   ├── inference/
│   │   └── predict.py         # Standalone single-video CLI prediction engine
│   └── utils/
│       ├── config.py          # Configuration parser & validator
│       └── logging_utils.py   # Structured logging configuration
│
├── backend/
│   └── app.py                 # FastAPI application (GET /health, POST /predict)
│
└── frontend/
    └── index.html             # Responsive Tailwind CSS web interface
```

---

## 5. Installation & Setup

### Operating System Support
Tested on macOS (Apple Silicon arm64 / Intel) and Linux.

### Step 1: Create and Activate Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Commands Reference

### 1. Generate Video-Level Splits
Discovers all dataset videos and generates zero-leakage splits:
```bash
python -m src.dataset.dataset_loader --create-splits
```

### 2. Test Preprocessing Pipeline
Extract faces and inspect metadata on a sample video:
```bash
python -m src.preprocessing.video_processor --video "Celeb-DF/Celeb-real/id0_0000.mp4" --frames 10
```

### 3. Train the Model
Run full training with configuration from `config/config.yaml`:
```bash
python -m src.training.train --config config/config.yaml
```

Run a quick controlled smoke test (ideal for CPU-only testing):
```bash
python -m src.training.train --smoke-test --epochs 2
```

### 4. Evaluate the Model
Evaluate video-level and frame-level metrics on the official test set:
```bash
python -m src.training.evaluate --config config/config.yaml
```

Evaluate on a subset of test videos:
```bash
python -m src.training.evaluate --max-videos 20
```

### 5. Run Single Video Inference
Run CLI prediction on any video file:
```bash
python src/inference/predict.py --video "Celeb-DF/Celeb-synthesis/id0_id16_0000.mp4"
```

Output structured JSON for script parsing:
```bash
python src/inference/predict.py --video "Celeb-DF/Celeb-real/id0_0000.mp4" --json
```

### 6. Start the FastAPI Backend
Start the high-performance uvicorn server:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Backend REST API Contract

### Health Check: `GET /health`
Verifies backend operational status and whether model weights are loaded in RAM.
- **Request:** `GET http://localhost:8000/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_path": "models/trained_model/vgg16_deepfake_detector.keras"
  }
  ```

### DeepFake Prediction: `POST /predict`
Uploads a video and returns full inference diagnostics.
- **Request:** `POST http://localhost:8000/predict`
  - `Content-Type`: `multipart/form-data`
  - Form field `video`: Binary video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
- **cURL Example:**
  ```bash
  curl -X POST http://localhost:8000/predict \
    -F "video=@Celeb-DF/Celeb-synthesis/id0_id16_0000.mp4"
  ```
- **Response (200 OK):**
  ```json
  {
    "prediction": "REAL",
    "confidence": 86.92,
    "frames_analyzed": 10,
    "valid_frames": 10,
    "average_probability": 0.1308,
    "frame_probabilities": [0.0478, 0.0998, 0.0954, 0.2182, 0.1126, 0.0704, 0.1562, 0.3031, 0.0784, 0.1262]
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Unsupported file extension.
  - `422 Unprocessable Entity`: Corrupted video file or processing failure.
  - `503 Service Unavailable`: Model weights not yet trained or loaded.

---

## 8. Web Frontend Dashboard

The project includes an interactive web dashboard in `frontend/index.html`.
When the FastAPI backend is running, open `http://localhost:8000/` in any modern web browser to access:
- Drag-and-drop video upload with real-time video player preview.
- Dynamic verdict banner (`SYNTHETIC / DEEPFAKE` or `AUTHENTIC / REAL`).
- Animated confidence score gauge.
- Breakdown of sampled frames, valid faces detected, and average probability.
- Per-frame probability badges showing individual frame scores.

---

## 9. Performance & Reproducibility

- **Face Extraction Caching:** Extracted face crops are compressed and cached in `data/processed/faces/` as `.npz` files, eliminating redundant OpenCV extraction across training epochs.
- **Random Seed:** Set deterministically (`seed=42`) across NumPy, TensorFlow, and Scikit-Learn.
- **Offline Pretrained Weights:** The ImageNet pretrained VGG16 base weights are cached locally in `models/pretrained/`, enabling 100% offline model initialization.
- **Resource Recommendations:**
  - Training on CPU: Feasible for smoke tests and small subsets (2-5 minutes per epoch on 20 videos).
  - Full dataset training (922 videos, 10 frames each = ~9,200 face images): Recommended to run on a CUDA-enabled GPU (NVIDIA RTX 3080/4090 or T4/A100) or Apple Silicon with MPS.

---

## 10. Limitations & Future Work

1. **Haar Cascade Sensitivity:** Haar cascades perform best on frontal faces with good lighting. Under extreme poses or occlusions, face detection may yield fewer valid frames. A future enhancement could integrate MediaPipe or MTCNN for enhanced pose tolerance.
2. **Audio Stream Analysis:** The current pipeline focuses exclusively on visual facial artifacts. Future iterations can incorporate audio deepfake detection to detect synthetic voice cloning.
3. **Temporal Sequencing:** While mean probability pooling across 10 frames is effective and robust, future extensions could add an LSTM or Transformer head to capture inter-frame temporal blinks and inconsistencies.
