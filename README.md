# Ballet Pose Classifier — Static Module

A machine learning pipeline that classifies ballet positions from a single image using pose estimation and joint angle features.

## How It Works

```
Image → MediaPipe Pose → 33 3D Keypoints → Joint Angle Normalization → ML Classifier → Ballet Position Label
```

Instead of training on raw pixels, the model uses **joint angles** derived from skeletal keypoints. This makes the classifier:
- **Rotation-invariant** — works from multiple camera angles
- **Scale-invariant** — works regardless of dancer's distance from camera
- **Body-type agnostic** — not biased toward a specific dancer's proportions

## Supported Positions

| Class | Label |
|---|---|
| First Position | `first_position` |
| Second Position | `second_position` |
| Third Position | `third_position` |
| Fourth Position | `fourth_position` |
| Fifth Position | `fifth_position` |
| Arabesque | `arabesque` |
| Attitude | `attitude` |
| Tendu | `tendu` |

## Project Structure

```
ballet-pose-classifier/
├── src/
│   ├── extraction/
│   │   ├── pose_extractor.py      # MediaPipe keypoint extraction
│   │   └── image_loader.py        # Image loading + validation
│   ├── preprocessing/
│   │   ├── normalizer.py          # Keypoints → joint angles
│   │   ├── augmentor.py           # Data augmentation
│   │   └── dataset_builder.py     # Build train/val/test splits
│   ├── models/
│   │   ├── classifier.py          # RF, SVM, MLP wrappers
│   │   └── train.py               # Training loop + MLflow logging
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, per-class F1, confusion matrix
│   │   └── visualize_results.py   # Plot confusion matrix and keypoints
│   └── utils/
│       ├── config.py              # Load YAML configs
│       └── helpers.py             # Shared utility functions
├── data/
│   ├── raw_images/                # One subfolder per class (gitignored)
│   │   ├── first_position/
│   │   ├── second_position/
│   │   └── ...
│   ├── keypoints/                 # Extracted CSVs (gitignored)
│   ├── labels/                    # labels.csv — committed to git
│   ├── augmented/                 # Augmented images (gitignored)
│   └── splits/                    # train/val/test CSVs (gitignored)
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   └── data_config.yaml           # Paths, class names, split ratios
├── scripts/
│   ├── collect_data.py            # Download frames from YouTube videos
│   ├── build_dataset.py           # Run extraction + build splits
│   ├── run_training.py            # Train all models + log to MLflow
│   └── predict.py                 # Run inference on a single image
├── demo/
│   └── app.py                     # Streamlit web demo
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   ├── test_extractor.py
│   ├── test_normalizer.py
│   └── test_classifier.py
├── mlruns/                        # MLflow experiment runs (gitignored)
├── requirements.txt
└── README.md

## Quickstart

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Collect data
Add images to `data/raw_images/<class_name>/` — at least 100 images per class.
Images should be JPG or PNG. Subfolders act as class labels automatically.

### 3. Build dataset
```bash
python scripts/build_dataset.py
```
This runs pose extraction on every image and builds train/val/test splits.

### 4. Train
```bash
python scripts/run_training.py
```
View experiment results:
```bash
mlflow ui
```

### 5. Predict
```bash
python scripts/predict.py --image path/to/your/image.jpg
```

### 6. Run demo
```bash
streamlit run demo/app.py
```

## Data Collection Tips

- **YouTube**: Search "ballet positions tutorial" and screenshot clean holds
- **yt-dlp**: `yt-dlp -x --write-thumbnail <url>` to grab frames
- **Angle variety**: Capture front, side (90°), and diagonal (45°) views for each position
- **Target**: 150-300 images per class minimum

## Results Tracking

All training runs are logged with MLflow. Models are compared on:
- Overall accuracy
- Per-class F1 score
- Confusion matrix

## Tech Stack

- **MediaPipe** — pose estimation
- **scikit-learn** — Random Forest, SVM, Gradient Boosting
- **TensorFlow/Keras** — MLP
- **MLflow** — experiment tracking
- **Streamlit** — demo UI
