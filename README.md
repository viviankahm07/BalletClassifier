# Ballet Pose Classifier

A machine learning pipeline that classifies ballet positions from a single image using pose estimation and joint angle features.

## How It Works

```
Image → MediaPipe Pose → 33 3D Keypoints → Joint Angle Features → ML Classifier → Ballet Position Label
```

Instead of training on raw pixels, the model uses **joint angles** derived from skeletal keypoints. This makes the classifier:
- **Rotation-invariant** — works from multiple camera angles
- **Scale-invariant** — works regardless of dancer's distance from camera
- **Side-invariant** — left arabesque and right arabesque are treated as the same pose

## Supported Positions

| Category | Classes |
|---|---|
| Positions | `first_position`, `second_position`, `third_position`, `fourth_position`, `fifth_position` |
| One-legged | `arabesque`, `attitude_derriere`, `attitude_devant`, `passe`, `penche` |
| Tendu | `tendu_devant`, `tendu_a_la_seconde`, `tendu_derriere` |
| Dégagé | `degage_devant`, `degage_a_la_seconde`, `degage_derriere` |
| Two-legged | `demi_plie`, `grand_plie`, `fondu`, `releve`, `saute` |

## Project Structure

```
BalletClassifier/
├── src/
│   ├── extraction/
│   │   └── pose_extractor.py      # MediaPipe keypoint extraction
│   ├── preprocessing/
│   │   ├── normalizer.py          # Keypoints → joint angle features
│   │   └── dataset_builder.py     # Build train/val/test splits
│   ├── models/
│   │   ├── classifier.py          # RF, SVM, Gradient Boosting wrappers
│   │   └── grouped_classifier.py  # Feature-group-aware classifier
│   └── utils/
│       ├── config.py              # Load YAML configs
│       └── feature_groups.py      # Which features each pose group uses
├── data/
│   ├── raw_images/                # One subfolder per class (gitignored)
│   └── splits/                    # train/val/test CSVs (gitignored)
├── app.py                         # Streamlit web demo
├── build_dataset.py               # Run pose extraction + build splits
├── run_training.py                # Train all models + log to MLflow
├── predict.py                     # Run inference on a single image
├── train.py                       # Training logic
├── data_config.yaml               # Class names, split ratios, paths
├── model_config.yaml              # Model hyperparameters
└── requirements.txt
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add training images
Add images to `data/raw_images/<class_name>/` — at least 100 images per class recommended. Images should be JPG or PNG.

### 3. Build dataset
```bash
python3 build_dataset.py
```
Runs pose extraction on every image and builds train/val/test splits.

### 4. Train
```bash
python3 run_training.py
```
View experiment results:
```bash
mlflow ui
```

### 5. Run demo
```bash
streamlit run app.py
```

### 6. Predict on a single image
```bash
python3 predict.py --image path/to/image.jpg
```

## Data Collection Tips

- Screenshot clear holds from YouTube ballet tutorials
- Aim for 100+ images per class
- Capture a mix of front, side, and diagonal views
- For side-invariant poses (arabesque, tendu, etc.) you can use images from either side

## Feature Design

The model uses 24 joint angle features per image:
- **12 directional angles** — left/right knee, hip, elbow, shoulder, ankle, hip abduction
- **4 symmetric arm angles** — sorted pairs so left-arm-up and right-arm-up look identical (used for third/fourth position)
- **8 symmetric leg angles** — sorted pairs so left-leg and right-leg versions look identical (used for all one-legged poses)

A **GroupedClassifier** trains separate sub-models per feature group so each pose type only uses the features relevant to it.

## Tech Stack

- **MediaPipe** — pose estimation (Tasks API)
- **scikit-learn** — Random Forest, SVM, Gradient Boosting
- **MLflow** — experiment tracking
- **Streamlit** — demo UI
