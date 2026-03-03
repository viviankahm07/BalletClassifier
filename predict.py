"""
predict.py
----------
Run inference on a single image using the best saved model.

Usage:
    python scripts/predict.py --image path/to/image.jpg
"""

import argparse
import numpy as np
from src.extraction.pose_extractor import PoseExtractor
from src.preprocessing.normalizer import extract_all_features
from src.models.classifier import BalletClassifierBase


def predict_single_image(image_path: str, model_path: str, classes_path: str):
    # Load model and class names
    model = BalletClassifierBase.load(model_path)
    class_names = np.load(classes_path, allow_pickle=True)

    # Extract pose
    extractor = PoseExtractor()
    result = extractor.extract(image_path)

    if not result.success:
        print(f"Could not detect a pose in {image_path}")
        return

    # Extract features and predict
    angles = extract_all_features(result.keypoints)
    proba = model.predict_proba([angles])[0]
    predicted_class = class_names[proba.argmax()]
    confidence = proba.max()

    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    print("\nAll probabilities:")
    for cls, prob in sorted(zip(class_names, proba), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<20} {prob:.1%}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model", default="models/saved/best_model_random_forest.pkl")
    parser.add_argument("--classes", default="data/splits/label_classes.npy")
    args = parser.parse_args()

    predict_single_image(args.image, args.model, args.classes)
