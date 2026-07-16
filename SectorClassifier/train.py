"""Train TF-IDF + LinearSVC models to predict Sector and Subsector
from an organization's Definition text."""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils import preprocess_definition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Sector_Classifier_Definition_Data.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TEXT_COLUMN = "Definition"
SECTOR_COLUMN = "Sector_of_Org"
SUBSECTOR_COLUMN = "Sub_Sector"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the training Excel dataset.

    Args:
        path: Path to the Excel file.

    Returns:
        DataFrame containing only the required columns, with nulls dropped.

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_excel(path)

    required_cols = [TEXT_COLUMN, SECTOR_COLUMN, SUBSECTOR_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].dropna()
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    return df.reset_index(drop=True)


def evaluate_model(model, X_test, y_test, label: str) -> None:
    """Print evaluation metrics for a trained classifier.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        label: Name of the target being evaluated (for display).
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"Evaluation Metrics: {label}")
    print(f"{'=' * 60}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print(f"\nClassification Report ({label}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Confusion Matrix ({label}):")
    print(confusion_matrix(y_test, y_pred))


def train_classifier(X_train, y_train) -> LinearSVC:
    """Train a LinearSVC classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Trained LinearSVC model.
    """
    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Run the full training pipeline: load, preprocess, vectorize,
    train, evaluate, and save both models."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    print("Preprocessing text...")
    df["clean_definition"] = df[TEXT_COLUMN].apply(preprocess_definition)

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["clean_definition"])

    y_sector = df[SECTOR_COLUMN]
    y_subsector = df[SUBSECTOR_COLUMN]

    # Stratified splitting requires at least 2 samples per class.
    # Report and drop any classes that do not meet this minimum.
    sector_counts = y_sector.value_counts()
    rare_sectors = sector_counts[sector_counts < 2].index.tolist()
    subsector_counts = y_subsector.value_counts()
    rare_subsectors = subsector_counts[subsector_counts < 2].index.tolist()

    if rare_sectors:
        print(f"Warning: dropping Sector classes with < 2 samples: {rare_sectors}")
    if rare_subsectors:
        print(f"Warning: dropping Subsector classes with < 2 samples: {rare_subsectors}")

    # --- Sector model ---
    print("\nSplitting data for Sector model...")
    sector_mask = ~y_sector.isin(rare_sectors)
    X_sector, y_sector_filtered = X[sector_mask.values], y_sector[sector_mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sector,
        y_sector_filtered,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sector_filtered,
    )
    print("Training Sector classifier (LinearSVC)...")
    sector_model = train_classifier(X_train, y_train)
    evaluate_model(sector_model, X_test, y_test, "Sector")

    # --- Subsector model ---
    print("\nSplitting data for Subsector model...")
    subsector_mask = ~y_subsector.isin(rare_subsectors)
    X_subsector, y_subsector_filtered = X[subsector_mask.values], y_subsector[subsector_mask]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_subsector,
        y_subsector_filtered,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_subsector_filtered,
    )
    print("Training Subsector classifier (LinearSVC)...")
    subsector_model = train_classifier(X_train_s, y_train_s)
    evaluate_model(subsector_model, X_test_s, y_test_s, "Subsector")

    # --- Save artifacts ---
    print("\nSaving models...")
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.pkl"))
    joblib.dump(sector_model, os.path.join(MODELS_DIR, "sector_model.pkl"))
    joblib.dump(subsector_model, os.path.join(MODELS_DIR, "subsector_model.pkl"))
    print(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
