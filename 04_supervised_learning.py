import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import load_data

# Import complex loader from 03 script
from importlib import import_module
complex_module = import_module('03_complex_embeddings')
load_data_complex = complex_module.load_data_complex

def evaluate_supervised(X, y, class_names, feature_name):
    print(f"\n=========================================")
    print(f"Supervised Learning: {feature_name}")
    print(f"=========================================\n")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

def main():
    print("Checking how supervised learning performs on the different feature sets...")

    # 1. Physics Baseline (2 features)
    X_physics, y_physics, _, class_names = load_data("dummy_audio", use_all_features=False)
    evaluate_supervised(X_physics, y_physics, class_names, "Physics Baseline (2 Features)")

    # 2. Expanded Features (13 features)
    X_expanded, y_expanded, _, _ = load_data("dummy_audio", use_all_features=True)
    evaluate_supervised(X_expanded, y_expanded, class_names, "Expanded Features (13 Features)")

    # 3. Complex Embeddings (YAMNet)
    X_complex, y_complex, _, _ = load_data_complex("dummy_audio")
    evaluate_supervised(X_complex, y_complex, class_names, "Complex Embeddings (YAMNet)")

if __name__ == "__main__":
    main()