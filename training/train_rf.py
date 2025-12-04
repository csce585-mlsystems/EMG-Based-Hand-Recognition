import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def main():
    print("Loading dataset...")

    data = np.load("data/processed_data_A1.npz")
    X = data["X"]
    y = data["y"]

    print("Loaded:", X.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print("Train / Test sizes:", X_train.shape, X_test.shape)

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc}")
    print(classification_report(y_test, preds))

    # Confusion Matrix 
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Confusion Matrix - Raw Counts", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_rf_raw.png", dpi=200)
    plt.close()
    print("Saved confusion_rf_raw.png")

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Confusion Matrix - Normalized (%)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_rf_norm.png", dpi=200)
    plt.close()
    print("Saved confusion_rf_norm.png")

    # --- Latency Test (REUSE X, DO NOT RELOAD ANYTHING) ---
    clf_full = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf_full.fit(X, y)

    N = 2000
    sample = X[0].reshape(1, -1)

    start = time.time()
    for _ in range(N):
        clf_full.predict(sample)
    end = time.time()

    avg_latency = (end - start) / N
    print(f"Average Prediction Latency per Window: {avg_latency*1000:.4f} ms")


if __name__ == "__main__":
    main()
