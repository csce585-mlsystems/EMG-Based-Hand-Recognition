import numpy as np
import scipy.io as sio
import glob
from tqdm import tqdm

# -----------------------------
# Feature Functions
# -----------------------------
def mav(x):
    return np.mean(np.abs(x))

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def wl(x):
    return np.sum(np.abs(np.diff(x)))

def zc(x, threshold=0.01):
    """Zero crossings with threshold to avoid noise."""
    return np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold))

def ssc(x, threshold=0.01):
    """Slope sign changes."""
    count = 0
    for i in range(1, len(x) - 1):
        prev = x[i] - x[i-1]
        nxt = x[i] - x[i+1]
        if (prev * nxt > 0) and (abs(prev) > threshold or abs(nxt) > threshold):
            count += 1
    return count


# -----------------------------
# Feature Extraction: 50 features (10 channels × 5 TD features)
# -----------------------------
def extract_features(window):
    feats = []
    for ch in range(window.shape[0]):
        x = window[ch]
        feats.append(mav(x))
        feats.append(rms(x))
        feats.append(wl(x))
        feats.append(zc(x))
        feats.append(ssc(x))
    return np.array(feats, dtype=np.float32)


# -----------------------------
# MAIN PREPROCESSING FUNCTION
# -----------------------------
def preprocess_ninapro_A1(
    mat_dir="raw_data/",
    save_name="processed_data_A1.npz",
    window_size=400,
    step_size=400
):
    """
    Preprocess NinaPro DB1 Exercise A1 ONLY.
    Produces dataset with:
      - X_raw: shape (N, 10, 400)
      - X:     shape (N, 50)
      - y:     shape (N,)
    """

    print("\n========== NinaPro A1 Preprocessing ==========\n")

    X_raw = []
    X_features = []
    y_all = []

    # Load ONLY A1 files by name pattern (E1_E1...E1_E6)
    files = sorted(glob.glob(mat_dir + "*E1_*"))  # DB1 A1 files look like s1_E1.mat, s2_E1.mat, etc.
    if len(files) == 0:
        files = sorted(glob.glob(mat_dir + "*.mat"))  # fallback
        print("⚠️  Warning: Using all .mat files. Make sure they are A1.")

    print(f"Found {len(files)} A1 files.")

    for f in tqdm(files, desc="Processing files"):
        mat = sio.loadmat(f)

        emg = mat['emg']  # shape (samples, 10)
        restimulus = mat['restimulus'].flatten()

        # ✔ Filter ONLY gesture labels 1–12 (A1 exercises)
        valid_idx = np.where((restimulus >= 1) & (restimulus <= 12))[0]
        emg = emg[valid_idx]
        restimulus = restimulus[valid_idx]

        # Windowing
        for i in range(0, len(emg) - window_size, step_size):
            window = emg[i:i+window_size]      # (400, 10)
            window = window.T                  # → (10, 400)

            label = int(restimulus[i + window_size//2])

            # Save raw window
            X_raw.append(window)

            # Extract TD features
            feats = extract_features(window)
            X_features.append(feats)

            y_all.append(label)

    # Convert lists to arrays
    X_raw = np.array(X_raw, dtype=np.float32)
    X_features = np.array(X_features, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)

    print("\nShapes:")
    print("X_raw:", X_raw.shape)
    print("X_features:", X_features.shape)
    print("y:", y_all.shape)

    # Save final dataset
    np.savez(save_name, X_raw=X_raw, X=X_features, y=y_all)

    print(f"\nSaved A1 preprocessed dataset → {save_name}")
    print("==============================================\n")


if __name__ == "__main__":
    preprocess_ninapro_A1()
