import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from pathlib import Path
import sys
sys.path.append('..')

from utils.dataloader import load_ninapro_subject
from src.preprocessing import EMGPreprocessor
from src.features import EMGFeatureExtractor

# Gesture names for NinaPro DB1 (after removing rest, remapped 0-11)
GESTURE_NAMES = [
    "0: Thumb up",
    "1: Index+Middle ext",
    "2: Ring+Little flex", 
    "3: Thumb to little",
    "4: Finger abduction",
    "5: Fist",
    "6: Pointing index",
    "7: Finger adduction",
    "8: Wrist supination",
    "9: Wrist pronation",
    "10: Wrist flexion",
    "11: Wrist extension"
]

def analyze_gesture_pair(emg, labels, gesture_a, gesture_b, output_dir):
    """
    Compare two gestures that are commonly confused.
    
    Args:
        emg: EMG data (samples x channels)
        labels: Gesture labels
        gesture_a: First gesture ID
        gesture_b: Second gesture ID
        output_dir: Where to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data for each gesture
    mask_a = labels == gesture_a
    mask_b = labels == gesture_b
    
    emg_a = emg[mask_a]
    emg_b = emg[mask_b]
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {GESTURE_NAMES[gesture_a]} vs {GESTURE_NAMES[gesture_b]}")
    print(f"{'='*70}")
    print(f"Gesture {gesture_a} samples: {len(emg_a)}")
    print(f"Gesture {gesture_b} samples: {len(emg_b)}")
    
    # 1. Channel-wise RMS comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"RMS Comparison: {GESTURE_NAMES[gesture_a]} vs {GESTURE_NAMES[gesture_b]}")
    
    for ch in range(8):
        ax = axes[ch // 4, ch % 4]
        
        rms_a = np.sqrt(np.mean(emg_a[:, ch]**2))
        rms_b = np.sqrt(np.mean(emg_b[:, ch]**2))
        
        ax.bar([0, 1], [rms_a, rms_b], color=['#1f77b4', '#ff7f0e'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'G{gesture_a}', f'G{gesture_b}'])
        ax.set_ylabel('RMS')
        ax.set_title(f'Channel {ch+1}')
        ax.grid(True, alpha=0.3)
        
        # Add percentage difference
        diff_pct = abs(rms_a - rms_b) / max(rms_a, rms_b) * 100
        ax.text(0.5, max(rms_a, rms_b) * 0.9, f'{diff_pct:.1f}% diff', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'rms_comparison_G{gesture_a}_G{gesture_b}.png', dpi=300)
    plt.close()
    
    # 2. Sample raw signals
    fig, axes = plt.subplots(8, 2, figsize=(14, 16))
    fig.suptitle(f"Sample Signals: {GESTURE_NAMES[gesture_a]} (left) vs {GESTURE_NAMES[gesture_b]} (right)")
    
    # Get 1 second samples
    sample_len = 2000  # 1 second at 2000 Hz
    start_a = len(emg_a) // 2  # Middle of gesture
    start_b = len(emg_b) // 2
    
    for ch in range(8):
        # Gesture A
        ax = axes[ch, 0]
        if start_a + sample_len < len(emg_a):
            ax.plot(emg_a[start_a:start_a+sample_len, ch], linewidth=0.5)
        ax.set_ylabel(f'Ch{ch+1}')
        if ch == 0:
            ax.set_title(f'Gesture {gesture_a}')
        if ch == 7:
            ax.set_xlabel('Sample')
        ax.grid(True, alpha=0.3)
        
        # Gesture B
        ax = axes[ch, 1]
        if start_b + sample_len < len(emg_b):
            ax.plot(emg_b[start_b:start_b+sample_len, ch], linewidth=0.5, color='orange')
        if ch == 0:
            ax.set_title(f'Gesture {gesture_b}')
        if ch == 7:
            ax.set_xlabel('Sample')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'raw_signals_G{gesture_a}_G{gesture_b}.png', dpi=300)
    plt.close()
    
    # 3. Power Spectral Density
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"PSD Comparison: {GESTURE_NAMES[gesture_a]} vs {GESTURE_NAMES[gesture_b]}")
    
    for ch in range(8):
        ax = axes[ch // 4, ch % 4]
        
        # Compute PSD
        if start_a + sample_len < len(emg_a):
            f_a, psd_a = signal.welch(emg_a[start_a:start_a+sample_len, ch], 
                                       fs=2000, nperseg=512)
            ax.semilogy(f_a, psd_a, label=f'G{gesture_a}', alpha=0.7)
        
        if start_b + sample_len < len(emg_b):
            f_b, psd_b = signal.welch(emg_b[start_b:start_b+sample_len, ch], 
                                       fs=2000, nperseg=512)
            ax.semilogy(f_b, psd_b, label=f'G{gesture_b}', alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title(f'Channel {ch+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 500])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'psd_comparison_G{gesture_a}_G{gesture_b}.png', dpi=300)
    plt.close()
    
    # 4. Statistical comparison
    print("\n" + "="*70)
    print("Statistical Analysis:")
    print("="*70)
    
    print("\nChannel-wise separability (higher = more distinct):")
    print(f"{'Channel':<10} {'RMS Diff':<12} {'Mean Diff':<12} {'Std Diff':<12} {'Score':<10}")
    print("-" * 70)
    
    separability_scores = []
    for ch in range(8):
        rms_a = np.sqrt(np.mean(emg_a[:, ch]**2))
        rms_b = np.sqrt(np.mean(emg_b[:, ch]**2))
        rms_diff = abs(rms_a - rms_b)
        
        mean_a = np.mean(np.abs(emg_a[:, ch]))
        mean_b = np.mean(np.abs(emg_b[:, ch]))
        mean_diff = abs(mean_a - mean_b)
        
        std_a = np.std(emg_a[:, ch])
        std_b = np.std(emg_b[:, ch])
        std_diff = abs(std_a - std_b)
        
        # Combined separability score (normalized)
        score = (rms_diff + mean_diff + std_diff) / 3
        separability_scores.append((ch, score))
        
        indicator = "★★★" if score > 0.5 else ("★★" if score > 0.2 else "★")
        print(f"Ch{ch+1:<8} {rms_diff:<12.4f} {mean_diff:<12.4f} {std_diff:<12.4f} {indicator}")
    
    # Rank channels by importance
    separability_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*70}")
    print("Channel Importance Ranking:")
    print(f"{'='*70}")
    for rank, (ch, score) in enumerate(separability_scores, 1):
        print(f"  {rank}. Channel {ch+1}: {score:.4f}")
    
    # 5. Feature space visualization (PCA)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Extract features for small subset
    preprocessor = EMGPreprocessor(sampling_rate=2000.0)
    extractor = EMGFeatureExtractor(sampling_rate=2000.0)
    
    features_a = []
    features_b = []
    
    window_size = 400  # 200ms
    n_windows = 50  # Sample 50 windows from each
    
    for i in range(min(n_windows, len(emg_a) // window_size)):
        start = i * window_size
        window = emg_a[start:start+window_size]
        if len(window) == window_size:
            feat = extractor.extract_multichannel_features(window, use_wavelets=False)
            features_a.append(feat)
    
    for i in range(min(n_windows, len(emg_b) // window_size)):
        start = i * window_size
        window = emg_b[start:start+window_size]
        if len(window) == window_size:
            feat = extractor.extract_multichannel_features(window, use_wavelets=False)
            features_b.append(feat)
    
    if len(features_a) > 0 and len(features_b) > 0:
        features_a = np.array(features_a)
        features_b = np.array(features_b)
        
        # Combine and reduce to 2D
        X_combined = np.vstack([features_a, features_b])
        y_combined = np.array([gesture_a]*len(features_a) + [gesture_b]*len(features_b))
        
        # Handle NaN values (from failed feature extraction)
        nan_mask = np.isnan(X_combined).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  Warning: Removing {nan_mask.sum()} samples with NaN features")
            X_combined = X_combined[~nan_mask]
            y_combined = y_combined[~nan_mask]
        
        if len(X_combined) < 10:
            print("  Not enough valid samples for PCA analysis")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            
            # Replace any remaining NaN/inf with 0
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter_a = plt.scatter(X_pca[y_combined==gesture_a, 0], 
                                X_pca[y_combined==gesture_a, 1],
                                label=GESTURE_NAMES[gesture_a], alpha=0.6, s=50)
        scatter_b = plt.scatter(X_pca[y_combined==gesture_b, 0], 
                                X_pca[y_combined==gesture_b, 1],
                                label=GESTURE_NAMES[gesture_b], alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'Feature Space Separation (PCA)\n{GESTURE_NAMES[gesture_a]} vs {GESTURE_NAMES[gesture_b]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'pca_G{gesture_a}_G{gesture_b}.png', dpi=300)
        plt.close()
        
        # Calculate overlap
        from scipy.spatial.distance import cdist
        center_a = np.mean(X_pca[y_combined==gesture_a], axis=0)
        center_b = np.mean(X_pca[y_combined==gesture_b], axis=0)
        separation = np.linalg.norm(center_a - center_b)
        
        print(f"\nFeature Space Analysis:")
        print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1]*100:.2f}%")
        print(f"  Center-to-center distance: {separation:.3f}")
        print(f"  {'Low overlap - well separated' if separation > 2 else 'High overlap - difficult to separate'}")
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Check {output_dir} for visualizations.")
    print(f"{'='*70}\n")


def analyze_all_confusions(data_dir, subject_id, output_base_dir):
    """
    Analyze the major confusion pairs from your confusion matrix.
    """
    # Load subject data
    from train import load_multiple_subjects, filter_gestures
    
    emg_list, labels_list, _ = load_multiple_subjects(data_dir, [subject_id])
    
    if len(emg_list) == 0:
        print(f"Error: Could not load subject {subject_id}")
        return
    
    emg = emg_list[0]
    labels = labels_list[0]
    
    # Filter and remap gestures
    emg, labels, label_mapping = filter_gestures(emg, labels, 
                                                   remove_rest=True, 
                                                   keep_only_basic=True)
    
    output_dir = Path(output_base_dir) / f"gesture_analysis_S{subject_id}"
    
    # Analyze major confusions
    confusion_pairs = [
        (2, 0, "35 misclass - Largest confusion"),
        (11, 9, "33 misclass - Wrist gestures"),
        (8, 9, "Wrist rotation confusion"),
        (3, 2, "13 misclass"),
        (10, 9, "17 misclass")
    ]
    
    for gesture_a, gesture_b, description in confusion_pairs:
        print(f"\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")
        analyze_gesture_pair(emg, labels, gesture_a, gesture_b, 
                            output_dir / f"G{gesture_a}_vs_G{gesture_b}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze EMG gesture confusions")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing NinaPro .mat files')
    parser.add_argument('--subject', type=int, default=1,
                       help='Subject ID to analyze')
    parser.add_argument('--output_dir', type=str, default='./gesture_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--gesture_a', type=int, default=None,
                       help='First gesture to compare (or analyze all confusions)')
    parser.add_argument('--gesture_b', type=int, default=None,
                       help='Second gesture to compare')
    
    args = parser.parse_args()
    
    if args.gesture_a is not None and args.gesture_b is not None:
        # Analyze specific pair
        from train import load_multiple_subjects, filter_gestures
        
        emg_list, labels_list, _ = load_multiple_subjects(args.data_dir, [args.subject])
        emg, labels, _ = filter_gestures(emg_list[0], labels_list[0], 
                                          remove_rest=True, keep_only_basic=True)
        
        analyze_gesture_pair(emg, labels, args.gesture_a, args.gesture_b, 
                            Path(args.output_dir) / f"S{args.subject}_G{args.gesture_a}_vs_G{args.gesture_b}")
    else:
        # Analyze all major confusions
        analyze_all_confusions(args.data_dir, args.subject, args.output_dir)