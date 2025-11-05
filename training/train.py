import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from glob import glob

import sys
sys.path.append('..')  # Add parent directory to path

from utils.dataloader import load_ninapro_subject
from src.preprocessing import EMGPreprocessor
from src.windowing import SlidingWindow
from src.features import EMGFeatureExtractor
from src.classifier import GestureClassifier


def load_multiple_subjects(data_dir, subject_ids):
    """
    Load data from multiple subjects.
    
    Args:
        data_dir: Directory containing .mat files
        subject_ids: List of subject IDs (e.g., [1, 2, 3, ...])
        
    Returns:
        all_emg: List of EMG arrays
        all_labels: List of label arrays
        subject_info: List of subject IDs for each array
    """
    all_emg = []
    all_labels = []
    subject_info = []
    
    for sid in subject_ids:
        # Try different filename patterns
        patterns = [
            f"S{sid}_A1_E1.mat",
            f"S{sid:02d}_A1_E1.mat",
            f"s{sid}_A1_E1.mat",
        ]
        
        filepath = None
        for pattern in patterns:
            candidate = Path(data_dir) / pattern
            if candidate.exists():
                filepath = candidate
                break
        
        if filepath is None:
            print(f"  Warning: Could not find file for subject {sid}, skipping...")
            continue
            
        try:
            emg, labels = load_ninapro_subject(str(filepath))
            original_gestures = len(np.unique(labels))
            emg, labels, label_mapping = filter_gestures(emg, labels, remove_rest=True, keep_only_basic=True)
            filtered_gestures = len(np.unique(labels))
            print(f"  Filtered: {original_gestures} → {filtered_gestures} gestures (removed rest)")
            all_emg.append(emg)
            all_labels.append(labels)
            subject_info.append(sid)
            print(f"  Loaded S{sid}: EMG shape {emg.shape}, {len(np.unique(labels))} unique gestures")
        except Exception as e:
            print(f"  Error loading S{sid}: {e}")
            continue
    
    return all_emg, all_labels, subject_info


def segment_emg_data(emg, labels, window_size, overlap=0.5):
    """
    Segment EMG data into windows with labels.
    
    Args:
        emg: EMG data (samples x channels)
        labels: Gesture labels for each sample
        window_size: Number of samples per window
        overlap: Overlap ratio between windows
        
    Returns:
        windows: Array of EMG windows
        window_labels: Labels for each window
    """
    windower = SlidingWindow(window_size=window_size, overlap=overlap)
    windows = []
    window_labels = []
    
    step_size = int(window_size * (1 - overlap))
    
    for idx, window in enumerate(windower.window_generator(emg)):
        window_start = idx * step_size
        window_end = window_start + window_size
        
        if window_end > len(labels):
            break
            
        window_label_segment = labels[window_start:window_end]
        
        # Get majority label (most common in window)
        if len(window_label_segment) > 0:
            majority_label = np.bincount(window_label_segment.astype(int)).argmax()
            windows.append(window)
            window_labels.append(majority_label)
    
    return np.array(windows), np.array(window_labels)


def process_subjects(emg_list, labels_list, subject_ids, preprocessor, 
                    window_size, overlap, extractor, use_wavelets):
    """
    Process multiple subjects: preprocess, segment, extract features.
    
    Args:
        emg_list: List of EMG arrays
        labels_list: List of label arrays
        subject_ids: List of subject IDs
        preprocessor: EMGPreprocessor instance
        window_size: Window size in samples
        overlap: Window overlap ratio
        extractor: EMGFeatureExtractor instance
        use_wavelets: Whether to use wavelet features
        
    Returns:
        X: Feature matrix
        y: Labels
        subject_labels: Subject ID for each window
    """
    all_features = []
    all_labels = []
    all_subject_labels = []
    
    for emg, labels, sid in zip(emg_list, labels_list, subject_ids):
        print(f"\n  Processing S{sid}...")
        
        # Preprocess
        emg_clean = preprocessor.preprocess(
            emg,
            bandpass=(20.0, 450.0),
            notch_freq=60.0,
            baseline_removal='highpass',
            normalize=True
        )
        
        # Segment into windows
        windows, window_labels = segment_emg_data(
            emg_clean, labels, window_size, overlap
        )
        print(f"    Created {len(windows)} windows")
        
        # Extract features
        for window in windows:
            features = extractor.extract_multichannel_features(
                window, use_wavelets=use_wavelets
            )
            all_features.append(features)
        
        all_labels.extend(window_labels)
        all_subject_labels.extend([sid] * len(windows))
    
    X = np.array(all_features)
    y = np.array(all_labels)
    subject_labels = np.array(all_subject_labels)
    
    print(f"\n  Total windows: {len(X)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Unique gestures: {np.unique(y)}")
    
    return X, y, subject_labels


def plot_results(results, save_dir):
    """Plot training results and save figures."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-validation scores
    plt.figure(figsize=(10, 6))
    cv_scores = results['cv_scores']
    plt.bar(range(len(cv_scores)), cv_scores)
    plt.axhline(y=results['cv_mean'], color='r', linestyle='--', 
                label=f"Mean: {results['cv_mean']:.3f}")
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'cv_scores.png', dpi=300)
    plt.close()
    
    print(f"\nResults saved to {save_dir}")


def plot_confusion_matrix(eval_results, save_dir):
    """Plot confusion matrix."""
    save_dir = Path(save_dir)
    
    cm = np.array(eval_results['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix (Test Accuracy: {eval_results['accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
    plt.close()


def plot_subject_comparison(y_test, y_pred, subject_labels_test, save_dir):
    """Plot per-subject accuracy comparison."""
    save_dir = Path(save_dir)
    
    unique_subjects = np.unique(subject_labels_test)
    accuracies = []
    
    for sid in unique_subjects:
        mask = subject_labels_test == sid
        acc = np.mean(y_test[mask] == y_pred[mask])
        accuracies.append(acc)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique_subjects)), accuracies)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.3f}')
    plt.xticks(range(len(unique_subjects)), [f'S{sid}' for sid in unique_subjects])
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('Per-Subject Test Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / 'per_subject_accuracy.png', dpi=300)
    plt.close()

def filter_gestures(emg, labels, remove_rest=True, keep_only_basic=True):
    """Filter and REMAP gestures."""
    # Create mask
    if remove_rest and keep_only_basic:
        mask = (labels >= 1) & (labels <= 12)
    elif remove_rest:
        mask = labels != 0
    elif keep_only_basic:
        mask = labels <= 12
    else:
        mask = np.ones(len(labels), dtype=bool)
    
    filtered_emg = emg[mask]
    filtered_labels = labels[mask]
    
    unique_labels = np.unique(filtered_labels)
    label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(sorted(unique_labels))}
    remapped_labels = np.array([label_mapping[lbl] for lbl in filtered_labels])
    
    print(f"  Original labels: {unique_labels}")
    print(f"  Remapped to: {np.unique(remapped_labels)}")
    print(f"  Label mapping: {label_mapping}")
    
    # Check class distribution
    unique, counts = np.unique(remapped_labels, return_counts=True)
    print(f"  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls}: {cnt} samples")
    
    return filtered_emg, remapped_labels, label_mapping

def balance_windows(X, y, subject_labels=None, method='downsample', min_samples=10):
    """Balance classes after feature extraction."""
    unique_classes, counts = np.unique(y, return_counts=True)
    
    print(f"\nClass distribution BEFORE balancing:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"  Class {cls}: {cnt} windows")
    
    # Remove classes with too few samples
    valid_classes = unique_classes[counts >= min_samples]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    if subject_labels is not None:
        subject_labels = subject_labels[mask]
    
    # Recalculate after filtering
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if method == 'downsample':
        min_count = counts.min()
        balanced_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) >= min_count:
                selected = np.random.choice(cls_indices, min_count, replace=False)
                balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        subject_labels_balanced = subject_labels[balanced_indices] if subject_labels is not None else None
        
    elif method == 'upsample':
        max_count = counts.max()
        balanced_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) < max_count:
                selected = np.random.choice(cls_indices, max_count, replace=True)
            else:
                selected = cls_indices
            balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        subject_labels_balanced = subject_labels[balanced_indices] if subject_labels is not None else None
    
    print(f"\nClass distribution AFTER balancing:")
    unique_classes, counts = np.unique(y_balanced, return_counts=True)
    for cls, cnt in zip(unique_classes, counts):
        print(f"  Class {cls}: {cnt} windows")
    
    if subject_labels is not None:
        return X_balanced, y_balanced, subject_labels_balanced
    return X_balanced, y_balanced

def main(args):
    """Main training pipeline for multi-subject classification."""
    
    print("="*70)
    print("Multi-Subject EMG Hand Gesture Recognition")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"multi_subject_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define train/test subject split
    all_subjects = list(range(1, args.num_subjects + 1))
    n_test = int(len(all_subjects) * args.test_subject_ratio)
    
    if args.train_subjects and args.test_subjects:
        # User-specified split
        train_subjects = [int(s) for s in args.train_subjects.split(',')]
        test_subjects = [int(s) for s in args.test_subjects.split(',')]
    else:
        # Automatic split
        test_subjects = all_subjects[-n_test:]
        train_subjects = all_subjects[:-n_test]
    
    print(f"\nTrain subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")
    
    # 1. Load data
    print("\n[1/5] Loading training subjects...")
    train_emg, train_labels, train_ids = load_multiple_subjects(
        args.data_dir, train_subjects
    )
    
    print("\n[1/5] Loading test subjects...")
    test_emg, test_labels, test_ids = load_multiple_subjects(
        args.data_dir, test_subjects
    )
    
    if len(train_ids) == 0 or len(test_ids) == 0:
        print("\nError: No data loaded. Check your data directory and file names.")
        return
    
    print(f"\nLoaded {len(train_ids)} training subjects")
    print(f"Loaded {len(test_ids)} test subjects")
    
    # 2. Initialize processors
    print("\n[2/5] Initializing preprocessor and feature extractor...")
    preprocessor = EMGPreprocessor(sampling_rate=args.sampling_rate)
    extractor = EMGFeatureExtractor(sampling_rate=args.sampling_rate)
    window_size = int(args.window_length / 1000.0 * args.sampling_rate)
    
    # 3. Process training data
    print("\n[3/5] Processing training data...")
    X_train, y_train, subject_labels_train = process_subjects(
        train_emg, train_labels, train_ids,
        preprocessor, window_size, args.overlap,
        extractor, args.use_wavelets
    )
    
    # 4. Process test data
    print("\n[4/5] Processing test data...")
    X_test, y_test, subject_labels_test = process_subjects(
        test_emg, test_labels, test_ids,
        preprocessor, window_size, args.overlap,
        extractor, args.use_wavelets
    )
    
    print(f"\nFinal dataset sizes:")
    print(f"  Training: {X_train.shape[0]} windows from {len(train_ids)} subjects")
    print(f"  Test: {X_test.shape[0]} windows from {len(test_ids)} subjects")

    # In main(), after balancing:
    print("\n[4/5] Balancing classes...")
    X_train, y_train, subject_labels_train = balance_windows(
        X_train, y_train, subject_labels_train, method='downsample', min_samples=20
    )
    X_test, y_test, subject_labels_test = balance_windows(
        X_test, y_test, subject_labels_test, method='downsample', min_samples=5
    )

    # 5. Train classifier
    print("\n[5/5] Training classifier...")
    classifier = GestureClassifier(classifier_type=args.classifier)

    # Set model params properly
    if args.classifier == 'svm':
        model_params = {
            'C': args.svm_C,
            'kernel': args.svm_kernel,
            'gamma': args.svm_gamma,
            'class_weight': 'balanced'  
        }
    else:
        model_params = {}

    results = classifier.train(
        X_train, y_train,
        cv_folds=args.cv_folds,
        **model_params
    )
    
    print(f"\n  Cross-validation accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    # 6. Evaluate on test subjects
    print("\n[6/6] Evaluating on test subjects...")
    eval_results = classifier.evaluate(X_test, y_test)
    
    # Make predictions for per-subject analysis
    y_pred = classifier.predict(X_test)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Training Subjects CV Accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"Test Subjects Accuracy: {eval_results['accuracy']:.4f}")
    print(f"{'='*70}")
    
    # Per-subject breakdown
    print("\nPer-Subject Test Accuracy:")
    for sid in test_ids:
        mask = subject_labels_test == sid
        acc = np.mean(y_test[mask] == y_pred[mask])
        print(f"  S{sid}: {acc:.4f} ({np.sum(mask)} windows)")
    
    # Save model
    model_path = output_dir / "gesture_model"
    classifier.save_model(str(model_path))
    print(f"\n✓ Model saved to: {model_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_results(results, output_dir)
    plot_confusion_matrix(eval_results, output_dir)
    plot_subject_comparison(y_test, y_pred, subject_labels_test, output_dir)
    
    # Save detailed summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Multi-Subject EMG Gesture Recognition Training Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Train Subjects: {train_ids}\n")
        f.write(f"Test Subjects: {test_ids}\n\n")
        f.write(f"Classifier: {args.classifier}\n")
        f.write(f"Window size: {args.window_length}ms\n")
        f.write(f"Overlap: {args.overlap}\n")
        f.write(f"Sampling rate: {args.sampling_rate}Hz\n")
        f.write(f"Wavelets: {args.use_wavelets}\n\n")
        f.write(f"Training Data: {X_train.shape[0]} windows from {len(train_ids)} subjects\n")
        f.write(f"Test Data: {X_test.shape[0]} windows from {len(test_ids)} subjects\n\n")
        f.write(f"CV Accuracy (Train Subjects): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
        f.write(f"Test Accuracy (Test Subjects): {eval_results['accuracy']:.4f}\n\n")
        f.write("Per-Subject Test Accuracy:\n")
        for sid in test_ids:
            mask = subject_labels_test == sid
            acc = np.mean(y_test[mask] == y_pred[mask])
            f.write(f"  S{sid}: {acc:.4f}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-subject EMG gesture recognition model"
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing NinaPro .mat files')
    parser.add_argument('--output_dir', type=str, default='./wavelength_testing',
                       help='Output directory for models and results')
    parser.add_argument('--num_subjects', type=int, default=27,
                       help='Total number of subjects available')
    parser.add_argument('--test_subject_ratio', type=float, default=0.2,
                       help='Ratio of subjects to use for testing')
    parser.add_argument('--train_subjects', type=str, default=None,
                       help='Comma-separated train subject IDs (e.g., "1,2,3,4,5")')
    parser.add_argument('--test_subjects', type=str, default=None,
                       help='Comma-separated test subject IDs (e.g., "6,7,8")')
    
    # Preprocessing arguments
    parser.add_argument('--sampling_rate', type=float, default=2000.0,
                       help='EMG sampling rate in Hz')
    parser.add_argument('--window_length', type=float, default=200.0,
                       help='Window length in milliseconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio (0.0 to 1.0)')
    
    # Feature extraction
    parser.add_argument('--use_wavelets', action='store_true',
                       help='Include wavelet features')
    
    # Classifier arguments
    parser.add_argument('--classifier', type=str, default='svm',
                       choices=['svm', 'rf', 'lda'],
                       help='Classifier type')
    parser.add_argument('--svm_C', type=float, default=10.0,
                       help='SVM regularization parameter')
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM kernel type')
    parser.add_argument('--svm_gamma', type=str, default='scale',
                       help='SVM gamma parameter')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Fast debug mode with minimal subjects for quick testing')
    args = parser.parse_args()
    if args.debug:
        print("\n" + "="*70)
        print("DEBUG")
        print("="*70)
        args.train_subjects = "1,2,3"
        args.test_subjects = "4"
        args.cv_folds = 3
        args.overlap = 0.25 
        args.svm_kernel = 'linear'
        print(f"  Train subjects: {args.train_subjects}")
        print(f"  Test subjects: {args.test_subjects}")
        print(f"  CV folds: {args.cv_folds}")
        print(f"  Overlap: {args.overlap}")
        print(f"  Kernel: {args.svm_kernel}")
        print("="*70 + "\n")
    
    main(args)

    