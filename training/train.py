import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.dataloader import load_ninapro_subject
from src.preprocessing import EMGPreprocessor
from src.windowing import SlidingWindow
from src.features import EMGFeatureExtractor
from src.classifier import GestureClassifier


def segment_emg_data(emg, labels, window_size, overlap=0.5):
    """
    Segment EMG data into windows with labels.
    
    Args:
        emg: EMG data (samples x channels)
        labels: Gesture labels for each sample
        window_size: Number of samples per window
        overlap: Overlap ratio between windows
        
    Returns:
        X: Feature matrix (n_windows x n_features)
        y: Labels (n_windows,)
    """
    print(f"Segmenting data with window_size={window_size}, overlap={overlap}")
    
    windower = SlidingWindow(window_size=window_size, overlap=overlap)
    windows = []
    window_labels = []
    
    for window in windower.window_generator(emg):
        # Get the most common label in this window (majority vote)
        window_start = len(windows) * int(window_size * (1 - overlap))
        window_end = window_start + window_size
        window_label_segment = labels[window_start:window_end]
        
        # Skip windows with label 0 (rest/no gesture)
        if len(window_label_segment) > 0:
            majority_label = np.bincount(window_label_segment).argmax()
            windows.append(window)
            window_labels.append(majority_label)

    
    print(f"Created {len(windows)} windows from {len(emg)} samples")
    return np.array(windows), np.array(window_labels)


def extract_features_from_windows(windows, extractor, use_wavelets=True):
    """
    Extract features from all windows.
    
    Args:
        windows: Array of EMG windows (n_windows x window_size x n_channels)
        extractor: EMGFeatureExtractor instance
        use_wavelets: Whether to include wavelet features
        
    Returns:
        Feature matrix (n_windows x n_features)
    """
    print(f"Extracting features from {len(windows)} windows...")
    
    features_list = []
    for i, window in enumerate(windows):
        if i % 100 == 0:
            print(f"  Processing window {i}/{len(windows)}")
        
        features = extractor.extract_multichannel_features(
            window, use_wavelets=use_wavelets
        )
        features_list.append(features)
    
    X = np.array(features_list)
    print(f"Feature matrix shape: {X.shape}")
    return X


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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=eval_results['classes'],
                yticklabels=eval_results['classes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix (Test Accuracy: {eval_results['accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
    plt.close()


def main(args):
    """Main training pipeline."""
    
    print("="*70)
    print("EMG-Based Hand Gesture Recognition Training Pipeline")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    emg, labels = load_ninapro_subject(args.data_path)
    print(f"  EMG shape: {emg.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique gestures: {np.unique(labels)}")
    print(f"  Number of gestures (excluding rest): {len(np.unique(labels[labels != 0]))}")
    
    # 2. Preprocess
    print("\n[2/6] Preprocessing signals...")
    preprocessor = EMGPreprocessor(sampling_rate=args.sampling_rate)
    emg_clean = preprocessor.preprocess(
        emg,
        bandpass=(args.bandpass_low, args.bandpass_high),
        notch_freq=args.notch_freq,
        baseline_removal='highpass',
        normalize=True
    )
    print(f"  Preprocessed EMG shape: {emg_clean.shape}")
    
    # 3. Segment into windows
    print("\n[3/6] Segmenting data into windows...")
    window_size = int(args.window_length / 1000.0 * args.sampling_rate)  # Convert ms to samples
    windows, window_labels = segment_emg_data(
        emg_clean, labels, window_size, overlap=args.overlap
    )
    
    # 4. Extract features
    print("\n[4/6] Extracting features...")
    extractor = EMGFeatureExtractor(sampling_rate=args.sampling_rate)
    X = extract_features_from_windows(windows, extractor, use_wavelets=args.use_wavelets)
    y = window_labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\n  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 5. Train classifier
    print("\n[5/6] Training classifier...")
    classifier = GestureClassifier(classifier_type=args.classifier)
    
    # Model parameters
    model_params = {
        'C': args.svm_C,
        'kernel': args.svm_kernel,
        'gamma': args.svm_gamma
    } if args.classifier == 'svm' else {}
    
    results = classifier.train(
        X_train, y_train, 
        cv_folds=args.cv_folds,
        **model_params
    )
    
    print(f"\n  Cross-validation accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    # 6. Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    eval_results = classifier.evaluate(X_test, y_test)
    print(f"\n  Test Accuracy: {eval_results['accuracy']:.4f}")
    print("\n  Classification Report:")
    
    # Print per-class metrics
    report = eval_results['classification_report']
    for gesture_class in sorted([k for k in report.keys() if k.isdigit()]):
        metrics = report[gesture_class]
        print(f"    Gesture {gesture_class}: "
              f"Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, "
              f"F1={metrics['f1-score']:.3f}")
    
    # Save model
    model_path = output_dir / "gesture_model"
    classifier.save_model(str(model_path))
    print(f"\n✓ Model saved to: {model_path}")
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_results(results, output_dir)
    plot_confusion_matrix(eval_results, output_dir)
    
    # Save summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EMG Gesture Recognition Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Classifier: {args.classifier}\n")
        f.write(f"Window size: {args.window_length}ms\n")
        f.write(f"Overlap: {args.overlap}\n")
        f.write(f"Sampling rate: {args.sampling_rate}Hz\n\n")
        f.write(f"CV Accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
        f.write(f"Test Accuracy: {eval_results['accuracy']:.4f}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EMG-based hand gesture recognition model"
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NinaPro .mat file')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for models and results')
    
    # Preprocessing arguments
    parser.add_argument('--sampling_rate', type=float, default=2000.0,
                       help='EMG sampling rate in Hz')
    parser.add_argument('--bandpass_low', type=float, default=20.0,
                       help='Bandpass filter low cutoff frequency')
    parser.add_argument('--bandpass_high', type=float, default=450.0,
                       help='Bandpass filter high cutoff frequency')
    parser.add_argument('--notch_freq', type=float, default=60.0,
                       help='Notch filter frequency (50 or 60 Hz)')
    
    # Windowing arguments
    parser.add_argument('--window_length', type=float, default=250.0,
                       help='Window length in milliseconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio (0.0 to 1.0)')
    
    # Feature extraction arguments
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
    
    # Train/test split
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    
    args = parser.parse_args()
    main(args)