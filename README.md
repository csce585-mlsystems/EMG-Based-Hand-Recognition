# Real-Time EMG-Based Hand Gesture Recognition using SVM

## Group Info
- Ashwin Sakthivel  
- Email: ashwins@email.sc.edu   

## Abstract
This project explores the development of a real-time hand gesture recognition system using electromyography (EMG) signals and Support Vector Machines (SVM). The pipeline processes raw EMG data, extracts relevant features, and classifies hand gestures with high accuracy. The system also evaluates latency and throughput to balance accuracy with responsiveness. This work demonstrates the feasibility of EMG-based human–computer interaction.

## Problem Description
- **Problem description:** Hand gesture recognition using EMG signals is challenged by noisy data, variable muscle signals, and the need for real-time performance. The problem is to design a pipeline that ensures accurate classification while maintaining low latency.

- **Motivation**
  - Enable accessible human–computer interaction (HCI) for prosthetics and rehabilitation.  
  - Enhance gesture-based control in wearable and AR/VR systems.  
  - Investigate efficient machine learning pipelines for real-time biomedical signal processing.  

- **Challenges**
  - Noise and variability in EMG signals.  
  - Real-time latency and throughput requirements.  
  - Designing robust feature extraction and classification methods.  

## Contribution
We extend prior research on EMG-based gesture recognition by integrating a real-time feedback loop to monitor latency and throughput. This ensures practical system deployment beyond offline analysis.  

- Contribution 1: Implementation of a preprocessing and feature extraction pipeline (RMS, waveform length).  
- Contribution 2: Integration of a latency/throughput monitor for real-time system optimization.  

---

## Dependencies

Python Packages

numpy
scipy
scikit-learn
matplotlib
seaborn Optional
jupyter
tqdm
You can install everything via:

pip install -r requirements.txt

## Project Directory

EMG-Based-Hand-Recognition/
│
├── data/
│   └── processed_data_A1.npz
│
├── docs/
│   ├── CurrentFindings.pdf
│   ├── FinalReport.pdf
│   ├── ProjectPresentation.pdf
│   └── proposal.pdf
│
├── results/
│   ├── confusion_rf_norm.png
│   ├── confusion_rf_raw.png
│   └── feature_analysis.png
│
├── src/
│   ├── __init__.py
│   ├── analyze_features.py
│   ├── gesture_signal_analyzer.py
│   └── gesture_analysis_S1/
│       ├── G10_vs_G9/
│       ├── G11_vs_G9/
│       ├── G2_vs_G0/
│       ├── G3_vs_G2/
│       └── G8_vs_G9/
│
├── training/
│   └── train_rf.py
│
├── utils/
│   ├── __init__.py
│   └── dataloader.py
│
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md

### How to Run

uv run training/train_rf.py

