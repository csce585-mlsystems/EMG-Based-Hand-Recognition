import numpy as np
from scipy import signal
from typing import Optional, Tuple, Union

class EMGPreprocessor:
    # Robust EMG signal preprocessing with configurable filters.
    
    
    def __init__(self, sampling_rate: float = 2000.0):
        self.sampling_rate = sampling_rate
        self.filters = {}
        
    def design_bandpass(self, lowcut: float = 20.0, highcut: float = 450.0, order: int = 4):
        # Design Butterworth bandpass filter.
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
            
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        self.filters['bandpass'] = sos
        return sos
    
    def design_notch(self, freq: float = 60.0, Q: float = 30.0):
        # Design notch filter for powerline interference.
        w0 = freq / (self.sampling_rate / 2)
        
        if w0 >= 1.0:
            return None
            
        b, a = signal.iirnotch(w0, Q)
        self.filters['notch'] = (b, a)
        return b, a
    
    def apply_bandpass(self, data: np.ndarray, lowcut: float = 20.0, 
                      highcut: float = 450.0) -> np.ndarray:
        # Apply bandpass filter to signal.
        if 'bandpass' not in self.filters:
            self.design_bandpass(lowcut, highcut)
        
        return signal.sosfiltfilt(self.filters['bandpass'], data, axis=0)
    
    def apply_notch(self, data: np.ndarray, freq: float = 60.0) -> np.ndarray:
        # Apply notch filter to remove powerline interference.
        if 'notch' not in self.filters:
            notch = self.design_notch(freq)
            if notch is None:
                return data
        
        b, a = self.filters['notch']
        return signal.filtfilt(b, a, data, axis=0)
    
    def remove_baseline(self, data: np.ndarray, method: str = 'highpass') -> np.ndarray:
        # Remove baseline wander from signal.
        if method == 'highpass':
            #High-pass filter at 5 Hz
            sos = signal.butter(4, 5.0 / (self.sampling_rate/2), btype='high', output='sos')
            return signal.sosfiltfilt(sos, data, axis=0)
        elif method == 'detrend':
            return signal.detrend(data, axis=0)
        elif method == 'median':
            # Moving median filter
            window_size = int(self.sampling_rate * 0.2)  #200ms window
            if data.ndim == 1:
                baseline = signal.medfilt(data, kernel_size=min(window_size, len(data)))
                return data - baseline
            else:
                result = np.zeros_like(data)
                for ch in range(data.shape[1]):
                    baseline = signal.medfilt(data[:, ch], 
                                            kernel_size=min(window_size, len(data)))
                    result[:, ch] = data[:, ch] - baseline
                return result
        return data
    
    def preprocess(self, emg_segment: np.ndarray, 
                  bandpass: Tuple[float, float] = (20, 450),
                  notch_freq: Optional[float] = 60.0,
                  baseline_removal: str = 'highpass',
                  normalize: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline for EMG signals.
        
        Args:
            emg_segment: Raw EMG data (samples x channels)
            bandpass: (lowcut, highcut) frequencies for bandpass filter
            notch_freq: Powerline frequency to remove (50 or 60 Hz)
            baseline_removal: Method for baseline removal
            normalize: Whether to normalize the signal
        
        Returns:
            Preprocessed EMG signal
        """
        data = emg_segment.copy()
        
        # Remove baseline
        if baseline_removal:
            data = self.remove_baseline(data, method=baseline_removal)
        
        # Apply bandpass filter
        if bandpass:
            data = self.apply_bandpass(data, bandpass[0], bandpass[1])
        
        # Apply notch filter
        if notch_freq:
            data = self.apply_notch(data, notch_freq)
        
        # Normalize if requested
        if normalize:
            # Z-score normalization per channel
            if data.ndim == 1:
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            else:
                for ch in range(data.shape[1]):
                    data[:, ch] = (data[:, ch] - np.mean(data[:, ch])) / \
                                 (np.std(data[:, ch]) + 1e-8)
        
        return data
