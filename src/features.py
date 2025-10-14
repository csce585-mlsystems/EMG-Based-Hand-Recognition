import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional, Union
import warnings

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class EMGFeatureExtractor:
    """
    Enhanced feature extraction with time, frequency, and time-frequency domains.
    """
    
    def __init__(self, sampling_rate: float = 2000.0):
        self.sampling_rate = sampling_rate
        self.feature_names = []
        
    # Time-domain features
    def mean_absolute_value(self, signal: np.ndarray) -> float:
        """MAV: Mean absolute value."""
        return np.mean(np.abs(signal))
    
    def root_mean_square(self, signal: np.ndarray) -> float:
        """RMS: Root mean square."""
        return np.sqrt(np.mean(signal**2))
    
    def waveform_length(self, signal: np.ndarray) -> float:
        """WL: Cumulative length of waveform."""
        return np.sum(np.abs(np.diff(signal)))
    
    def zero_crossing(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """ZC: Number of zero crossings with threshold."""
        signal = signal - np.mean(signal)
        crossings = 0
        for i in range(len(signal) - 1):
            if (abs(signal[i]) > threshold and abs(signal[i+1]) > threshold and
                np.sign(signal[i]) != np.sign(signal[i+1])):
                crossings += 1
        return crossings
    
    def slope_sign_change(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """SSC: Number of slope sign changes."""
        count = 0
        for i in range(1, len(signal) - 1):
            if ((signal[i] - signal[i-1]) * (signal[i] - signal[i+1]) > threshold**2):
                count += 1
        return count
    
    def variance(self, signal: np.ndarray) -> float:
        """VAR: Statistical variance."""
        return np.var(signal)
    
    def log_detector(self, signal: np.ndarray) -> float:
        """Log detector (for muscle contraction intensity)."""
        return np.exp(np.mean(np.log(np.abs(signal) + 1e-8)))
    
    # Frequency-domain features
    def frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1/self.sampling_rate)
        psd = np.abs(fft_vals)**2
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        features = {}
        
        # Mean frequency
        features['mean_freq'] = np.sum(fft_freq * psd_norm)
        
        # Median frequency
        cumsum = np.cumsum(psd_norm)
        median_idx = np.where(cumsum >= 0.5)[0]
        features['median_freq'] = fft_freq[median_idx[0]] if len(median_idx) > 0 else 0
        
        # Peak frequency
        features['peak_freq'] = fft_freq[np.argmax(psd)]
        
        # Spectral moments
        features['spectral_moment_1'] = np.sum(fft_freq * psd_norm)
        features['spectral_moment_2'] = np.sum((fft_freq**2) * psd_norm)
        
        # Band power ratios (useful for fatigue detection)
        low_band = np.sum(psd[(fft_freq >= 20) & (fft_freq <= 100)])
        high_band = np.sum(psd[(fft_freq >= 100) & (fft_freq <= 450)])
        features['band_power_ratio'] = low_band / (high_band + 1e-8)
        
        return features
    
    # Autoregressive coefficients
    def ar_coefficients(self, signal: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Extract AR coefficients using multiple methods with fallbacks.
        """
        coeffs = np.zeros(order)
        
        if len(signal) < order * 2:
            return coeffs
            
        try:
            if STATSMODELS_AVAILABLE:
                # Use statsmodels AutoReg
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = AutoReg(signal, lags=order, old_names=False)
                    model_fit = model.fit()
                    coeffs = model_fit.params[1:order+1]  # Exclude intercept
            else:
                # Fallback: Yule-Walker equations
                coeffs = self._yule_walker(signal, order)
        except:
            # Final fallback: simple linear prediction
            coeffs = self._simple_ar(signal, order)
            
        return coeffs
    
    def _yule_walker(self, signal: np.ndarray, order: int) -> np.ndarray:
        """Yule-Walker method for AR coefficients."""
        r = np.correlate(signal, signal, mode='full')
        r = r[len(r)//2:]  # Take positive lags
        r = r[:order+1]
        
        # Toeplitz matrix
        R = np.array([r[abs(i-j)] for i in range(order) for j in range(order)])
        R = R.reshape((order, order))
        
        # Solve Yule-Walker equations
        try:
            coeffs = np.linalg.solve(R, r[1:order+1])
        except:
            coeffs = np.zeros(order)
            
        return coeffs
    
    def _simple_ar(self, signal: np.ndarray, order: int) -> np.ndarray:
        """Simple AR estimation using least squares."""
        X = np.array([signal[i:i+order] for i in range(len(signal)-order)])
        y = signal[order:]
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            coeffs = np.zeros(order)
            
        return coeffs
    
    # Wavelet features
    def wavelet_features(self, signal: np.ndarray, wavelet: str = 'db4', 
                        level: int = 4) -> Dict[str, float]:
        """Extract wavelet-based features."""
        features = {}
        
        if not PYWT_AVAILABLE:
            # Fallback: use simple energy bands from FFT
            fft_features = self.frequency_features(signal)
            for i in range(level):
                features[f'wavelet_energy_{i}'] = fft_features.get('band_power_ratio', 0)
            return features
        
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # Energy in each level
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                features[f'wavelet_energy_{i}'] = energy
                
            # Relative wavelet energy
            total_energy = sum(features.values())
            for key in features:
                features[f'{key}_relative'] = features[key] / (total_energy + 1e-8)
                
        except:
            # Return zeros if wavelet transform fails
            for i in range(level + 1):
                features[f'wavelet_energy_{i}'] = 0
                features[f'wavelet_energy_{i}_relative'] = 0
                
        return features
    
    def extract_all_features(self, signal: np.ndarray, 
                            use_wavelets: bool = True) -> np.ndarray:
        """
        Extract comprehensive feature vector from single channel.
        
        Args:
            signal: 1D EMG signal
            use_wavelets: Whether to include wavelet features
            
        Returns:
            Feature vector
        """
        features = []
        
        # Time-domain features
        features.append(self.mean_absolute_value(signal))
        features.append(self.root_mean_square(signal))
        features.append(self.waveform_length(signal))
        features.append(self.zero_crossing(signal))
        features.append(self.slope_sign_change(signal))
        features.append(self.variance(signal))
        features.append(self.log_detector(signal))
        features.append(stats.skew(signal))
        features.append(stats.kurtosis(signal))
        
        # Frequency-domain features
        freq_features = self.frequency_features(signal)
        features.extend(freq_features.values())
        
        # AR coefficients
        ar_coeffs = self.ar_coefficients(signal, order=4)
        features.extend(ar_coeffs)
        
        # Wavelet features (optional)
        if use_wavelets:
            wavelet_features = self.wavelet_features(signal, level=3)
            features.extend(wavelet_features.values())
        
        return np.array(features)
    
    def extract_multichannel_features(self, emg_segment: np.ndarray,
                                     use_wavelets: bool = True) -> np.ndarray:
        """
        Extract features from multi-channel EMG segment.
        
        Args:
            emg_segment: 2D array (samples x channels)
            use_wavelets: Whether to include wavelet features
            
        Returns:
            Concatenated feature vector
        """
        if emg_segment.ndim == 1:
            return self.extract_all_features(emg_segment, use_wavelets)
        
        all_features = []
        n_channels = emg_segment.shape[1]
        
        for ch in range(n_channels):
            channel_features = self.extract_all_features(
                emg_segment[:, ch], use_wavelets
            )
            all_features.extend(channel_features)
        
        return np.array(all_features)
    
    def get_feature_names(self, n_channels: int = 8, 
                         use_wavelets: bool = True) -> List[str]:
        """Generate feature names for interpretation."""
        base_names = [
            'MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'LogDet', 
            'Skew', 'Kurt', 'MeanFreq', 'MedianFreq', 'PeakFreq',
            'SpectMom1', 'SpectMom2', 'BandPowerRatio',
            'AR1', 'AR2', 'AR3', 'AR4'
        ]
        
        if use_wavelets:
            for i in range(4):
                base_names.extend([f'WavEnergy{i}', f'WavEnergyRel{i}'])
        
        feature_names = []
        for ch in range(n_channels):
            for name in base_names:
                feature_names.append(f'Ch{ch}_{name}')
                
        return feature_names
