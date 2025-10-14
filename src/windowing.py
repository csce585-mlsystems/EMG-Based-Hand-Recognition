import numpy as np
from typing import Generator, Tuple, Optional
from collections import deque

class SlidingWindow:
    """
    Sliding window implementation for streaming EMG data.
    """
    
    def __init__(self, window_size: int, overlap: float = 0.5):
        """
        Initialize sliding window.
        
        Args:
            window_size: Number of samples per window
            overlap: Overlap ratio (0.0 to 1.0)
        """
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        
    def window_generator(self, data: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Generate windows from data.
        
        Args:
            data: Input data (samples x channels)
            
        Yields:
            Windows of shape (window_size, channels)
        """
        n_samples = len(data)
        
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            yield data[start_idx:end_idx]

class CircularBuffer:
    """
    Circular buffer for real-time streaming with automatic windowing.
    """
    
    def __init__(self, capacity: int, n_channels: int, 
                 window_size: int, overlap: float = 0.5):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum buffer capacity in samples
            n_channels: Number of EMG channels
            window_size: Samples per window
            overlap: Window overlap ratio
        """
        self.capacity = capacity
        self.n_channels = n_channels
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        
        self.buffer = np.zeros((capacity, n_channels))
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0
        self.windows_generated = 0
        
    def add_samples(self, samples: np.ndarray):
        """Add new samples to buffer."""
        n_new = len(samples)
        
        if n_new > self.capacity:
            # Take only the most recent samples
            samples = samples[-self.capacity:]
            n_new = self.capacity
            
        # Calculate indices
        end_idx = (self.write_idx + n_new) % self.capacity
        
        if end_idx > self.write_idx:
            # No wrap-around
            self.buffer[self.write_idx:end_idx] = samples
        else:
            # Wrap-around
            split_idx = self.capacity - self.write_idx
            self.buffer[self.write_idx:] = samples[:split_idx]
            self.buffer[:end_idx] = samples[split_idx:]
        
        self.write_idx = end_idx
        self.size = min(self.size + n_new, self.capacity)
        
    def get_window(self) -> Optional[np.ndarray]:
        """
        Get next window if available.
        
        Returns:
            Window array or None if insufficient data
        """
        if self.size < self.window_size:
            return None
            
        # Check if we have enough new data for next window
        samples_since_last = (self.write_idx - self.read_idx) % self.capacity
        if samples_since_last < self.step_size and self.windows_generated > 0:
            return None
            
        # Extract window
        window = np.zeros((self.window_size, self.n_channels))
        for i in range(self.window_size):
            idx = (self.read_idx + i) % self.capacity
            window[i] = self.buffer[idx]
            
        # Update read index
        self.read_idx = (self.read_idx + self.step_size) % self.capacity
        self.windows_generated += 1
        
        return window
    
    def reset(self):
        """Reset buffer state."""
        self.buffer.fill(0)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0
        self.windows_generated = 0
