"""
Audio Input Layer — Real-time Microphone Capture & Feature Extraction
=====================================================================
Simulates MEMS microphone input for the VirtualESP32Node.
Uses sounddevice for real-time capture and librosa for MFCC extraction.
"""

import numpy as np
import librosa
import threading
import queue
import time
import logging
from typing import Optional, Tuple, Callable

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Real-time audio capture from laptop microphone.
    Simulates MEMS microphone behavior for edge AI deployment.
    
    Supports two modes:
      - LIVE: Captures from system microphone via sounddevice
      - REPLAY: Replays audio files from the dataset (fallback)
    """
    
    SAMPLE_RATE = 22050          # Hz — standard for audio classification
    FRAME_DURATION = 1.0         # seconds per frame
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
    N_MFCC = 40                  # MFCC coefficients
    N_MELS = 128                 # Mel bands
    HOP_LENGTH = 512
    N_FFT = 2048
    ENERGY_THRESHOLD = 0.005     # Wake-on-sound energy threshold
    
    def __init__(self, mode: str = "live", replay_dir: Optional[str] = None):
        self.mode = mode.lower()
        self.replay_dir = replay_dir
        self.is_running = False
        self._audio_queue = queue.Queue(maxsize=30)
        self._stream = None
        self._thread = None
        self._current_frame = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        self._replay_files = []
        self._replay_index = 0
        self._lock = threading.Lock()
        
        if self.mode == "replay" and replay_dir:
            self._load_replay_files()
    
    def _load_replay_files(self):
        """Load audio file paths for replay mode."""
        import glob
        import os
        patterns = ["*.wav", "*.WAV"]
        for pattern in patterns:
            self._replay_files.extend(
                glob.glob(os.path.join(self.replay_dir, "**", pattern), recursive=True)
            )
        if self._replay_files:
            import random
            random.shuffle(self._replay_files)
            logger.info(f"[AUDIO] Loaded {len(self._replay_files)} files for replay mode")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream."""
        if status:
            logger.warning(f"[AUDIO] Stream status: {status}")
        audio_data = indata[:, 0].copy().astype(np.float32)
        try:
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass
    
    def start(self):
        """Start audio capture."""
        self.is_running = True
        if self.mode == "live" and SOUNDDEVICE_AVAILABLE:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=self.FRAME_SIZE,
                    callback=self._audio_callback
                )
                self._stream.start()
                logger.info("[AUDIO] Live microphone capture started")
            except Exception as e:
                logger.error(f"[AUDIO] Failed to start microphone: {e}")
                logger.info("[AUDIO] Falling back to replay mode")
                self.mode = "replay"
                if not self._replay_files and self.replay_dir:
                    self._load_replay_files()
        
        if self.mode == "replay":
            self._thread = threading.Thread(target=self._replay_loop, daemon=True)
            self._thread.start()
            logger.info("[AUDIO] Replay mode started")
    
    def _replay_loop(self):
        """Background thread for replaying dataset audio files."""
        while self.is_running and self._replay_files:
            try:
                file_path = self._replay_files[self._replay_index % len(self._replay_files)]
                audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.FRAME_DURATION)
                
                # Pad or truncate to FRAME_SIZE
                if len(audio) < self.FRAME_SIZE:
                    audio = np.pad(audio, (0, self.FRAME_SIZE - len(audio)), mode='constant')
                else:
                    audio = audio[:self.FRAME_SIZE]
                
                try:
                    self._audio_queue.put(audio, timeout=1.0)
                except queue.Full:
                    pass
                
                self._replay_index += 1
                time.sleep(self.FRAME_DURATION)
            except Exception as e:
                logger.error(f"[AUDIO] Replay error: {e}")
                self._replay_index += 1
                time.sleep(0.5)
    
    def stop(self):
        """Stop audio capture."""
        self.is_running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("[AUDIO] Audio capture stopped")
    
    def get_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """
        Get the next audio frame.
        
        Returns:
            numpy array of shape (FRAME_SIZE,) or None if timeout
        """
        try:
            frame = self._audio_queue.get(timeout=timeout)
            with self._lock:
                self._current_frame = frame.copy()
            return frame
        except queue.Empty:
            return None
    
    def get_current_frame(self) -> np.ndarray:
        """Get the most recent frame (non-blocking)."""
        with self._lock:
            return self._current_frame.copy()
    
    def compute_energy(self, frame: np.ndarray) -> float:
        """Compute RMS energy of audio frame (wake-on-sound simulation)."""
        return float(np.sqrt(np.mean(frame ** 2)))
    
    def is_sound_detected(self, frame: np.ndarray) -> bool:
        """Check if frame energy exceeds wake-on-sound threshold."""
        return self.compute_energy(frame) > self.ENERGY_THRESHOLD
    
    def extract_mfcc(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio frame.
        
        Returns:
            numpy array of shape (N_MFCC, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=frame,
            sr=self.SAMPLE_RATE,
            n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS
        )
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        return mfcc
    
    def extract_mel_spectrogram(self, frame: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram from audio frame."""
        mel_spec = librosa.feature.melspectrogram(
            y=frame,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess_for_model(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for model input.
        
        Returns:
            numpy array of shape (1, N_MFCC, time_steps, 1) ready for CNN
        """
        mfcc = self.extract_mfcc(frame)
        # Pad/truncate time dimension to fixed size (44 frames for 1s @ 22050Hz)
        target_length = 44
        if mfcc.shape[1] < target_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :target_length]
        
        # Reshape for CNN: (batch, height, width, channels)
        return mfcc.reshape(1, self.N_MFCC, target_length, 1).astype(np.float32)


def get_audio_devices() -> list:
    """List available audio input devices."""
    if not SOUNDDEVICE_AVAILABLE:
        return []
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': d['name'],
                    'channels': d['max_input_channels'],
                    'sample_rate': d['default_samplerate']
                })
        return input_devices
    except Exception:
        return []
