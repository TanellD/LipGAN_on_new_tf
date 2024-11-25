from dataclasses import dataclass
from glob import glob
import os, pickle


# Default hyperparameters using dataclass
@dataclass
class HParams:
    num_mels: int = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
    rescale: bool = True  # Whether to rescale audio prior to preprocessing
    rescaling_max: float = 0.9  # Rescaling value

    max_mel_frames: int = 900
    use_lws: bool = False

    n_fft: int = 800  # Extra window size is filled with 0 paddings to match this parameter
    hop_size: int = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size: int = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate: int = 16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms: float = None  # Can replace hop_size parameter. (Recommended: 12.5)

    signal_normalization: bool = True
    allow_clipping_in_normalization: bool = True  # Only relevant if mel_normalization = True
    symmetric_mels: bool = True
    max_abs_value: float = 4.0
    normalize_for_wavenet: bool = True
    clip_for_wavenet: bool = True

    preemphasize: bool = True  # whether to apply filter
    preemphasis: float = 0.97  # filter coefficient.

    min_level_db: int = -100
    ref_level_db: int = 20
    fmin: int = 55
    fmax: int = 7600  # To be increased/reduced depending on data.

    power: float = 1.5
    griffin_lim_iters: int = 60  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.


# Instantiate hyperparameters
hparams = HParams()

# Function to get a debug string with the hyperparameters
def hparams_debug_string():
    # Iterate through all parameters and format as a string
    return f"Hyperparameters:\n" + "\n".join([f"  {key}: {value}" for key, value in hparams.__dict__.items()])

# Example usage
# print(hparams_debug_string())
