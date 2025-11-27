# Acoustic Signal Processing

A Python library for acoustic signal analysis and visualization, designed for research in auditory neuroscience and audio processing.

## Features

### File Operations
- **Audio I/O**: Read, save, and play audio files
- **Format Support**: Compatible with standard audio formats via `librosa` and `soundfile`

### Amplitude Analysis
- Mean power calculation
- RMS (Root Mean Square) computation
- SPL (Sound Pressure Level) in dB

### Frequency Analysis
- **FFT**: Fast Fourier Transform with magnitude and dB conversion
- **Welch PSD**: Power Spectral Density estimation using Welch's method

### Time-Frequency Analysis
- **Spectrogram**: Time-frequency representation with dB scaling

### Signal Transforms
- **Amplitude**: 
  - dB scaling
  - Gaussian noise addition with specified SNR
- **Frequency Filters**:
  - Highpass, lowpass, bandpass, bandstop (Butterworth filters)
- **Temporal**:
  - Resampling
  - Silence trimming

### Visualization
- **Temporal Analysis Plot**: Waveform + Spectrogram with shared x-axis
- **Frequency Analysis Plot**: FFT + Welch PSD comparison
- Interactive plots using Plotly with Cividis colormap
