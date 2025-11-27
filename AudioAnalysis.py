import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.signal import welch, spectrogram, butter, filtfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ======================================================= FILE ========================================================== #
def read(data_fpath):
    data_fpath = data_fpath
    y, sr = librosa.load(data_fpath, sr=None, mono=True)
    N = len(y)
    t = np.arange(N) / sr
    print(f'-------- Audio Information --------')
    print(f'- audio filename : {os.path.basename(data_fpath)}')
    print(f'- audio length : {N / sr} seconds')
    print(f'- audio sampling rate : {sr} Hz')
    print(f'- audio shape : {N}\n')
    return t, y, sr

def listen(y, sr):
    print(f'-------- Listening Audio --------')
    sd.play(y, samplerate=sr)
    sd.wait()
    print(f'Listening Finished\n\n')

def save(y, sr, save_fpath):
    sf.write(save_fpath, y, sr)
    print(f'--------Saved to {save_fpath} ---------\n\n')


# ================================================= Amplitude Analysis ===================================================== #
def cal_mean_power(y, print_val=False):
    pwr = np.mean(y ** 2)
    if print_val:
        print(f'signal power : {pwr}')
    return pwr

def cal_rms(y, print_val=False):
    rms = np.sqrt(cal_mean_power(y))
    if print_val:
        print(f"signal rms : {rms}")
    return rms

def cal_spl(y, print_val=False):
    p_ref = 20 * 1e-6
    rms = cal_rms(y)
    spl = 20 * np.log10(rms / p_ref)
    if print_val:
        print(f'signal dB : {spl}')
    return spl


# ================================================= Temporal Analysis ===================================================== #
def do_gcc_phat(y1, y2, sr, max_time=None):
    n = y1.shape[0] + y2.shape[0]

    Y1 = np.fft.rfft(y1, n)  # n : odd, shape : (n+1)/2
    Y2 = np.fft.rfft(y2, n)  # n : even, shape : (n/2)+1

    R = Y1 * np.conj(Y2)
    R /= np.abs(R)

    cc = np.fft.ifft(R, n=n)
    max_shift = int(n / 2)
    if max_time:
        max_shift = np.minimum(int(sr * max_time), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    estimated_delay = shift / sr  # (-) : y2 is more delayed, (+) : y1 is more delayed
    return estimated_delay


# ================================================= Frequency Analysis ===================================================== #
def cal_fft(y, sr):
    p_ref = 20 * 1e-6
    N = len(y)
    freq = np.arange(N) * sr / N
    fft_comp = np.fft.fft(y) / N
    fft_mag = 2 * np.abs(fft_comp)
    fft_db = 20 * np.log10(fft_mag / p_ref)
    return freq, fft_comp, fft_mag, fft_db

def cal_welch(y, sr, window='hann', nperseg=1024, noverlap=512):
    p_ref = 20 * 1e-6
    fpsd, psd = welch(y, sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    psd_db = 20 * np.log10(np.sqrt(psd) / p_ref)
    return fpsd, psd, psd_db


# ================================================= Time-Frequency Analysis ================================================= #
def cal_spectrogram(y, sr, window='hann', nperseg=1024, noverlap=512):
    Sfreq, Stime, Sxx = spectrogram(y, sr, window=window, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx)
    return Sfreq, Stime, Sxx, Sxx_db


# ================================================= Amplitude Transform ================================================= #
def scale_to_db(y, db, print_val=False):
    p_ref = 20 * 1e-6
    current_rms = cal_rms(y)
    spl_rms = p_ref * 10 ** (db / 20)
    scale = spl_rms / current_rms
    if print_val:
        print(f'previous spl : {cal_spl(y)} dB / after spl : {db} dB')
    return y * scale

def make_noisy_gaussian(y, snr, print_val=False):
    rms = cal_rms(y)
    snr_linear = 10 ** (snr / 20)
    rms_ns = rms / snr_linear
    gauss = rms_ns * np.random.normal(size=len(y))
    noisy_signal = y + gauss
    if print_val:
        print(f'signal spl : {cal_spl(y)} dB / noise spl : {cal_spl(gauss)} dB')
    return noisy_signal


# ============================================= Frequency Transform (Filter) ============================================= #
def highpass(y, sr, f_cut, order=8):
    nyquist = sr / 2
    normalized_cutoff = f_cut / nyquist
    b, a = butter(order, normalized_cutoff, btype='high')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def lowpass(y, sr, f_cut, order=8):
    nyquist = sr / 2
    normalized_cutoff = f_cut / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def bandpass(y, sr, f_low, f_high, order=8):
    nyquist = sr / 2
    normalized_freqs = [f_low / nyquist, f_high / nyquist]
    b, a = butter(order, normalized_freqs, btype='band')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

# ================================================= Temporal Transform  ================================================== #
def resample_audio(y, sr_orig, sr_target):
    y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)
    return y_resampled


# ================================================= Plotly Visualization ================================================= #
def plot_temporal_analysis(t, y, sr, nperseg=1024, noverlap=512):
    Sfreq, Stime, _, Sxx_db = cal_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        subplot_titles=('Waveform', 'Spectrogram'),
        specs=[[{'type': 'scatter'}],
               [{'type': 'heatmap'}]],
        vertical_spacing=0.08
    )

    fig.add_trace(
        go.Scatter(x=t, y=y, mode='lines', line=dict(color='#1f77b4', width=1)),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    fig.add_trace(
        go.Heatmap(x=Stime, y=Sfreq, z=Sxx_db, colorscale='Cividis',
                   colorbar=dict(title='dB', x=1.02, len=0.7, y=0.35)),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)

    fig.update_xaxes(matches='x', row=1, col=1)
    fig.update_xaxes(matches='x', row=2, col=1)

    fig.update_layout(height=700, template='plotly_white', showlegend=False)

    return fig


def plot_frequency_analysis(y, sr, nperseg=1024, noverlap=512, freq_range=(20, None)):
    freq, _, _, fft_db = cal_fft(y, sr)
    fpsd, _, psd_db = cal_welch(y, sr, nperseg=nperseg, noverlap=noverlap)

    fig = go.Figure()

    N = len(freq)
    fig.add_trace(
        go.Scatter(x=freq[:N // 2], y=fft_db[:N // 2], mode='lines',
                   name='FFT', line=dict(color='#ff7f0e', width=1))
    )

    fig.add_trace(
        go.Scatter(x=fpsd, y=psd_db, mode='lines',
                   name='Welch', line=dict(color='#2ca02c', width=1))
    )

    freq_min = freq_range[0] if freq_range[0] is not None else 20
    freq_max = freq_range[1] if freq_range[1] is not None else sr / 2

    fig.update_layout(
        title='Frequency Analysis',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude (dB)',
        xaxis_type='log',
        xaxis_range=[np.log10(freq_min), np.log10(freq_max)],
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig