import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import convolve, correlate
from scipy.stats import linregress

from plotter import *
from read_listen_save import *

# %% Step 1. Trim relative region (overlapped) to input signal from measured signal
def trim_by_xcorr(input_sig, measured_sig, sr, width=800, height=500):
    rir_full = correlate(measured_sig, input_sig, mode='full', method='fft')
    peak_idx = np.argmax(rir_full) - (len(input_sig) - 1)
    print(f'len(rir_full) = {len(rir_full)}')
    print(f'len(input_sig) = {len(input_sig)}')
    print(f'len(measured_sig) = {len(measured_sig)}')
    print(f'peak idx: {peak_idx}')

    x = np.linspace(0, (len(measured_sig)+len(input_sig)-1)/sr, len(measured_sig)+len(input_sig)-1)
    y_input = input_sig
    y_measured = measured_sig
    y_trimmed = y_measured[peak_idx:]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('<b>Input and Measured</b>', '<b>Cross Correlation</b>', '<b>Input and Trimmed Measured</b>'),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_input,
            mode='lines',
            name='input',
            line=dict(color='black', width=1),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_measured,
            mode='lines',
            name='measured',
            line=dict(color='red', width=1),
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=peak_idx / sr,
        line_color='green',
        annotation_text='Peak',
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rir_full,
            mode='lines',
            name='correlation',
            line=dict(color='green', width=1),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_input,
            mode='lines',
            name='input',
            line=dict(color='black', width=1),
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_trimmed,
            mode='lines',
            name='measured_trim',
            line=dict(color='red', width=1),
        ),
        row=3, col=1
    )
    fig.update_xaxes(
        title_text='<b>Time (sec)</b>',
        title_font=dict(family='Arial', size=16, color='black'),
        showgrid=True,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text='<b>Amplitude</b>',
        title_font=dict(family='Arial', size=16, color='black'),
        showgrid=True,
        gridcolor='lightgray',
        zerolinecolor='lightgray'
    )

    fig.update_layout(
        plot_bgcolor='white',
        width=width,
        height=height
    )
    fig.show()
    return y_trimmed


# %% Step 2. Calculate rir by fft (Wiener deconvolution) from input signal and measured signal
def cal_rir_fft(input_sig, measured_sig, sr, trim=2, plot=True):
    # FFT
    n_fft = len(measured_sig) + len(input_sig) - 1
    Y = np.fft.rfft(measured_sig, n=n_fft)
    X = np.fft.rfft(input_sig, n=n_fft)

    # Wiener Deconvolution
    signal_power = np.abs(X) ** 2
    noise_power = 0.1 * np.mean(signal_power)
    H = Y * np.conj(X) / (signal_power + noise_power)

    # IFFT
    rir = np.fft.irfft(H, n=n_fft)

    # Direct sound detection
    peak_idx = np.argmax(np.abs(rir))

    # Trim (2 seconds)
    start = max(0, peak_idx - int(0.01 * sr))
    end = start + int(sr * trim)
    rir = rir[start:end]

    # Normalize
    rir = rir / np.max(np.abs(rir))

    if plot:
        plot_wave(rir, sr, name='rir', title='room impulse response', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)

    return rir


# %% Step 3. Calculate RT60 from rir using Schroeder integration method
def cal_rt60(rir, sr, start_db=-5, end_db=-35, plot=True):
    # Find peak position
    peak_idx = np.argmax(np.abs(rir))
    rir = rir[peak_idx:]  # Trim to start from peak

    # Schroeder backward integration
    energy = rir**2
    edc = np.cumsum(energy[::-1])[::-1]  # Backward cumulative sum

    # Convert to dB
    edc_db = 10 * np.log10(edc / edc[0] + 1e-10)

    # Time axis
    time = np.arange(len(edc)) / sr

    # Find indices for linear regression range
    idx_start = np.where(edc_db >= start_db)[0][-1] if np.any(edc_db >= start_db) else 0
    idx_end = np.where(edc_db >= end_db)[0][-1] if np.any(edc_db >= end_db) else len(edc_db) - 1

    # Linear regression on the specified range
    if idx_end > idx_start:
        slope, intercept, r_value, p_value, std_err = linregress(
            time[idx_start:idx_end],
            edc_db[idx_start:idx_end]
        )

        # Calculate RT60 from slope, RT60 = time for 60dB decay
        rt60 = -60 / slope

        # Fitted line for visualization
        fit_line = slope * time + intercept

        if plot:
            fig = go.Figure()

            # EDC curve
            fig.add_trace(go.Scatter(
                x=time, y=edc_db,
                mode='lines',
                name='Energy Decay Curve',
                line=dict(color='blue', width=2)
            ))

            # Linear regression range
            fig.add_trace(go.Scatter(
                x=time[idx_start:idx_end],
                y=edc_db[idx_start:idx_end],
                mode='markers',
                name=f'Regression range ({start_db} to {end_db} dB)',
                marker=dict(color='red', size=3)
            ))

            # Fitted line
            fig.add_trace(go.Scatter(
                x=time, y=fit_line,
                mode='lines',
                name=f'Linear fit (RT60={rt60:.3f}s)',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Horizontal lines at start_db and end_db
            fig.add_hline(y=start_db, line_dash="dot", line_color="green",
                         annotation_text=f"{start_db} dB")
            fig.add_hline(y=end_db, line_dash="dot", line_color="orange",
                         annotation_text=f"{end_db} dB")

            fig.update_layout(
                title=f'<b>Energy Decay Curve - RT60 = {rt60:.3f} s</b>',
                xaxis_title='<b>Time (s)</b>',
                yaxis_title='<b>Level (dB)</b>',
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                width=1000,
                height=600
            )

            fig.show()

        print(f'RT60 = {rt60:.3f} seconds')
        print(f'R² = {r_value**2:.4f}')

        return rt60, edc_db


# %% Step 4. Convolution with anechoic signal to generate signal sounds like in a room
def rir_convolve(anechoic_sig, sr, rir, plot=True):
    """Convolution of anechoic siganl wifh room impulse response"""
    output = convolve(anechoic_sig, rir, mode='full', method='fft')
    # normalisation to decent level
    output = 0.7 * output / np.max(np.abs(output))
    if plot:
        plot_wave(output, sr, name='convolved', title='convolved signal with rir', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)
    return output


# %% Main function
if __name__ == "__main__":
    # Step 0. Read audio
    input_fpath = 'data/sweep_input.wav'
    measured_fpath = 'data/HATS_L.wav'
    input, sr_i = read_audio(input_fpath)
    measured, sr_m = read_audio(measured_fpath)
    plot_wave(input, sr_i, name='input', title='input signal', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)
    plot_wave(measured, sr_m, name='meausred', title='measured_signal', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)

    input = input[:sr_i*15]
    measured = measured[:sr_m*15]
    # # Step 1. Trim audio to matching
    # measured_trim = trim_by_xcorr(input, measured, sr_i, width=1200, height=1200)

    # Step 2. Calculate rir by Wiener deconvolution
    rir = cal_rir_fft(input, measured, sr_m)
    rir_save_fpath = './data/rir_L.wav'
    save_audio(rir, sr_m, rir_save_fpath)

    # Step 3. Calculate RT60
    rt60, edc_db = cal_rt60(rir, sr_i, start_db=-5, end_db=-35, plot=True)

    # Step 4. Convolve anechoic signal with rir
    anechoic_fpath = 'data/anechoic_orchestra.wav'
    anechoic, sr_a = read_audio(anechoic_fpath)
    plot_wave(anechoic, sr_a, name='anechoic', title='anechoic_signal', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)
    orchestra_conv = rir_convolve(anechoic, sr_a, rir)
    orchestra_conv_save_fpath = './data/orchestra_at_stoller.wav'
    save_audio(orchestra_conv, sr_a, orchestra_conv_save_fpath)