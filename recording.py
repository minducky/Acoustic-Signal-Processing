import sounddevice as sd
from scipy.io import wavfile

from plotter import *


# %% Record audio using SoundDevice
def record_audio(sr=None, duration=None, save_fpath=None, plot=True):
    # Recording
    print("Start Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')[:, 0]
    sd.wait()  # Wait Until Finished
    print("Recording Finished")

    mean_power = np.mean(audio ** 2)
    p_ref = 20e-6
    mean_db = 10 * np.log10(mean_power / (p_ref**2))
    print(f'========= Audio mean dB : {mean_db} dB =========')

    # Save
    wavfile.write(save_fpath, sr, audio)
    print(audio.shape)
    print(sr)
    # Plot
    if plot:
        plot_wave(sig=audio, sr=sr, name=f'{save_fpath}', title=f'Recorded {save_fpath}', xaxis_title='Time (sec)', yaxis_title='Amplitude', width=800, height=600)


# %% Main function
if __name__ == '__main__':
    sr = 48000
    duration = 10
    save_fpath = 'data/recorded.wav'
    record_audio(sr=sr, duration=duration, save_fpath=save_fpath)

