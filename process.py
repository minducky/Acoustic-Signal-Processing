# %%
# process recordings to obtain impulse responses and plot; save them to wav files
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
import pandas as pd 
from os import listdir 
from os.path import isfile, join

f0, f1 = 20,20000

def process_raw_recordings(path, n, f0=f0, f1=f1):
    # path : file path
    # n : int, number of sweeps (assumed equal length)
    # f0, f1 : frequency limits of input sweep / excitation 

    files = [join(path,f) for f in listdir(path) if (isfile(join(path, f))) and ('.wav' in f)]

    h_list = []
    input_path = [f for f in files if "Input.wav" in f][0]
    fs, input_signal = wavfile.read(input_path)
    
    for fname in files:
        if "Input.wav" in fname:
            continue 

        fs, data = wavfile.read(fname)

        N = int(data.shape[0] // n)  
        N_tot = N * n 
        
        x =input_signal[:N_tot].reshape((n,N))
        y =data[:N_tot].reshape((n,N))
        
        X = np.fft.rfft(x, axis=1, n=N)
        Y = np.fft.rfft(y, axis=1, n=N) 
        
        Gxy = (Y*np.conj(X)).mean(axis=0)
        Gxx = (np.abs(X)**2).mean(axis=0)
        
        H = Gxy / Gxx

        H[np.isnan(H)]=0
        H[np.isinf(H)]=0
        
        freq = np.fft.rfftfreq(N, 1/fs)
        
        H[(np.abs(freq) < f0) | (np.abs(freq) > f1)] = 0
        
        time = np.arange(N)/fs 
        h = pd.Series(np.fft.irfft(H), index=time)
        
        if "SF_W.wav" in fname:
            h.name= 'W'
            h = h*np.sqrt(2) 
        elif "SF_X.wav" in fname:
            h.name= 'X'
        elif "SF_Y.wav" in fname:
            h.name= 'Y'
        elif "SF_Z.wav" in fname:
            h.name= 'Z' 
        elif "HATS_L.wav" in fname:
            h.name= 'L'
        elif "HATS_R.wav" in fname:
            h.name='R'
        
        h_list.append(h)

    data_out = pd.concat(h_list, axis=1)
    data_out.fs = fs 

    return data_out 

path = 'posA'
name = 'posA'

data = process_raw_recordings(path, 4)
fs = data.fs
data = data.iloc[:fs*3,:]  # first 3 seconds

HL = np.abs(np.fft.rfft(data.L))
HR = np.abs(np.fft.rfft(data.R))
HW = np.abs(np.fft.rfft(data.W))
HX = np.abs(np.fft.rfft(data.X))
HY = np.abs(np.fft.rfft(data.Y))
HZ = np.abs(np.fft.rfft(data.Z))

freq = np.fft.rfftfreq(len(data.L), 1/fs)

fig = plt.figure()
axt = fig.add_subplot(221)

axt.plot(data.index, data.L,'k', linewidth=0.5)
axt.plot(data.index, data.R,'r', linewidth=0.5)
axt.set_title("HATS, time domain")

axtsf = fig.add_subplot(222)
axtsf.plot(data.index, data.W,'k', linewidth=0.5)
axtsf.plot(data.index, data.X,'r', linewidth=0.5)
axtsf.plot(data.index, data.Y,'g', linewidth=0.5)
axtsf.plot(data.index, data.Z,'b', linewidth=0.5)
axtsf.set_title("SF, time domain")

axf = fig.add_subplot(223)
axf.plot(freq, 10*np.log10(HL),'k', linewidth=0.5)
axf.plot(freq, 10*np.log10(HR),'r', linewidth=0.5)
axf.set_title("HATS, frequency domain")

axfsf = fig.add_subplot(224)
axfsf.plot(freq, 10*np.log10(HW),'k', linewidth=0.5)
axfsf.plot(freq, 10*np.log10(HX),'r', linewidth=0.5)
axfsf.plot(freq, 10*np.log10(HY),'g', linewidth=0.5)
axfsf.plot(freq, 10*np.log10(HZ),'b', linewidth=0.5)
axfsf.set_title("SF, frequency domain")
fig.tight_layout()

#%%
#save to wav files

intfact = 2**31 - 1  # int32 max

norm = np.max([np.max(np.abs(data.L)), np.max(np.abs(data.R))])

hats_l = data.L / norm * 0.9 * intfact  
hats_r = data.R / norm * 0.9 * intfact

hw = data.W 
hx = data.X
hy = data.Y
hz = data.Z

norm = np.max([np.max(np.abs(hw)), np.max(np.abs(hx)), np.max(np.abs(hy)), np.max(np.abs(hz))])

hw = hw / norm * 0.9 * intfact / np.sqrt(2)
hx = hx / norm * 0.9 * intfact
hy = hy / norm * 0.9 * intfact
hz = hz / norm * 0.9 * intfact

# write two channel hats wav
hats_stereo = np.vstack((hats_l, hats_r)).T
wavfile.write(f'hats_{name}.wav', fs, hats_stereo.astype(np.int32))

sf_multi = np.vstack((hw, hx, hy, hz)).T
wavfile.write(f'sf_{name}.wav', fs, sf_multi.astype(np.int32))
