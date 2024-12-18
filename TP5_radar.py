# -*- coding: utf-8 -*-
"""

@author: fdadam
"""

#%% Libs and functions
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq

def fastconv(A,B):
    out_len = len(A)+len(B)-1
    
    # Next nearest power of 2
    sizefft = int(2**(np.ceil(np.log2(out_len))))
    
    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))
    
    fftA = fft(Afilled)
    fftB = fft(Bfilled)
    
    fft_out = fftA * fftB
    out = ifft(fft_out)
    
    out = out[0:out_len]
    
    return out

#%% Parameters

c = 3e8 # speed of light [m/s]
k = 1.380649e-23 # Boltzmann

fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 100 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs

Te = 5e-6 # Tx recovery Time[s]
Tp = 10e-6 # Tx Pulse Width [s]
BW = 2e6 # Tx Chirp bandwidth [Hz]
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength [m]
kwave = 2*np.pi/wlen # Wavenumber [rad/m]
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range [m]
vu_ms = wlen*PRF/2 # Unambigous Velocity [m/s]
vu_kmh = vu_ms*3.6 # Unambigous Velocity [km/h]

rank_min = (Tp/2+Te)*c/2 # Minimum Range [m]
rank_max = 30e3 # Maximum Range [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2 # Range Step [m]
tmax = 2*rank_max/c # Maximum Simulation Time

radar_signal = pd.read_csv('signal_1.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)
radar_signal_f=fft(radar_signal,norm='ortho') # radar_signal (f)
radar_signal_f_l=radar_signal_f[0:50]
radar_signal_f_r=radar_signal_f[50:100]
radar_signal_ff=np.concatenate((radar_signal_f_r,radar_signal_f_l))




#%%

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')
matched_filter_f_l=matched_filter_f[0:1000]
matched_filter_f_r=matched_filter_f[1000:2000]
matched_filter_ff=np.concatenate((matched_filter_f_r,matched_filter_f_l))

compressed_signal_ff = np.zeros_like(radar_signal_ff)
for row in range(Np):
    compressed_signal_ff[row] = matched_filter_ff*radar_signal_ff[row]
    
compressed_signal=ifft(compressed_signal_ff)

compressed_signal_t = np.zeros_like(radar_signal)
for row in range(Np):
    compressed_signal_t[row] = np.convolve(matched_filter,radar_signal[row],mode="same")
    






#%% Plot Signals

fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.real(radar_signal[0]))
ax.plot(ranks/1e3,np.imag(radar_signal[0]))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Uncompressed and compressed Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx Raw signal')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(compressed_signal_t[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx compressed signal')
ax.grid(True)



#%% MTI Canceller

# Inicializar la señal MTI
mti_signal = np.zeros_like(radar_signal) #Crea un array del mismo tamaño de la señal

# Aplicar el MTI cancelador simple
for i in range(1, Np):
    mti_signal[i] = radar_signal[i] - radar_signal[i-1] ##Se utiliza en Doppler

# Señal comprimida después del MTI cancelador
compressed_mti_signal = np.zeros_like(compressed_signal_t)
for row in range(1, Np):
    compressed_mti_signal[row] = np.convolve(matched_filter, mti_signal[row], mode="same")
    
    

#%% CFAR - Ventana

cantCeros=30
cantUnos=100
ventana=np.concatenate((np.ones(cantUnos), np.zeros(cantCeros), np.ones(cantUnos)))
ventana/=200    #Altura ventana

#%% MTI - CANCELADOR SIMPLE - PARA DOS MUESTRAS
k=3.8             #Ajuste de Ganancia
compressed_signal_mti= np.abs(compressed_signal_t[1]-compressed_signal_t[0])
MTI_threshold= k*np.convolve( compressed_signal_mti , ventana , mode='same')

# Aplicar el umbral a la señal comprimida después del MTI

detection_cfar = np.zeros_like(compressed_signal_mti)

# Detectar si la señal supera el umbral
for row in range(1, len(compressed_signal_mti)):
    if compressed_signal_mti[row] > MTI_threshold[row]:
        detection_cfar[row] = compressed_signal_mti[row]

#Copio la señal CFAR
detection_cfar_mti=detection_cfar 

#Proceso la señal detection_cfar_mti sin np.sign
detection_cfar_mti=np.diff(detection_cfar)
detection_cfar_mti= np.concatenate((np.zeros(1),detection_cfar_mti))
detection_cfar_mti[np.where(detection_cfar_mti<0)] = 0
detection_cfar_mti = detection_cfar_mti.reshape(1, 2000)

#Procesamiento normal de la señal para obtener un solo resultado por elemento movil
detection_cfar=np.sign(detection_cfar)
detection_cfar=np.diff(detection_cfar)
detection_cfar = np.concatenate((np.zeros(1),detection_cfar))
detection_cfar[np.where(detection_cfar<0)] = 0
detection_cfar = detection_cfar.reshape(1, 2000)




# for idx in 
#%% PLOT SIGNAL 0 Y 1
fig, axes = plt.subplots(4,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.plot(ranks/1e3,np.abs(radar_signal[1]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)


#Compressed Signal

ax = axes[1]
ax.plot(ranks/1e3,np.abs(compressed_signal_t[0]))
ax.plot(ranks/1e3,np.abs(compressed_signal_t[1]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)


#Compressed MTI Signal

ax = axes[2]
ax.plot(ranks/1e3,np.abs(compressed_mti_signal[1]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[2]                                #PLOT threshold
ax.plot(ranks/1e3,np.abs(MTI_threshold))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

#Deteccion CFAR

ax = axes[3]
ax.plot(ranks/1e3, (detection_cfar[0]),marker='o')
ax.set_ylabel('Abs value')
ax.set_xlabel('Detected Signal after MTI')
ax.grid(True)

#%% Doppler

mti_signal_doppler=np.zeros_like(radar_signal)

for i in range(1, Np):
    mti_signal_doppler[i] = compressed_signal_t[i] - compressed_signal_t[i-1] 

mti_signal_doppler_T=(mti_signal_doppler).T

cols = np.arange(1,Np+1)
rows = np.arange(1,Np+1).reshape(Np,-1)

FD = np.exp(-1j*2*np.pi*(cols*rows)/Np)


filtered_signaL_doppler=np.matmul(mti_signal_doppler_T,FD).T


# Plot 3D

# Definir los valores X e Y
y = ranks
x = -np.linspace(-vu_ms/2, vu_ms/2, Np)
X, Y = np.meshgrid(x, y)

# Definir Z 
Z=np.abs(fftshift(filtered_signaL_doppler,axes=0)).T

# Crear la figura y los ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear la gráfica de superficie
ax.plot_surface(X, Y, Z, cmap='coolwarm')

# Mostrar la gráfica
plt.show()


#%% MTI - CANCELADOR DOBLE - PARA TRES MUESTRAS
k=3.95             #Ajuste de Ganancia
compressed_signal_mti_dc= np.abs(compressed_signal_t[2]-2*compressed_signal_t[1]+compressed_signal_t[0])

MTI_threshold= k*np.convolve( compressed_signal_mti_dc , ventana , mode='same')

# Aplicar el umbral a la señal comprimida después del MTI

detection_cfar = np.zeros_like(compressed_signal_mti_dc)

# Detectar si la señal supera el umbral
for row in range(1, len(compressed_signal_mti_dc)):
    if compressed_signal_mti_dc[row] > MTI_threshold[row]:
        detection_cfar[row] = compressed_signal_mti_dc[row]


detection_cfar=np.sign(detection_cfar)
detection_cfar=np.diff(detection_cfar)
# detection_cfar=np.insert(detection_cfar,[0],[0])
detection_cfar = np.concatenate((np.zeros(1),detection_cfar))

detection_cfar[np.where(detection_cfar<0)] = 0
detection_cfar = detection_cfar.reshape(1, 2000)


# for idx in 
#%% PLOT SIGNAL 0 Y 1
fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')


#Compressed MTI Signal

compressed_signal_mti_dc_T=compressed_signal_mti_dc.reshape(1, 2000)

ax = axes[0]
ax.plot(ranks/1e3,np.abs(compressed_signal_mti_dc_T[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[0]                                #PLOT threshold
ax.plot(ranks/1e3,np.abs(MTI_threshold))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

#Deteccion CFAR

ax = axes[1]
ax.plot(ranks/1e3, (detection_cfar[0]),marker='o')
ax.set_ylabel('Abs value')
ax.set_xlabel('Detected Signal after MTI - Cancelador Doble')
ax.grid(True)




#%% STI - SIMPLE CANCELADOR - PARA DOS MUESTRAS
k=10            #Ajuste de Ganancia
compressed_signal_mti_sc= np.abs(compressed_signal_t[1]+compressed_signal_t[0])

MTI_threshold= k*np.convolve( compressed_signal_mti_sc , ventana , mode='same')

# Aplicar el umbral a la señal comprimida después del MTI

detection_cfar = np.zeros_like(compressed_signal_mti_sc)

# Detectar si la señal supera el umbral
for row in range(1, len(compressed_signal_mti_sc)):
    if compressed_signal_mti_sc[row] > MTI_threshold[row]:
        detection_cfar[row] = compressed_signal_mti_sc[row]


detection_cfar=np.sign(detection_cfar)
detection_cfar=np.diff(detection_cfar)
# detection_cfar=np.insert(detection_cfar,[0],[0])
detection_cfar = np.concatenate((np.zeros(1),detection_cfar))

detection_cfar[np.where(detection_cfar<0)] = 0
detection_cfar = detection_cfar.reshape(1, 2000)


# for idx in 
#%% PLOT SIGNAL 0 Y 1
fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')


#Compressed MTI Signal

compressed_signal_mti_sc_T=compressed_signal_mti_sc.reshape(1, 2000)

ax = axes[0]
ax.plot(ranks/1e3,np.abs(compressed_signal_mti_sc_T[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[0]                                #PLOT threshold
ax.plot(ranks/1e3,np.abs(MTI_threshold))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

#Deteccion CFAR

ax = axes[1]
ax.plot(ranks/1e3, (detection_cfar[0]),marker='o')
ax.set_ylabel('Abs value')
ax.set_xlabel('Detected Signal after STI')
ax.grid(True)

#Deteccion CFAR - TEST

detection_cfar=detection_cfar-100000*detection_cfar_mti
detection_cfar[np.where(detection_cfar<0)] = 0



ax = axes[2]
ax.plot(ranks/1e3, (detection_cfar[0]),marker='o')
ax.set_ylabel('Abs value')
ax.set_xlabel('Detected Signal after STI')
ax.grid(True)

