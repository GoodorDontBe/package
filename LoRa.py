# code:utf-8  	
# 2022-01-24 v3
# This script simulates LoRa transmission and Reception based on basic Chirps 

import struct
import math
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt
from numpy.lib.function_base import average

## Prepare class
class LoRa_parameter:
    def _init_(self,name,bw,fs,sf,a1,a2,t,upChirp,downChirp,N_up,N_down,N_sym,snr,preamble,data):
        # basic
        self.name = name
        self.bw = bw    # Chirp bandwidth
        self.fs = fs    # Physical sampling rate
        self.sf = sf    # Spread factor 
        self.a1 = a1
        self.a2 = a2
        self.t = t
        self.upChirp = upChirp
        self.downChirp = downChirp
        # trasmission
        self.N_up = N_up
        self.N_down = N_down
        self.N_sym = N_sym
        self.snr = snr
        self.preamble = preamble
        self.data = data
    def test_show(self):
        print('Hello World!')

## Prepare function
def sim_init():
    '''
    Instantiate class LoRa_parameter
    configure all the parameters
    '''
    # Instantiate
    param = LoRa_parameter()
    # Basic
    param.name = 'LoRa_parameter'
    param.bw = 500e3     # Chirp bandwidth
    param.fs = 10e6      # Physical sampling rate
    param.sf = 10        # Spread factor 
    param.a1 = 0/(2**param.sf)*param.bw - param.bw/2
    param.a2 = (param.bw**2)/(2**param.sf)
    param.t = np.linspace(0, (2**param.sf-1)/param.bw, num=1024)
    param.upChirp = np.exp(1j*2*math.pi*(0.5*param.a2*(param.t**2) + param.a1*param.t + 0))
    param.downChirp = np.exp(1j*2*math.pi*(-0.5*param.a2*(param.t**2) - param.a1*param.t + 0))
    # Trasmission
    param.N_up = 8
    param.N_down = 2.25
    param.N_sym = 10
    param.snr = 20       # SNR
    param.preamble = None
    param.data = None
    return param

def sim_tx(param):
    '''
    generate tx waveform, including preamble & payload
    configurate sim_param.preamble & data
    '''
    ## Build a complete LoRa preamble
    for i in range(0,param.N_up):
        if i==0:
            preamble_up = param.upChirp
        else:
            preamble_up = np.hstack((preamble_up, param.upChirp))   # combine by row
    for i in range(0,int(math.floor(param.N_down))):
        if i==0:
            preamble_down = param.downChirp
        else:
            preamble_down = np.hstack((preamble_down, param.downChirp))
    tmp_ind1 = round((param.N_down - math.floor(param.N_down))*(2**param.sf))
    preamble_down_residual = param.downChirp[0:tmp_ind1]
    preamble = np.hstack((preamble_up,preamble_down,preamble_down_residual))

    ## LoRa payload
    for i in range(0,param.N_sym):
        tmp_data = np.random.randint(0, high=2**param.sf, size=None, dtype='l') # left close right open 0-1023
        if i==0:
            data = tmp_data
            if(tmp_data==0):
                tmp_wave = param.upChirp
                wave_payload = tmp_wave
            else:
                tmp_wave = np.hstack((param.upChirp[2**param.sf-tmp_data:2**param.sf],param.upChirp[0:2**param.sf-tmp_data]))   # left close right open
                wave_payload = tmp_wave
        else:
            data = np.hstack((data,tmp_data))
            if(tmp_data==0):
                tmp_wave = param.upChirp
                wave_payload = np.hstack((wave_payload, tmp_wave)) 
            else:
                tmp_wave = np.hstack((param.upChirp[2**param.sf-tmp_data:2**param.sf],param.upChirp[0:2**param.sf-tmp_data])) 
                wave_payload = np.hstack((wave_payload, tmp_wave)) 
    tx_vec = np.hstack((preamble,wave_payload))

    ## Build tx frame (resample)
    len_baseline = len(tx_vec)
    len_upsample = int(len(tx_vec)*(param.fs/param.bw))
    t_baseline = np.linspace(0, (len_baseline-1)/param.bw, num=len_baseline)
    t_upsample = np.linspace(0, (len_baseline-1)/param.bw, num=len_upsample)
    # resample I
    y_linear_I = interpolate.interp1d(t_baseline,tx_vec.real)  
    tck1 = interpolate.splrep(t_baseline,tx_vec.real) 
    y_spline_I = interpolate.splev(t_upsample,tck1)
    # resample Q
    y_linear_Q = interpolate.interp1d(t_baseline,tx_vec.imag)    
    tck2 = interpolate.splrep(t_baseline,tx_vec.imag)
    y_spline_Q = interpolate.splev(t_upsample,tck2)

    tx_vec_air = y_spline_I + 1j*y_spline_Q

    ## Display information
    param.preamble = preamble
    param.data = data
    # print('preamble:',preamble.shape)
    # print('wave_payload:',wave_payload.shape)
    # print('tx_vec:',tx_vec.shape)
    # print('tx_vec_air:',tx_vec_air.shape)
    print('tx_data:',data,data.dtype)

    return tx_vec_air, param

def sim_channel(param, tx_vec_air):
    '''
    Simulation for an AWGN channel
    get SNR from sim_param.snr
    '''
    tmp_len = len(tx_vec_air)
    snr_dB = param.snr
    snr = 10**(snr_dB/10.0)
    xpower_I = np.sum(tx_vec_air.real**2)/tmp_len
    xpower_Q = np.sum(tx_vec_air.imag**2)/tmp_len
    npower_I = xpower_I / snr
    npower_Q = xpower_Q / snr
    n_I = np.random.randn(tmp_len) * np.sqrt(npower_I)
    n_Q = np.random.randn(tmp_len) * np.sqrt(npower_Q)
    rx_vec_air = (tx_vec_air.real + n_I) + 1j*(tx_vec_air.imag + n_Q)
    
    return rx_vec_air

def sim_rx(param, rx_vec_air):
    '''
    To do rx digital siganl processing
    step1: Synchronization by corr method
    step2: Demodulation by fft peak
    '''
    ## Synchronization with LoRa signal
    len_baseline = len(param.preamble)
    len_upsample = int(len(param.preamble)*(param.fs/param.bw))
    t_baseline = np.linspace(0, (len_baseline-1)/param.bw, num=len_baseline)
    t_upsample = np.linspace(0, (len_baseline-1)/param.bw, num=len_upsample)
    # resample the preamble
    y_linear_I = interpolate.interp1d(t_baseline,param.preamble.real)    
    y_linear_Q = interpolate.interp1d(t_baseline,param.preamble.imag) 
    tck1 = interpolate.splrep(t_baseline,param.preamble.real)   
    tck2 = interpolate.splrep(t_baseline,param.preamble.imag) 
    y_spline_I = interpolate.splev(t_upsample,tck1)
    y_spline_Q = interpolate.splev(t_upsample,tck2)

    preamble_air = y_spline_I + 1j*y_spline_Q

    ## Sync correlation using preamble
    tmp_reversed = preamble_air[::-1]
    tmp_conj = tmp_reversed.conjugate()
    tmp_conv = signal.convolve(tmp_conj, rx_vec_air)
    preamble_corr = abs(tmp_conv)

    ## Get the payload waveform
    ind = preamble_corr.argmax()    # index is from 0-1023
    ind1 = ind + 1
    ind2 = int(ind + (param.fs/param.bw)*param.N_sym*(2**param.sf) + 1)
    rx_payload_air = rx_vec_air[ind1:ind2] 

    ## Timing sync (resample the rx_payload_air)
    offset_sync = 15
    for i in range(0,int(param.N_sym*2**param.sf)):
        tmp_ind = offset_sync + i*int(param.fs/param.bw)
        if i==0:
            rx_vec = rx_payload_air[tmp_ind]
        else:
            rx_vec = np.hstack((rx_vec,rx_payload_air[tmp_ind]))
    
    ## Demodulate payload
    demod_result = np.zeros(param.N_sym)
    for i in range(0,param.N_sym):
        tmp = rx_vec[i*1024:(i+1)*1024] * param.downChirp
        tmp_fft = abs(fft(tmp))
        max_ind = tmp_fft.argmax()              # 0-1023
        if(max_ind==0):
            demod_result[i] = 0
        else:
            demod_result[i] = 1024 - max_ind    # 1023-0
    demod_result = demod_result.astype(np.int32)

    ## Display information
    print('rx_data:',demod_result,demod_result.dtype)
    # print('rx_payload_air:',rx_payload_air.shape)
    # print('rx_vec:',rx_vec.shape)
    # print('ind1:',ind1)
    # print('ind2:',ind2)

    return demod_result

def sim_er(param, demod_result):
    '''
    calculate ber & ser for this simulation
    '''
    ser = 0
    ber = 0
    for i in range(0,param.N_sym):
        if demod_result[i] != param.data[i]:
            ser = ser + 1
        else:
            ser = ser
    ser = ser / param.N_sym
    return ber, ser

def simulation_LoRa():
    print("#### Simulation start! ####")
    sim_param = sim_init()                          # init
    tx_vec_air, sim_param = sim_tx(sim_param)       # tx
    rx_vec_air = sim_channel(sim_param, tx_vec_air) # awgn
    demod_result = sim_rx(sim_param, rx_vec_air)    # rx
    ber, ser = sim_er(sim_param, demod_result)      # ber
    print('SNR =',sim_param.snr)
    print('ser =',ser)
    print('#### Simulation finish! ####')

    return 0


if __name__ == '__main__':
    ## Wrapper
    simulation_LoRa()
