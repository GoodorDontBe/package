# code:utf-8  	
# c.r. Kevin Lee
# 2022-01-22
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
    param = LoRa_parameter()
    # basic
    param.name = 'LoRa_parameter'
    param.bw = 500e3     # Chirp bandwidth
    param.fs = 10e6      # Physical sampling rate
    param.sf = 10        # Spread factor 
    param.a1 = 0/(2**param.sf)*param.bw - param.bw/2
    param.a2 = (param.bw**2)/(2**param.sf)
    # param.t = np.arange(0, (2**param.sf-1)/param.bw, 1/param.bw)  
    param.t = np.linspace(0, (2**param.sf-1)/param.bw, num=1024)
    # np.arange存在一定隐患，可能是内部运算精度的问题，不能完全保证点数为1024
    param.upChirp = np.exp(1j*2*math.pi*(0.5*param.a2*(param.t**2) + param.a1*param.t + 0))
    param.downChirp = np.exp(1j*2*math.pi*(-0.5*param.a2*(param.t**2) - param.a1*param.t + 0))
    # trasmission
    param.N_up = 8
    param.N_down = 2.25
    param.N_sym = 10
    param.snr = 20
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
    # plt.plot(np.arange(len(preamble)),preamble.real)
    # plt.show()
    ## LoRa payload
    test_data = np.array([0,256,3,4,5,6,7,8,9,10])
    # test_data = np.array([1,2,3,4,5,6,7,8,9,10])
    for i in range(0,param.N_sym):
        # tmp_data = np.random.randint(0, high=2**param.sf, size=None, dtype='l') # left close right open 0-1023
        tmp_data = test_data[i]
        if i==0:
            data = tmp_data
            if(tmp_data==0):
                tmp_wave = param.upChirp
                wave_payload = tmp_wave
            else:
                # tmp_wave = np.hstack((param.upChirp[0+tmp_data:2**param.sf], param.upChirp[0:tmp_data])) # left close right open
                tmp_wave = np.hstack((param.upChirp[2**param.sf-tmp_data:2**param.sf],param.upChirp[0:2**param.sf-tmp_data]))   # left close right open
                wave_payload = tmp_wave
        else:
            data = np.hstack((data,tmp_data))
            if(tmp_data==0):
                tmp_wave = param.upChirp
                wave_payload = np.hstack((wave_payload, tmp_wave)) 
            else:
                # tmp_wave = np.hstack((param.upChirp[0+tmp_data:2**param.sf], param.upChirp[0:tmp_data])) 
                tmp_wave = np.hstack((param.upChirp[2**param.sf-tmp_data:2**param.sf],param.upChirp[0:2**param.sf-tmp_data])) 
                wave_payload = np.hstack((wave_payload, tmp_wave)) 
    tx_vec = np.hstack((preamble,wave_payload))

    ## Build tx frame
    param.preamble = preamble
    param.data = data
    len_baseline = len(tx_vec)
    len_upsample = int(len(tx_vec)*(param.fs/param.bw))
    t_baseline = np.linspace(0, (len_baseline-1)/param.bw, num=len_baseline)
    t_upsample = np.linspace(0, (len_baseline-1)/param.bw, num=len_upsample)
    # print(t_baseline.shape)
    # print(t_upsample.shape)
    y_linear_I = interpolate.interp1d(t_baseline,tx_vec.real)    #这里的y_linear已经是插值后的结果了
    tck1 = interpolate.splrep(t_baseline,tx_vec.real) #这个必须有，splrep()的结果作为splev()的第二个参数
    y_spline_I = interpolate.splev(t_upsample,tck1)

    y_linear_Q = interpolate.interp1d(t_baseline,tx_vec.imag)    #这里的y_linear已经是插值后的结果了
    tck2 = interpolate.splrep(t_baseline,tx_vec.imag) #这个必须有，splrep()的结果作为splev()的第二个参数
    y_spline_Q = interpolate.splev(t_upsample,tck2)

    tx_vec_air = y_spline_I + 1j*y_spline_Q
    plt.subplot(2,1,1)
    plt.plot(t_baseline[0:32],tx_vec.real[0:32],linewidth='0.5',linestyle='dotted',marker='v',markersize=5)
    # plt.plot(t_upsample[0:32*20],tx_vec_air.real[0:32*20],linewidth='0.5',linestyle='dotted',marker='o',markersize=2)
    # plt.show()

    # plt.plot(np.arange(128),preamble.real[0:128])
    # plt.show() 
    # how to resample?
    # in MATLAB:
    # %   [Y,Ty] = RESAMPLE(X,Tx,Fs) uses interpolation and an anti-aliasing filter 
    # %   to resample the signal at a uniform sample rate, Fs, expressed in hertz (Hz).

    # print('preamble:',preamble.shape)
    # print('wave_payload:',wave_payload.shape)
    # print('tx_vec',tx_vec.shape)
    # print('tx_vec_air',tx_vec_air.shape)

    print('tx_data',data,data.dtype)
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
    
    # tmp_len = 1024*1
    # plt.plot(np.arange(tmp_len),tx_vec_air.real[0:tmp_len],linewidth='0.5')
    # plt.plot(np.arange(tmp_len),rx_vec_air.real[0:tmp_len],linewidth='0.5')
    # plt.show()
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
    
    y_linear_I = interpolate.interp1d(t_baseline,param.preamble.real)    
    tck1 = interpolate.splrep(t_baseline,param.preamble.real) 
    y_spline_I = interpolate.splev(t_upsample,tck1)
    y_linear_Q = interpolate.interp1d(t_baseline,param.preamble.imag)   
    tck2 = interpolate.splrep(t_baseline,param.preamble.imag) 
    y_spline_Q = interpolate.splev(t_upsample,tck2)

    preamble_air = y_spline_I + 1j*y_spline_Q

    # preamble_air = param.preamble   # Not resample yet
    tmp_reversed = preamble_air[::-1]
    tmp_conj = tmp_reversed.conjugate()
    tmp_conv = signal.convolve(tmp_conj, rx_vec_air)
    preamble_corr = abs(tmp_conv)
    ## preamble_corr = abs(np.correlate(preamble_air, rx_vec_air, mode='valid')) # not work
    # plt.plot(np.arange(len(preamble_corr)),preamble_corr)
    # plt.show()
    # print('pre:',preamble_air.shape)
    # print('rec:',rx_vec_air.shape)
    # print('corr:',preamble_corr.shape)

    ind = preamble_corr.argmax()    # index is from 0-1023
    ind1 = ind + 1
    ind2 = int(ind + (param.fs/param.bw)*param.N_sym*(2**param.sf) + 1)
    # ind2 = ind + (param.fs/param.bw)*len(wave_payload)
    # ind2 = ind1 + 10240             # len of payload

    rx_vec_tmp = rx_vec_air[ind1:ind2] 
    # test_len = 20
    # plt.subplot(2,1,2)
    # plt.plot(np.arange(test_len),rx_vec_tmp.real[0:test_len],linewidth='0.5',linestyle='dotted',marker='o',markersize=2)
    # plt.show()

    # print('rx_vec_tmp:',rx_vec_tmp.shape)
    # rx_vec = rx_vec_air[ind1:ind2]  # left close right open
    offset_sync = 15
    for i in range(0,int(param.N_sym*2**param.sf)):
        tmp_ind = offset_sync + i*int(param.fs/param.bw)
        if i==0:
            rx_vec = rx_vec_tmp[tmp_ind]
        else:
            rx_vec = np.hstack((rx_vec,rx_vec_tmp[tmp_ind]))
    
    # print('rx_vec:',rx_vec.shape)
    test_len = 32
    plt.subplot(2,1,2)
    plt.plot(np.arange(test_len),rx_vec.real[0:test_len],linewidth='0.5',linestyle='dotted',marker='v',markersize=5)
    plt.show()

    # print('rx_vec:',rx_vec.shape)
    # print('ind1',ind1)
    # print('ind2',ind2)

    ## Demodulate payload
    demod_result = np.zeros(param.N_sym)
    for i in range(0,param.N_sym):
        tmp = rx_vec[i*1024:(i+1)*1024] * param.downChirp
        tmp_fft = abs(fft(tmp))
        max_ind = tmp_fft.argmax()          # 0-1023
        if(max_ind==0):
            demod_result[i] = 0
        else:
            demod_result[i] = 1024 - max_ind    # 1023-0
        # demod_result[i] = max_ind

    demod_result = demod_result.astype(np.int32)
    print('rx_data',demod_result,demod_result.dtype)
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
        elif demod_result[i] == param.data[i]:
            ser = ser
        else:
            ser = 1000000
    ser = ser / param.N_sym
    return ber, ser

def simulation_LoRa():
    print("#### Simulation begin! ####")
    sim_param = sim_init()                          # init
    tx_vec_air, sim_param = sim_tx(sim_param)       # tx
    rx_vec_air = sim_channel(sim_param, tx_vec_air) # awgn
    demod_result = sim_rx(sim_param, rx_vec_air)    # rx
    ber, ser = sim_er(sim_param, demod_result)      # show
    print('ser=',ser)
    # sim_param.test_show()                           # test
    print('#### Simulation finish! ####')
    return 0

def test():
    a = np.array([1,2,3,4])
    b = np.array([[1],[2],[3],[4]])
    c = a[0:1]
    d = a[::-1]
    print('d:',d)
    maxindex = a.argmax()
    print('maxindex',maxindex)

    return 0

if __name__ == '__main__':
    ## Wrapper
    simulation_LoRa()
    # test()



    '''
    Matlab:

    preamble_air = resample(preamble,fs,bw);
    preamble_corr = abs(conv(conj(fliplr(preamble_air)), rx_vec_air));
    [~,ind] = max(abs(preamble_corr));
    ind1 = ind+1;
    ind2 = ind+(fs/bw)*length(wave_payload);
    rx_vec = resample(rx_vec_air(ind1:ind2),bw,fs);

    demod_result = zeros(1,N_sym);
    for i = 1:N_sym
        tmp =  rx_vec(2^sf*(i-1)+1:2^sf*i).*downChirp;
        tmp_fft = fft(tmp);
        [~,max_ind] = max(abs(tmp_fft)); 
        demod_result(i) = 1025-max_ind;
    end
    '''