import struct
import numpy as np
from matplotlib import pyplot as plt

from numpy.lib.function_base import average
# from client_trans import *

_interp = 12.5/10 # the sampling rate of N300 is 12.5Mbps, while the sampling rate of tag is 10Mbps


def parse(path):
    '''
    decode the file and get the payload
    path: the file path, binary file
    '''
    f = open(path, 'rb')
    content = f.read()
    f.close()
    # 4 bit represent 1 data, complex data consists of 2 data (real and imag)
    num = len(content)//8
    # 8976 = 32+64+80+352*20*_interp
    if num <= 8976:
        return
    data = []
    for i in range(8976,num): # directly get the payload
        data.append(struct.unpack('f',content[i*8:i*8+4])[0]+1j*struct.unpack('f',content[i*8+4:i*8+8])[0])

    return np.array(data)

def get_bits(data, gi, gp):
    '''
    Args:\\
    gi: weights gate\\
    gp: update gate\\ 
    return a dictionary: key is bin and value is relevant information
    '''
    offset = 6
    cp_len = int(16*_interp)
    N = len(data)//128-cp_len
    data = data.reshape(128,N+cp_len)
    data = data[:,offset:offset+N]
    dec_data = np.fft.fft(data)

    ## eliminate the first and last 10 bin 
    start = 10
    # determine whether the bin is transmitting using the first 8 bits
    avg_data = average(abs(dec_data[:,start:-1*start]),axis=0)
    # maxi = np.argmax(sum_data)
    # maximum = max(np.delete(sum_data,maxi))
    maximum = max(avg_data)
    # print('avg_data: ', avg_data)
    # print('maximum: ', maximum)
    
    # if maximum < 1:
    #     return

    plt.plot(20*np.log10(average(abs(dec_data),axis=0)))
    plt.show() 
    # plt.plot(np.arange(len(sum_data)),abs(sum_data))
    # plt.show()
    # plt.savefig('1.png')

    bins = []
    for i in range(len(avg_data)):
        if avg_data[i] >= 0.7*maximum and i!=N//2-start: #and i!=100-start and i!=156-start:
            bins.append(i+start)
    
    ## equalization
    # eqlist = np.array([1,1])
    # eqlist = np.array([1,1,-1,1,1,-1,-1,1])
    eqlist = np.array([1,1,1,1,1,1,1,1])
    n = len(eqlist)
    H_est = np.matmul(eqlist[:], dec_data[:n,bins])/n
    zero_index = np.where(H_est==0)[0]
    H_est[zero_index] = 1
    eq_data = dec_data[:,bins]/H_est
    # eq_data = dec_data[:,bins]
    # the phase of the first one should be 0
    eq_data = eq_data*np.exp(-1j*np.angle(eq_data[0,:]))
    # get the phase only
    eq_data = np.exp(1j*np.angle(eq_data))
    # eq_data = np.exp(1j*np.angle(dec_data[:,bins]))

    ## show dynamically
    # fig = plt.figure(tight_layout=True)
    # point_ani, = plt.plot(np.real(eq_data[0,0])/np.mean(abs(eq_data[:,0])),np.imag(eq_data[0,0])/np.mean(abs(eq_data[:,0])),'ro',\
    #     markersize=3)
    # def update(num):
    #     point_ani.set_data(np.real(eq_data[num,0])/np.mean(abs(eq_data[:,0])),np.imag(eq_data[num,0])/np.mean(abs(eq_data[:,0])))
    #     return point_ani,
    
    # ani = animation.FuncAnimation(fig, update, np.arange(0, 100), interval=1000)
    # plt.xticks(np.arange(-2,2,0.5))
    # plt.yticks(np.arange(-2,2,0.5))
    # plt.grid()
    # plt.show()


    ## phase correction and decode
    bin_bits = {}
    # p = Pool()
    # for i in range(eq_data.shape[1]):
    #     bits = p.apply_async(bpsk_decode,(eqlist,eq_data[:,i],gi, gp))
    #     bin_bits[bins[i]] = bits.get()
    # print(bin_bits)
    # print(np.angle(eq_data))
    for i in range(eq_data.shape[1]):
        bin_bits[bins[i]] = {}
        bin_bits[bins[i]]['dist'] = avg_data[bins[i]-start]
        bin_bits[bins[i]]['temp'] = 0
        bin_bits[bins[i]]['humi'] = 0
        bin_bits[bins[i]]['bits'] = []

        # get phase correction from the reference signal
        delta_fi_est = 0
        for k in range(8):
            x = eq_data[k][i]*np.exp(-1j*delta_fi_est)
            eq_data[k][i] = eq_data[k][i]*np.exp(-1j*delta_fi_est)
            delta_fi = gi*(np.angle(x)+np.pi*int(eqlist[k]==-1))
            delta_fi_est = gp*delta_fi_est+delta_fi

        # phase correction and demodalute from BPSK to bits
        for j in range(8,eq_data.shape[0]):
            x = eq_data[j][i]*np.exp(-1j*delta_fi_est)
            eq_data[j][i] = eq_data[j][i]*np.exp(-1j*delta_fi_est)
            if np.real(x) >= 0:
                bin_bits[bins[i]]['bits'].append(1)
                delta_fi = gi*(np.angle(x)+0)
            else:
                bin_bits[bins[i]]['bits'].append(0)
                delta_fi = gi*(np.angle(x)+np.pi)
            delta_fi_est = gp*delta_fi_est+delta_fi


    ## show dynamically
    # fig = plt.figure(tight_layout=True)
    # point_ani, = plt.plot(np.real(eq_data[0,0])/np.mean(abs(eq_data[:,0])),np.imag(eq_data[0,0])/np.mean(abs(eq_data[:,0])),'ro')
    # def update(num):
    #     point_ani.set_data(np.real(eq_data[:num,0])/np.mean(abs(eq_data[:,0])),np.imag(eq_data[:num,0])/np.mean(abs(eq_data[:,0])))
    #     return point_ani,
    
    # ani = animation.FuncAnimation(fig, update, np.arange(0, 1000), interval=1000)
    # plt.xticks(np.arange(-2,2,0.5))
    # plt.yticks(np.arange(-2,2,0.5))
    # plt.grid()
    # plt.show()

    return bin_bits


def decode(path,gi,gp):
    print('Start!')
    # start = time.time_ns()
    data = parse(path)
    bin_bits = get_bits(data, gi, gp)
    # transmit(bin_bits)
    # f = open('/home/peacer/桌面/N300_Tranceiver_v0/binbit/rlt_{}.json'.format(time.time_ns()),'w')
    # json.dump(bin_bits,f)
    # f.close()
    # json.dump(bin_bits,f)
    # f.close()
    # print(bin_bits.keys())
    # print(time.time_ns()-start)
    # print('Finish Decoding!')
    print(bin_bits)
    print('Finish!')

    return bin_bits
    # time.sleep(0.0001)

if __name__ == '__main__':
    # data = parse('/home/peacer/桌面/N300_Tranceiver_v0/recv/file10_25_0.49923568.dat')
    # bin_bits = get_bits(data,1,1)
    decode('C:/Users/91020/Desktop/py_learning/OFDMA/file8_20_4.31426998.dat',0.2,0.2)
    # decode('/home/peacer/桌面/N300_Tranceiver_v0/recv/file9_0_3.15051690.dat',0.2,0.2)



