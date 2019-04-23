import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftshift,ifftshift
from scipy import signal
from scipy.interpolate import interp1d

def cconv(a,b):
    if len(a)>=len(b):
        y = np.convolve(a,b,mode='full')
        y[:len(b)-1]+= y[-len(b)+1:]
    return y

def myconvolve(a,b):
    a,b = a.copy(),b.copy()
    lenA,lenB = len(a),len(b)
    lenX = lenA+lenB-1
    x = np.zeros(lenX)
    a = np.hstack([a,np.zeros(lenB-1)])
    b = np.hstack([np.zeros(lenA-1),b[::-1]])
    for i in range(lenX):
        x[i] = a[:i+1].dot(b[-(i+1):].T)
    return x

'''convolving two random signals using myconvolve and np.convolve'''
random1 = np.random.randn(100)
random2 = np.random.randn(120)
plt.figure(11)
plt.plot(np.convolve(random1,random2),label='np.convolve')
plt.plot(myconvolve(random1,random2),label='myconvolve')
plt.legend()
plt.title('Verifying myconvolve function')
plt.show()

'''Low Pass Filter with Linear Phase'''
LPfilter = np.loadtxt('h.csv')
w,h = signal.freqz(LPfilter,[1],None)
S,phi = 20*np.log10(abs(h)),np.angle(h)
plt.figure(1)
plt.suptitle('BodePlot of filter')
plt.subplot(211).plot(w,S,color='#663399',lw=1.8)
plt.ylabel('|H|')
_3dB_y = max(S)-3
_3dB_x = interp1d(S,w)(_3dB_y)
plt.axhline(y=_3dB_y,ls='--',c='black')
plt.axvline(x=_3dB_x,ls='--',c='black')
plt.subplot(212).plot(w,phi,color='#663399',lw=1.8)
plt.ylabel('Phase of  H')
plt.xlabel(r'$\omega \longrightarrow$')
plt.show()

# Input signal
n = np.arange(1,1025)
L,P=2**4,len(LPfilter)# L is length of segment for circular convolution
xi = np.cos(0.85*np.pi*n) + np.cos(0.2*np.pi*n)#np.random.randn(1024)

plt.figure(2)   #Input plot
plt.title(r'$\cos(0.85\pi n) + \cos(0.2\pi n)$')
plt.plot(xi)

plt.figure(3)   #Using np.convolve
plt.title('Linear convolution')
plt.plot(np.convolve(LPfilter,xi),color='#663399')

plt.figure(4)   #Circular in Freq domain
plt.title('Circular convolution using DFT and IDFT')
y = ifft(fft(xi)*fft(np.hstack([LPfilter,np.zeros(len(xi)-len(LPfilter))])))
plt.plot(y.real,c='#c21fbc')
plt.show()

'''Dividing the signal into pieces of length L and finding linear convolution for
each of them and overlapping the result to get the overall result'''
plt.figure(5)
y1=np.zeros(len(xi)+P-1)
for i in range(int(len(xi)/L)):
    y1[i*L:((i+1)*L)+P-1] += np.convolve(xi[i*L:(i+1)*L],LPfilter)
plt.title('piecewise linear convolution')# Linear using Linear
plt.plot(y1,color='#e35e20')

'''Dividing the signal into pieces of length L and finding circular convolution for
each of them by adding P-1 samples from previous segment and discarding the P-1 after convolving
a zero padding of length L is added to get the last P-1 values of linear convolution'''
plt.figure(6)
xi1 = np.hstack([np.zeros(P-1),xi,np.zeros(L)])
y2=np.zeros(len(xi)+L,dtype=np.complex)
for i in range(int(len(xi1)/L)):
    y2[i*L:(i+1)*L] += ifft(fft(xi1[i*L:(i+1)*L+P-1])*fft(np.hstack([LPfilter,np.zeros(L+P-1-len(LPfilter))])))[P-1:]
plt.title('linear convolution using circular convolution')
plt.plot(y2[:-L+P-1].real,color='#663399')

'''Comparing the output of the above operation with actual linear convolution.'''
plt.figure(7)
plt.title('comparing linear with circular')
plt.plot(y2[:-L+P-1].real,color='#663399',label='linear convolution using circular')
plt.plot(np.convolve(xi,LPfilter),color='#c21fbc',label='expected linear convolution')
plt.legend()
plt.show()

#zadoff chou
x = np.array([complex(i.replace('i','j')) for i in np.loadtxt('x1.csv',dtype='str')])

y = ifft(fft(np.roll(x,5))*np.conjugate(fft(x)))
plt.figure(8)
plt.plot(np.abs(y),c='b')
plt.title('correlation with rotated version(zadoff chu)')
plt.axhline(y=839,ls='--',c='#c21fbc')
plt.axvline(x=5,ls='--',c='#c21fbc')
plt.xlim([0,100])   #remove this for zoomed out version
plt.show()

