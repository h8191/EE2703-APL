import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from numpy.fft import fft,fftshift,ifft,ifftshift
import random

pi = np.pi

def findCos(w,Ymag,Yphi,truevals=[],Sup=1e-2):
    '''To find the phase and freq of a given cosine'''
    ii = np.where(np.logical_and(Ymag>Sup,w>0))
    Mag = np.abs(Ymag[ii])**2
    w0 = np.sum(w[ii]*Mag)/np.sum(Mag)         
    if np.allclose(Yphi[w==1], Yphi[w==2], atol=1e-1):
        delta = np.sum(Yphi[ii]*Mag)/np.sum(Mag)
    else:
        delta = Yphi[w==1][0]
    return [w0,delta],np.abs(np.array(truevals)-np.array([w0,delta]))

def FFT(t,y,wnd=True):
    y = y.copy()
    N = len(t)          # length of sequence
    if wnd==True:
        window = fftshift(0.54 + 0.46*np.cos(2*pi*np.arange(N)/(N-1)))
        y *= window
    # plt.plot(t,y)
    # plt.figure()
    dt = t[1]-t[0]      #sampling period of the sequence
    f_max = 1/dt        #sampling frequency half of this is max freq signal that can be sampled 
    # y[0] =     0            # for pure imag transform(-tmax => 0)
    y = ifftshift(y)            # shift signal from -pi to pi to 0 to 2*pi
    Y = fftshift(fft(y))/N      # shift fft from -0 to 2pi to -pi to pi          
    w = np.linspace(-f_max*pi,f_max*pi,N,endpoint=False)
    return w,abs(Y),np.angle(Y)

def Plot(w,Ymag,Yphi,title,xlims=[],unwrap=False,dots='go'):
    plt.suptitle(r'Spectrum of ${}$'.format(title))
    plt.subplot(211).plot(w,Ymag,c='#663399',lw=2);plt.grid();plt.ylabel('|Y|')
    if xlims:   plt.xlim(xlims)# if ylims:  plt.ylim(ylims)
    ii = np.where(Ymag>1e-3)# to find points which have considerble magnitude

    phase = np.unwrap(Yphi[ii]) if unwrap else Yphi[ii]
    plt.subplot(212).plot(w[ii],phase,'go',lw=1,label = 'Mag > {}'.format(1e-3))
    plt.xlim(xlims if xlims else [np.min(w),np.max(w)])
    # we want X-axis of this to be in sync with the |Y|
    # if it is not given when taking points with mag>Sup lims of plot will change
    plt.grid();plt.ylabel('Phase of Y');plt.xlabel(r'$\omega \longrightarrow$')
    plt.legend(loc='upper right')

t = np.linspace(-4*pi,4*pi,128,endpoint=False)
wnd1 = fftshift(0.54 + 0.46*np.cos(2*pi*np.arange(128)/(127)))
y1 = np.cos(0.86*t)**3
# plt.figure(1)
# Plot(*FFT(t,y1,False),title=r'\cos^3(0.86t)',xlims=[-5,5])
# plt.figure(2)
# Plot(*FFT(t,y1,True),title=r'\cos^3(0.86t)*w(t)',xlims=[-5,5])

# t = np.linspace(-4*pi,4*pi,128,endpoint=False)
# iters = 1000
# values, errors = np.zeros((iters,2)),np.zeros((iters,2))
# valuesNoise, errorsNoise = np.zeros((iters,2)),np.zeros((iters,2))
# w0,delta = random.random()+0.5,(random.random()-0.5)*2*pi
# ycos = np.cos(w0*t+delta)

# # plt.figure(3)
# # Plot(*FFT(t,ycos,False),title=r'\cos(0.86t)',xlims=[-5,5])

# for i in range(iters):
#     yNoise = ycos+0.1*np.random.randn(128)  # max mag of noise is around 0.08
#     values[i],errors[i] = findCos(*FFT(t,ycos,wnd=True),[w0,delta])
#     valuesNoise[i],errorsNoise[i] = findCos(*FFT(t,yNoise,wnd=True),[w0,delta])

# print('True value of w {} and delta {}'.format(w0,delta))
# print(errors[0])
# print('without noise')
# print('Mean value of w0 {} and delta {}'.format(*np.mean(values,axis=0)))
# print('Mean error in W0 {},delta {}'.format(*np.mean(errors,axis=0)))
# print('Max error in W0 {},delta {}'.format(*np.max(errors,axis=0)))
# print(errorsNoise[0])
# print('with noise')
# print('Mean value of w0 {} and delta {}'.format(*np.mean(valuesNoise,axis=0)))
# print('Mean error in W0 {},delta {}'.format(*np.mean(errorsNoise,axis=0)))
# print('Max error in W0 {},delta {}'.format(*np.max(errorsNoise,axis=0)))

t = np.linspace(-pi,pi,1024,endpoint=False)
ChirpedSignal = np.cos(16*t*(1.5+t/(2*pi)))
# plt.figure()
# Plot(*FFT(t,ChirpedSignal,wnd=True),title=r'\cos(16t(1.5+\frac{t}{2\pi}))*w(t)',xlims=[-60,60],unwrap=True)
# plt.figure()
# Plot(*FFT(t,ChirpedSignal,wnd=False),title=r'\cos(16t(1.5+\frac{t}{2\pi}))',xlims=[-60,60],unwrap=True)

for window in (True,False):
    ts = np.hsplit(t,16)
    Ymag,Yphi = np.zeros((64,16)),np.zeros((64,16))# DFT for feach piece in column
  
    for tt,k in zip(ts,range(16)):
        ws,Ymag[:,k],Yphi[:,k]=FFT(tt,np.cos(16*tt*(1.5+tt/(2*pi))),wnd=window)
        T,W = np.meshgrid(t[::64],ws)

    Yphi[Ymag<1e-1] = 0

    for i in (['Mag',Ymag],['Phi',Yphi]):
        plt.figure()
        ax = p3.Axes3D(plt.gcf())
        plt.title('{} with{}window'.format(i[0],' ' if window else ' out '))
        surface = ax.plot_surface(T,W,i[1],rstride=1,cstride=1,cmap=plt.cm.jet)
        plt.ylim([-100,100])
        # plt.figure()
        # plt.title('{} with{}window'.format(i[0],' ' if window else ' out '))
        # plt.imshow(i[1].T)

# for i in range(4):
#     plt.figure(i)
#     plt.savefig('Figure_{}.png'.format(i+10))

plt.show()
   w