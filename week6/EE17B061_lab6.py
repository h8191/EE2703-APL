import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy import poly1d
from math import sqrt

def make_f(t,w,a):              # to eval the value of f at different time instants for different w and a
    return np.cos(w*t)*np.exp(-a*t)

def make_X(w,a):                # to make transfer function object for x(s)
    temp = poly1d([1,a])
    return sp.lti(temp,(temp**2+w**2)*poly1d([1,0,2.25]))

plt.figure(1)
#impulse => inverse laplace transform of X(s) 
t,x1 = sp.impulse(make_X(1.5,0.5),None,np.linspace(0,50,1000)) 
t,x2 = sp.impulse(make_X(1.5,0.05),None,np.linspace(0,50,1000))
plt.plot(t,x1,label='decay=0.5')
plt.plot(t,x2,label='decay=0.05')
plt.title('response x(t) for different decay rates of f(t)')

XF = sp.lti([1],[1,0,2.25]) # X(s)/F(s)
t = np.linspace(0,50,1000)   # time interval where we want to evaluate x(t)

plt.figure(2)
for i in np.arange(1.4,1.6+0.01,0.05):  #varying frequency with decay kept constant
    t,y,svec = sp.lsim(XF,make_f(t,i,0.05),t)
    plt.plot(t,y,label=r'$\omega ={}$'.format(i))
# plt.plot(t,np.exp(0.05*t),linestyle='--',alpha=0.5)
# plt.plot(t,-np.exp(0.05*t),linestyle='--',alpha=0.5)
plt.title('response for different frequencies')

R,L,C = 100,1e-6,1e-6

h_RLC = sp.lti([1],[L*C,R*C,1]) # transfer function for the given circuit
w,S,phi = h_RLC.bode()

plt.figure(3)
plt.suptitle(r'bode plot for H(S) = $\frac{1}{S^{2}LC + SRC + 1}$ or $\frac{1}{S^{2}10^{-12} + S10^{-4} + 1}$')
plt.subplot(211)
plt.semilogx(w,S)
plt.ylabel(r'$magnitude(dB) \longrightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.ylabel(r'$phase(\degree s) \longrightarrow$')
plt.xlabel(r'$w \longrightarrow$')

plt.figure(4)
t = np.linspace(0,20,1000)
x = (2+np.cos(sqrt(3)*t))/3
y = 2*(1-np.cos(sqrt(3)*t))/3
plt.plot(t,x,c='#663399',label=r'x = $\frac{2+\cos(\sqrt{3}t)}{3}$')
plt.plot(t,y,c='#339966',label=r'y = $\frac{2-2\cos(\sqrt{3}t)}{3}$')
# t,x = sp.impulse(sp.lti([1,0,2,0],[1,0,3,0,0]),None,t)
# t,y = sp.impulse(sp.lti([2,0],[1,0,3,0,0]),None,t)
# plt.plot(t,x,c='#663399',label=r'x = 1$\frac{2+\cos(\sqrt{3}t)}{3}$')
# plt.plot(t,y,c='#339966',label=r'y = 1$\frac{2-2\cos(\sqrt{3}t)}{3}$')
plt.title('coupled spring mass system')

vi6 = lambda t: np.cos(10**3*t)-np.cos(10**6*t)

plt.figure(5)
t = np.linspace(0,0.01,10**5)
t,vo6,svec = sp.lsim(h_RLC,vi6(t),t)
# t1,vo7,svec = sp.lsim(h_RLC,np.cos(10**3*t),t)
plt.title(r'low pass filter output for $\cos(10^{3}t)-\cos(10^{6}t)$')
plt.plot(t,vo6)#,t,vo7)

plt.figure(6)
t = np.linspace(0,0.000030,10**3)
t,vo6,svec = sp.lsim(h_RLC,vi6(t),t)
plt.title(r'low pass filter output for $\cos(10^{3}t)-\cos(10^{6}t)$ for 30$\mu$s')
plt.plot(t,vo6)#,t,vo7)

for i in (1,2,4,5,6):   
	plt.figure(i)
	plt.legend()
	plt.xlabel(r'$t \longrightarrow$')
	plt.ylabel(r'$amplitude \longrightarrow$')
# for i in range(1,6):    plt.figure(i);plt.savefig('Laplace_{}.png'.format(i))
plt.show()