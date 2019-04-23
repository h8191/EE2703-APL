#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sp
import sympy as sm
from cmath import phase
from IPython.display import display
sm.init_session
sm.init_printing(use_latex='mathjax')
#sm.init_printing(use_unicode=True)
my_phase = np.vectorize(phase)


# In[2]:


# symbols used throughout this code
s = sm.symbols('s')
t = sm.Symbol('t',positive=True,real=True)
w = sm.Symbol('w',real=True)


# In[3]:


#functions used through out this code
def lowpass(R1,R2,C1,C2,G,Vi):
    '''opamp circuit that acts as a lowpass filter'''
    s = sm.symbols('s')
    A = sm.Matrix([[0,0,1,-1/G],
                   [-1/(1+s*R2*C2),1,0,0],
                   [0,-G,G,1],
                   [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sm.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

def highpass(R1,R2,C1,C2,G,Vi):
    '''opamp circuit that acts as a highpass filter'''
    s = sm.symbols('s')
    A = sm.Matrix([[0,0,1,-1/G],
                   [-s*R2*C2/(1+s*R2*C2),1,0,0],
                   [0,-G,G,1],
                   [-1/R1-s*C2-s*C1,s*C2,0,1/R1]])
    b = sm.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)

def makeLti(sympy_expression):
    '''returns a lti object corresponding to the given sympy transfer function'''
    #s1 = sm.simplify(sympy_expression)
    #sm.factor(s1)
    n,d = sympy_expression.as_numer_denom()
    n_coeffs,d_coeffs = [np.array(sm.Poly(expr,s).all_coeffs(),dtype=np.float) for expr in (n,d)]
    # use Poly(more powerful)(not poly) and all_coeffs => includes zeros("not" coeffs => only non zero coeffs)
    return sp.lti(n_coeffs,d_coeffs)


# In[4]:


A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
sym_lowpass = sm.simplify(V[3])
H_lowpass = makeLti(sym_lowpass)
print('low pass filter transfer function')
display(sym_lowpass)

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1)
print('high pass filter transfer function')
sym_highpass = sm.simplify(V[3])
H_highpass = makeLti(sym_highpass)
display(sym_highpass)


# In[5]:


#bodeplots
w = np.logspace(0,8,801)
ss = 1j*w
hf_lowpass,hf_highpass = sm.lambdify(s,sym_lowpass,'numpy'),sm.lambdify(s,sym_highpass,'numpy')
lowpass_data,highpass_data = [[abs(value),my_phase(value)*180/np.pi] for value in (hf_lowpass(ss),hf_highpass(ss))]
plt.figure(1,figsize=(8,6),dpi=72)
plt.suptitle('magnitude plot for low pass and high pass filter')
plt.subplot(211).loglog(w,lowpass_data[0])
plt.xlabel(r'$\omega \longrightarrow$')
plt.ylabel(r'$magnitude \longrightarrow$')
plt.subplot(212).loglog(w,highpass_data[0])
plt.ylabel(r'$magnitude \longrightarrow$')
plt.xlabel(r'$\omega \longrightarrow$')


# In[6]:


#step response
time = np.linspace(0,1.5e-4,1001)
A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
step_response_high = makeLti(V[3])
#lowpass
plt.figure(2,figsize=(13,5),dpi=72)
plt.suptitle('step response for lowpass and highpass filter')
#plt.subplot(121).plot(*sp.lsim(H_lowpass,np.ones(time.shape),time)[:-1])
plt.subplot(121).plot(*sp.step(H_lowpass,T=time))
plt.xlabel(r'$time \longrightarrow$')
plt.ylabel(r'$step;response;value \longrightarrow$')
plt.subplot(122).plot(*sp.impulse(step_response_high,None,time))
plt.xlabel(r'$time \longrightarrow$')
plt.ylabel(r'$step;response;value \longrightarrow$')


# In[7]:


#response to sinusoidal input for lowpass filters
time = np.linspace(0,1e-2,8001)
pi = round(np.pi,3)
display(pi)
sin_t = lambda t: np.sin(2e3*pi*t)+np.cos(2e6*pi*t)

plt.figure(3,figsize=(13,5),dpi=72)
plt.suptitle('input and output of lowpass filter')
plt.subplot(121).plot(time,sin_t(time))
plt.xlabel(r'$time;\longrightarrow$')
plt.ylabel(r'$output;value \longrightarrow$')
plt.title(r'$(\sin(2.10^3\pi t)+\cos(2.10^6\pi t))u_0(t)$')
plt.subplot(122).plot(*sp.lsim(H_lowpass,sin_t(time),time)[:-1])
plt.xlabel(r'$time;\longrightarrow$')
plt.ylabel(r'$output;value \longrightarrow$')
plt.title('filtered output')

# In[8]:


#response to decaying sinusoidal input for highpass filter
time = np.linspace(0,1e-4,2001)
d_sin = lambda t:np.exp(-5*10**4*time)*np.sin(10**6*np.pi*t)
plt.figure(4,figsize=(13,5),dpi=72)
plt.suptitle('response to decaying sinusoidal input for highpass filter')
plt.subplot(121).plot(time,d_sin(time))
plt.plot(time,np.exp(-5*10**4*time))
plt.legend(['decaying sinusoid','corresponding exponential'])
plt.title(r'$\exp(-5x10^4t)\sin(10^6\pi t)$')
plt.subplot(122).plot(*sp.lsim(H_highpass,d_sin(time),time)[:-1])
plt.plot(time,0.8*np.exp(-5*10**4*time))
plt.title('output of the highpass filter')
plt.legend(['decaying sinusoid','corresponding exponential'])


# In[9]:


#impulse responses
#lowpass filter
time = np.linspace(0,1e-4,1000)
plt.figure(5,figsize=(13,5),dpi=72)
plt.suptitle('Impulse response for lowpass and highpass filters')
plt.subplot(121).plot(*sp.impulse(H_lowpass,None,time))
plt.xlabel(r'$time;\longrightarrow$')
plt.ylabel(r'$output;value \longrightarrow$')
plt.title(sym_lowpass)
plt.subplot(122).plot(*sp.impulse(H_highpass,None,time))
plt.xlabel(r'$time;\longrightarrow$')
plt.ylabel(r'$output;value \longrightarrow$')
plt.title(sym_highpass)

plt.show()

# for i in range(1,6):
#     plt.figure(i)
#     plt.savefig('Figure_{}.png'.format(i))
