from pylab import *
from scipy.integrate import quad
from scipy.linalg import lstsq

fexp = np.exp
fcos = lambda x:np.cos(np.cos(x))

def FourierCoeffs(func=fcos,N=25):# To N+1 a's and N b's for a given func in [0,2*pi]
    u,v = lambda x,k:func(x)*cos(k*x),lambda x,k:func(x)*sin(k*x)
    c = [quad(u,0,2*pi,0)[0]/(2*pi)]#constant or a0
    for i in range(1,N+1):   c.extend([quad(u,0,2*pi,i)[0]/pi,quad(v,0,2*pi,i)[0]/pi])
    return array(c).reshape(2*N+1,1)

def makeA(t,N=25):
    '''rtns a matrix with cos and sin vals for Fs'''
    A = np.ones(t.shape)
    for i in range(1,26):   A = hstack([A,cos(i*t),sin(i*t)])
    return A

N = 400
t = np.linspace(0,2*pi,N+1)[:-1].reshape(N,1)#column vector #to find the fourier coeffs
t1 = np.linspace(-2*pi,4*pi,N,endpoint = False).reshape(N,1)# to plot curves 

A,A1 = makeA(t),makeA(t1)
c1,c2 = FourierCoeffs(fexp), FourierCoeffs(fcos)# fourier coeffs
est1,est2 = lstsq(A,fexp(t))[0],lstsq(A,fcos(t))[0]# lstsq coeffs
#print(c1,c2,est1,est2)
plt.figure(1)
aexp = fexp(t)
plt.semilogy(t1,fexp(t1),label='fexp_Normal')   #original curve
plt.semilogy(r_[t-2*np.pi,t,t+2*np.pi],r_[aexp,aexp,aexp],label='expected fourier series')
plt.semilogy(t1,dot(A1,c1),label='fexp_FS')     #fourier series fit
plt.semilogy(t1,dot(A1,est1),label='fexp_lstsq')#lstsq fit
plt.xticks(np.arange(-2*pi,4*pi+0.1,pi),[r'${}\pi$'.format(i) for i in range(-2,5)])

plt.figure(2)
ACOS = fcos(t)
plt.plot(t1,fcos(t1),label='fcos_Normal')
plt.plot(r_[t-2*np.pi,t,t+2*np.pi],r_[ACOS,ACOS,ACOS],label='expected fourier series')
plt.plot(t1,dot(A1,c2),label='fcos_FS')
plt.plot(t1,dot(A1,est2),label='fcos_lstsq')
plt.xticks(np.arange(-2*pi,4*pi+0.1,pi),[r'${}\pi$'.format(i) for i in range(-2,5)])

XF = arange(1,52)
'''
#fourier coefficients
figure(3);semilogy(XF,abs(c1),'o--',c='r',label='fexpFcoeffs')
figure(4);loglog(XF,abs(c1),'o--',c='r',label='fexpFcoeffs')
figure(5);semilogy(XF,abs(c2),'o--',c='r',label='fcosFcoeffs')
figure(6);loglog(XF,abs(c2),'o--',c='r',label='fcosFcoeffs')
#least square estimates
figure(3);semilogy(XF,abs(est1),'o--',c='g',label='fexpLstsq')
figure(4);loglog(XF,abs(est1),'o--',c='g',label='fexpLstsq')
figure(5);semilogy(XF,abs(est2),'o--',c='g',label='fcosLstsq')
figure(6);loglog(XF,abs(est2),'o--',c='g',label='fcosLstsq')
'''
figure(3);semilogy();scatter(XF,abs(c1),c='r',label='fexpFcoeffs')
figure(4);loglog();scatter(XF,abs(c1),c='r',label='fexpFcoeffs')
figure(5);semilogy();scatter(XF,abs(c2),c='r',label='fcosFcoeffs')
figure(6);loglog();scatter(XF,abs(c2),c='r',label='fcosFcoeffs')

figure(3);semilogy();scatter(XF,abs(est1),c='g',label='fexpLstsq')
figure(4);loglog();scatter(XF,abs(est1),c='g',label='fexpLstsq')
figure(5);semilogy();scatter(XF,abs(est2),c='g',label='fcosLstsq')
figure(6);loglog();scatter(XF,abs(est2),c='g',label='fcosLstsq')

print("max diff b/w Fcoeffs and lstsq in fexp is {[0]} \
and fcos is {[0]}".format(max(abs(c1-est1)),max(abs(c2-est2))))
print('Close' if allclose(A.dot(est1),fexp(t),atol=1e-8) else 'not Close')

for i in range(1,7):
    figure(i);legend();grid(True);#savefig('Figure_%s.png'%i)
show()