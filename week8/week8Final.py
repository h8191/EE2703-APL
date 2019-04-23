import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
pltCount = 1

gaussianW = lambda w: np.sqrt(2*pi)*np.exp(-w**2/2)

def my_plot(w,y,title,phases='Sup',Sup=1e-3,bandLimited=True,xlims=None,ylims=[-3.7,3.7],SF=1,plot=True):
	global pltCount
	samples = len(y)
	if bandLimited:
		Y = np.fft.fftshift(np.fft.fft(y))/samples
	else:
		Y=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))/SF#/samples
	if not plot:
		return Y,w
	plt.figure(pltCount)
	pltCount += 1
	title = r'Spectrum of ${}$'.format(title)
	plt.suptitle(title)
	plt.subplot(211).plot(w,abs(Y))
	if xlims:	plt.xlim(xlims)
	# if ylims:	plt.ylim(ylims)
	plt.grid()
	plt.ylabel('|Y|')

	ii = np.where(np.abs(Y)>Sup)# to find points which have considerble magnitude
	plt.subplot(212)

	if phases == 'All' or phases =='both':# all points
		plt.plot(w,np.angle(Y),'ro',label='all Mags')
	if phases == 'Sup' or phases =='both':# point with mag > Sup
		plt.plot(w[ii],np.angle(Y[ii]),'go',label = 'Mag > {}'.format(Sup))
	
	plt.xlim(xlims if xlims else [np.min(w),np.max(w)])
	if ylims:	plt.ylim(ylims)
	plt.grid()
	plt.ylabel('Phase of Y')
	plt.xlabel(r'$\omega \longrightarrow$')
	plt.legend(loc='upper right')

	return Y,w

N,k = 512,4
t = np.linspace(-k*pi,k*pi,N,endpoint=False)
w = np.linspace(-N/2/k,N/2/k,N,endpoint=False)
print('extremes of w are {}'.format(N/2/k))
y1, y2, y3 = np.sin(t)**3, np.cos(t)**3, np.cos(20*t+5*np.cos(t))

my_plot(w,y1,'\sin^{3}(t)',xlims=[-15,15])
my_plot(w,y2,'\cos^{3}(t)',xlims=[-15,15])
my_plot(w,y3,'\cos(20t + 5 \cos(t))',xlims=[-35,35],phases='Sup')

def ErrorEStimation(k,N,PLOT=True):
	t = np.linspace(-k*pi,k*pi,N,endpoint=False)
	w = np.linspace(-N/2/k,N/2/k,N,endpoint=False)
	y5 = np.exp(-t**2/2)
	yo1 = gaussianW(w)
	Y,w = my_plot(w,y5,'gaussian k=%d'%k,Sup=1e-6,bandLimited=False,SF=N/(k*2*np.pi),plot=k==3,xlims=[-20,20])
	if k==1 or k==2 and PLOT:
		plt.figure(100+k)
		plt.plot(w,abs(Y),label='fft from function')
		plt.plot(w,yo1,label='fft from formula')
		plt.xlim(-20,20)
		plt.legend()
		plt.title('gaussian Xc and Xs*T at k={} N={}'.format(k,N))
		# plt.xlim([1000,1050])
	if not np.allclose(abs(Y),yo1,atol=1e-6):
		print('error is too large') 
	error = np.max(np.abs(Y-yo1))
	print('error for k={} and N={} is {}'.format(k,N,error))
	C = 2*k*pi/N
	print('Error Xc is {} Xs is {}'.format(error,error/C))

for k,N in zip([1,2,3,4],[512]*4):
	ErrorEStimation(k,N)


for i in list(range(1,pltCount))+[101,102]:
	plt.figure(i)
	plt.savefig('Figure_{}.png'.format(i))
plt.show()
