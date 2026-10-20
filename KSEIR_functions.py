import numpy as np
from scipy.integrate import odeint
from scipy.stats import norm
import matplotlib.pyplot as plt
import pylab
from integracion import *


golden_mean = (np.sqrt(5)-1.0)/2.0        # Aesthetic ratio
fig_width = 3+3/8                # width  in inches
fig_height = fig_width*golden_mean          # height in inches
#fig_width = 2.8
#fig_height = fig_width*1.2
fig_size =  [fig_width,fig_height]


params = {'backend': 'ps','axes.titlesize': 8,'axes.labelsize': 9,'axes.linewidth': 0.5,'axes.grid': False,
          'axes.labelweight': 'normal','font.family': 'serif','font.size': 8.0,'font.weight': 'normal','text.color': 'black',
          'xtick.labelsize': 8,'ytick.labelsize': 8,'text.usetex': True,'legend.fontsize': 8,'figure.dpi': 300,
          'figure.figsize': fig_size,'savefig.dpi': 500,}
pylab.rcParams.update(params)


#La distribucion de grado:
def pi_power_law(k_max,gamma):
	pi = []
	for k in range(1,k_max+1):
		pi += [k**(-gamma)]
	pi = np.array(pi)/sum(pi)
	pi = list(pi)
	k_mean = np.sum( [k * pi[k-1] for k in range(1,k_max+1) ] )
	return pi, k_mean

#Distribucion uniforme, solo para probar:
def pi_uniforme(k_max,k0):
	pi = [0]*(k_max)
	pi[k0-1] = 1
	return pi , k0

def pi_power_law_gaussiana(k_max,gamma,k_pico,sigma_pico):
	pi_LeyPotencia,k_mean = pi_power_law(k_max,gamma)
	gaussiana = norm.pdf(range(1,k_max+1),k_pico, sigma_pico)
	pi = pi_LeyPotencia + gaussiana/20
	pi = np.array(pi)/sum(pi)
	pi = list(pi)
	k_mean = np.sum( [k * pi[k-1] for k in range(1,k_max+1) ] )
	return pi, k_mean

def pi_bimodal_apantallamiento(k_max,gamma_viejo,gamma_nuevo,k_pico,sigma_pico):
	pi_1,k_mean = pi_power_law(k_max,gamma_viejo)
	pi_2,k_mean = pi_power_law(k_max,gamma_nuevo)
	kk = np.array(range(1,k_max+1))
	gaussiana = np.exp(-0.5*((kk-k_pico)/sigma_pico)**2)
	beta_numerador = 1-pi_2[k_pico]/pi_1[k_pico]
	beta_denominador = 1/pi_1[k_pico]-sum(gaussiana)
	beta = beta_numerador/beta_denominador
	pi_nuevo = np.array(pi_2) + beta*gaussiana
	pi_nuevo /= sum(pi_nuevo)
	k_mean = np.sum( [k * pi_nuevo[k-1] for k in range(1,k_max+1) ] )
	return list(pi_nuevo) , k_mean

#Las ecuaciones diferenciales:
def deriv(y,t,k_max,gamma,pi_power_law,T_inf,T_inc_inv,r):
	pi, k_mean = pi_power_law(k_max,gamma)
	S = y[0:k_max]
	E = y[k_max:k_max*2]
	I = y[k_max*2:3*k_max]
	R = y[-k_max:]
	suma=sum([ I[k]*(k)*pi[k] for k in range(k_max) ])
	dSdt = [ -r*(k+1)*S[k]/k_mean*suma for k in range(k_max)]
	dEdt = [( r*(k+1)*S[k]/k_mean*suma - E[k]*T_inc_inv) for k in range(k_max)]
	dIdt = [ T_inc_inv * E[k] - I[k]/T_inf  for k in range(k_max)]
	dRdt = [I[k]/T_inf  for k in range(k_max)]
	deriv_vector = dSdt + dEdt + dIdt + dRdt
	return np.array(deriv_vector,dtype='float64')

def deriv_args(y,t,args):
	k_max,gamma,pi_power_law,T_inf,T_inc_inv,r = args
	pi, k_mean = pi_power_law(k_max,gamma)
	S = y[0:k_max]
	E = y[k_max:k_max*2]
	I = y[k_max*2:3*k_max]
	R = y[-k_max:]
	suma=sum([ I[k]*(k)*pi[k] for k in range(k_max) ])
	dSdt = [ -r*(k+1)*S[k]/k_mean*suma for k in range(k_max)]
	dEdt = [( r*(k+1)*S[k]/k_mean*suma - E[k]*T_inc_inv) for k in range(k_max)]
	dIdt = [ T_inc_inv * E[k] - I[k]/T_inf  for k in range(k_max)]
	dRdt = [I[k]/T_inf  for k in range(k_max)]
	deriv_vector = dSdt + dEdt + dIdt + dRdt
	return np.array(deriv_vector,dtype='float64')
################################################################################
def resuelve(t,deriv,k_max,gamma,y0,pi_power_law,T_inf,T_inc,r):
	T_inc_inv = 1.0/T_inc
	#Integra las ecuaciones en el tiempo
	#args son los argumentos que se pasan a deriv, aparte de y,t. 
	'''original:
	y_sol = odeint(deriv, y0, t, args=(k_max,gamma,pi_power_law,T_inf,T_inc_inv,r))
	S = y_sol.T[0:k_max]
	E = y_sol.T[k_max:k_max*2]
	I = y_sol.T[k_max*2:3*k_max]
	R = y_sol.T[-k_max:]'''
	args = k_max,gamma,pi_power_law,T_inf,T_inc_inv,r
	y_sol = integrar_euler(deriv_args, y0, t, args )
	S = y_sol[0:k_max]
	E = y_sol[k_max:k_max*2]
	I = y_sol[k_max*2:3*k_max]
	R = y_sol[-k_max:]
	return S,E,I,R



def plot_separado(t,k_max,k0,S,E,I,R,title,figname,tmax=100,Reff=False):
	fig = plt.figure()
	ax = fig.add_subplot()
	Imax = max(list(I[k0-1]))
	Imax_index = list(I[k0-1]).index(Imax)
	t_Imax = t[Imax_index]
	pylab.axvline(x=t_Imax, color='k', alpha=0.3, lw=1.5)
	for i in range(k0-2,k0+1):
		ax.plot(t, S[i], 'C%i'%(i), alpha=0.7, lw=2, label=r'$S_{}$'.format(i+1),linestyle='--')
		ax.plot(t, E[i], 'C%i'%(i), alpha=0.7, lw=2, label=r'$E_{}$'.format(i+1),linestyle='-.')
		ax.plot(t, I[i], 'C%i'%(i), alpha=0.7, lw=2, label=r'$I_{}$'.format(i+1))
		ax.plot(t, R[i], 'C%i'%(i), alpha=0.7, lw=2, label=r'$R_{}$'.format(i+1),linestyle='dotted')
	if Reff == True:
		ventana = 100
		delay = 100
		R_eff = calcula_Reff(I,t,ventana,delay)
		ax.plot(t, R_eff, 'k', alpha=0.7, lw=2, label='$R_{eff}$')
	ax.set_xlabel('Tiempo (días)')
	ax.set_ylabel('Fraccion')
	ax.yaxis.set_tick_params(length=0)
	ax.xaxis.set_tick_params(length=0)
	ax.grid(which='major', alpha=0.5, lw=1.2)
	ax.legend(loc='center right')
	ax.set_title(title)
	ax.set_ylim([0,1.1])
	ax.set_xlim([0,tmax])
	ax.set_yticks(np.arange(0,1.01,0.1))
	pylab.savefig(figname, dpi=300, bbox_inches='tight')
	pylab.clf()
	pylab.close(fig)

def plot_tot(t,k_max,S_tot,E_tot,I_tot,R_tot,title,figname,tmax=100,Reff=False,tiempos_inicio_fin=[[0],[0]]):
	fig = plt.figure()
	ax = fig.add_subplot()
	Imax = max(list(I_tot))
	Imax_index = list(I_tot).index(Imax)
	t_Imax = t[Imax_index]
	#pylab.axhline(y=0.05,c='k',linestyle="dashed",lw=0.75,alpha=0.65)
	#pylab.axvline(x=t_Imax, color='k', alpha=0.3, lw=1)
	#pylab.axhline(y=Imax,   color='k', alpha=0.3, lw=1)
	for m in range(len(tiempos_inicio_fin[0])):
		ax.fill_betweenx([0,1], tiempos_inicio_fin[0][m],  tiempos_inicio_fin[1][m],
				   alpha=0.4,color='orange')
	ax.plot(t, S_tot, 'b' , alpha=0.7, lw=2, label='$S$')
	ax.plot(t, E_tot, 'C1', alpha=0.7, lw=2, label='$E$')
	ax.plot(t, I_tot, 'r' , alpha=0.7, lw=2, label='$I$')
	ax.plot(t, R_tot, 'g' , alpha=0.7, lw=2, label='$R$')
	if Reff == True:
		ventana = 100
		delay = 100
		R_eff = calcula_Reff(I_tot,t,ventana,delay)
		ax.plot(t, R_eff, 'k', alpha=0.7, lw=2, label='$R_{eff}$')
	ax.set_xlabel('Tiempo (días)')
	#ax.set_xlabel('Time (days)')
	ax.set_ylabel('Fracción de individuos')
	#ax.set_ylabel('Fraction of individuals')
	ax.yaxis.set_tick_params(length=0)
	ax.xaxis.set_tick_params(length=0)
	ax.grid(which='major', alpha=0.5, lw=1.2)
	ax.legend(loc='upper center',fontsize=7)
	#ax.set_title(title)
	ax.set_ylim([0,1])
	if tmax==0:
		tmax = max(t)
	ax.set_xlim([0,tmax])
	ax.set_yticks(np.arange(0,1.01,0.1))
	ax.set_xticks(np.arange(0,tmax+0.001,25))
	pylab.savefig(figname, bbox_inches='tight',pad_inches = 0.01)
	pylab.clf()
	pylab.close(fig)
	
def calcula_Reff(I,t,ventana,delay):
	R_eff = [1]*len(t)
	for i in range(ventana+delay+1, len(t)-1):
		I_post = 0
		I_pre = 0
		for j in range(ventana):
			I_post += I[i-j]
			I_pre += I[i-j-delay]
		R_eff[i]=I_post/I_pre
	for ti in range(3,int(len(t))):
		if I[ti-1]<=0.1 and I[ti]>=0.1:
			t_line=t[ti]
			pylab.axvline(x=t_line, color='k', alpha=0.3, lw=1.5)
	return R_eff


def calculo_observables(I,R,t):
	Imax = max(I)
	Imax_index = np.argmax(I)
	t_Imax = t[Imax_index]
	Rinf_index = -1
	Rinf = R[-1]
	for i in range(Imax_index+100,len(t)):
		if abs(R[i-100] - Rinf)<10**(-4):
			if abs(R[i]-R[i-100])<10**(-5):
				if Rinf_index == -1:
					Rinf_index=i
	t_Rinf = t[Rinf_index]
	return Imax , t_Imax , Rinf, t_Rinf





def recalcular_compartimentos_prueba1(S,E,I,R,gamma_viejo,gamma_nuevo,pi_power_law,k_max):
	pi_nuevo, k_mean_nuevo = pi_power_law(k_max,gamma_nuevo)
	pi_viejo, k_mean_viejo = pi_power_law(k_max,gamma_viejo)
	S_nuevo = [S[k]*pi_viejo[k]/pi_nuevo[k] for k in range(k_max)]
	E_nuevo = [E[k]*pi_viejo[k]/pi_nuevo[k] for k in range(k_max)]
	I_nuevo = [I[k]*pi_viejo[k]/pi_nuevo[k] for k in range(k_max)]
	R_nuevo = [R[k]*pi_viejo[k]/pi_nuevo[k] for k in range(k_max)]
	return S_nuevo, E_nuevo, I_nuevo, R_nuevo


def recalcular_compartimentos_prueba2(S,E,I,R,gamma_viejo,gamma_nuevo,pi_power_law,k_max):
	pi_nuevo, k_mean_nuevo = pi_power_law(k_max,gamma_nuevo)
	pi_viejo, k_mean_viejo = pi_power_law(k_max,gamma_viejo)
	S_tot = sum([S[k]*pi_viejo[k] for k in range(k_max)])
	E_tot = sum([E[k]*pi_viejo[k] for k in range(k_max)])
	I_tot = sum([I[k]*pi_viejo[k] for k in range(k_max)])
	R_tot = sum([R[k]*pi_viejo[k] for k in range(k_max)])
	S_nuevo = [S_tot/pi_nuevo[k] for k in range(k_max)]
	E_nuevo = [E_tot/pi_nuevo[k] for k in range(k_max)]
	I_nuevo = [I_tot/pi_nuevo[k] for k in range(k_max)]
	R_nuevo = [R_tot/pi_nuevo[k] for k in range(k_max)]
	return S_nuevo, E_nuevo, I_nuevo, R_nuevo



def recalcular_compartimentos_prueba3(S,E,I,R,gamma_viejo,gamma_nuevo,pi_power_law,k_max):
	pi_nuevo, k_mean_nuevo = pi_power_law(k_max,gamma_nuevo)
	pi_viejo, k_mean_viejo = pi_power_law(k_max,gamma_viejo)
	S_nuevo = [0]*k_max
	E_nuevo = [0]*k_max
	I_nuevo = [0]*k_max
	R_nuevo = [0]*k_max

	S_nuevo[-1] = S[-1]
	E_nuevo[-1] = E[-1]
	I_nuevo[-1] = I[-1]
	R_nuevo[-1] = R[-1]
	
	entran_S = (pi_viejo[-1]-pi_nuevo[-1])/(k_max-1)*S[-1]
	entran_E = (pi_viejo[-1]-pi_nuevo[-1])/(k_max-1)*E[-1]
	entran_I = (pi_viejo[-1]-pi_nuevo[-1])/(k_max-1)*I[-1]
	entran_R = (pi_viejo[-1]-pi_nuevo[-1])/(k_max-1)*R[-1]
	#Esto llega hasta k=2. el k=1 (i=0) lo hago a mano despues del loop
	for i in range(2,k_max):
		S_nuevo[-i] =  (pi_viejo[-i]*S[-i]+entran_S)/pi_nuevo[-i]
		entran_S += (pi_viejo[-i]-pi_nuevo[-i])/(k_max-i)*S[-i]
		
		E_nuevo[-i] =  (pi_viejo[-i]*E[-i]+entran_E)/pi_nuevo[-i]
		entran_E += (pi_viejo[-i]-pi_nuevo[-i])/(k_max-i)*E[-i]
		
		I_nuevo[-i] =  (pi_viejo[-i]*I[-i]+entran_I)/pi_nuevo[-i]
		entran_I += (pi_viejo[-i]-pi_nuevo[-i])/(k_max-i)*I[-i]
		
		R_nuevo[-i] =  (pi_viejo[-i]*R[-i]+entran_R)/pi_nuevo[-i]
		entran_R += (pi_viejo[-i]-pi_nuevo[-i])/(k_max-i)*R[-i]
		
	S_nuevo[0] = (pi_viejo[-0]*S[0]+entran_S)/pi_nuevo[0]
	E_nuevo[0] = (pi_viejo[-0]*E[0]+entran_S)/pi_nuevo[0]
	I_nuevo[0] = (pi_viejo[-0]*I[0]+entran_S)/pi_nuevo[0]
	R_nuevo[0] = (pi_viejo[-0]*R[0]+entran_S)/pi_nuevo[0]
	return S_nuevo, E_nuevo, I_nuevo, R_nuevo



def recalcular_compartimentos(S,E,I,R,gamma_viejo,gamma_nuevo,pi_power_law,k_max):
	pi_p, k_mean_p = pi_power_law(k_max,gamma_nuevo)
	pi_a, k_mean_a = pi_power_law(k_max,gamma_viejo)
	pi_p = np.array(pi_p)
	pi_a = np.array(pi_a)
	
	X = np.array(range(k_max))
	grupo1 = X[pi_a>pi_p]
	grupo3 = X[pi_a<pi_p]
	
	f_i = pi_a - pi_p
	g_j = pi_p - pi_a

	Phi1 = sum(f_i[grupo1])

	##### sigma[i,j] = f_i[i]*g_j[j]  /Phi
	sigma = np.outer(f_i,g_j)/Phi1
	
	SS_P = S*pi_a
	EE_P = E*pi_a
	II_P = I*pi_a
	RR_P = R*pi_a
	
	S_p = S.copy()
	E_p = E.copy()
	I_p = I.copy()
	R_p = R.copy()

	for j in grupo3:
		for i in grupo1:
			SS_P[j] += sigma[i,j]*S[i]
			EE_P[j] += sigma[i,j]*E[i]
			II_P[j] += sigma[i,j]*I[i]
			RR_P[j] += sigma[i,j]*R[i]
		S_p[j] = SS_P[j]/pi_p[j]
		E_p[j] = EE_P[j]/pi_p[j]
		I_p[j] = II_P[j]/pi_p[j]
		R_p[j] = RR_P[j]/pi_p[j]
	return S_p, E_p, I_p, R_p

def recalcular_compartimentos_general(S,E,I,R,pi_a,pi_p):
	pi_p = np.array(pi_p)
	pi_a = np.array(pi_a)
	k_max = len(pi_a)
	X = np.array(range(k_max))
	grupo1 = X[pi_a>pi_p]
	grupo3 = X[pi_a<pi_p]
	
	f_i = pi_a - pi_p
	g_j = pi_p - pi_a

	Phi1 = sum(f_i[grupo1])

	##### sigma[i,j] = f_i[i]*g_j[j]  /Phi
	sigma = np.outer(f_i,g_j)/Phi1
	
	SS_P = S*pi_a
	EE_P = E*pi_a
	II_P = I*pi_a
	RR_P = R*pi_a
	
	S_p = S.copy()
	E_p = E.copy()
	I_p = I.copy()
	R_p = R.copy()

	for j in grupo3:
		for i in grupo1:
			SS_P[j] += sigma[i,j]*S[i]
			EE_P[j] += sigma[i,j]*E[i]
			II_P[j] += sigma[i,j]*I[i]
			RR_P[j] += sigma[i,j]*R[i]
		S_p[j] = SS_P[j]/pi_p[j]
		E_p[j] = EE_P[j]/pi_p[j]
		I_p[j] = II_P[j]/pi_p[j]
		R_p[j] = RR_P[j]/pi_p[j]
	return S_p, E_p, I_p, R_p

