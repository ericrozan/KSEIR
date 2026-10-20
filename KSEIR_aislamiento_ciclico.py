from KSEIR_functions import *
import argparse
from integracion import *
import os

os.chdir('Ciclico')

def deriv_aislamiento_ciclico(y,t,args):
	pi_normal,k_mean_normal,pi_cuarentena,k_mean_cuarentena,T_inf,T_inc_inv,r,dt,d1,d2,tmax,tau = args 
	k_max = len(pi_cuarentena)
	global cuarentena
	global q
	global lista_t_final
	global lista_t_inicio
	#Primero: estamos en cuarentena o no? Calculo en funcion de eso todo
	if cuarentena:
		pi , k_mean = pi_cuarentena, k_mean_cuarentena
	else:
		pi , k_mean = pi_normal, k_mean_normal
	S = y[0:k_max]
	E = y[k_max:k_max*2]
	I = y[k_max*2:3*k_max]
	R = y[-k_max:]
	I_tot = sum([I[k]*pi[k] for k in range(k_max)])
	cambio = False
	# Segundo: hay que cambiar el estado de la cuarentena? Si a no o viceversa. 
	# Si cambia el estado, hay que recalcular unas cuantas cosas EN EL MISMO PASO.
	# Si cambia el estado de la cuarentena, ese timestep se usa para cambiar
	# el grado. O sea, no evoluciona.
	if not cuarentena and q==0:
		#Activacion por primera vez
		if (I_tot >= tau):
			cuarentena = True
			pi_a = pi
			pi , k_mean = pi_cuarentena, k_mean_cuarentena
			S_nuevo,E_nuevo,I_nuevo,R_nuevo = recalcular_compartimentos_general(S,E,I,R,pi_a,pi)
			#Ya me armo toda la lista de t_inicio y t_fin:
			lista_t_inicio += [t]
			q += 1
			cambio = True
		#Activacion el resto de las veces:
	if not cuarentena and q>0:
		if abs(t-lista_t_final[-1]-d2)<=0.0001:
			if  I_tot>=tau:
				cuarentena = True
				lista_t_inicio += [t]
				pi_a = pi
				pi , k_mean = pi_cuarentena, k_mean_cuarentena
				S_nuevo,E_nuevo,I_nuevo,R_nuevo = recalcular_compartimentos_general(S,E,I,R,pi_a,pi)
				q += 1
				cambio = True
	if cuarentena:
		if abs(t-lista_t_inicio[-1] -d1) <=0.0001:
			lista_t_final += [t]
			cuarentena = False
			pi_a = pi
			pi , k_mean = pi_normal, k_mean_normal
			S_nuevo,E_nuevo,I_nuevo,R_nuevo = recalcular_compartimentos_general(S,E,I,R,pi_a,pi)
			cambio = True
	if not cambio:
		suma = sum([ I[k]*(k)*pi[k] for k in range(k_max) ])
		dSdt = [ -r*(k+1)*S[k]/k_mean*suma for k in range(k_max)]
		dEdt = [( r*(k+1)*S[k]/k_mean*suma - E[k]*T_inc_inv) for k in range(k_max)]
		dIdt = [ T_inc_inv * E[k] - I[k]/T_inf  for k in range(k_max)]
		dRdt = [I[k]/T_inf  for k in range(k_max)]
		deriv_vector = dSdt + dEdt + dIdt + dRdt
	else:
		dSdt = list((S_nuevo-S)/dt)
		dEdt = list((E_nuevo-E)/dt)
		dIdt = list((I_nuevo-I)/dt)
		dRdt = list((R_nuevo-R)/dt)
		deriv_vector = dSdt + dEdt + dIdt + dRdt
	return np.array(deriv_vector,dtype='float64')
	
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--write', dest='write', action='store_true') 
parser.add_argument('-p', '--plot', dest='plot', action='store_true') 
parser.add_argument('-I', '--I_compara', dest='I', action='store_true') 
argsparser = parser.parse_args()

#Parametros
T_inc = 5.2
T_inc_inv = 1.0/T_inc
T_inf = 14.0-T_inc
r = 0.2
gamma0 = 2.5
gamma1 = 3.0
k_max = 200
tau = 0.05
pi_normal,k_mean_normal = pi_power_law(k_max,gamma0)
pi_cuarentena,k_mean_cuarentena = pi_power_law(k_max,gamma1)


E0 = [0]*k_max
I0 = [0.01]*k_max
R0 = [0]*k_max
S0 = [ 1-E0[k]-I0[k]-R0[k] for k in range(k_max) ]
y0 = S0 + E0 + I0 + R0
tmax = 200
t = np.arange(0, tmax,0.01)
dt = t[2]-t[1]

d1_list = np.arange(1, 25, 1, dtype=int)
d2_list = np.arange(1, 25, 1, dtype=int)
d1_list = [15]
d2_list = [12,13]

if argsparser.plot:
	folder = 'Figuras gamma0={} gamma1={}'.format(gamma0,gamma1)
	if not os.path.exists(folder): os.makedirs(folder)

if argsparser.write:
	f=open('output_ciclico_g0_{}_g1_{}.table'.format(gamma0,gamma1),'w')
	f.write('gamma0\tgamma1\td1\td2\tRinf\tImax\ttImax\tq\n')
if argsparser.I:
	f2 = open('many_I_ciclico_gamma0={}_gamma1={}.table'.format(gamma0,gamma1),'w')


for d2 in d2_list:
	for d1 in d1_list:
		print('d1=',d1,'d2=',d2)
		cuarentena = False
		q = 0
		lista_t_inicio = []
		lista_t_final = []
		#Integra las ecuaciones en el tiempo
		#args son los argumentos que se pasan a deriv, aparte de y,t.
		args = pi_normal,k_mean_normal,pi_cuarentena,k_mean_cuarentena,T_inf,T_inc_inv,r,dt,d1,d2,tmax,tau
		y_sol = integrar_euler(deriv_aislamiento_ciclico, y0, t, args )
		S = y_sol[0:k_max]
		E = y_sol[k_max:k_max*2]
		I = y_sol[k_max*2:3*k_max]
		R = y_sol[-k_max:]
		S_tot = sum([S[k]*pi_normal[k] for k in range(k_max)])
		E_tot = sum([E[k]*pi_normal[k] for k in range(k_max)])
		I_tot = sum([I[k]*pi_normal[k] for k in range(k_max)])
		R_tot = sum([R[k]*pi_normal[k] for k in range(k_max)])
		if len(lista_t_final)<len(lista_t_inicio):
			lista_t_final += [tmax]
		for tt in range(len(t)):
			for m in range(len(lista_t_final)):
				if (t[tt] >= lista_t_inicio[m]) and (t[tt] < lista_t_final[m]):
					S_tot[tt] = sum([S[k][tt]*pi_cuarentena[k] for k in range(k_max)])
					E_tot[tt] = sum([E[k][tt]*pi_cuarentena[k] for k in range(k_max)])
					I_tot[tt] = sum([I[k][tt]*pi_cuarentena[k] for k in range(k_max)])
					R_tot[tt] = sum([R[k][tt]*pi_cuarentena[k] for k in range(k_max)])
		Imax , t_Imax , Rinf, t_Rinf = calculo_observables(I_tot,R_tot,t)
		if argsparser.plot:
			plt.plot()
			figname=folder+'/d1{}_d2{}.png'.format(d1,d2)
			title= r'$\gamma_a$={} $\gamma_p$={} $d_1$={} $d_2$={}'.format(gamma0,gamma1,d1,d2)
			plot_tot(t,k_max,S_tot,E_tot,I_tot,R_tot,title,figname,tmax=150,Reff=False,tiempos_inicio_fin=[lista_t_inicio,lista_t_final])
		if argsparser.write:
			f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(gamma0, gamma1, d1, d2, Rinf, Imax, t_Imax,q))
		if argsparser.I:
			f2.write('%i %i'%(d1,d2))
			for iii in I_tot:
				f2.write(' {:.6f}'.format(iii))
			f2.write('\n')