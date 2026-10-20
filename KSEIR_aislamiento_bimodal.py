from KSEIR_functions import *
import argparse
from integracion import *
import os

os.chdir('Bimodal')


def deriv_aislamiento(y,t,args):
	pi_normal,k_mean_normal,pi_cuarentena,k_mean_cuarentena,T_inf,T_inc_inv,r,dt,threshold,duracion,repeticion = args 
	k_max = len(pi_cuarentena)
	global t_inicio
	global t_fin
	global cuarentena
	global duracion_total
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
	if not cuarentena:
		if I_tot > threshold:
			if repeticion or q==0:
				cuarentena = True
				lista_t_inicio += [t]
				pi_a = pi
				pi , k_mean = pi_cuarentena, k_mean_cuarentena
				S_nuevo,E_nuevo,I_nuevo,R_nuevo = recalcular_compartimentos_general(S,E,I,R,pi_a,pi)
				t_inicio = t
				q += 1
				cambio = True
	if cuarentena:
		if I_tot > threshold:
			t_fin = t #t_fin se va a actualizar cada vez, si I>tau
		elif t-t_fin > duracion:
			cuarentena = False
			lista_t_final += [t]
			pi_a = pi
			pi , k_mean = pi_normal, k_mean_normal
			S_nuevo,E_nuevo,I_nuevo,R_nuevo = recalcular_compartimentos_general(S,E,I,R,pi_a,pi)
			duracion_total += t - t_inicio
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
parser.add_argument('-r', '--repeticion', dest='repeticion', action='store_true') 
parser.add_argument('-w', '--write', dest='write', action='store_true') 
parser.add_argument('-p', '--plot', dest='plot', action='store_true') 
parser.add_argument('-I', '--I_compara', dest='I', action='store_true') 
argsparser = parser.parse_args()
repeticion = argsparser.repeticion


#Parametros
T_inc = 5.2
T_inc_inv = 1.0/T_inc
T_inf = 14.0-T_inc
r = 0.2
gamma0 = 2.5
#gamma1_list = [i/50 for i in range(100,251)]
#gamma1_list = [i/10 for i in range(21,51)]
gamma1 = 3.0

k_pico = 100
k_pico_list = [i*5 for i in range(2,40)]
sigma_pico = 3
k_max = 200
E0 = [0]*k_max
I0 = [0.01]*k_max
R0 = [0]*k_max
S0 = [ 1-E0[k]-I0[k]-R0[k] for k in range(k_max) ]
y0 = S0 + E0 + I0 + R0
t = np.arange(0, 500,0.01)
dt = t[2]-t[1]

if repeticion:
	os.chdir('Aislamiento con repeticion')
	if argsparser.write:
		f=open('output_aislamiento_bimodal_g0={}_g1={}.txt'.format(gamma0,gamma1),'w')
		f.write('gamma0\tgamma1\tpico\tsigma_pico\td\ttau\tRinf\tImax\ttImax\tq\tduracion_total\n')
	if argsparser.I:
		f2 = open('many_I_conn_repe_g0={}_g1={}.table'.format(gamma0,gamma1),'w')
else:
	os.chdir('Aislamiento sin repeticion')
	if argsparser.write:
		f=open('output_aislamiento_bimodal_g0={}_g1={}.txt'.format(gamma0,gamma1),'w')
		f.write('gamma0\tgamma1\tpico\tsigma_pico\td\ttau\tRinf\tImax\ttImax\tduracion_total\n')
	if argsparser.I:
		f2 = open('many_I_sin_repe_r={}_gamma0={}.table'.format(r,gamma0),'w')

threshold_list = [0.05]
duracion_list = [7]

if argsparser.plot:
	folder = 'Figuras g0={} g1={} pico={} sigma={}'.format(gamma0,gamma1,k_pico,sigma_pico)
	if not os.path.exists(folder): os.makedirs(folder)

for duracion in duracion_list:
	for threshold in threshold_list:
		for i in range(len(k_pico_list)): 
			#gamma1 = gamma1_list[i]
			k_pico = k_pico_list[i]
			print('k_pico:',k_pico)
			t_inicio = 0
			t_fin = 0
			lista_t_inicio = []
			lista_t_final = []
			duracion_total = 0
			cuarentena = False
			q = 0
			pi,k_mean = pi_power_law(k_max,gamma0)
			pi1,k_mean1 = pi_bimodal_apantallamiento(k_max,gamma0,gamma1,k_pico,sigma_pico)
			#Integra las ecuaciones en el tiempo
			#args son los argumentos que se pasan a deriv, aparte de y,t.
			args = pi,k_mean,pi1,k_mean1,T_inf,T_inc_inv,r,dt,threshold,duracion,repeticion
			y_sol = integrar_euler(deriv_aislamiento, y0, t, args )
			S = y_sol[0:k_max]
			E = y_sol[k_max:k_max*2]
			I = y_sol[k_max*2:3*k_max]
			R = y_sol[-k_max:]
			S_tot = sum([S[k]*pi[k] for k in range(k_max)])
			E_tot = sum([E[k]*pi[k] for k in range(k_max)])
			I_tot = sum([I[k]*pi[k] for k in range(k_max)])
			R_tot = sum([R[k]*pi[k] for k in range(k_max)])
			for tt in range(len(t)):
				for m in range(len(lista_t_final)):
					if (t[tt] >= lista_t_inicio[m]) and (t[tt] < lista_t_final[m]):
						S_tot[tt] = sum([S[k][tt]*pi1[k] for k in range(k_max)])
						E_tot[tt] = sum([E[k][tt]*pi1[k] for k in range(k_max)])
						I_tot[tt] = sum([I[k][tt]*pi1[k] for k in range(k_max)])
						R_tot[tt] = sum([R[k][tt]*pi1[k] for k in range(k_max)])
			Imax , t_Imax , Rinf, t_Rinf = calculo_observables(I_tot,R_tot,t)
			#if argsparser.plot & ((gamma1%0.1)<0.001):
			if argsparser.plot:
				figname=folder+'/d{}_g0={:.2f}_g1={:.2f}_pico{}.png'.format(duracion,gamma0,gamma1,k_pico)
				title= r'KSEIR con $K$={} $\gamma_0$={} $\gamma_1$={} pico en $k$={}'.format(k_max,gamma0,gamma1,k_pico)
				plot_tot(t,k_max,S_tot,E_tot,I_tot,R_tot,title,figname,tmax=100,Reff=False,tiempos_inicio_fin=[lista_t_inicio,lista_t_final])
			if argsparser.write:
				if repeticion:
					f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(gamma0, gamma1, k_pico,sigma_pico, duracion, threshold, Rinf, Imax, t_Imax, q, duracion_total))
				else:
					f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(gamma0, gamma1, k_pico,sigma_pico, duracion, threshold, Rinf, Imax, t_Imax, duracion_total))
			if argsparser.I:
				if (gamma1 % 0.1) <= 0.001:
					f2.write('%i %i'%(gamma1,duracion))
					for iii in I_tot:
						f2.write(' {:.6f}'.format(iii))
					f2.write('\n')