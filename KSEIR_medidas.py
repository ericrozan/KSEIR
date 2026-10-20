from KSEIR_functions import *
import os
import argparse

os.chdir('Sin Aislamiento')
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repeticion', dest='repeticion', action='store_true') 
parser.add_argument('-w', '--write', dest='write', action='store_true') 
parser.add_argument('-p', '--plot', dest='plot', action='store_true') 
parser.add_argument('-I', '--I_compara', dest='I', action='store_true') 
args = parser.parse_args()

#Parametros
T_inc = 5.2
T_inf = 14.0-T_inc
#r = 0.2
#gamma_list = np.arange(0.25,4.1,0.25)

#gamma_list = [g/4 for g in range(1,17)]
#gamma_list = [i/20 for i in range(4,81)]
gamma_list = [2.5]

#k_list = [i*2 for i in range(5,126)]
k_list = [200]
#k_list = [150,200,210,225,250]

#r_list = [j/100 for j in range(1,101)]
r_list = [0.2]

t = np.arange(0, 200,0.01)

#f1 = open('tabla_vs_kmax_gamma_r=0.2_final.table','w')
#f1.write('gamma\tkmax\tRinf\tImax\ttImax\ttRinf\n')
if args.write:
	f2 = open('tabla_observables_vs_gamma_vs_r_K200.table','w')
	f2.write('r\tgamma\tkmax\tRinf\tImax\ttImax\ttRinf\n')
	contador=0

for i in range(len(gamma_list)):
	gamma=gamma_list[i]
	for r in r_list:
		for j in range(len(k_list)):
			k_max=k_list[j]
			print('r',r,'gamma',gamma,'kmax',k_max)
			pi,k_mean = pi_power_law(k_max,gamma)
			E0 = [0]*k_max
			I0 = [0.01]*k_max
			R0 = [0]*k_max
			S0 = [ 1-E0[k]-I0[k]-R0[k] for k in range(k_max) ]
			y0 = S0 + E0 + I0 + R0
			S,E,I,R=resuelve(t,deriv,k_max,gamma,y0,pi_power_law,T_inf,T_inc,r)
			E_tot=sum([E[k]*pi[k] for k in range(k_max)])
			S_tot=sum([S[k]*pi[k] for k in range(k_max)])
			I_tot=sum([I[k]*pi[k] for k in range(k_max)])
			R_tot=sum([R[k]*pi[k] for k in range(k_max)])
			Imax , t_Imax , Rinf, t_Rinf = calculo_observables(I_tot,R_tot,t)
			if args.write:
				#f1.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(gamma,k_max,Rinf,Imax,t_Imax,t_Rinf))
				f2.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(r,gamma,k_max,Rinf,Imax,t_Imax,t_Rinf))
				contador +=1
				if contador%10==0:
					f2.flush()
					os.fsync(f2)
					print('contador:',contador)
			if args.plot:
			#if k_max%10==0:
				folder = 'Figuras r={}'.format(r)
				if not os.path.exists(folder): os.makedirs(folder)
				figname=folder+'/KSEIR_kmax%i_gamma%.2f.png'%(k_max,gamma)
				#title=r'KSEIR con $K$=%i, $\gamma$=%.2f'%(k_max,gamma)
				title=r'$\gamma$=%.2f'%(gamma)
				plot_tot(t,gamma,S_tot,E_tot,I_tot,R_tot,title,figname,tmax=100)
