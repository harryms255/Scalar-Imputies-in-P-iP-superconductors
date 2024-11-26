# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:18:03 2024

@author: Harry MullineauxSanders
"""
from p_ip_functions_file import *

omega=0
kx_values=np.linspace(-np.pi,np.pi,101)
t=1
mu=+1
Delta=0.1

analytic_spectrum=np.zeros((2,len(kx_values)),dtype=float)
numeric_spectrum=np.zeros((2,len(kx_values)),dtype=float)

for kx_indx,kx in enumerate(tqdm(kx_values)):
    analytic_spectrum[:,kx_indx]=np.linalg.eigvalsh(analytic_GF(omega, kx, t, mu, Delta))
    numeric_spectrum[:,kx_indx]=np.linalg.eigvalsh(numeric_GF(omega, kx, 0, t, mu, Delta))
    
plt.figure()
for i in range(2):
    if i==0:
        plt.plot(kx_values/np.pi,analytic_spectrum[i,:],"k",label="Analytic")
        plt.plot(kx_values/np.pi,numeric_spectrum[i,:],"b.",label="Numeric")
    else:
        plt.plot(kx_values/np.pi,analytic_spectrum[i,:],"k")
        plt.plot(kx_values/np.pi,numeric_spectrum[i,:],"b.")
    
plt.legend()   
plt.xlabel(r"$k_x/\pi$")
plt.ylabel(r"$\lambda_{g(\omega=0,k_x,y=0)}$")
plt.xlim(left=-1,right=1)

