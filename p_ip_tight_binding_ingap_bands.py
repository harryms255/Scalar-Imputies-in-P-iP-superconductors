# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:57:55 2024

@author: Harry MullineauxSanders
"""
from p_ip_functions_file import *
 
Ny=101
t=1
mu_values=np.linspace(-5,0,21)
Delta=0.1
V=0
mu=-2
kx_values=np.linspace(-np.pi,np.pi,101)
spectrum=np.zeros((2*Ny,len(kx_values)))
gap_values=np.zeros(len(kx_values))

#for mu in mu_values:
plt.figure("mu={:.2f}t".format(mu))
for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, V))
    gap_values[kx_indx]=gap(kx, t, mu, Delta)

for i in range(2*Ny):
    plt.plot(kx_values,spectrum[i,:],"k")
#plt.title(r"$\nu={}$".format(invariant(Ny, t, mu, Delta, V)))


plt.plot(kx_values,gap_values,"r-")
