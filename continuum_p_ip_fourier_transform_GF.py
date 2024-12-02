# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:36:52 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
plt.close("all")

def bulk_p_ip_superconductor(kx,ky,mu,Delta):
    H=np.array(([-mu+kx**2+ky**2,Delta*(kx-1j*ky)],
                [Delta*(kx+1j*ky),mu-kx**2-ky**2]))
    return H

def bulk_GF(omega,kx,ky,mu,Delta):
    G=np.linalg.inv((omega+0.00001j)*np.identity(2)-bulk_p_ip_superconductor(kx,ky,mu,Delta))
    
    return G

def mixed_GF(omega,kx,y,mu,Delta):
    ky_values=np.linspace(-50,50,10001)
    dk=ky_values[1]-ky_values[0]
    GF=np.zeros((2,2),dtype=complex)
    
    for ky in ky_values:
        GF+=1/(2*np.pi)*bulk_GF(omega, kx, ky, mu, Delta)*dk
        
    return GF

def interface_GF(omega,kx,y1,y2,mu,Delta,V):
    Hv=V*np.array(([1,0],[0,-1]))
    if y1==y2:
        g_0=mixed_GF(omega, kx, 0, mu, Delta)
        g_1=mixed_GF(omega, kx, y1, mu, Delta)
        g_2=mixed_GF(omega, kx, -y2, mu, Delta)
        g_12=g_0
    else:
        g_0=mixed_GF(omega, kx, 0, mu, Delta)
        g_1=mixed_GF(omega, kx, y1, mu, Delta)
        g_2=mixed_GF(omega, kx, -y2, mu, Delta)
        g_12=mixed_GF(omega, kx, y1-y2, mu, Delta)
    
    T=np.linalg.inv(np.linalg.inv(Hv)-g_0)
    
    G=g_12+g_1@T@g_2
    
    return G

def phase_boundary(mu,Delta):
    g=mixed_GF(0, 0, 0, mu, Delta)
    
    return np.real(1/g[0,0])
    

mu_values=np.linspace(-2,2,21)
Delta=1
y=0
omega=0
kx_values=np.linspace(-np.pi,np.pi,101)
ky_values=np.linspace(-np.pi,np.pi,101)

# spectrum=np.zeros((len(kx_values),2))
# for mu in mu_values:
#     plt.figure(r"$\mu={:.2f}$".format(mu))
#     for kx_indx,kx in enumerate(tqdm(kx_values)):
#         spectrum[kx_indx,:]=np.linalg.eigvalsh(mixed_GF(omega,kx,y,mu,Delta))
        
#     for i in range(2):
#         plt.plot(kx_values,spectrum[:,i],"k")
plt.figure()
mu_values=np.linspace(-5,5,101)
phase_boundaries_values=np.zeros(len(mu_values))
for mu_indx,mu in enumerate(tqdm(mu_values)):
    phase_boundaries_values[mu_indx]=phase_boundary(mu, Delta)
    
plt.plot(mu_values,phase_boundaries_values)
        