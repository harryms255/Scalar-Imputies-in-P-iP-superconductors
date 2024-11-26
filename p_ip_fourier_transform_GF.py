# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:26:59 2024

@author: hm255
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
plt.close("all")

def bulk_p_ip_superconductor(kx,ky,t,mu,Delta):
    H=np.array(([-2*t*(np.cos(kx)+np.cos(ky))-mu,Delta*(np.sin(kx)-1j*np.sin(ky))],
                [Delta*(np.sin(kx)+1j*np.sin(ky)),2*t*(np.cos(kx)+np.cos(ky))+mu]))
    return H

def bulk_GF(omega,kx,ky,t,mu,Delta):
    G=np.linalg.inv((omega+0.00001j)*np.identity(2)-bulk_p_ip_superconductor(kx, ky, t, mu, Delta))
    
    return G

def mixed_GF(omega,kx,y,t,mu,Delta):
    ky_values=np.linspace(-np.pi,np.pi,1001)
    dk=ky_values[1]-ky_values[0]
    GF=np.zeros((2,2),dtype=complex)
    
    for ky in ky_values:
        GF+=1/(2*np.pi)*bulk_GF(omega, kx, ky, t, mu, Delta)*dk
        
    return GF

def interface_GF(omega,kx,y1,y2,t,mu,Delta,V):
    Hv=V*np.array(([1,0],[0,-1]))
    if y1==y2:
        g_0=mixed_GF(omega, kx, 0, t, mu, Delta)
        g_1=mixed_GF(omega, kx, y1, t, mu, Delta)
        g_2=mixed_GF(omega, kx, -y2, t, mu, Delta)
        g_12=g_0
    else:
        g_0=mixed_GF(omega, kx, 0, t, mu, Delta)
        g_1=mixed_GF(omega, kx, y1, t, mu, Delta)
        g_2=mixed_GF(omega, kx, -y2, t, mu, Delta)
        g_12=mixed_GF(omega, kx, y1-y2, t, mu, Delta)
    
    T=np.linalg.inv(np.linalg.inv(Hv)-g_0)
    
    G=g_12+g_1@T@g_2
    
    return G
    

t=1
# mu_values=np.linspace(-5,5,21)
mu_values=[-2]
Delta=0.1
y=0
omega=0
kx_values=np.linspace(-np.pi,np.pi,101)
ky_values=np.linspace(-np.pi,np.pi,101)

spectrum=np.zeros((len(kx_values),2))
for mu in mu_values:
    plt.figure(r"$\mu={:.2f}t$".format(mu))
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        spectrum[kx_indx,:]=np.linalg.eigvalsh(mixed_GF(omega, kx, y, t, mu, Delta))
        
    for i in range(2):
        plt.plot(kx_values,spectrum[:,i],"k")
        
    