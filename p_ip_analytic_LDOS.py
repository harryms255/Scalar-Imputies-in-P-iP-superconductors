# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:29:39 2024

@author: Harry MullineauxSanders
"""

from p_ip_functions_file import *

t=1
mu=2
Delta=0.1
omega_values=np.linspace(-(4*t+abs(mu)),4*t+abs(mu),1001)
kx_values=np.linspace(-np.pi,np.pi,1001)


LDOS_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx,kx in enumerate(kx_values):
        LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, t, mu, Delta)
        
plt.figure()
sns.heatmap(LDOS_values,cmap="viridis",vmin=0,vmax=2)
plt.gca().invert_yaxis()
