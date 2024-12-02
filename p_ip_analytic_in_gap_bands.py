# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:52:08 2024

@author: Harry MullineauxSanders
"""

from p_ip_functions_file import *

Ny=101
t=1
mu_values=np.linspace(-5,5,51)
mu_values=[-3.5]
Delta=0.1
V=-1
kx_values=np.linspace(-np.pi,np.pi,101)
numeric_spectrum=np.zeros((2*Ny,len(kx_values)))
analytic_spectrum=np.zeros((2,len(kx_values)))
#gap_values=np.zeros(len(kx_values))

for mu in tqdm(mu_values):
    fig,axs=plt.subplots(1,1,num=r"$\mu={}t$".format(mu))
    ax=axs
    for kx_indx,kx in enumerate(kx_values):
        x0=0.99*np.min(abs(np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, 0))))
        numeric_spectrum[:,kx_indx]=np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, V))
    
        analytic_spectrum[:,kx_indx]=analytic_ingap_band(kx, t, mu, Delta, V,x0=x0)
        
#        gap_values[kx_indx]=gap(kx, t, mu, Delta)
    for i in range(2*Ny):
        ax.plot(kx_values/np.pi,numeric_spectrum[i,:],"k-")
    for i in range(2):
        ax.plot(kx_values/np.pi,analytic_spectrum[i,:],"b.",linewidth=4)
    
    
    #ax.plot(kx_values,gap_values,"g")
    
    ax.set_xlabel(r"$k_x/\pi$")
    ax.set_ylabel(r"E/t")
    ax.set_ylim(bottom=-(4*t+abs(mu)),top=(4*t+abs(mu)))
    ax.set_xlim(left=-1,right=1)