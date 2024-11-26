# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:37:10 2024

@author: Harry MullineauxSanders
"""
from p_ip_functions_file import *

Nx=101
Ny=101
t=1
mu_values=np.linspace(-5,5,101)
Delta=0.1
V=-2
Nev=100
colors = plt.cm.winter(np.linspace(0, 0, len(mu_values)))

fig,axs=plt.subplots(1,2)
spectrum_PBC=np.zeros((Nev,len(mu_values)))

for mu_indx,mu in enumerate(tqdm(mu_values)):
    spectrum_PBC[:,mu_indx]=spl.eigsh(p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=Nx//2,BC="PBC",sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
ax=axs[0]
for n in range(Nev):
    ax.scatter(mu_values,spectrum_PBC[n,:],color=colors)
    
ax.set_xlabel(r"$\mu/t$")
ax.set_ylabel(r"$E/t$")


#real space spectrum of a 
spectrum_OBC=np.zeros((Nev,len(mu_values)))

for mu_indx,mu in enumerate(tqdm(mu_values)):
    spectrum_OBC[:,mu_indx]=spl.eigsh(p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=Nx,BC="OBC",sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
ax=axs[1]
for n in range(Nev):
    ax.scatter(mu_values,spectrum_OBC[n,:],c=colors)
    
ax.set_xlabel(r"$\mu/t$")
ax.set_ylabel(r"$E/t$")



        
    
    
    
    
    
    