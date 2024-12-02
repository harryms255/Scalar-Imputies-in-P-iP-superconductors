# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:37:10 2024

@author: Harry MullineauxSanders
"""
from p_ip_functions_file import *

Nx=101
Ny=101
t=1
Delta=0.1
mu=-1
V_values=np.linspace(-5,0,101)
Nev=20
colors = plt.cm.winter(np.linspace(0, 0, len(V_values)))

fig,axs=plt.subplots(1,2)
spectrum_PBC=np.zeros((Nev,len(V_values)))

for V_indx,V in enumerate(tqdm(V_values)):
    spectrum_PBC[:,V_indx]=spl.eigsh(p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=Nx//2,BC="PBC",sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
ax=axs[0]
for n in range(Nev):
    ax.scatter(V_values,spectrum_PBC[n,:],color=colors)
    
ax.set_xlabel(r"$V/t$")
ax.set_ylabel(r"$E/t$")
ax.set_title("PBC")
ax.set_xlim(left=min(V_values),right=max(V_values))
for kx in [0,np.pi]:
    phase_boundary=analytic_phase_boundaries(t, mu, Delta,kx=kx)
    if phase_boundary>=min(V_values) and phase_boundary<=max(V_values):
        ax.axvline(x=phase_boundary,linestyle="dashed",color="black",linewidth=4)


#real space spectrum of a 
spectrum_OBC=np.zeros((Nev,len(V_values)))

for V_indx,V in enumerate(tqdm(V_values)):
    spectrum_OBC[:,V_indx]=spl.eigsh(p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=Nx,BC="OBC",sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
ax=axs[1]
for n in range(Nev):
    ax.scatter(V_values,spectrum_OBC[n,:],c=colors)
    
ax.set_xlabel(r"$V/t$")
ax.set_ylabel(r"$E/t$")
ax.set_xlim(left=min(V_values),right=max(V_values))
ax.set_title("OBC")
for kx in [0,np.pi]:
    phase_boundary=analytic_phase_boundaries(t, mu, Delta,kx=kx)
    if phase_boundary>=min(V_values) and phase_boundary<=max(V_values):
        ax.axvline(x=phase_boundary,linestyle="dashed",color="black",linewidth=4)



        
    
    
    
    
    
    