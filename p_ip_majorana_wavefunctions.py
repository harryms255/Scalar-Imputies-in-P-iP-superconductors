# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:55:27 2024

@author: hm255
"""

from p_ip_functions_file import *
plt.close("all")

Nx=101
Ny=101
t=1
Delta=0.1
mu=-1
V_values=[-4,-1]
BC_values=["PBC","OBC"]
chain_length_values=[Nx//2,Nx]
fig,axs=plt.subplots(1,2)

for i in tqdm(range(2)):
    V=V_values[i]
    BC=BC_values[i]
    ax=axs[i]
    chain_length=chain_length_values[i]
    eigenvalues,eigenstates=spl.eigsh(p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,sparse=True,BC=BC,chain_length=chain_length),k=2,sigma=0,which="LM")
    
    majorana_mode_1=eigenstates[:,0]
    majorana_mode_2=eigenstates[:,1]
    mod_sqr_wavefunction=0.5*(abs(majorana_mode_1[::2])**2+abs(majorana_mode_1[1::2])**2+abs(majorana_mode_2[::2])**2+abs(majorana_mode_2[1::2])**2)
    real_space_wavefunction=np.reshape(mod_sqr_wavefunction, (Ny,Nx))
    sns.heatmap(real_space_wavefunction,cmap="viridis",vmin=0,ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    