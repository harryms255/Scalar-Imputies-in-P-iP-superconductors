# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:07:58 2024

@author: Harry MullineauxSanders
"""

from p_ip_functions_file import *
plt.close("all")

Nx=101
Ny=101
t=1
Delta=0.1
mu=-1
V=-4

x_values=np.linspace(0,Nx-1,Nx)



fig,axs=plt.subplots(1,2)
invariant_values_PBC=np.zeros(len(x_values))
for x_indx,x in enumerate(tqdm(x_values)):
   
    invariant_values_PBC[x_indx]=one_D_class_D_invariant(x, Nx, Ny, t, mu, Delta, V,sparse=True,BC="PBC",chain_length=Nx//2)

ax=axs[0]
ax.plot(x_values,invariant_values_PBC,"b-o")
ax.set_title("PBC")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\nu(x)$")
ax.axvline(x=Nx//2-(Nx//2)/2,linestyle="dashed",color="black",linewidth=3)
ax.axvline(x=Nx//2+(Nx//2)/2,linestyle="dashed",color="black",linewidth=3)




invariant_values_OBC=np.zeros(len(x_values))
for x_indx,x in enumerate(tqdm(x_values)):
    
    invariant_values_OBC[x_indx]=one_D_class_D_invariant(x, Nx, Ny, t, mu, Delta, V,sparse=True,BC="OBC",chain_length=Nx)

ax=axs[1]
ax.plot(x_values,invariant_values_OBC,"b-o")
ax.set_title("OBC")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\nu(x)$")
ax.axvline(x=0,linestyle="dashed",color="black",linewidth=3)
ax.axvline(x=Nx-1,linestyle="dashed",color="black",linewidth=3)
