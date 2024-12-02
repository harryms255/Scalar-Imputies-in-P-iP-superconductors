# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:40:31 2024

@author: Harry MullineauxSanders
"""

from p_ip_functions_file import *
plt.close("all")

Nx=75
Ny=75
t=1
Delta=0.1
mu=-1
V=-4

x_values=np.linspace(0,Nx-1,Nx)
y_values=np.linspace(0,Ny-1,Ny)
#y_values=[Ny//2]


fig,axs=plt.subplots(1,1)
# invariant_values_PBC=np.zeros((len(y_values),len(x_values)))
# for x_indx,x in enumerate(tqdm(x_values)):
#     for y_indx,y in enumerate(y_values):
#         invariant_values_PBC[y_indx,x_indx]=two_D_class_D_invariant(x, y, 0, Nx, Ny, t, mu, Delta, V,sparse=False,BC="PBC",chain_length=Nx//2)

# ax=axs[0]
# sns.heatmap(invariant_values_PBC,cmap="viridis",ax=ax)
# ax.invert_yaxis()
# ax.set_title("PBC")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# np.savetxt("PBC_two_dim_invariant_values.txt",invariant_values_PBC)



invariant_values_OBC=np.zeros((len(y_values),len(x_values)))
for x_indx,x in enumerate(tqdm(x_values)):
    for y_indx,y in enumerate(y_values):
        invariant_values_OBC[y_indx,x_indx]=two_D_class_D_invariant(x, y, 0, Nx, Ny, t, mu, Delta, V,sparse=False,BC="OBC",chain_length=Nx)

#ax=axs[1]
ax=axs
sns.heatmap(invariant_values_OBC,cmap="viridis",ax=ax)
ax.invert_yaxis()
ax.set_title("OBC")
ax.set_xlabel("x")
ax.set_ylabel("y")

np.savetxt("OBC_two_dim_invariant_values.txt",invariant_values_OBC)