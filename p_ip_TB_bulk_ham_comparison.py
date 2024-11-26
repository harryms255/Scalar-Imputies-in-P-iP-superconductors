# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:47:04 2024

@author: hm255
"""

from p_ip_functions_file import *



Ny=101
t=1
mu=-2
Delta=0.1

kx_values=np.linspace(-np.pi,np.pi,501)
ky_values=2*np.pi/Ny*np.linspace(0,Ny-1,Ny)

TB_bands=np.zeros((2*Ny,len(kx_values)))
bulk_bands=np.zeros((2*Ny,len(kx_values)))


# for kx_indx,kx in enumerate(tqdm(kx_values)):
#     TB_bands[:,kx_indx]=np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta))
    
#     for ky_indx,ky in enumerate(ky_values):
#         bulk_bands[2*ky_indx:2*ky_indx+2,kx_indx]=np.linalg.eigvalsh(p_ip(kx, ky, t, mu, Delta))

# plt.figure()
# for i in range(2*Ny):
#     plt.plot(kx_values,TB_bands[i,:],"k-")
# for i in range(2*Ny):
#     plt.plot(kx_values,bulk_bands[i,:],"r--")
    
    
mu_values=np.linspace(-5,0,501)
TB_bands=np.zeros((2*Ny,len(kx_values)))
bulk_bands=np.zeros((2*Ny,len(kx_values)))
for mu_indx,mu in enumerate(tqdm(mu_values)):
    TB_bands[:,mu_indx]=np.linalg.eigvalsh(p_ip_interface_model(0, Ny, t, mu, Delta,0))
    for ky_indx,ky in enumerate(ky_values):
             bulk_bands[2*ky_indx:2*ky_indx+2,mu_indx]=np.linalg.eigvalsh(p_ip(0, ky, t, mu, Delta))
    
plt.figure()
for i in range(2*Ny):
    plt.plot(mu_values,TB_bands[i,:],"k")
    
for i in range(2*Ny):
    plt.plot(mu_values,bulk_bands[i,:],"r--")


# for ky in ky_values:
#     E=np.sqrt((2*t*(np.cos(0)+np.cos(ky))+mu_values)**2+Delta**2*np.sin(ky)**2)
    
#     plt.plot(mu_values,E,"r--")
#     plt.plot(mu_values,-E,"r--")

