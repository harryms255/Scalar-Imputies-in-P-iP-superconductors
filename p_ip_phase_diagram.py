# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:42:28 2024

@author: hm255
"""
from p_ip_functions_file import *
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import seaborn as sns
# import pfapack.pfaffian as pf

# plt.close("all")
# plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
# plt.rc('text', usetex=True)
# plt.rcParams.update({'font.size': 30})

# def p_ip_interface_model(kx,Ny,t,mu,Delta,V):
    
#     h=np.zeros((2*Ny,2*Ny),dtype=complex)
    
#     #Y hopping
#     for y in range(Ny):
#         h[2*((y+1)%Ny),2*y]=-t
#         h[2*((y+1)%Ny)+1,2*y+1]=t
        
#     #x Pairing
        
#         h[2*y,2*y+1]=-1j*Delta*np.sin(kx)
        
#     #y Pairing
#         #Have to include the 1/2 as it has been artifically doubled
#         h[2*((y+1)%Ny),2*y+1]=1j*Delta/2
#         h[2*((y+1)%Ny)+1,2*y]=1j*Delta/2
        
#     #H is made hermitian
    
#     h+=np.conj(h.T)
    
#     for y in range(Ny):
#         h[2*y,2*y]=-2*t*np.cos(kx)-mu
#         h[2*y+1,2*y+1]=+2*t*np.cos(kx)+mu
    
#     y=Ny//2
    
#     h[2*y,2*y]+=V
#     h[2*y+1,2*y+1]+=-V
#     return h

# def majorana_unitary(Ny):
#     U=1/np.sqrt(2)*np.array(([1,-1j],[1,1j]))
#     U_tot=np.kron(np.identity(Ny),U)
#     return U_tot

# def invariant(Ny,t,mu,Delta,V):
    
#     invariant=1
#     kx_values=[0,np.pi]
#     for kx in kx_values:
#         H=p_ip_interface_model(kx, Ny, t, mu, Delta, V)    
#         U=majorana_unitary(Ny)
#         H_majorana=np.conj(U.T)@H@U
#         invariant*=np.sign(pf.pfaffian(H_majorana))
#     return np.real(invariant)
    
Ny=101
t=1
mu=-2
Delta=0.1
V_values=np.linspace(-5,5,41)
kx_values=np.linspace(-np.pi,np.pi,101)
spectrum=np.zeros((2*Ny,len(kx_values)))

# for V in V_values:
#     plt.figure("V={:.2f}t".format(V))
#     for kx_indx,kx in enumerate(tqdm(kx_values)):
#         spectrum[:,kx_indx]=np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, V))
    
#     for i in range(2*Ny):
#         plt.plot(kx_values,spectrum[i,:],"k")
#     plt.title(r"$\nu={}$".format(invariant(Ny, t, mu, Delta, V)))
    

mu_values=np.linspace(-5,5,251)
#V_values=np.linspace(-5,5,251)
V_values=[0]

invariant_values=np.zeros((len(V_values),len(mu_values)))

for mu_indx,mu in enumerate(tqdm(mu_values)):
    for V_indx,V in enumerate(V_values):
        invariant_values[V_indx,mu_indx]=TB_invariant(Ny, t, mu, Delta, V)
        
plt.figure("phase_diagram")
sns.heatmap(invariant_values,cmap="viridis",vmin=-1,vmax=1)
plt.gca().invert_yaxis()


x_ticks=[]
x_labels=[]
y_ticks=[]
y_labels=[]

for i in range(11):
    y_ticks.append(i/10*len(V_values))
    y_labels.append(str(np.round(min(V_values)+i/10*(max(V_values)-min(V_values)),2)))
    
for i in range(11):
    x_ticks.append(i/10*len(mu_values))
    x_labels.append(str(np.round(min(mu_values)+i/10*(max(mu_values)-min(mu_values)),2)))
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)

plt.ylabel(r"$V/t$")
plt.xlabel(r"$\mu/t$")