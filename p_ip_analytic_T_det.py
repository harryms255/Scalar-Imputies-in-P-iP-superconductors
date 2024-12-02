# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:44:04 2024

@author: hm255
"""

from p_ip_functions_file import *


t=1
mu_values=np.linspace(-5,5,1001)
Delta=0.1
V_values=np.linspace(-5,5,1001)
kx_values=[0,np.pi]

det_values=np.zeros((len(V_values),len(mu_values)))
phase_boundaries_1=np.zeros((len(V_values),len(mu_values)))
phase_boundaries_2=np.zeros((len(V_values),len(mu_values)))
fig,axs=plt.subplots(1,2)

for kx_indx,kx in enumerate(kx_values):
    ax=axs[kx_indx]
    for mu_indx,mu in enumerate(tqdm(mu_values)):
        for V_indx,V in enumerate(V_values):
            det_values[V_indx,mu_indx]=np.linalg.det(analytic_T_matrix(0, kx, t, mu, Delta, V,eta=10**(-3)))
            phase_boundaries_1[V_indx,mu_indx]=analytic_phase_boundaries_numpy(t, mu, Delta, V,kx=kx)
            #phase_boundaries_2[V_indx,mu_indx]=analytic_phase_boundaries_numpy(t, mu, Delta, V,kx=np.pi)
            
    
    sns.heatmap(np.log10(abs(det_values)),cmap="plasma",ax=ax)
    contour_1=ax.contour(phase_boundaries_1,levels=[0],linestyles="dashed",colors="black",linewidths=4)
    #contour_2=ax.contour(phase_boundaries_2,levels=[0],linestyles="dashed",colors="black",linewidths=4)
    ax.invert_yaxis()
    
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
    
    ax.set_yticks(ticks=y_ticks,labels=y_labels)
    ax.set_xticks(ticks=x_ticks,labels=x_labels)
    
    ax.set_ylabel(r"$V/t$")
    ax.set_xlabel(r"$\mu/t$")
    ax.set_title(r"$k_x={:.0f}\pi$".format(kx/np.pi))
fig.suptitle(r"$\log_{10}(|\det(H_V^{-1}-g(\omega=0,k))|)$")