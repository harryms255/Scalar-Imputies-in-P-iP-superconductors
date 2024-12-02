# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:30:23 2024

@author: hm255
"""

from p_ip_functions_file import *

t=1
mu_values=np.linspace(-5,5,501)
Delta=0.1
V_values=np.linspace(-5,5,501)

invariant_values=np.zeros((len(V_values),len(mu_values)))
phase_boundaries_1=np.zeros((len(V_values),len(mu_values)))
phase_boundaries_2=np.zeros((len(V_values),len(mu_values)))

for mu_indx,mu in enumerate(tqdm(mu_values)):
    for V_indx,V in enumerate(V_values):
        invariant_values[V_indx,mu_indx]=top_ham_invariant(t, mu, Delta, V)
        phase_boundaries_1[V_indx,mu_indx]=analytic_phase_boundaries_numpy(t, mu, Delta, V)
        phase_boundaries_2[V_indx,mu_indx]=analytic_phase_boundaries_numpy(t, mu, Delta, V,kx=np.pi)
        
plt.figure()
sns.heatmap(invariant_values,cmap="viridis",vmin=-1,vmax=1)
contour_1=plt.contour(phase_boundaries_1,levels=[0],linestyles="dashed",colors="black",linewidths=4)
contour_2=plt.contour(phase_boundaries_2,levels=[0],linestyles="dashed",colors="black",linewidths=4)
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