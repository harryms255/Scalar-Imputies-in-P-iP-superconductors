# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:10:56 2024

@author: Harry MullineauxSanders
"""
from p_ip_functions_file import *

t=1
mu_values=np.linspace(-5,5,501)
Delta=0.1

V_crit_1=np.zeros(len(mu_values))
V_crit_2=np.zeros(len(mu_values))
for mu_indx,mu in enumerate(tqdm(mu_values)):
    V_crit_1[mu_indx]=analytic_phase_boundaries(t, mu, Delta)
    V_crit_2[mu_indx]=analytic_phase_boundaries(t, mu, Delta,kx=np.pi)

plt.figure()
plt.plot(mu_values,V_crit_1,label=r"$k_x=0$")
plt.plot(mu_values,V_crit_2,label=r"$k_x=\pi$")
plt.legend()
plt.xlabel(r"$\mu/t$")
plt.ylabel(r"$V_{crit}$")
plt.ylim(top=5,bottom=-5)
