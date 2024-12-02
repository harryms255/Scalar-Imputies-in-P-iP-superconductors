# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:48:09 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})

def coeff_a1(Delta):
    return Delta**2/4-1

def coeff_a2(kx,mu):
    return -2*(mu+2*np.cos(kx))

def coeff_a3(omega,kx,mu,Delta):
    return omega**2-2-(mu+2*np.cos(kx))**2-Delta**2*np.sin(kx)**2-Delta**2/2

def poles(omega,kx,mu,Delta,pm1,pm2):
    
    omega+=0.00001j
    a=coeff_a1(Delta)
    b=coeff_a2(kx, mu)
    c=coeff_a3(omega, kx, mu, Delta)
    
    
    pole=(-b+pm1*np.emath.sqrt(8*a**2-4*a*c+b**2))/(4*a)+pm2/2*np.emath.sqrt(b**2/(2*a**2)-c/a-2-pm1*b/(2*a**2)*np.emath.sqrt(b**2-4*a*c+8*a**2))
    return pole

def pole_condition(z,omega,kx,mu,Delta):
    a1=coeff_a1(Delta)
    a2=coeff_a2(kx, mu)
    a3=coeff_a3(omega, kx, mu, Delta)
    
    return a1*(z**4+1)+a2*(z**3+z)+a3*z**2


kx_values=np.linspace(-np.pi,np.pi,10001)

mu_values=[-5,-2,2,5]
Delta=0.1
omega=0
pm=[1,-1]
color=["k","g","r","b"]

fig,axs=plt.subplots(2,2)
#axs_choice=[axs[0,0],axs[0,1],axs[0,2],axs[0,3],axs[1,0],axs[1,1],axs[1,2],axs[1,3]]
axs_choice=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]

for mu_indx,mu in enumerate(mu_values):
    
    
        
    # ax.plot(np.cos(np.linspace(-np.pi,np.pi)),np.sin(np.linspace(-np.pi,np.pi)),linestyle="dashed",color="black")
    # ax.plot(np.cos(np.linspace(-np.pi,np.pi)),-np.sin(np.linspace(-np.pi,np.pi)),linestyle="dashed",color="black")
    
    color_indx=0
    for pm1 in pm:
        for pm2 in pm:
            pole=poles(omega, kx_values, mu, Delta, pm1, pm2)
            ax=axs_choice[mu_indx]
            ax.set_title(r"$\mu={:.0f}$".format(mu))
            ax.set_xlabel(r"$k_x/\pi$")
            ax.set_ylabel(r"$|p_{\pm_1,\pm_2}|$")
            ax.set_xlim(left=-1,right=1)
            ax.set_ylim(bottom=0,top=8)
            ax.axhline(y=1,linestyle="dashed")
            if mu_indx==0:
                ax.plot(kx_values/np.pi,abs(pole),color=color[color_indx],label=r"$\pm_1,\pm_2={},{}$".format(pm1,pm2))
            else:
                ax.plot(kx_values/np.pi,abs(pole),color=color[color_indx])
                
            # ax=axs_choice[4+mu_indx]
            # ax.set_xlabel(r"$k_x/\pi$")
            # ax.set_ylabel(r"$arg(z_{\pm_1,\pm_2})/\pi$")
            # ax.plot(kx_values/np.pi,np.angle(pole)/np.pi,color=color[color_indx])
            # ax.set_xlim(left=-1,right=1)
            # ax.set_ylim(bottom=-1.1,top=1.1)

            color_indx+=1
            
            
fig.legend()



