# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:07:59 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root
from tqdm import tqdm
import seaborn as sns
plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})

def p_ip(kx,ky,t,mu,Delta):
    H=np.array(([-2*t*np.cos(kx)-2*t*np.cos(ky)-mu,Delta*(np.sin(kx)-1j*np.sin(ky))],[Delta*(np.sin(kx)+1j*np.sin(ky)),mu+2*t*(np.cos(kx)+np.cos(ky))]))
    return H

def numeric_g(omega,kx,Ny,y,t,mu,Delta):
    eta=0.001j
    g=lambda ky:np.linalg.inv((omega+eta)*np.identity(2)-p_ip(kx, ky, t, mu, Delta))
    
    ky_values=2*np.pi/Ny*np.linspace(0,Ny-1,Ny)
    dk=ky_values[1]-ky_values[0]
    g_y=np.zeros((2,2),dtype=complex)
    for ky in ky_values:
        g_y+=np.e**(1j*ky*y)/(2*np.pi)*g(ky)*dk
        
    return g_y


    
def p_ip_interface_model(kx,Ny,t,mu,Delta,V):
    
    h=np.zeros((2*Ny,2*Ny),dtype=complex)
    
    #Y hopping
    for y in range(Ny):
        h[2*((y+1)%Ny),2*y]=-t
        h[2*((y+1)%Ny)+1,2*y+1]=t
        
    #x Pairing
        
        h[2*y,2*y+1]=1j*Delta*np.sin(kx)
        
    #y Pairing
        
        h[2*((y+1)%Ny),2*y+1]=1j*Delta/2
        h[2*((y+1)%Ny)+1,2*y]=1j*Delta/2
        
    #H is made hermitian
    
    h+=np.conj(h.T)
    
    for y in range(Ny):
        h[2*y,2*y]=-2*t*np.cos(kx)-mu
        h[2*y+1,2*y+1]=+2*t*np.cos(kx)+mu
        
    y=Ny//2
    
    h[2*y,2*y]+=V
    h[2*y+1,2*y+1]+=-V
    return h

def TB_g(omega,Ny,kx,y,t,mu,Delta):
    H=p_ip_interface_model(kx, Ny, t, mu, Delta, 0)
    G=np.linalg.inv((omega+0.001j)*np.identity(2*Ny)-H)
    
    g=G[y:y+2,:2]
    
    return g

def GF_fourier_transform(omega,kx,y,t,mu,Delta):
    h_11=lambda ky: -2*t*(np.cos(kx)+np.cos(ky))-mu
    h_12=lambda ky: Delta*(np.sin(kx)-1j*np.sin(ky))
    h_21=lambda ky: Delta*(np.sin(kx)+1j*np.sin(ky))
    h_22=lambda ky: 2*t*(np.cos(kx)+np.cos(ky))+mu
    
    eta=0.00001j
    g_11=0
    g_12=0
    g_21=0
    g_22=0
    ky_values=np.linspace(-np.pi,np.pi,101)
    dky=ky_values[1]-ky_values[0]
    for ky in ky_values:
        g_11+=1/(2*np.pi)*np.e**(1j*ky*y)*(omega+eta+h_11(ky))/((omega+eta)**2-h_11(ky)**2-abs(h_12(ky))**2)*dky
        g_12+=1/(2*np.pi)*np.e**(1j*ky*y)*(h_21(ky))/((omega+eta)**2-h_11(ky)**2-abs(h_12(ky))**2)*dky
        g_21+=1/(2*np.pi)*np.e**(1j*ky*y)*(h_12(ky))/((omega+eta)**2-h_11(ky)**2-abs(h_12(ky))**2)*dky
        g_22+=1/(2*np.pi)*np.e**(1j*ky*y)*(omega+eta+h_22(ky))/((omega+eta)**2-h_11(ky)**2-abs(h_12(ky))**2)*dky
        
    return g_11,g_12,g_21,g_22

def T_matrix(omega,kx,t,mu,Delta,V):
    g_11,g_12,g_21,g_22=GF_fourier_transform(omega,kx,0,t,mu,Delta)
    
    if V==0:
        T_11=0
        T_12=0
        T_21=0
        T_22=0
        
    else:
        T_11=1/V-g_11
        T_12=-g_12
        T_21=-g_21
        T_22=-1/V-g_22
    
    return T_11,T_12,T_21,T_22
def gap(kx,t,mu,Delta):
    
    mu_kx=mu+2*t*np.cos(kx)
    Delta_x=Delta*np.sin(kx)
    
    gap_values=np.array(([]))
    gap_values=np.append(gap_values,[(2*t+mu_kx)**2+Delta_x**2,(-2*t+mu_kx)**2+Delta_x**2])
    
    cos_ky=mu_kx*t/(Delta**2-2*t**2)
    if abs(cos_ky)<1:
        gap_values=np.append(gap_values,(2*t*cos_ky+mu_kx)**2+Delta_x**2+Delta**2*(1-cos_ky**2))
    gap_value=np.sqrt(np.min(gap_values))

    return gap_value  

def pole_condition(omega,kx,t,mu,Delta,V):
    #g=TB_g(omega,Ny,kx,0,t,mu,Delta)
    g=numeric_g(omega, kx,Ny, 0, t, mu, Delta)
    
    # T_11,T_12,T_21,T_22=T_matrix(omega, kx, t, mu, Delta, V)
    
    # pole_condition=T_11*T_22-T_12*T_21
    if V==0:
        T=10**6*np.array(([1,0],[0,-1]))
    else:
        T=1/V*np.array(([1,0],[0,-1]))-g
        
    pole_condition=np.linalg.det(T)
    
        
    pole_condition_return=abs(pole_condition)**2
    # pole_condition_return=np.zeros(2,dtype=float)
    # pole_condition_return[0],pole_condition_return[1]=np.real(pole_condition),np.imag(pole_condition)
    
    return pole_condition_return


def ingap_band(kx,t,mu,Delta,V,x0=0):
    
    det_T=lambda omega:pole_condition(omega, kx, t, mu, Delta, V)
    if x0==0:
        x0=0.99*gap(kx, t, mu, Delta)
    
    p_pole=fsolve(det_T,x0=x0,full_output=True)
    h_pole=fsolve(det_T,x0=-x0,full_output=True)
    
    return p_pole[0][0],h_pole[0][0]

def LDOS(omega,kx,Ny,t,mu,Delta):
    g=numeric_g(omega, kx,Ny, 0, t, mu, Delta)
    #g=TB_g(omega, Ny,kx,0, t, mu, Delta)
    
    LDOS_value=-1/np.pi*np.imag(np.trace(g))
    
    return LDOS_value


Ny=101
t=1
mu_values=np.linspace(-5,0,51)
#mu_values=[-1]
Delta=1
V=-2
mu=-2
kx_values=np.linspace(-np.pi,np.pi,101)
numeric_spectrum=np.zeros((2*Ny,len(kx_values)))
analytic_spectrum=np.zeros((2,len(kx_values)))
gap_values=np.zeros(len(kx_values))

for mu in mu_values:
#plt.figure("mu={:.2f}t".format(mu))
    fig,axs=plt.subplots(1,1,num=r"$\mu={}t$".format(mu))
    #ax=axs[0]
    ax=axs
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        numeric_spectrum[:,kx_indx]=np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, V))
        
        x0=0.99*min(abs(np.linalg.eigvalsh(p_ip_interface_model(kx, Ny, t, mu, Delta, 0))))
        analytic_spectrum[:,kx_indx]=ingap_band(kx, t, mu, Delta, V,x0=x0)
    for i in range(2*Ny):
        ax.plot(kx_values,numeric_spectrum[i,:],"k")
    
    for i in range(2):
        ax.plot(kx_values,analytic_spectrum[i,:],"r-.")
    
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"E/t")
    ax.set_ylim(bottom=-5*Delta,top=5*Delta)
    
    # ax=axs[1]
    # omega_values=np.linspace(-5*Delta,5*Delta,101)
    # det_T_values=np.zeros((len(omega_values),len(kx_values)),dtype=float)
    # LDOS_values=np.zeros((len(omega_values),len(kx_values)),dtype=float)
    # for omega_indx,omega in enumerate(tqdm(omega_values)):
    #     for kx_indx,kx in enumerate(kx_values):
    #         #det_T_values[omega_indx,kx_indx]=1/sum(abs(pole_condition(omega, kx, t, mu, Delta, V))**2)
    #         LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx,Ny, t, mu, Delta)
            
            
    
    # sns.heatmap(LDOS_values,cmap="viridis",ax=ax,vmax=1,cbar=True)





    