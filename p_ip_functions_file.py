# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:51:41 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools as itr
import seaborn as sns
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from scipy.optimize import fsolve
import pfapack.pfaffian as pf

plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})

#bulk substrate model--------------------------------------------------------------------
def p_ip(kx,ky,t,mu,Delta):
    H=np.array(([-2*t*np.cos(kx)-2*t*np.cos(ky)-mu,Delta*(np.sin(kx)-1j*np.sin(ky))],[Delta*(np.sin(kx)+1j*np.sin(ky)),mu+2*t*(np.cos(kx)+np.cos(ky))]))
    return H

#k space Tight binding models--------------------------------------------------
def p_ip_interface_model(kx,Ny,t,mu,Delta,V):
    
    h=np.zeros((2*Ny,2*Ny),dtype=complex)
    
    #Y hopping
    for y in range(Ny):
        h[2*((y+1)%Ny),2*y]=-t
        h[2*((y+1)%Ny)+1,2*y+1]=t
        
    #x Pairing
        
        h[2*y,2*y+1]=-1j*Delta*np.sin(kx)
        
    #y Pairing
        #Have to include the 1/2 as it has been artifically doubled
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

#Real spacce tight binding models----------------------------------------------

def p_ip_interface_real_space(Nx,Ny,t,mu,Delta,V,chain_length=0,BC="PBC",sparse=False):
    
    if sparse==False:
        H=np.zeros((2*Nx*Ny,2*Nx*Ny),dtype=complex)
    if sparse==True:
        H=dok_matrix((2*Nx*Ny,2*Nx*Ny),dtype=complex)
    
    if chain_length==0:
        chain_length=Nx
        
    try:
        if BC=="PBC":
            
            #hoppings
            for x in range(Nx):
                for y in range(Ny):
                    #hoppings
                    H[2*(x+Nx*y),2*((x+1)%Nx+Nx*y)]=-t
                    H[2*(x+Nx*y)+1,2*((x+1)%Nx+Nx*y)+1]=t
                    
                    
                    #pairing
                    H[2*(x+Nx*y),2*((x+1)%Nx+Nx*y)+1]=Delta
                    H[2*((x+1)%Nx+Nx*y),2*(x+Nx*y)+1]=-Delta
                    
                    
        elif BC=="OBC":
            
            #hoppings
            for x in range(Nx):
                for y in range(Ny):
                    #hoppings
                    if x!=Nx-1:
                        H[2*(x+Nx*y),2*((x+1)+Nx*y)]=-t
                        H[2*(x+Nx*y)+1,2*((x+1)+Nx*y)+1]=t
                    
                    
                    #pairing
                    if x!=Nx-1:
                        H[2*(x+Nx*y),2*((x+1)%Nx+Nx*y)+1]=Delta
                        H[2*((x+1)%Nx+Nx*y),2*(x+Nx*y)+1]=-Delta
                        
                        
        else:
            raise ValueError
    
    except ValueError:
        print("\nInvalid Boundary Conditions Chosen, it must be either 'OBC' or 'PBC'\n")
    
    #y-terms
    for x in range(Nx):
        for y in range(Ny):
                
            #y-hoppings
            H[2*(x+Nx*y),2*(x+Nx*((y+1)%Ny))]=-t
            H[2*(x+Nx*y)+1,2*(x+Nx*((y+1)%Ny))+1]=t
            
            
            #pairing
            
            H[2*(x+Nx*y),2*(x+Nx*((y+1)%Ny))+1]=1j*Delta
            H[2*(x+Nx*((y+1)%Ny)),2*(x+Nx*y)+1]=-1j*Delta
                
    
    H+=np.conj(H.T)
    #chemical potential
    for x in range(Nx):
        for y in range(Ny):
            H[2*(x+Nx*y),2*(x+Nx*y)]=-mu
            H[2*(x+Nx*y)+1,2*(x+Nx*y)+1]=mu
            
    #Impurity interface
    x_min=Nx//2-chain_length//2
    x_max=Nx//2+chain_length//2
    y=Ny//2
    for x in range(x_min,x_max):
        H[2*(x+Nx*y),2*(x+Nx*y)]+=V
        H[2*(x+Nx*y)+1,2*(x+Nx*y)+1]+=-V

    if sparse==True:
        H=H.tocsc()
        
    return H

#Numeric Green's Functions-----------------------------------------------------

def numeric_bulk_GF(omega,kx,ky,t,mu,Delta):
    H=np.array(([-2*t*(np.cos(kx)+np.cos(ky))-mu,Delta*(np.sin(kx)-1j*np.sin(ky))],
                [Delta*(np.sin(kx)+1j*np.sin(ky)),2*t*(np.cos(kx)+np.cos(ky))+mu]))
    G=np.linalg.inv((omega+0.00001j)*np.identity(2)-H)
    
    return G

def numeric_GF(omega,kx,y,t,mu,Delta):
    ky_values=np.linspace(-np.pi,np.pi,1001)
    dk=ky_values[1]-ky_values[0]
    GF=np.zeros((2,2),dtype=complex)
    
    for ky in ky_values:
        GF+=1/(2*np.pi)*numeric_bulk_GF(omega, kx, ky, t, mu, Delta)*dk
        
    return GF

def numeric_interface_GF(omega,kx,y1,y2,t,mu,Delta,V):
    Hv=V*np.array(([1,0],[0,-1]))
    if y1==y2:
        g_0=numeric_GF(omega, kx, 0, t, mu, Delta)
        g_1=numeric_GF(omega, kx, y1, t, mu, Delta)
        g_2=numeric_GF(omega, kx, -y2, t, mu, Delta)
        g_12=g_0
    else:
        g_0=numeric_GF(omega, kx, 0, t, mu, Delta)
        g_1=numeric_GF(omega, kx, y1, t, mu, Delta)
        g_2=numeric_GF(omega, kx, -y2, t, mu, Delta)
        g_12=numeric_GF(omega, kx, y1-y2, t, mu, Delta)
    
    T=np.linalg.inv(np.linalg.inv(Hv)-g_0)
    
    G=g_12+g_1@T@g_2
    
    return G

#Analytic Green's Function-----------------------------------------------------

def d_sigma(kx,z,t,mu,Delta):

    dx=Delta*np.sin(kx)
    dy=Delta*(z-1/z)/(2j)
    dz=-(mu+2*np.cos(kx))-z-1/z
    
    sigma_x=np.array(([0,1],[1,0]),dtype=complex)
    sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
    sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
    return dx*sigma_x+dy*sigma_y+dz*sigma_z

def coeff_a1(Delta):
    return Delta**2/4-1

def coeff_a2(kx,mu):
    return -2*(mu+2*np.cos(kx))

def coeff_a3(omega,kx,mu,Delta):
    eta=0.00001j
    return (omega+eta)**2-2-(mu+2*np.cos(kx))**2-Delta**2*np.sin(kx)**2-Delta**2/2

def poles(omega,kx,mu,Delta,pm1,pm2):
    

    a=coeff_a1(Delta)
    b=coeff_a2(kx, mu)
    c=coeff_a3(omega, kx, mu, Delta)
    
    
    pole=(-b+pm1*np.emath.sqrt(8*a**2-4*a*c+b**2))/(4*a)+pm2/2*np.emath.sqrt(b**2/(2*a**2)-c/a-2-pm1*b/(2*a**2)*np.emath.sqrt(b**2-4*a*c+8*a**2))
    return pole


def analytic_GF(omega,kx,t,mu,Delta):
    pm=[1,-1]
    pole_values=np.array(([poles(omega, kx, mu, Delta, pm1, pm2) for pm1,pm2 in itr.product(pm,pm)]))
    g=np.zeros((2,2),dtype=complex)
    #pole_number=0
    for p_indx,p in enumerate(pole_values):
        if abs(p)<1:
            #pole_number+=1
            rm_pole_values=np.delete(pole_values,p_indx)
            
            denominator=np.prod(p-rm_pole_values)
            
            g+=1/(t*(Delta**2/4-1)*denominator)*(p*((omega+0.00001j)*np.identity(2)+d_sigma(kx, p, t, mu, Delta)))
    #print(kx,pole_number)
    return g

#Analytic Ingap Bands----------------------------------------------------------

def pole_condition(omega,kx,t,mu,Delta,V):
    g=analytic_GF(omega, kx, t, mu, Delta)
    
    if V==0:
        T=10**6*np.array(([1,0],[0,-1]))
    else:
        T=1/V*np.array(([1,0],[0,-1]))-g
        
    pole_condition=np.linalg.det(T)
    
    pole_condition_return=abs(pole_condition)**2
    return pole_condition_return

def analytic_ingap_band(kx,t,mu,Delta,V,x0=0):
    
    det_T=lambda omega:pole_condition(omega, kx, t, mu, Delta, V)
    if x0==0:
        x0=0.99*gap(kx, t, mu, Delta)
    
    p_pole=fsolve(det_T,x0=x0,full_output=True)
    h_pole=fsolve(det_T,x0=-x0,full_output=True)
    
    return p_pole[0][0],h_pole[0][0]

#Numeric Ingap Bands----------------------------------------------------------

def numeric_pole_condition(omega,kx,t,mu,Delta,V):
    g=numeric_GF(omega, kx,0, t, mu, Delta)
    
    if V==0:
        T=10**6*np.array(([1,0],[0,-1]))
    else:
        T=1/V*np.array(([1,0],[0,-1]))-g
        
    pole_condition=np.linalg.det(T)
    
    pole_condition_return=abs(pole_condition)**2
    return pole_condition_return

def numeric_ingap_band(kx,t,mu,Delta,V,x0=0):
    
    det_T=lambda omega:numeric_pole_condition(omega, kx, t, mu, Delta, V)
    if x0==0:
        x0=0.99*gap(kx, t, mu, Delta)
    
    p_pole=fsolve(det_T,x0=x0,full_output=True)
    h_pole=fsolve(det_T,x0=-x0,full_output=True)
    
    return p_pole[0][0],h_pole[0][0]

#Electronic Structure----------------------------------------------------------

def LDOS(omega,kx,t,mu,Delta):
    g=analytic_GF(omega, kx, t, mu, Delta)
    
    LDOS_value=-1/np.pi*np.imag(np.trace(g))
    
    return LDOS_value

def gap(kx,t,mu,Delta):
    
    mu_kx=mu+2*t*np.cos(kx)
    Delta_x=Delta*np.sin(kx)
    
    gap_values=np.array(([]))
    gap_values=np.append(gap_values,[(2*t+mu_kx)**2+Delta_x**2,(-2*t+mu_kx)**2+Delta_x**2])
    
    cos_ky=mu_kx*t/(Delta**2-2*t**2)
    if abs(cos_ky).any()<1:
        gap_values=np.append(gap_values,(2*t*cos_ky+mu_kx)**2+Delta_x**2+Delta**2*(1-cos_ky**2))
    gap_value=np.sqrt(np.min(gap_values))

    return gap_value  



#Topological Properties--------------------------------------------------------

def majorana_unitary(Ny):
    U=1/np.sqrt(2)*np.array(([1,-1j],[1,1j]))
    U_tot=np.kron(np.identity(Ny),U)
    return U_tot

def TB_invariant(Ny,t,mu,Delta,V):
    
    invariant=1
    kx_values=[0,np.pi]
    for kx in kx_values:
        H=p_ip_interface_model(kx, Ny, t, mu, Delta, V)    
        U=majorana_unitary(Ny)
        H_majorana=np.conj(U.T)@H@U
        invariant*=np.sign(pf.pfaffian(H_majorana))
    return np.real(invariant)


