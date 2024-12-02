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
#import qdldl

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
            #if y!=Ny-1:
            H[2*(x+Nx*y),2*(x+Nx*((y+1)%Ny))]=-t
            H[2*(x+Nx*y)+1,2*(x+Nx*((y+1)%Ny))+1]=t
            
            
            #pairing
            #if y!=Ny-1:
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
    ky_values=np.linspace(-np.pi,np.pi,10001)
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

def d_sigma(kx,z,t,mu,Delta,sign_y):

    dx=Delta*np.sin(kx)
    dy=Delta*(z-1/z)/(2j)*sign_y
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


def analytic_GF(omega,kx,y,t,mu,Delta,eta=0.00001j):
    # pm=[1,-1]
    # pole_values=np.array(([poles(omega, kx, mu, Delta, pm1, pm2) for pm1,pm2 in itr.product(pm,pm)]))
    g=np.zeros((2,2),dtype=complex)
    
    # for p_indx,p in enumerate(pole_values):
    #     if abs(p)<1:
            
    #         rm_pole_values=np.delete(pole_values,p_indx)
            
    #         denominator=np.prod(p-rm_pole_values)
            
    #         g+=1/(t*(Delta**2/4-1)*denominator)*(p*((omega+eta)*np.identity(2)+d_sigma(kx, p, t, mu, Delta)))
    if y==0:
        sign_y=1
    else:
        sign_y=np.sign(y)
        
    if mu==0:
        sign_mu=1
    else:
        sign_mu=np.sign(mu)
        
    p1= poles(omega,kx,mu,Delta,1,sign_mu)
    p2= poles(omega,kx,mu,Delta,-1,sign_mu)
    p3= poles(omega,kx,mu,Delta,1,-sign_mu)
    p4= poles(omega,kx,mu,Delta,-1,-sign_mu)
    
    g+=1/(t*(Delta**2/4-1)*(p1-p2)*(p1-p3)*(p1-p4))*(p1**(abs(y)+1)*((omega+eta)*np.identity(2)+d_sigma(kx, p1, t, mu, Delta,sign_y)))
    g+=1/(t*(Delta**2/4-1)*(p2-p1)*(p2-p3)*(p2-p4))*(p2**(abs(y)+1)*((omega+eta)*np.identity(2)+d_sigma(kx, p2, t, mu, Delta,sign_y)))
   
    return g

def analytic_T_matrix(omega,kx,t,mu,Delta,V,eta=0.00001j):
    g=analytic_GF(omega, kx,0, t, mu, Delta,eta=eta)
    if V==0:
        T=10**3*np.array(([1,0],[0,-1]))
    else:
        T=1/V*np.array(([1,0],[0,-1]))-g

    return T

def total_analytic_GF(omega,kx,y1,y2,t,mu,Delta,V,eta=0.00001j):
    g_y1=analytic_GF(omega, kx, y1, t, mu, Delta,eta=eta)
    g_y2=analytic_GF(omega, kx, -y2, t, mu, Delta,eta=eta)
    g_y1y2=analytic_GF(omega, kx, y1-y2, t, mu, Delta)
    T=analytic_T_matrix(omega, kx, t, mu, Delta, V,eta=eta)
    
    G=g_y1y2+g_y1@T@g_y2
    
    return G
    
    
#Analytic Ingap Bands----------------------------------------------------------

def pole_condition(omega,kx,t,mu,Delta,V):
    g=analytic_GF(omega, kx,0, t, mu, Delta)
    
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
    
    p_pole_ret,h_pole_ret=p_pole[0][0],h_pole[0][0]
    
    #This determines if the solution is good or not
    #This is generally true when the in-gap band is not seperated from the bulk
    #If it is bad, the intial guess is simply returned
    if abs(p_pole[1]["fvec"])>10**(-5):
        p_pole_ret=x0
    if abs(h_pole[1]["fvec"])>10**(-5):
        h_pole_ret=-x0
    
    return p_pole_ret,h_pole_ret

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
    
    p_pole_ret,h_pole_ret=p_pole[0][0],h_pole[0][0]
    
    #This determines if the solution is good or not
    #This is generally true when the in-gap band is not seperated from the bulk
    #If it is bad, the intial guess is simply returned
    if abs(p_pole[1]["fvec"])>10**(-5):
        p_pole_ret=x0
    if abs(h_pole[1]["fvec"])>10**(-5):
        h_pole_ret=-x0
    
    return p_pole_ret,h_pole_ret

#Electronic Structure----------------------------------------------------------

def subtrate_LDOS(omega,kx,t,mu,Delta):
    g=analytic_GF(omega, kx, 0,t, mu, Delta)
    
    LDOS_value=-1/np.pi*np.imag(np.trace(g))
    
    return LDOS_value

def LDOS(omega,kx,y,t,mu,Delta,V):
    G=total_analytic_GF(omega, kx, y, y, t, mu, Delta, V,eta=0.001j)
    
    LDOS_value=-1/np.pi*np.imag(np.trace(G))
    
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

def analytic_phase_boundaries(t,mu,Delta,kx=0):
    g=analytic_GF(0, kx,0, t, mu, Delta,eta=0)
    
    return np.real(1/g[0,0])

def analytic_phase_boundaries_numpy(t,mu,Delta,V,kx=0):
    g=analytic_GF(0, kx,0, t, mu, Delta,eta=0)
    g+=np.conj(g.T)
    g*=1/2
    
    return V-np.real(1/g[0,0])

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

def top_ham_invariant(t,mu,Delta,V):
    kx_values=[0,np.pi]
    U=majorana_unitary(1)
    invariant=1
    for kx in kx_values:
        T=analytic_T_matrix(0, kx, t, mu, Delta, V,eta=0)
        T_majorana=np.round(np.conj(U.T)@T@U,decimals=6)
        invariant*=np.sign(pf.pfaffian(T_majorana))
    return np.real(invariant)


#Spectral Localiser------------------------------------------------------------
def x_operator(Nx,Ny,sparse=True):
    if sparse==True:
        X=dok_matrix((2*Nx*Ny,2*Nx*Ny))
    if sparse==False:
        X=np.zeros((2*Nx*Ny,2*Nx*Ny))
        
    for x in range(Nx):
        for y in range(Ny):
            X[2*(x+Nx*y),2*(x+Nx*y)]=x
            X[2*(x+Nx*y)+1,2*(x+Nx*y)+1]=x
            
    if sparse==True:
        X=X.tocsc()
        
    return X

def y_operator(Nx,Ny,sparse=True):
    if sparse==True:
        Y=dok_matrix((2*Nx*Ny,2*Nx*Ny))
    if sparse==False:
        Y=np.zeros((2*Nx*Ny,2*Nx*Ny))
        
    for x in range(Nx):
        for y in range(Ny):
            Y[2*(x+Nx*y),2*(x+Nx*y)]=y
            Y[2*(x+Nx*y)+1,2*(x+Nx*y)+1]=y
            
    if sparse==True:
        Y=Y.tocsc()
        
    return Y


def spectral_localiser(x,y,E,Nx,Ny,t,mu,Delta,V,BC="PBC",chain_length=0,sparse=True):
    H=p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=chain_length,BC=BC,sparse=sparse)
    X=x_operator(Nx, Ny,sparse=sparse)
    Y=y_operator(Nx, Ny,sparse=sparse)
    
    kappa=0.01
    
    if sparse==True:
        L=dok_matrix((4*Nx*Ny,4*Nx*Ny),dtype=complex)
        iden=sp.identity(2*Nx*Ny,format="csc")
    if sparse==False:
        L=np.zeros((4*Nx*Ny,4*Nx*Ny),dtype=complex)
        iden=np.identity(2*Nx*Ny)
        
    L[:2*Nx*Ny,:2*Nx*Ny]=H-E*iden
    L[:2*Nx*Ny,2*Nx*Ny:]=kappa*(X-x*iden-1j*(Y-y*iden))
    L[2*Nx*Ny:,:2*Nx*Ny]=kappa*(X-x*iden+1j*(Y-y*iden))
    L[2*Nx*Ny:,2*Nx*Ny:]=-H+E*iden
    
    if sparse==True:
        L=L.tocsc()
    
    return L

def one_D_class_D_invariant(x,Nx,Ny,t,mu,Delta,V,BC="PBC",chain_length=0,sparse=True):
    H=p_ip_interface_real_space(Nx, Ny, t, mu, Delta, V,chain_length=chain_length,BC=BC,sparse=sparse)
    X=x_operator(Nx, Ny,sparse=sparse)
    
    
    kappa=0.001
    
    if sparse==False:
    
        C=kappa*(X-x*np.identity(2*Nx*Ny))+1j*H
        
        invariant,det=np.linalg.slogdet(C)
        
    if sparse==True:
        C=kappa*(X-x*sp.identity(2*Nx*Ny,format="csc"))+1j*H
        
        #LU_decom=spl.splu(C,permc_spec="NATURAL")
        LU_decom=spl.splu(C, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
        
        L=LU_decom.L
        U=LU_decom.U
        
        invariant=1
        for i in range(2*Nx*Ny):
            invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
   
    return np.real(invariant)

def two_D_class_D_invariant(x,y,E,Nx,Ny,t,mu,Delta,V,BC="PBC",chain_length=0,sparse=True):
    L=spectral_localiser(x, y, E, Nx, Ny, t, mu, Delta, V,BC=BC,chain_length=chain_length,sparse=sparse)
    
    if sparse==False:
        eigenvalues=np.linalg.eigvalsh(L)
        invariant=1/2*(len(eigenvalues[eigenvalues>=0])-len(eigenvalues[eigenvalues<0]))
        
    if sparse==True:
        pos_eigvals=spl.eigsh(L,k=2*Nx*Ny,sigma=0,which="LA",return_eigenvectors=False)
        neg_eigvals=-spl.eigsh(-L,k=2*Nx*Ny,sigma=0,which="LA",return_eigenvectors=False)
        if len(pos_eigvals[pos_eigvals>=0])==len(neg_eigvals[neg_eigvals<0]):
            invariant=0
            
        elif len(pos_eigvals[pos_eigvals>=0])>len(neg_eigvals[neg_eigvals<0]):
            invariant=(4*Nx*Ny-2*len(neg_eigvals[neg_eigvals<0]))/2
        elif len(pos_eigvals[pos_eigvals>=0])<len(neg_eigvals[neg_eigvals<0]):
            invariant=(2*len(pos_eigvals[pos_eigvals>=0])-4*Nx*Ny)/2
            
    
    return invariant


