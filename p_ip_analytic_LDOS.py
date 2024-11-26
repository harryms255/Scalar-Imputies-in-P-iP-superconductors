# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:29:39 2024

@author: Harry MullineauxSanders
"""

from p_ip_functions_file import *

# def numeric_bulk_GF(omega,kx,ky,t,mu,Delta):
#     H=np.array(([-2*t*(np.cos(kx)+np.cos(ky))-mu,Delta*(np.sin(kx)-1j*np.sin(ky))],
#                 [Delta*(np.sin(kx)+1j*np.sin(ky)),2*t*(np.cos(kx)+np.cos(ky))+mu]))
#     G=np.linalg.inv((omega+0.00001j)*np.identity(2)-H)
    
#     return G

# def numeric_GF(omega,kx,y,t,mu,Delta):
#     ky_values=np.linspace(-np.pi,np.pi,1001)
#     dk=ky_values[1]-ky_values[0]
#     GF=np.zeros((2,2),dtype=complex)
    
#     for ky in ky_values:
#         GF+=1/(2*np.pi)*numeric_bulk_GF(omega, kx, ky, t, mu, Delta)*dk
        
#     return GF

# def d_sigma(kx,z,t,mu,Delta):

#     dx=Delta*np.sin(kx)
#     dy=Delta*(z-1/z)/(2j)
#     dz=-(mu+2*np.cos(kx))-z-1/z
    
#     sigma_x=np.array(([0,1],[1,0]),dtype=complex)
#     sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
#     sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
#     return dx*sigma_x+dy*sigma_y+dz*sigma_z

# def coeff_a1(Delta):
#     return Delta**2/4-1

# def coeff_a2(kx,mu):
#     return -2*(mu+2*np.cos(kx))

# def coeff_a3(omega,kx,mu,Delta):
#     return omega**2-2-(mu+2*np.cos(kx))**2-Delta**2*np.sin(kx)**2-Delta**2/2

# def poles(omega,kx,mu,Delta,pm1,pm2):
    
#     omega+=0.00001j
#     a=coeff_a1(Delta)
#     b=coeff_a2(kx, mu)
#     c=coeff_a3(omega, kx, mu, Delta)
    
    
#     pole=(-b+pm1*np.emath.sqrt(8*a**2-4*a*c+b**2))/(4*a)+pm2/2*np.emath.sqrt(b**2/(2*a**2)-c/a-2-pm1*b/(2*a**2)*np.emath.sqrt(b**2-4*a*c+8*a**2))
#     return pole


# def analytic_GF(omega,kx,t,mu,Delta):
#     pm=[1,-1]
#     pole_values=np.array(([poles(omega, kx, mu, Delta, pm1, pm2) for pm1,pm2 in itr.product(pm,pm)]))
#     g=np.zeros((2,2),dtype=complex)
#     for p_indx,p in enumerate(pole_values):
#         if abs(p)<1:
#             rm_pole_values=np.delete(pole_values,p_indx)
            
#             denominator=np.prod(p-rm_pole_values)
            
#             g+=1/(t*(Delta**2/4-1)*denominator)*(p*((omega+0.00001j)*np.identity(2)+d_sigma(kx, p, t, mu, Delta)))
#     return g

# def LDOS(omega,kx,t,mu,Delta):
#     g=analytic_GF(omega, kx, t, mu, Delta)
    
#     LDOS_value=-1/np.pi*np.imag(np.trace(g))
    
#     return LDOS_value


t=1
mu=2
Delta=0.1
omega_values=np.linspace(-(4*t+abs(mu)),4*t+abs(mu),1001)
kx_values=np.linspace(-np.pi,np.pi,1001)


LDOS_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx,kx in enumerate(kx_values):
        LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, t, mu, Delta)
        
plt.figure()
sns.heatmap(LDOS_values,cmap="viridis",vmin=0,vmax=2)
plt.gca().invert_yaxis()
