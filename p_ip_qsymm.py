# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:40:40 2024

@author: hm255
"""

import numpy as np
import qsymm
import sympy

ham_p_ip="""(-2*t*(cos(kx)+cos(ky))-mu)*sigma_z+Delta*sin(kx)*sigma_x+Delta*sin(ky)*sigma_y


"""    
ham_class_D="""(-2*t*(cos(kx)+cos(ky))-mu)*kron(sigma_z,eye(2))+Delta_T*sin(kx)*kron(sigma_x,sigma_x)+Delta*sin(ky)*kron(sigma_x,sigma_y)+Delta_S*kron(sigma_x,eye(2))


"""    


square_group=qsymm.groups.square()
print(r"Symmetries of $p+ip$ superconductor")
H_p_ip=qsymm.Model(ham_p_ip, momenta=['kx',"ky"])
discrete_symm, continuous_symm = qsymm.symmetries(H_p_ip,square_group)

for i in range(len(discrete_symm)):
    display(discrete_symm[i])
    print(np.round(discrete_symm[i].U,decimals=10))
    print("Conjugate={}".format(discrete_symm[i].conjugate))



# print(r"Symmetries of $s+p$ superconductor")
# H_D=qsymm.Model(ham_class_D, momenta=['kx',"ky"])
# discrete_symm, continuous_symm = qsymm.symmetries(H_D,square_group)

# for i in range(len(discrete_symm)):
#     display(discrete_symm[i])
#     print(np.round(discrete_symm[i].U,decimals=10))
#     print("Conjugate={}".format(discrete_symm[i].conjugate))