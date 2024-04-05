
# https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression
# -that-includes-user-defined-functions-in-sympy

"""
Example: Complex Ginzburgh-Landau (CGL) model from Wilson and Ermentrout RSTA
2019

"""



# user-defined
#from nBodyCoupling import nBodyCoupling
#from nBodyCouplingbk0317 import nBodyCoupling
from nBodyCoupling import nBodyCoupling



#import matplotlib
import numpy as np
from sympy import Matrix
import sympy as sym
import time

import matplotlib.pyplot as plt

#import matplotlib
#matplotlib.use('QtAgg')

from scipy.integrate import solve_ivp

import argparse

def rhs(t,z,pdict,option='value',idx=''):
    """
    Right-hand side of the Complex Ginzburgh-Landau (CGL) model from
    Wilson and Ermentrout RSTA 2019 

    Parameters

        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the thalamic model v, h, r, w.
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.

    Returns

        numpy array or sympy Matrix
            returns numpy array if option == 'val'
            returns sympy Matrix if option == 'sym'

    """

    idx = str(idx)
    
    x,y = z
    R2 = x**2 + y**2
    mu = pdict['mu'+idx]
    sig = pdict['sig'+idx]
    rho = pdict['rho'+idx]

    if option == 'value' or option == 'val':
        return np.array([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                         sig*y*(mu-R2)+x*(1+rho*(R2-mu))])
    elif option == 'sym' or option == 'symbolic':
        return Matrix([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                       sig*y*(mu-R2)+x*(1+rho*(R2-mu))])

def coupling(vars_pair, pdict, option='value'):
    """

    Diffusive coupling function between Complex Ginzburgh Landau
    (CGL) oscillators.

    E.g.,this Python function is the function $G(x_i,x_j)$
    in the equation
    $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon \\sum_{j=1}^N G(x_i,x_j)$

    Parameters

        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            x1,y1,x2,y2
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.

    Returns

        * numpy array or sympy Matrix
            * returns numpy array if option == 'val'. 
            returns sympy Matrix if option == 'sym'

    """
    x1,y1,x2,y2 = vars_pair

    if option == 'value':
        return np.array([(x2-x1)-pdict['d']*(y2-y1),
                         (y2-y1)+pdict['d']*(x2-x1)])
    elif option == 'sym':
        return Matrix([(x2-x1)-pdict['d']*(y2-y1),
                       (y2-y1)+pdict['d']*(x2-x1)])

def coupling_mat(N,option='val'):
    """
    define coupling matrix.
    """

    if option == 'val':
        a = np.ones((N,N))/N
        for i in range(N):
            a[i,i] = 0
        
        return a

    elif option == 'sym':
        #a = sym.MatrixSymbol('a',N,N)
        a = sym.Matrix(N,N, lambda i,j:sym.symbols('a%d%d' % (i,j),real=True))
        
        print('a',a)
            
        return a


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute_h',action='store_true')
    args = parser.parse_args()

    print(args)

    var_names = ['x','y']

    pardict = {'q_val':1,
               'eps_val':0,
               'd_val':.9,
               'sig_val':.1,
               'rho_val':.15,
               'mu_val':1}

    pdict = {}
    for key in pardict.keys():
        key2 = key[:-4]
        pdict[key2] = pardict[key]

    kwargs = {'recompute_list':['pg'],
              
              'g_forward':False,
              'i_forward':False,
              'z_forward':False,
              'dense':True,

              'N':3,
              'coupling_mat':coupling_mat,
              'dir':'data/dat_cgl3/',
              
              'max_n':-1,
              'trunc_order':2,
              'trunc_deriv':1,
              'NA':101,
              'NG':101,
              'NP':101,
              'gij_parallel':False,
              'p_iter':5,
              'TN':5000,
              'rtol':1e-8,
              'atol':1e-8,
              'rel_tol':1e-6,
              'method':'LSODA',
              'load_all':True,
              'plot':False,
              'log_level':'INFO'}

    T_init = 2*np.pi
    lc_init = np.array([1,0,T_init])

    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = nBodyCoupling(rhs,coupling,lc_init,var_names,pardict,**kwargs)

    
if __name__ == "__main__":
    __spec__ = None
    main()
