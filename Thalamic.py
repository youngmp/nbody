"""
Example: Thalamic model from Wilson and Ermentrout RSTA 2019,
Rubin and Terman JCNS 2004

"""


# user-defined
from nBodyCoupling import nBodyCoupling

import numpy as np
import sympy as sym
from sympy import Matrix

from extensisq import SWAG, CK5, Ts5

import pathos

from pathos.pools import ProcessPool

import matplotlib.pyplot as plt

import argparse

def rhs(t,z,pdict,option='val'):
        """
        Right-hand side of the Thalamic model from Wilson and Ermentrout
        RSTA 2019 and Rubin and Terman JCNS 2004
        
        
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
        
        if option == 'val' or option == 'value':
            exp = np.exp
        else:
            exp = sym.exp
        
        v, h, r, w = z
        
        v *= 100
        r /= 100
        
        ah = 0.128*exp(-(v+46)/18)  #
        bh = 4/(1+exp(-(v+23)/5))  #
        
        minf = 1/(1+exp(-(v+37)/7))  #
        hinf = 1/(1+exp((v+41)/4))  #
        rinf = 1/(1+exp((v+84)/4))  #
        pinf = 1/(1+exp(-(v+60)/6.2))  #
        #print(pinf)
        
        tauh = 1/(ah+bh)  #
        taur = 28+exp(-(v+25)/10.5)  #
        
        iL = pdict['gL']*(v-pdict['eL'])  #
        ina = pdict['gna']*(minf**3)*h*(v-pdict['ena'])  #
        ik = pdict['gk']*((0.75*(1-h))**4)*(v-pdict['ek'])  #
        it = pdict['gt']*(pinf**2)*r*(v-pdict['et'])  #
        
        
        dv = (-iL-ina-ik-it+pdict['ib'])/pdict['c']
        dh = (hinf-h)/tauh
        dr = (rinf-r)/taur
        dw = pdict['alpha']*(1-w)/(1+exp(-(v-pdict['vt'])/pdict['sigmat']))\
            -pdict['beta']*w
        #dw = alpha*(1-w)-beta*w
        
        if option == 'val':
            return np.array([dv/100,dh,dr*100,dw])
            #return np.array([dv/100,dh,dr*100,dw])
            #return np.array([dv,dh,dr])
        else:
            return Matrix([dv/100,dh,dr*100,dw])
            #return Matrix([dv,dh,dr])

def coupling(vars_pair,pdict,option='val'):
        """
        
        Synaptic coupling function between Thalamic oscillators.
        
        E.g.,this Python function is the function $G(x_i,x_j)$
        in the equation
        $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$
        
        Parameters
        
            vars_pair : list or array
                contains state variables from oscillator A and B, e.g.,
                vA, hA, rA, wA, vB, hB, rB, wB  
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
                returns numpy array if option == 'val'. 
                returns sympy Matrix if option == 'sym'
    
        """
        vA, hA, rA, wA, vB, hB, rB, wB = vars_pair

        if option == 'val':
            return -np.array([wB*(vA*100-pdict['esyn']),0,0,0])/pdict['c']/100
        else:
            return -Matrix([wB*(vA*100-pdict['esyn']),0,0,0])/pdict['c']/100

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
        a = sym.Matrix(N,N, lambda i,j:sym.var('a_%d%d' % (i,j)))
        
        for i in range(N):
            a[i,i] = 0
            
        return a

import numba as nb


def eval_lam_arr(pts,lam=None):
    """
    use this for vectorizing lamban until i figure out something better
    """
    
    pts = list(pts)
    sol = list(map(lam,pts))
    
    return sol

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute_h',action='store_true')
    args = parser.parse_args()
    
    var_names = ['v','h','r','w']
    
    pardict = {'gL_val':0.05,
               'gna_val':3,
               'gk_val':5,
               'gt_val':5,
               'eL_val':-70,
               'ena_val':50,
               'ek_val':-90,
               'et_val':0,
               'esyn_val':0,
               'c_val':1,
               'alpha_val':3,
               'beta_val':2,
               'sigmat_val':0.8,
               'vt_val':-20,
               'ib_val':3.5}
    
    
    kwargs = {'recompute_LC':False,
              'recompute_monodromy':False,
              'recompute_g_sym':False,
              'recompute_g':False,
              'recompute_het_sym':False,
              'recompute_z':False,
              'recompute_i':False,

              
              'recompute_k_sym':False,
              'recompute_p_sym':False,
              
              'recompute_gij':False,
              
              'recompute_h_sym':False,
              'recompute_h':args.recompute_h,
              'z_forward':[False,True,True,True],
              'i_forward':[False,False,True,False,False,False,False],
              'i_bad_dx':True,

              'N':3,
              'coupling_mat':coupling_mat,
              'dir':'data/dat_thalamicn/',
              'trunc_order':2,
              'max_n':25,
              
              'LC_rate':1,
              'ignore_var':True,

              'NG':100,
              'NA':500,
              'p_iter':25,
              'max_iter':20,
              'TN':10000,
              
              
              'rtol':1e-9,
              'atol':1e-9,
              'rel_tol':1e-7,
              'LC_tol':1e-10,
              'method':'LSODA',
              'load_all':True,
              'LC_long_sim_time':1500,
              'log_level':'DEBUG'}
    
    #T_init = 10.6
    #LC_init = np.array([-.64,0.71,0.25,0,T_init])
    
    #pardict['beta_val'] = .2
    #pardict['ib_val'] = 0.8
    #pardict['esyn_val'] = -100
    #T_init = 34.74
    #LC_init = np.array([-.64,0.98,0.48,.1,T_init])
    
    pardict['beta_val'] = .2
    pardict['ib_val'] = 0.8
    pardict['esyn_val'] = -100
    kwargs['z_forward'] = [False,False,False]
    kwargs['i_forward'] = [False,False,False]
    T_init = 34.7
    
    LC_init = np.array([-.05467127, 0.3326327, 0.3430555, 0.4488427,T_init])
    #LC_init = np.array([-.5927,0.99,0.507,.006,T_init]) # for ib=.6
    
    a = nBodyCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)

    pdict = {}
    for key in pardict.keys():
        key2 = key[:-4]
        pdict[key2] = pardict[key]
    #print(a.h[0]['lam'][0]([1,0,0]))
    #print(a.h[1]['lam'][0]([0,1,0]))
    #print(a.h[2]['lam'][0]([0,0,1]))

    if False:
        fig,axs = plt.subplots(nrows=a.N,ncols=a.trunc_order+1)
        t = np.linspace(0,2*np.pi,1000)
        p = np.zeros(len(t))
        for i in range(a.N):
            for j in range(a.trunc_order+1):

                
                lam = a.h[i]['lam'][j]

                if a.N == 2:
                    #axs[i,j].plot(t,lam(t,0)-lam(-t,0))
                    #axs[i,j].plot(t,lam(0,t)-lam(0,-t))
                    
                    #fn1 = eval_lam_arr(zip(t,p),lam)
                    #fn2 = eval_lam_arr(zip(p,t),lam)
                    #axs[i,j].plot(t,fn1)
                    #axs[i,j].plot(t,fn2)
                    
                    in1 = np.column_stack((t,p))
                    in2 = np.column_stack((p,t))
                    
                    fn1 = eval_lam_arr(in1,lam=lam)
                    fn2 = eval_lam_arr(in2,lam=lam)
                    
                    axs[i,j].plot(t,fn1)
                    axs[i,j].plot(t,fn2)
                    
                if a.N == 3:
                
                    in1 = np.column_stack((t,p,p))
                    in2 = np.column_stack((p,t,p))
                    in3 = np.column_stack((p,p,t))
                
                    fn1 = eval_lam_arr(in1,lam=lam)
                    fn2 = eval_lam_arr(in2,lam=lam)
                    fn3 = eval_lam_arr(in3,lam=lam)
                    axs[i,j].plot(t,fn1)
                    axs[i,j].plot(t,fn2)
                    axs[i,j].plot(t,fn3)
                    
                    #axs[i,j].plot(t,lam(t,0,0))
                    #axs[i,j].plot(t,lam(0,t,0))
                    #axs[i,j].plot(t,lam(0,0,t))


    
        plt.show()
    
if __name__ == "__main__":
    
    __spec__ = None

    main()
