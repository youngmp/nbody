
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

def rhs(t, z, pdict, option='value'):
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

    x,y = z
    R2 = x**2 + y**2
    mu = pdict['mu']
    sig = pdict['sig']
    rho = pdict['rho']


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
        a = sym.Matrix(N,N, lambda i,j:sym.var('a_%d%d' % (i,j)))
        
        for i in range(N):
            a[i,i] = 0

        print('a',a)
            
        return a

def phase_phi(t,y,o,eps):
    """
    right-hand side of phase equations
    t: time
    y: phase diff variables
    o: object
    """
    
    p1=y[0];p2=y[1];p3=y[2]
    
    hi0k1 = o.h[0]['lam'][0]
    hi1k1 = o.h[1]['lam'][0]
    hi2k1 = o.h[2]['lam'][0]
    
    h11 = hi1k1(0,p1,p2,p3) - hi0k1(0,p1,p2,p3)
    h21 = hi2k1(0,p1,p2,p3) - hi0k1(0,p1,p2,p3)

    dp1 = np.real(h11)
    dp2 = np.real(h21)
    
    return eps*np.array([dp1,dp2])
    #return np.array([dt0,dt1,dt2,dt3])


def phase_o1_th(t,y,o,eps):
    """
    phase theta (as opposed to phase diffs phi)
    right-hand side of phase equations
    t: time
    y: phase diff variables
    o: object
    """
    
    t0=y[0];t1=y[1];t2=y[2];t3=y[3]
    
    hi0k1 = o.h[0]['lam'][0]
    hi1k1 = o.h[1]['lam'][0]
    hi2k1 = o.h[2]['lam'][0]
    hi3k1 = o.h[3]['lam'][0]

    h0 = hi0k1(t0,t1,t2,t3)
    h1 = hi1k1(t0,t1,t2,t3)
    h2 = hi2k1(t0,t1,t2,t3)
    h3 = hi3k1(t0,t1,t2,t3)

    dt0 = np.real(h0)
    dt1 = np.real(h1)
    dt2 = np.real(h2)
    dt3 = np.real(h3)
    
    return eps*np.array([dt0,dt1,dt2,dt3])

def phase_o2_th(t,y,o,eps):
    """
    2nd lowest order
    phase theta (as opposed to phase diffs phi)
    right-hand side of phase equations
    t: time
    y: phase diff variables
    o: object
    """
    
    t0=y[0];t1=y[1];t2=y[2]#;t3=y[3]
    
    hi0k0 = o.h[0]['lam'][0];hi0k1 = o.h[0]['lam'][1]
    hi1k0 = o.h[1]['lam'][0];hi1k1 = o.h[1]['lam'][1]
    hi2k0 = o.h[2]['lam'][0];hi2k1 = o.h[2]['lam'][1]
    #hi3k0 = o.h[3]['lam'][0];hi3k1 = o.h[3]['lam'][1]

    h0 = eps*hi0k0(t0,t1,t2)+eps**2*hi0k1(t0,t1,t2)
    h1 = eps*hi1k0(t0,t1,t2)+eps**2*hi1k1(t0,t1,t2)
    h2 = eps*hi2k0(t0,t1,t2)+eps**2*hi2k1(t0,t1,t2)
    #h3 = eps*hi3k0(t0,t1,t2,t3)+eps**2*hi3k1(t0,t1,t2,t3)

    dt0 = np.real(h0)
    dt1 = np.real(h1)
    dt2 = np.real(h2)
    #dt3 = np.real(h3)
    
    #return np.array([dt0,dt1,dt2,dt3])
    return np.array([dt0,dt1,dt2])


def phase_o3_th(t,y,o,eps):
    """
    2nd lowest order
    phase theta (as opposed to phase diffs phi)
    right-hand side of phase equations
    t: time
    y: phase diff variables
    o: object
    """
    
    t0=y[0];t1=y[1];t2=y[2]#;t3=y[3]
    
    hi0k0 = o.h[0]['lam'][0];hi0k1 = o.h[0]['lam'][1];hi0k2 = o.h[0]['lam'][2]
    hi1k0 = o.h[1]['lam'][0];hi1k1 = o.h[1]['lam'][1];hi1k2 = o.h[1]['lam'][2]
    hi2k0 = o.h[2]['lam'][0];hi2k1 = o.h[2]['lam'][1];hi2k2 = o.h[2]['lam'][2]
    #hi3k0 = o.h[3]['lam'][0];hi3k1 = o.h[3]['lam'][1]

    h0 = eps*hi0k0(t0,t1,t2)+eps**2*hi0k1(t0,t1,t2)+eps**3*hi0k2(t0,t1,t2)
    h1 = eps*hi1k0(t0,t1,t2)+eps**2*hi1k1(t0,t1,t2)+eps**3*hi1k2(t0,t1,t2)
    h2 = eps*hi2k0(t0,t1,t2)+eps**2*hi2k1(t0,t1,t2)+eps**3*hi2k2(t0,t1,t2)

    dt0 = np.real(h0)
    dt1 = np.real(h1)
    dt2 = np.real(h2)
    #dt3 = np.real(h3)
    
    #return np.array([dt0,dt1,dt2,dt3])
    return np.array([dt0,dt1,dt2])


def cgl4(t,y,pdict,eps):
    """
    assume y has 8 coordinates.
    (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    """
    x1,y1,x2,y2,x3,y3,x4,y4 = y

    g = coupling

    z1=(x1,y1);z2=(x2,y2);z3=(x3,y3);z4=(x4,y4)
    g1 = g(z1+z1,pdict)+g(z1+z2,pdict)+g(z1+z3,pdict)+g(z1+z4,pdict)
    g2 = g(z2+z1,pdict)+g(z2+z2,pdict)+g(z2+z3,pdict)+g(z2+z4,pdict)
    g3 = g(z3+z1,pdict)+g(z3+z2,pdict)+g(z3+z3,pdict)+g(z3+z4,pdict)
    g4 = g(z4+z1,pdict)+g(z4+z2,pdict)+g(z4+z3,pdict)+g(z4+z4,pdict)
    
    dz1 = rhs(t, z1, pdict, option='value') + eps*g1/4
    dz2 = rhs(t, z2, pdict, option='value') + eps*g2/4
    dz3 = rhs(t, z3, pdict, option='value') + eps*g3/4
    dz4 = rhs(t, z4, pdict, option='value') + eps*g4/4
    
    return np.array([dz1[0],dz1[1],dz2[0],dz2[1],dz3[0],dz3[1],dz4[0],dz4[1]])

def cgl3(t,y,pdict,eps):
    """
    assume y has 8 coordinates.
    (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    """
    x1,y1,x2,y2,x3,y3 = y

    g = coupling

    z1=(x1,y1);z2=(x2,y2);z3=(x3,y3)
    g1 = g(z1+z1,pdict)+g(z1+z2,pdict)+g(z1+z3,pdict)
    g2 = g(z2+z1,pdict)+g(z2+z2,pdict)+g(z2+z3,pdict)
    g3 = g(z3+z1,pdict)+g(z3+z2,pdict)+g(z3+z3,pdict)
    
    dz1 = rhs(t, z1, pdict, option='value') + eps*g1/4
    dz2 = rhs(t, z2, pdict, option='value') + eps*g2/4
    dz3 = rhs(t, z3, pdict, option='value') + eps*g3/4
    
    return np.array([dz1[0],dz1[1],dz2[0],dz2[1],dz3[0],dz3[1]])


def h_at_0(a,eps):
        
    # check that zero is a fixed point        
    hi0k0 = a.h[0]['lam'][0];hi0k1 = a.h[0]['lam'][1]
    hi1k0 = a.h[1]['lam'][0];hi1k1 = a.h[1]['lam'][1]
    hi2k0 = a.h[2]['lam'][0];hi2k1 = a.h[2]['lam'][1]
    #hi3k0 = a.h[3]['lam'][0];hi3k1 = a.h[3]['lam'][1]

    h0 = eps*hi0k0(0,0,0)+eps**2*hi0k1(0,0,0)
    h1 = eps*hi1k0(0,0,0)+eps**2*hi1k1(0,0,0)
    h2 = eps*hi2k0(0,0,0)+eps**2*hi2k1(0,0,0)
    #h3 = eps*hi3k0(0,0,0,0)+eps**2*hi3k1(0,0,0,0)
    #print('equilib check',np.abs([h0,h1,h2,h3]))
    print('equilib check',np.abs([h0,h1,h2]))

    
def d_estimate(a,eps,dx=1e-5):
    """
    estimate derivate of function of 4 vars
    a: object
    """

    hi0k0 = a.h[0]['lam'][0];hi0k1 = a.h[0]['lam'][1];hi0k2 = a.h[0]['lam'][2]
    hi1k0 = a.h[1]['lam'][0];hi1k1 = a.h[1]['lam'][1];hi1k2 = a.h[1]['lam'][2]
    hi2k0 = a.h[2]['lam'][0];hi2k1 = a.h[2]['lam'][1];hi2k2 = a.h[2]['lam'][2]
    #hi3k0 = a.h[3]['lam'][0];hi3k1 = a.h[3]['lam'][1]

    dx11 = 0
    for i in range(a.truc_order+1):
        dx11 += eps**(i+1)*a.h[0]['lam'][i](dx,0,0)
        dx12 += eps**(i+1)*a.h[0]['lam'][i](0,dx,0)
        dx13 += eps**(i+1)*a.h[0]['lam'][i](0,0,dx)

        dx21 += eps**(i+1)*a.h[1]['lam'][i](dx,0,0)
        dx22 += eps**(i+1)*a.h[1]['lam'][i](0,dx,0)
        dx23 += eps**(i+1)*a.h[1]['lam'][i](0,0,dx)

        dx31 += eps**(i+1)*a.h[2]['lam'][i](dx,0,0)
        dx32 += eps**(i+1)*a.h[2]['lam'][i](0,dx,0)
        dx33 += eps**(i+1)*a.h[2]['lam'][i](0,0,dx)
        
    dx11/=dx;dx12/=dx;dx13/=dx
    dx21/=dx;dx22/=dx;dx23/=dx
    dx31/=dx;dx32/=dx;dx33/=dx

    return np.array([[dx11,dx12,dx13],
                     [dx21,dx22,dx23],
                     [dx31,dx32,dx33]])#,
                     #[dx41,dx42,dx43,dx44]])


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

    kwargs = {'recompute_LC':False,
              'recompute_monodromy':False,
              'recompute_g_sym':False,
              'recompute_g':False,
              'recompute_het_sym':False,
              'recompute_z':False,
              'recompute_i':False,
              'recompute_k_sym':False,
              'recompute_p_sym':False,
              'recompute_p':False,
              'recompute_gij':False,
              'recompute_h_sym':False,
              
              'recompute_h':args.recompute_h,
              'g_forward':False,
              'i_forward':False,
              'z_forward':False,
              'dense':True,

              'N':3,
              'coupling_mat':coupling_mat,
              'dir':'data/dat_cgln/',
              
              'max_n':11,
              'trunc_order':2,
              'trunc_deriv':1,
              'NA':101,
              'NG':101,
              'gij_parallel':False,
              'p_iter':5,
              'TN':5000,
              'rtol':1e-8,
              'atol':1e-8,
              'rel_tol':1e-6,
              'method':'LSODA',
              'load_all':True}

    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])

    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = nBodyCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)

    if True:
        fig,axs = plt.subplots(nrows=a.N,ncols=a.trunc_order+1)
        p = np.pi
        for i in range(a.N):
            for j in range(a.trunc_order+1):

                t = a.Aarr_noend
                lam = a.h[i]['lam'][j]

                if a.N == 2:
                    #axs[i,j].plot(t,lam(t,0)-lam(-t,0))
                    #axs[i,j].plot(t,lam(0,t)-lam(0,-t))

                    axs[i,j].plot(t,lam(t,p))
                    axs[i,j].plot(t,lam(p,t))
                    
                if a.N == 3:
                    axs[i,j].plot(t,lam(t,p,p))
                    axs[i,j].plot(t,lam(p,t,p))
                    axs[i,j].plot(t,lam(p,p,t))

    #plt.show()


    T = 5000
    dt = .02
    t = np.linspace(0,T,int(T/dt))
    #init = [0,2*np.pi/3+.1,-2*np.pi/3-.1]
    
    #init = [3,-.12,np.pi/2-.001]
    #init = [0.01,0.02,0.03]
    init = [.1,.2,.3]
    #init = [.5,1,1.5]
    eps = .03
    
    #sol_phi = solve_ivp(phase_o1_phi,[0,T],init,args=(a,eps),
    #                    t_eval=t,method='LSODA')

    sol_th_o2 = solve_ivp(phase_o2_th,[0,T],init,args=(a,eps),
                          t_eval=t,method='LSODA')

    if a.trunc_order == 2:
        sol_th_o3 = solve_ivp(phase_o3_th,[0,T],init,args=(a,eps),
                              t_eval=t,method='LSODA')

    #init_cgl = [np.cos(0),np.sin(0),np.cos(np.pi/2),np.sin(np.pi/2),
    #            np.cos(np.pi),np.sin(np.pi),np.cos(3*np.pi/2),np.sin(3*np.pi/2)]

    init_cgl = [np.cos(init[0]),np.sin(init[0]),
                np.cos(init[1]),np.sin(init[1]),
                np.cos(init[2]),np.sin(init[2])]
    
    sol_f = solve_ivp(cgl3,[0,T],init_cgl,args=(pdict,eps),t_eval=t,method='LSODA')

    # get angle for full model
    th_f0 = np.arctan2(sol_f.y.T[:,1],sol_f.y.T[:,0])
    th_f1 = np.arctan2(sol_f.y.T[:,3],sol_f.y.T[:,2])
    th_f2 = np.arctan2(sol_f.y.T[:,5],sol_f.y.T[:,4])

    if False:
        np.random.seed(0)
        pts = np.random.rand(20,3)*2*np.pi

        #t2 = np.linspace(0,a.T,100)
        vals = []
        osc_idx = 0
        k_idx = 1
        for pt in pts:
            fn_val = a.h[osc_idx]['lam'][k_idx](*pt)
            vals.append(fn_val)
            
        fig2,axs2 = plt.subplots()
        axs2.plot(vals)
        axs2.set_title('fourier i{}_k{}'.format(osc_idx,k_idx))

        
    fig, axs = plt.subplots(nrows=1,ncols=3)

    p1 = sol_th_o2.y.T[:,1]-sol_th_o2.y.T[:,0]
    p2 = sol_th_o2.y.T[:,2]-sol_th_o2.y.T[:,0]
    axs[0].plot(sol_th_o2.t,np.mod(p1-np.pi,2*np.pi)-np.pi,label=r'$\phi_1$')
    axs[0].plot(sol_th_o2.t,np.mod(p2-np.pi,2*np.pi)-np.pi,label=r'$\phi_2$')

    if a.trunc_order == 2:
        p1 = sol_th_o3.y.T[:,1]-sol_th_o3.y.T[:,0]
        p2 = sol_th_o3.y.T[:,2]-sol_th_o3.y.T[:,0]
        axs[1].plot(sol_th_o3.t,np.mod(p1-np.pi,2*np.pi)-np.pi,label=r'$\phi_1$')
        axs[1].plot(sol_th_o3.t,np.mod(p2-np.pi,2*np.pi)-np.pi,label=r'$\phi_2$')

    pf1 = np.mod(th_f1-th_f0-np.pi,2*np.pi)-np.pi
    pf2 = np.mod(th_f2-th_f0-np.pi,2*np.pi)-np.pi
    axs[2].plot(sol_f.t,np.mod(pf1-np.pi,2*np.pi)-np.pi,label='$\phi_1$')
    axs[2].plot(sol_f.t,np.mod(pf2-np.pi,2*np.pi)-np.pi,label='$\phi_2$')

    axs[0].set_title(r'Order 2 ($\varepsilon^2$)')
    axs[1].set_title(r'Order 3 ($\varepsilon^3$)')
    axs[2].set_title('Full CGL')
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    plt.savefig('figs_temp/out.png')

    plt.show()

    
if __name__ == "__main__":
    __spec__ = None
    main()
