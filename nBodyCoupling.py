"""
StrongCoupling.py computes the higher-order interaction functions from
Park and Wilson 2020 for $N=2$ models and one Floquet multiplier.
In broad strokes, this library computes functions in the following order:

* Use the equation for $\Delta x$ (15) to produce a hierarchy of
ODEs for $g^{(k)}$ and solve. (Wilson 2020)
* Do the same using (30) and (40) to generate a hierarchy of ODEs
for $Z^{(k)}$ and $I^{(k)}$, respectively. (Wilson 2020)
* Solve for $\phi$ in terms of $\\theta_i$, (13), (14) (Park and Wilson 2020)
* Compute the higher-order interaction functions (15) (Park and Wilson 2020)

Assumptions:
-homogeneous vector fields
-NO SELF COUPLING

TODO                 # get list of all cases for 4 vars
OEIS - for counting integers
just enumerate and count up to k=4.

Notes:
- ``pA`` requires endpoint=False. make sure corresponding `dxA`s are used.

"""



import copy
import lib.lib_sym as slib


#import lib.lib as lib
from lib import lib
from lib.fast_interp import interp1d,interp2d
from lib.lambdifyn import lambdifyn as ldn
#from lib.fast_splines import interp2d
#from lam_vec import lam_vec

#import inspect
import time
import os

import math
import sys
#import multiprocessing as multip
import tqdm
from pathos.pools import ProcessPool
from pathos.pools import _ProcessPool
import multiprocessing as mp
from multiprocessing import shared_memory

import scipy.interpolate as si
import numpy as np
from numpy.fft import fft,ifft,fftfreq,fft2

#import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dill
dill.settings['recurse'] = True

from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.matrices.dense import matrix_multiply_elementwise as me
from sympy.tensor.array import derive_by_array

from scipy.signal import fftconvolve
#import pdoc

from itertools import chain
import numexpr as ne

imp_fn = implemented_function

import scipy as sp
from scipy.interpolate import RectBivariateSpline as interp2rbso
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.integrate import solve_ivp, trapezoid

from collections.abc import Iterable

import logging # for debugging
import warnings

logging.getLogger('matplotlib.font_manager').disabled = True

np.set_printoptions(linewidth=np.inf)

import warnings

warnings.filterwarnings('ignore',category=dill.PicklingWarning)
warnings.filterwarnings('ignore',category=dill.UnpicklingWarning)

import pyfftw

fftw = pyfftw.interfaces.numpy_fft.fft
ifftw = pyfftw.interfaces.numpy_fft.ifft
fftw2 = pyfftw.interfaces.numpy_fft.fft2
ifft2 = pyfftw.interfaces.numpy_fft.ifft2
fftwn = pyfftw.interfaces.numpy_fft.fftn
ifftn = pyfftw.interfaces.numpy_fft.ifftn

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    import scipy.fft
    scipy.fft.set_global_backend(cufft)
except ImportError:
    warnings.warn('Warning: cupy not available. This is ok'
                  'if you are not calculating G and H functions')

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


class nBodyCoupling(object):
    
    def __init__(self,rhs,coupling,lc_init,var_names,pardict,**kwargs):

        """
        See the defaults dict below for allowed kwargs.
        
        All model parameters must follow the convention
        'parameter_val'. No other underscores should be used.
        the script splits the parameter name at '_' and uses the
        string to the left as the sympy parmeter name.
        
        Reserved names: ...
            
            rhs: callable.
                right-hand side of a model
            coupling: callable.
                coupling function between oscillators
            lc_init: list or numpy array.
                initial condition of limit cycle (must be found manually).
                XPP is useful, otherwise integrate your RHS for various
                initial conditions for long times and extract an initial
                condition close to the limit cycle.

            lc_long_sim_time: float or int.
                Simulation time to compute the trjactory of an initial 
                condition near the limit cycle solution. Used to
                estimate limit cycle initial condition for use 
                in the Newton method. Default: 500
            lc_eps_time: float.
                Approximation of the time error estimate in Newton's
                method, e.g., (difference between initial conditions)/
                (lc_eps_time). Default: 1e-4
            lc_tol: float:
                Error tolerance to stop Newton's method when computing
                the limit cycle. Default: 1e-13

            var_names: list.
                list of variable names as strings
            pardict: dict.
                dictionary of parameter values. dict['par1_val'] = float.
                Make sure to use par_val format, where each parameter name is
                followed by _val.
                
            g_forward: list or bool.
                If bool, integrate forwards or backwards
                when computing g^k. If list, integrate g^k forwards or
                backwards based on bool value g_forward[k].
                Default: False.
            z_forward: list or bool.
                Same idea as g_forward for PRCS. Default: False.

            i_forward: list or bool.
                Same idea as g_forward for IRCS. Default: False.
            dense: bool.
                If True, solve_ivp uses dense=True and evaluate solution
                along tlc.
            dir: str.
                Location of data directory. Please choose carefully
                because some outputs may be on the order of gigabytes
                if NA >= 5000. Write 'home+data_dir/' to save to the folder
                'data_dir' in the home directory. Otherwise the script
                will use the current working directory by default uless
                an absolute path is used. The trailing '/' is
                required. Default: None.

            trunc_order: int.
                Highest order to truncate the expansion. For example, 
                trunc_order = 3 means the code will compute up to and 
                including order 3. Default: 3.
            NA: int.
                Number of partitions to discretize phase when computing p.
                Default: 500.
            p_iter: int.
                Number of periods to integrate when computing the time 
                interal in p. Default: 10.

            max_iter: int.
                Number of Newton iterations. Default: 20.
            TN: int.
                Total time steps when computing g, z, i.
            rtol, atol: float.
                Relative and absolute tolerance for ODE solvers.
                Defaults: 1e-7, 1e-7.

            rel_tol: float.
                Threshold for use in Newton scheme. Default: 1e-6.
            method: string.
                Specify the method used in scipy.integrate.solve_ivp.
                Default: LSODA.
            g_bad_dx: list or bool. If bool, use another variable to increase
                the magnitude of the Newton derivative. This can only be
                determined after attempting to run simulations and seeing that
                the Jacobian for the Newton step is ill-conditioned. If list,
                check for ill-conditioning for each order k.
                For example, we use g_small_dx = [False,True,False,...,False]
                for the thalamic model. The CGL model only needs
                g_small_idx = False

            z_bad_idx: same idea as g_small_idx for PRCs
            i_bad_idx: same idea as g_small_idx for IRCs

            N: total number of oscillators
            coupling_mat: function, coupling matrix. shape
                must equal N, i.e., coupling_mat().shape[i] == N
                for i = 0, 1.
        
        """
        
        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        defaults = {
            'trunc_order':3,'trunc_deriv':3,'coupling_mat':None,
            'coupling_pars':'','N':2,'TN':20000,'dir':None,
            'lc_long_sim_time':500,'lc_eps_time':1e-4,'lc_tol':1e-13,
            'NG':500,'NA':500,'NP':500,'p_iter':10,'max_iter':100,'rtol':1e-7,
            
            'atol':1e-7,'rel_tol':1e-6,'method':'LSODA','g_forward':True,
            'z_forward':True,'i_forward':True,'g_bad_dx':False,'z_bad_dx':False,
            'i_bad_dx':False,'ignore_var':False,'dense':False,'max_n':-1,
            
            'g_jac_eps':1e-3,'z_jac_eps':1e-3,'i_jac_eps':1e-3,'lc_rate':1,
            
            'load_all':True,
            'load_test':False,'gij_parallel':True,
            
            'recompute_list':[],
            
            'processes':5,'chunksize':500,'chunk_g':500,
            'process_g':5,

            'log_level':'INFO',
            'log_file':'log_default.log',
            'plot':False,
            }

        
        
        self.rhs = rhs
        self.coupling = coupling
        self.lc_init = lc_init
    
        self.rule_par = {}
        
        # if no kwarg for default, use default. otherwise use input kwarg.
        for (prop, default) in defaults.items():
            value = kwargs.get(prop, default)
            setattr(self, prop, value)
        

        if self.log_level == 'DEBUG':
            self.log_level = logging.DEBUG
        elif self.log_level == 'INFO':
            self.log_level = logging.INFO
        elif self.log_level == 'WARNING':
            self.log_level = logging.WARNING
        elif self.log_level == 'ERROR':
            self.log_level = logging.ERROR
        elif self.log_level == 'CRITICAL':
            self.log_level = logging.CRITICAL

        
        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename=self.log_file,level=self.log_level,
                            format=FORMAT)

        assert(self.trunc_order < 10)
        #########
        #  need to implement indexing for oscillator number to allow for
        # general trunc_order >=10
        #########
        
        assert((type(self.g_forward) is bool) or\
               (type(self.g_forward) is list))
        assert((type(self.z_forward) is bool) or\
               (type(self.z_forward) is list))
        assert((type(self.i_forward) is bool) or
               (type(self.i_forward) is list))
        
        assert((type(self.g_jac_eps) is float) or\
               (type(self.g_jac_eps) is list))
        assert((type(self.z_jac_eps) is float) or\
               (type(self.z_jac_eps) is list))
        assert((type(self.i_jac_eps) is float) or\
               (type(self.i_jac_eps) is list))
            
        # update self with model parameters and save to dict
        self.pardict_sym = {}
        self.pardict_val = {}
        for (prop, value) in pardict.items():
            
            # define sympy names, and parameter replacement rule.
            if prop.split('_')[-1] == 'val':
                parname = prop.split('_')[0]
                
                # save parname_val
                setattr(self,prop,value)
                
                # sympy name using parname
                symvar = symbols(parname)
                setattr(self,parname,symvar)
                
                # define replacement rule for parameters
                # i.e. parname (sympy) to parname_val (float/int)
                self.rule_par.update({symvar:value})
                self.pardict_sym.update({parname:symvar})
                self.pardict_val.update({parname:value})

        # symbolic coupling matrix
        self.cmat = self.coupling_mat(self.N,option='sym')
        self.cmat_num = self.coupling_mat(self.N,option='val')

        # replacement rule for coupling coefficient
        for i in range(self.N):
            for j in range(self.N):
                self.rule_par.update({self.cmat[i,j]:self.cmat_num[i,j]})

        # variable names
        self.var_names = var_names
        self.dim = len(self.var_names)
        
        # max iter number
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(self.dim)
        self.psi, self.eps = sym.symbols('psi eps',real=True)
        
        self.kappa = sym.symbols('kappa',negative=True)
        
        # all symbolic variables
        # loc == local, glb == global
        self.syms = {}; self.dsyms = {}
        self.syms['loc'] = sym.zeros(1,self.dim)

        # all oscillator variables and coupling variables
        self.syms['glo'] = sym.zeros(self.N,self.dim)
        self.dsyms['glo'] = sym.zeros(self.N,self.dim)

        for j,name in enumerate(var_names):
            self.syms['loc'][j] = symbols(name)
            
            for i in range(self.N):
                self.syms['glo'][i,j] = symbols(name+str(i))
                self.dsyms['glo'][i,j] = symbols('d'+name+str(i))

        # save var11, var12, ..., var1n,
        #      var21, var22, ..., var2n
        #      ...
        #      varN1, varN2, ..., varNn
        # where n is the single oscillator dimension
            
        self.t = symbols('t',real=True)
        
        # coupling variables
        # ths stands for thetas and pss stands for psis
        ths_str = ' '.join(['th'+str(i) for i in range(self.N)])
        pss_str = ' '.join(['ps'+str(i) for i in range(self.N)])
        ts_str = ' '.join(['t'+str(i) for i in range(self.N)])

        self.ths = symbols(ths_str,real=True)
        self.pss = symbols(pss_str,real=True)
        self.ts = symbols(ts_str,real=True)

        self.psij = []
        
        for i in range(self.N):
            self.psij.append([])
            for j in range(self.N):
                self.psij[i].append(symbols('ps{}{}'.format(i,j)))

        
        # function dicts
        # individual functions
        self.lc = {}
        self.g = {}
        self.z = {}
        self.i = {}
        
        # for coupling
        self.c = {}        
        self.k = {}
        self.pg = {}
        self.p = {}
        self.h = {}
        
        #dims = (self.N,self.N,self.dim)
        #self.c['sym'] = 
        #self.p['dat'] = 
        
        # make dicts for each oscillator
        # terrible naming convention...... will fix later if lucky
        
        for i in range(self.N):
            self.g[i] = {}
            self.z[i] = {}
            self.i[i] = {}
            self.c[i] = {}
            self.k[i] = {}
            self.p[i] = {}
            self.c[i]['vec'] = []

            self.h[i] = {}
            
        self.rule_pg = {}
        dims = (self.N,self.N,self.miter)
        self.p['imp'] = sym.MutableDenseNDimArray(np.zeros(dims))
        
        #from os.path import expanduser
        #home = expanduser("~")
        

        # filenames and directories
        # I do this because my code is in dropbox but the data is elsewhere
        if self.dir is None:
            raise ValueError('Please define a data directory using \
                             the keyword argument \'dir\'.\
                             Write dir=\'home+file\' to save to file in the\
                             home directory. Write dir=\'file\' to save to\
                             file in the current working directory.')
            
        elif self.dir.split('+')[0] == 'home':
            from pathlib import Path
            home = str(Path.home())
            self.dir = home+'/'+self.dir.split('+')[1]
            
        else:
            self.dir = self.dir

        logging.debug('Saving data to '+self.dir)
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
        
        if self.coupling_pars == '':
            logging.info('NOTE: coupling_pars set to default empty string.'
                         'Please specify coupling_pars in kwargs if'
                         'varying parameters.')
        
        lib.generate_fnames(self,coupling_pars=self.coupling_pars)

        
        # make rhs callable
        self.rhs_sym = rhs(0,self.syms['loc'],self.pardict_sym,
                                option='sym')
        
        self.load_limit_cycle()
        
        self.A_array,self.dxA = np.linspace(0,self.T,self.NA,
                                            retstep=True,
                                            endpoint=True)
        
        self.Aarr_noend,self.dxA_noend = np.linspace(0,self.T,self.NA,
                                                     retstep=True,
                                                     endpoint=False)
                                                     
        self.garr,self.dg = np.linspace(0,self.T,self.NG,
                                         retstep=True,
                                         endpoint=True)
                                                     
        self.garr_noend,self.dg_noend = np.linspace(0,self.T,self.NG,
                                                     retstep=True,
                                                     endpoint=False)

        self.pn,self.dpn = np.linspace(0,self.T,self.NP,
                                         retstep=True,
                                         endpoint=False)

        self.T1,self.T2 = np.meshgrid(self.Aarr_noend,self.Aarr_noend,
                                      indexing='ij')
        self.T1R = np.reshape(self.T1,(self.NA**2,))
        self.T2R = np.reshape(self.T2,(self.NA**2,))
        
        if self.load_all:
                
            slib.generate_expansions(self)
            slib.load_coupling_expansions(self,recompute=True)            
            
            slib.load_jac_sym(self)
            
            rule = {**self.rule_lc['loc'],**self.rule_par}
            
            # callable jacobian matrix evaluated along limit cycle
            self.jaclc = lambdify((self.t),self.jac_sym.subs(rule),
                                  modules='numpy')
            # get monodromy matrix
            self.load_monodromy()

            # get heterogeneous terms for g, floquet e. fun.
            self.load_g_sym()

            # get g
            self.load_g()
            
            # get het. terms for z and i
            self.load_het_sym()
            
            # get iPRC, iIRC.
            self.load_z()
            self.load_i()
            
            
            self.load_k_sym()
            
            self.load_p_sym()
            
            #self.load_pg()
            
            self.load_h_sym()
        
        if not(self.load_test):
            #self.load_h()
            pass
        
    def monodromy(self,t,z):
        """
        calculate right-hand side of system
        
        
        $\dot \Phi = J\Phi, \Phi(0)=I$,
        
        where $\Phi$ is a matrix solution
        
        jaclc is the jacobian evaluated along the limit cycle
        """
        
        jac = self.jaclc(t)
        #lc_vec = np.array([self.lc['lam_v'](t),
        #                   self.lc['lam_h'](t),
        #                   self.lc['lam_r'](t),
        #                   self.lc['lam_w'](t)])
        
        #jac = self.numerical_jac(rhs,self.lc_vec(t))
        
        n = int(np.sqrt(len(z)))
        z = np.reshape(z,(n,n))
        
        dy = np.dot(jac,z)
        
        return np.reshape(dy,n*n)

    def numerical_jac(self,fn,x,eps=1e-7):
        """
        return numerical Jacobian function
        """
        n = len(x)
        J = np.zeros((n,n))
        
        PM = np.zeros_like(J)
        PP = np.zeros_like(J)
        
        for k in range(n):
            epsvec = np.zeros(n)
            epsvec[k] = eps
            PP[:,k] = fn(0,x+epsvec)
            PM[:,k] = fn(0,x-epsvec)
            
        J = (PP-PM)/(2*eps)
        
        return J

        
    def load_limit_cycle(self):
        
        self.lc['dat'] = []
        
        for key in self.var_names:
            self.lc['imp_'+key] = []
            self.lc['lam_'+key] = []
            
        
        file_dne = not(os.path.isfile(self.lc['dat_fname']))
        
        if 'lc' in self.recompute_list or file_dne:
            print('* Computing lc data...')
            sol,t_arr = self.generate_limit_cycle()
            
            np.savetxt(self.lc['dat_fname'],sol)
            np.savetxt(self.lc['t_fname'],t_arr)
            
        else:
            sol = np.loadtxt(self.lc['dat_fname'])
            t_arr = np.loadtxt(self.lc['t_fname'])
                
        self.lc['dat'] = sol
        self.lc['t'] = t_arr
        
        # define basic variables
        self.T = self.lc['t'][-1]
        self.tlc,self.dtlc = np.linspace(0,self.T,self.TN,retstep=True)
        self.omega = 2*np.pi/self.T
        
        logging.debug('* lc period = '+str(self.T))
            
        # Make lc data callable from inside sympy
        imp_lc = sym.zeros(self.dim,1)
        for i,key in enumerate(self.var_names):
            fn = interp1d(self.tlc[0],self.tlc[-2],self.dtlc,
                          self.lc['dat'][:-1,i],p=True,k=5)
                          
            self.lc['imp_'+key] = imp_fn(key,fn)
            self.lc['lam_'+key] = fn
            
            imp_lc[i] = self.lc['imp_'+key](self.t)
            
        self.lc_vec = lambdify(self.t,imp_lc,modules='numpy')
        
        if self.plot:
            self.save_temp_figure(sol,0,'lc')
        
        # local and global lc replacement rules
        self.rule_lc = {}
        self.rule_lc['loc'] = {}
        self.rule_lc['glo'] = {}
        
        for j,key in enumerate(self.var_names):
            d = {self.syms['loc'][j]:self.lc['imp_'+key](self.t)}
            self.rule_lc['loc'].update(d)

        for i in range(self.N):
            th = self.ths[i]
            d = {self.syms['glo'][i,j]:self.lc['imp_'+key](th)
                 for j,key in enumerate(self.var_names)}
            self.rule_lc['glo'].update(d)
            
    def generate_limit_cycle(self):
        
        tol = self.lc_tol
        
        #T_init = 5.7
        eps = np.zeros(self.dim) + 1e-4
        epstime = self.lc_eps_time
        dy = np.zeros(self.dim+1) + 10

        # rough init found using XPP
        init = self.lc_init
        T_init = init[-1]
        
        # run for a while to settle close to limit cycle
        sol = solve_ivp(self.rhs,[0,self.lc_long_sim_time],init[:-1],
                        method=self.method,dense_output=True,
                        rtol=1e-13,atol=1e-13,args=(self.pardict_val,))
        
        if self.plot:
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol.t,sol.y.T[:,0])
            plt.savefig('figs_temp/plot_limit_cycle_long.png')
        
        tn = len(sol.y.T)
        
        peak_idxs = sp.signal.find_peaks(sol.y.T[:,0])[0]
        
        maxidx = peak_idxs[-2]
        maxidx_prev = peak_idxs[-3]

        def sol_min(t):
            return -sol.sol(t)[0]

        # get stronger estimates of max values
        pad1lo = (sol.t[maxidx]-sol.t[maxidx-1])/2
        pad1hi = (sol.t[maxidx+1]-sol.t[maxidx])/2
        bounds1 = [sol.t[maxidx]-pad1lo,sol.t[maxidx]+pad1hi]
        res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

        pad2lo = (sol.t[maxidx_prev]-sol.t[maxidx_prev-1])/2
        pad2hi = (sol.t[maxidx_prev+1]-sol.t[maxidx_prev])/2
        bounds2 = [sol.t[maxidx_prev]-pad2lo,sol.t[maxidx_prev]+pad2hi]
        res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

        #maxidx = np.argmax(sol.y.T[int(.2*tn):,0])+int(.2*tn)
        #maxidx_prev = np.argmax(sol.y.T[int(.2*tn):maxidx-1,0])+int(.2*tn)

        # estimate initial time
        #T_init = sol.t[maxidx] - sol.t[maxidx_prev]
        #init = np.append(sol.y.T[maxidx,:],T_init)
        
        T_init = res1.x - res2.x
        init = np.append(sol.sol(res1.x),T_init)
        
        
        counter = 0
        while np.linalg.norm(dy) > tol:
            
            J = np.zeros((self.dim+1,self.dim+1))
            t = np.linspace(0,init[-1],self.TN)
            
            if self.ignore_var:
                dd = self.dim - 1
            else:
                dd = self.dim
            
            for p in range(dd):
                pertp = np.zeros(self.dim)
                pertm = np.zeros(self.dim)
                
                pertp[p] = eps[p]
                pertm[p] = -eps[p]
                
                initp = init[:-1] + pertp
                initm = init[:-1] + pertm
                
                # get error in position estimate
                solp = solve_ivp(self.rhs,[0,t[-1]],initp,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 args=(self.pardict_val,))
                
                solm = solve_ivp(self.rhs,[0,t[-1]],initm,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 args=(self.pardict_val,))
            
                yp = solp.y.T
                ym = solm.y.T

                J[:-1,p] = (yp[-1,:]-ym[-1,:])/(2*eps[p])
                
            J[:-1,:-1] = J[:-1,:-1] - np.eye(self.dim)
            
            tp = np.linspace(0,init[-1]+epstime,self.TN)
            tm = np.linspace(0,init[-1]-epstime,self.TN)
            
            # get error in time estimate
            solp = solve_ivp(self.rhs,[0,tp[-1]],initp,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(self.pardict_val,))
            
            solm = solve_ivp(self.rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(self.pardict_val,))
            
            yp = solp.y.T
            ym = solm.y.T
            
            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)
            
            J[-1,:] = np.append(self.rhs(0,init[:-1],self.pardict_val),0)
            
            sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(self.pardict_val,),
                             t_eval=t)
                             
            
            y_final = sol.y.T[-1,:]

            b = np.append(init[:-1]-y_final,0)
            #dy = np.dot(np.linalg.inv(J),b)
            dy = np.linalg.solve(J,b)
            init += dy
            
            to_disp = (counter,np.linalg.norm(dy))

            str1 = 'iter={}, lc rel. err ={:.2e}     '
            #logging.debug(str1.format(*to_disp))
            print(str1.format(*to_disp))
            counter += 1

        # find index of peak voltage and initialize.
        peak_idx = np.argmax(sol.y.T[:,0])
        init = sol.y.T[peak_idx,:]
        
        # run finalized limit cycle solution
        sol = solve_ivp(self.rhs,[0,sol.t[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,sol.t[-1],self.TN),
                        rtol=1e-13,atol=1e-13,
                        args=(self.pardict_val,))
        
        
        return sol.y.T,sol.t


    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or 
        recompute required, compute here.
        """
        
        file_dne = not(os.path.isfile(self.monodromy_fname))
        if 'm' in self.recompute_list or file_dne:

            initm = copy.deepcopy(self.eye)
            r,c = np.shape(initm)
            init = np.reshape(initm,r*c)
            
            sol = solve_ivp(self.monodromy,[0,self.tlc[-1]],init,
                            t_eval=self.tlc,
                            method=self.method,dense_output=True,
                            rtol=1e-13,atol=1e-13)
            
            self.sol = sol.y.T
            self.M = np.reshape(self.sol[-1,:],(r,c))
            np.savetxt(self.monodromy_fname,self.M)
            
        else:
            self.M = np.loadtxt(self.monodromy_fname)
        
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)

        logging.debug(str(self.eigenvalues)+str(self.eigenvectors))
        
        # get smallest eigenvalue and associated eigenvector
        self.min_lam_idx = np.argsort(self.eigenvalues)[-2]

        #logging.debug()
        logging.debug('min_lam_idx='+str(self.min_lam_idx))
        logging.debug('eigenstuff'+str(self.eigenvalues[self.min_lam_idx]))
        
        self.lam = self.eigenvalues[self.min_lam_idx]  # floquet mult.
        self.kappa_val = np.log(self.lam)/self.T  # floquet exponent
        
        if np.sum(self.eigenvectors[:,self.min_lam_idx]) < 0:
            self.eigenvectors[:,self.min_lam_idx] *= -1
        
        #einv = np.linalg.inv(self.eigenvectors/2)
        einv = np.linalg.inv(self.eigenvectors)
        idx = np.argsort(np.abs(self.eigenvalues-1))[0]
            
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]
        self.z0_init = einv[idx,:]
        self.i0_init = einv[self.min_lam_idx,:]

        print('* Floquet Exponent kappa ='+str(self.kappa_val))
        print('eigenvectors'+str(self.eigenvectors))
        print('g1_init'+str(self.g1_init))
        print('z0_init'+str(self.z0_init))
        print('i0_init'+str(self.i0_init))
        
        
        #logging.debug('eigenvectors'+str(self.eigenvectors))
        
        #logging.debug('g1_init'+str(self.g1_init))
        #logging.debug('z0_init'+str(self.z0_init))
        #logging.debug('i0_init'+str(self.i0_init))
        
        #logging.info('* Floquet Exponent kappa ='+str(self.kappa_val))
        
        
    def load_g_sym(self):
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        # create dict of gv0=0,gh0=0,etc for substitution later.
        self.rule_g0 = {sym.Indexed('g'+name,0):
                        sympify(0) for name in self.var_names}

        for key in self.var_names:
            self.g['sym_'+key] = []
        #self.g_sym = {k: [] for k in self.var_names}
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.g['sym_fnames_'+key]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
        
        
        if 'g_sym' in self.recompute_list or files_do_not_exist:
            print('* Computing g symbolic...')
            
            # create symbolic derivative
            sym_collected = slib.generate_g_sym(self)  
            
            for i in range(self.miter):
                for key in self.var_names:
                    expr = sym_collected[key].coeff(self.psi,i)
        
                    self.g['sym_'+key].append(expr)
                    dill.dump(self.g['sym_'+key][i],
                              open(self.g['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
        else:
            print('* Loading g symbolic...')
            for key in self.var_names:
                self.g['sym_'+key] = lib.load_dill(self.g['sym_fnames_'+key])
        
    
    def load_g(self):
        """
        load all Floquet eigenfunctions g or recompute
        """
        
        self.g['dat'] = []
        
        for key in self.var_names:
            self.g['imp_'+key] = []
            self.g['lam_'+key] = []
        
        print('* Computing g...')
        for i in range(self.miter):

            fname = self.g['dat_fnames'][i]
            
            file_does_not_exist = not(os.path.isfile(fname))
            if 'g' in self.recompute_list or file_does_not_exist:
                
                het_vec = self.interp_lam(i,self.g,fn_type='g')
                
                data = self.generate_g(i,het_vec)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            self.g['dat'].append(data)
            
            self.save_temp_figure(data,i,'g')
            
            for j,key in enumerate(self.var_names):
                #fn = interpb(self.tlc,data[:,j],self.T)
                #fn = interp1d(self.tlc,data[:,j],self.T,kind='cubic')
                
                fn = interp1d(self.tlc[0],self.tlc[-2],self.dtlc,data[:-1,j],p=True,k=5)
                imp = imp_fn('g'+key+'_'+str(i),fn)
                #imp = imp_fn('g'+key+'_'+str(i),self.fmod(fn))
                
                self.g['imp_'+key].append(imp)
                self.g['lam_'+key].append(fn)                
        
        # replacement rules.        
        self.rule_g = {}  # g function
        self.rule_g['loc'] = {}; self.rule_g['glo'] = {}

        # messy but keeps global and local indices clear
        t = self.t
        for key in self.var_names:
            for k in range(self.miter):
                
                fn_loc = sym.Indexed('g'+key,k)
                d_loc = {fn_loc:self.g['imp_'+key][k](t)}
                self.rule_g['loc'].update(d_loc) # local
                
                for i in range(self.N):
                    
                    ti = self.ths[i]
                    fn_glo = sym.Indexed('g'+key+str(i),k)
                    d_glo = {fn_glo:self.g['imp_'+key][k](ti)}
                    self.rule_g['glo'].update(d_glo) # global

    
    def generate_g(self,k,het_vec):
        """
        generate Floquet eigenfunctions g
        
        uses Newtons method
        """
        
        if type(self.g_forward) is bool:
            backwards = not(self.g_forward)
        elif type(self.g_forward) is list:
            backwards = not(self.g_forward[k])
        else:
            raise ValueError('g_forward must be bool or list, not',
                             type(self.g_forward))
            
        
        if type(self.g_jac_eps) is float:
            eps = self.g_jac_eps
        elif type(self.g_jac_eps) is list:
            eps= self.g_jac_eps[k]
        else:
            raise ValueError('g_jac_eps must be float or list or floats, not',
                             type(self.g_jac_eps))
        
        # load kth expansion of g for k >= 0
        if k == 0:
            # g0 is 0. do this to keep indexing simple.
            return np.zeros((self.TN,len(self.var_names)))
        
        if k == 1:
            # pick correct normalization
            init = copy.deepcopy(self.g1_init)
        else:
            init = np.zeros(self.dim)
            init = lib.run_newton2(self,self._dg,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  alpha=1,backwards=backwards,
                                  dense=self.dense)
        
        # get full solution
        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc

        sol = solve_ivp(self._dg,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        t_eval=tlc,
                        method=self.method,
                        dense_output=True,
                        rtol=self.rtol,atol=self.atol)
        
        
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        return gu


    def load_het_sym(self):
        # load het. for z and i if they exist. otherwise generate.
        for key in self.var_names:
            self.z['sym_'+key] = []
            self.i['sym_'+key] = []
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.z['sym_fnames_'+key]))
            val += not(lib.files_exist(self.i['sym_fnames_'+key]))
        
        val += not(lib.files_exist([self.A_fname]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
            
        
        if 'het' in self.recompute_list or files_do_not_exist:
            print('* Computing heterogeneous terms...')
            sym_collected = self.generate_het_sym()
            
            for i in range(self.miter):
                for key in self.var_names:
                    
                    expr = sym_collected[key].coeff(self.psi,i)
                    
                    self.z['sym_'+key].append(expr)
                    self.i['sym_'+key].append(expr)
                    
                    dill.dump(self.z['sym_'+key][i],
                              open(self.z['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    dill.dump(self.i['sym_'+key][i],
                              open(self.i['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            
        else:
            print('* Loading heterogeneous terms...')
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.z['sym_'+key] = lib.load_dill(self.z['sym_fnames_'+key])
                self.i['sym_'+key] = lib.load_dill(self.i['sym_fnames_'+key])
        
    def generate_het_sym(self):
        """
        Generate heterogeneous terms for integrating the Z_i and I_i terms.

        Returns
        -------
        None.

        """
        
        # get the general expression for h in z before plugging in g,z.        
        # column vectors ax ay for use in matrix A = [ax ay]
        self.a = {k: sym.zeros(self.dim,1) for k in self.var_names}
     
        for i in range(1,self.miter):
            logging.debug('z,i het sym deriv order='+str(i))
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.dim))

            for j,key in enumerate(self.var_names):
                logging.debug('\t var='+str(key))
                d1 = lib.vec(lib.df(self.rhs_sym[j],self.x_vec,i+1))
                self.a[key] += (1/math.factorial(i))*(p2*d1)
                
        self.A = sym.zeros(self.dim,self.dim)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]
        
        het = self.A*self.z['vec']
        
        # expand all terms
        out = {}
        
        rule = {**self.rule_g0,**self.rule_d2g}
        
        rule_trunc = {}
        for k in range(self.miter,self.miter+200):
            rule_trunc.update({self.psi**k:0})
            
        for i,key in enumerate(self.var_names):
            logging.info('z,i het sym subs key='+str(key))
            tmp = het[i].subs(rule)
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            tmp = sym.expand(tmp).subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            
            out[key] = tmp
            
        return out
        
    def load_z(self):
        """
        load all PRCs z or recompute
        """
        
        self.z['dat'] = []
        
        for key in self.var_names:
            self.z['imp_'+key] = []
            self.z['lam_'+key] = []
            
        print('* Computing z...')
        for i in range(self.miter):
            
            fname = self.z['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            if 'z' in self.recompute_list or file_dne:
                
                het_vec = self.interp_lam(i,self.z,fn_type='z')
                
                data = self.generate_z(i,het_vec)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            self.z['dat'].append(data)
            
            if self.plot:
                self.save_temp_figure(data,i,'z')
            
            for j,key in enumerate(self.var_names):
                
                fn = interp1d(self.tlc[0],self.tlc[-2],self.dtlc,data[:-1,j],p=True,k=5)
                imp = imp_fn('z'+key+'_'+str(i),fn)
                 
                #fn = interpb(self.tlc,data[:,j],self.T)
                #fn = interp1d(self.tlc,data[:,j],self.T,kind='cubic')
                #imp = imp_fn('z'+key+'_'+str(i),self.fmod(fn))
                self.z['imp_'+key].append(imp)
                self.z['lam_'+key].append(fn)

        
        # coupling
        #thA = self.thA
        #thB = self.thB

        self.rule_z = {}
        self.rule_z['loc'] = {}; self.rule_z['glo'] = {}
        # messy but keeps global and local indices clear
        for key in self.var_names:
            for k in range(self.miter):                
                for i in range(self.N):
                    ti = self.ths[i]
                    fn_glo = sym.Indexed('z'+key+str(i),k)
                    d_glo = {fn_glo:self.z['imp_'+key][k](ti)}
                    self.rule_z['glo'].update(d_glo) # global
        
        #self.rule_z_AB = {}
        #for key in self.var_names:
        #    for i in range(self.miter):
        #        dictA = {Indexed('z'+key+'A',i):self.z['imp_'+key][i](thA)}
        #        dictB = {Indexed('z'+key+'B',i):self.z['imp_'+key][i](thB)}
        #        
        #        self.rule_z_AB.update(dictA)
        #        self.rule_z_AB.update(dictB)

        
    def generate_z(self,k,het_vec):
        if type(self.z_forward) is bool:
            backwards = not(self.z_forward)
        elif type(self.z_forward) is list:
            backwards = not(self.z_forward[k])
        else:
            raise ValueError('z_forward must be bool or list, not',
                             type(self.z_forward))
            
        if type(self.z_jac_eps) is float:
            eps = self.z_jac_eps
        elif type(self.z_jac_eps) is list:
            eps= self.z_jac_eps[k]
        else:
            raise ValueError('z_jac_eps must be bool or list, not',
                             type(self.z_jac_eps))
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            #init = [-1.389, -1.077, 9.645, 0]
        else:
            
            init = np.zeros(self.dim)
            
            init = lib.run_newton2(self,self._dz,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,alpha=1,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  backwards=backwards,dense=self.dense)
        
        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc
            
        sol = solve_ivp(self._dz,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        method=self.method,dense_output=True,
                        t_eval=tlc,
                        rtol=self.rtol,atol=self.atol)
        
        if backwards:
            zu = sol.y.T[::-1,:]
            
        else:
            zu = sol.y.T
        
        if k == 0:
            # normalize
            if len(self.lc_vec(0)[0]) == 1:
                in1 = np.squeeze(self.lc_vec(0))
            else:
                in1 = self.lc_vec(0)[0]
            dlc = self.rhs(0,in1,self.pardict_val)
            zu = zu/(np.dot(dlc,zu[0,:]))*2*np.pi/self.T
            
        
        return zu
    

    def load_i(self):
        """
        load all IRCs i or recomptue
        """
        
        self.i['dat'] = []
        
        for key in self.var_names:
            self.i['imp_'+key] = []
            self.i['lam_'+key] = []
        
        print('* Computing i...')
        for i in range(self.miter):

            fname = self.i['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            
            if 'i' in self.recompute_list or file_dne:

                het_vec = self.interp_lam(i,self.i,fn_type='i')
                data = self.generate_i(i,het_vec)
                np.savetxt(fname,data)
                
            else:
                data = np.loadtxt(fname)

            self.i['dat'].append(data)
            
            if self.plot:
                self.save_temp_figure(data,i,'i')
            
            for j,key in enumerate(self.var_names):
                #fn = interpb(self.tlc,data[:,j],self.T)
                #fn = interp1d(self.tlc,data[:,j],self.T,kind='linear')
                #imp = imp_fn('i'+key+'_'+str(i),self.fmod(fn))
                
                fn = interp1d(self.tlc[0],self.tlc[-2],self.dtlc,data[:-1,j],p=True,k=5)
                imp = imp_fn('i'+key+'_'+str(i),fn)
                
                self.i['imp_'+key].append(imp)
                self.i['lam_'+key].append(fn)
                
                #lam_temp = lambdify(self.t,self.i['imp_'+key][i](self.t))
                
        
        # coupling
        # messy but keeps global and local indices clear

        self.rule_i = {}
        self.rule_i['loc'] = {}; self.rule_i['glo'] = {}
        for key in self.var_names:
            for k in range(self.miter):                
                for i in range(self.N):
                    ti = self.ths[i]
                    fn_glo = sym.Indexed('i'+key+str(i),k)
                    d_glo = {fn_glo:self.i['imp_'+key][k](ti)}
                    self.rule_i['glo'].update(d_glo) # global

        #thA = self.thA
        #thB = self.thB
        
        #self.rule_i_AB = {}
        #for key in self.var_names:
        #    for i in range(self.miter):
        #        dictA = {Indexed('i'+key+'A',i):self.i['imp_'+key][i](thA)}
        #        dictB = {Indexed('i'+key+'B',i):self.i['imp_'+key][i](thB)}
                
        #        self.rule_i_AB.update(dictA)
        #        self.rule_i_AB.update(dictB)
        
    
    def generate_i(self,k,het_vec):
        """
        i0 equation is stable in forwards time
        i1, i2, etc equations are stable in backwards time.

        """
        
        if type(self.i_forward) is bool:
            backwards = not(self.i_forward)
        elif type(self.i_forward) is list:
            backwards = not(self.i_forward[k])
        else:
            raise ValueError('i_forward must be bool or list, not',
                             type(self.i_forward))
        
        if type(self.i_bad_dx) is bool:
            exception = self.i_bad_dx
        elif type(self.i_bad_dx) is list:
            exception = self.i_bad_dx[k]
        else:
            raise ValueError('i_bad_dx must be bool or list, not',
                             type(self.i_bad_dx))
            
        if type(self.i_jac_eps) is float:
            eps = self.i_jac_eps
        elif type(self.i_jac_eps) is list:
            eps= self.i_jac_eps[k]
        else:
            raise ValueError('i_jac_eps must be bool or list, not',
                             type(self.i_jac_eps))
        
        if k == 0:
            init = copy.deepcopy(self.i0_init)
        else:
            
            init = np.zeros(self.dim)
            init = lib.run_newton3(self,self._di,init,k,het_vec,
                                   max_iter=self.max_iter,rel_tol=self.rel_tol,
                                   eps=eps,alpha=1,
                                   backwards=backwards,
                                   exception=exception,
                                   dense=self.dense)

        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc
        
        sol = solve_ivp(self._di,[0,tlc[-1]],init,
                        args=(k,het_vec),
                        t_eval=tlc,
                        method=self.method,dense_output=True,
                        rtol=self.rtol,atol=self.atol)
    
        if backwards:
            iu = sol.y.T[::-1,:]
            
        else:
            iu = sol.y.T
                
        if k == 0:
            # normalize. classic weak coupling theory normalization
            c = np.dot(self.g1_init,iu[0,:])
            iu /= c
            
            logging.debug('norm const i0'+str(c))
    
        if k == 1:  # normalize
        
            # kill off nonzero v
            #if np.sum(self.g['dat'][1][:,-1]) < 1e-20:
            #    iu[:,-1] = 0
            
            # see Wilson 2020 PRE for normalization formula.
            lc0 = []
            g10 = []
            z00 = []
            i00 = []
        
            for varname in self.var_names:
                key = 'lam_'+varname
                lc0.append(self.lc[key](0))
                g10.append(self.g[key][1](0))
                z00.append(self.z[key][0](0))
                i00.append(self.i[key][0](0))
                
            F = self.rhs(0,lc0,self.pardict_val)
            g1 = np.array(g10)
            z0 = np.array(z00)
            i0 = np.array(i00)
            
            J = self.jaclc(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa_val - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0

            sol = solve_ivp(self._di,[0,tlc[-1]],init,
                            args=(k,het_vec),
                            t_eval=tlc,
                            method=self.method,dense_output=True)
            
            iu = sol.y.T[::-1]
            logging.debug('norm const i1='+str(be))
        return iu

    def load_k_sym(self):
        
        """
        k[i]['type_'+key][j][k]
        oscillator i, function type, oscillator j, order k, variable name 'key'.
        
        c contains the derivatives of the coupling fn given
        oscillators i,j and variable name 'key'.
        c[i][key][j]
        same as k just without terms collected.

        """
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}

        for key in self.var_names:
            for i in range(self.N):
                self.k[i]['sym_'+key] = []
                self.k[i]['imp_'+key] = []
            
        # check that files exist
        val = 0
        for key in self.var_names:
            for i in range(self.N):
                for j in range(self.N):
            
                    val += not(lib.files_exist(self.k[i]['sym_fnames_'+key][j]))
                    val += not(os.path.isfile(self.c[i]['sym_fname'][j]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False

        rule_trunc = {}
        for k in range(self.miter,self.miter+500):
            rule_trunc.update({self.eps**k:0})
            
        
        if 'k_sym' in self.recompute_list or files_do_not_exist:
            print('* Computing K symbolic...')
            
            self.generate_k_sym()
            
            for i in range(self.N):
                for j in range(self.N):
                    
                    self.c[i]['vec'].append(sym.zeros(self.dim,1))
                    
                    print(i,j,self.c[i]['vec'])
                    # save c in vector format
                    for key_idx,key in enumerate(self.var_names):
                        self.c[i]['vec'][j][key_idx] = self.c[i]['sym_'+key][j]

                    dill.dump(self.c[i]['vec'][j],
                              open(self.c[i]['sym_fname'][j],'wb'),
                              recurse=True) # dump

            # now collect each order k given i,j,key.
            for key in self.var_names:
                logging.info('Collecting for variable'+str(key))
                
                for i in range(self.N):
                    
                    self.k[i]['sym_'+key] = []
                    self.c[i][key+'_collected'] = []
                    
                    for j in range(self.N):
                        tmp = expand(self.c[i]['sym_'+key][j],
                                     **self._expand_kws)
                        tmp = tmp.subs(rule_trunc)
                
                        self.c[i][key+'_collected'].append(tmp)
                        self.k[i]['sym_'+key].append([])
                        
                        for k in range(self.miter):

                            eps_k_term = self.c[i][key+'_collected'][j].coeff(self.eps,k)
                            self.k[i]['sym_'+key][j].append(eps_k_term)
                        
                            dill.dump(self.k[i]['sym_'+key][j][k],
                                      open(self.k[i]['sym_fnames_'+key][j][k],'wb'),
                                      recurse=True)
                    
                
        else:
            print('* Loading K symbolic...')
            for key in self.var_names:
                for i in range(self.N):
                    self.k[i]['sym_'+key] = []
                    
                    for j in range(self.N):
                    
                        self.c[i]['vec'].append(sym.zeros(self.dim,1))
                        
                        dat = lib.load_dill(self.k[i]['sym_fnames_'+key][j])
                        self.k[i]['sym_'+key].append(dat)
                        
                        self.c[i]['vec'][j] = lib.load_dill([self.c[i]['sym_fname'][j]])

            for i in range(self.N):
                fnames = [self.c[i]['sym_fname'][j] for j in range(self.N)]
        
    
    def generate_k_sym(self):
        """
        generate terms involving the coupling term (see K in paper).
        """

       
        psym = self.pardict_sym # shorten for readability
        a = self.cmat
        fn = self.coupling # coupling function
        
        # shape (N,N,dim)
        # perform sum over j at end
        #c_all = sym.zeros(self.N,self.N,self.dim)
        
        dims = (self.N,self.N,self.dim)
        c_all = sym.MutableDenseNDimArray(np.zeros(dims))
        self.c['sym'] = sym.MutableDenseNDimArray(np.zeros(dims))

        print('self c before',self.c)
         
        # get full coupling term for each oscillator i
        for i in range(self.N):
            for j in range(self.N):
                
                # oscillator i coords; oscillator j coords.
                z1 = self.syms['glo'][i,:];z2 = self.syms['glo'][j,:]
                z = z1.T.col_join(z2.T)
                
                # get symbolic coupling
                out = sym.flatten(a[i,j]*fn(z,psym,option='sym'))
                c_all[i,j,:] = out

        print('')
        print('self c 1',self.c)
        # 0 and 1st derivative
        c_temp = {}
        for i in range(self.N):
            c_temp[i] = {} # messy but it'll do...
            
            for j in range(self.N):
                c_temp[i][j] = {}
                                    
                # i,j oscillator coordinates
                z1=self.syms['glo'][i,:];z2=self.syms['glo'][j,:]
                z = z1.T.col_join(z2.T)
                
                dz1=self.dsyms['glo'][i,:];dz2=self.dsyms['glo'][j,:]
                dz = dz1.T.col_join(dz2.T)

                for key_idx,key in enumerate(self.var_names):
                    c_temp[i][j][key] = c_all[i,j,key_idx]
                    
                    d = lib.df(c_all[i,j,key_idx],z.T,1).dot(dz.T)
                    c_temp[i][j][key] += d
        print('')
        print('self c 2',self.c)
        
        # 2nd + derivative
        for i in range(self.N):
            for j in range(self.N):
                
                # i,j oscillator coordinates
                z1=self.syms['glo'][i,:];z2=self.syms['glo'][j,:]
                z = z1.T.col_join(z2.T)
                
                dz1=self.dsyms['glo'][i,:];dz2=self.dsyms['glo'][j,:]
                dz = dz1.T.col_join(dz2.T)

                for key_idx,key in enumerate(self.var_names):
                    for k in range(2,self.trunc_deriv+1):
                        
                        kp = lib.kProd(k,dz)
                        da = lib.vec(lib.df(c_all[i,j,key_idx],z,k))
                        c_temp[i][j][key] += (1/math.factorial(k))*kp.dot(da)

        print('')
        print('self c 3',self.c)
        # save to c
        for key in self.var_names:
            
            for i in range(self.N):
                self.c[i]['sym_'+key] = []
                for j in range(self.N):
                    self.c[i]['sym_'+key].append(c_temp[i][j][key])
                        
        # create rule to replace dx with g expansion in eps.
        rule = {}

        for i in range(self.N):
            for key_idx,key in enumerate(self.var_names):
                rule.update({self.dsyms['glo'][i,key_idx]:self.g[i][key+'_eps']})

        for key in self.var_names:
            for i in range(self.N):
                for j in range(self.N):
                    self.c[i]['sym_'+key][j] = self.c[i]['sym_'+key][j].subs(rule)
        print('self c',self.c)

        return self.c
    
    def load_p_sym(self):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.        
        p['sym'][i,j,k] is the forcing function for oscillator i of order k
        and specifically the term involving i and j.
        
        so the full forcing function for a given i and k would be the sum
        over j. but we need to keep these terms distinct.
        
        After using the integrating factor method, each $p_i^{(k)}$ takes
        the form
        
        $p_i^{(k)}(t) = \sum_{j=1}^N g_{ij}^{(k}(\theta_i(t),\theta_j(t))$.
        
        so a substitution of the above needs to be done for $k>1$.
        
        """
        

        val = 0
        for i in range(self.N):
            for j in range(self.N):
                val += not(lib.files_exist(list(self.p['sym_fnames'][i,j,:])))

        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False

        dims = (self.N,self.N,self.miter)
        self.p['sym'] = sym.MutableDenseNDimArray(np.zeros(dims))

        eps = self.eps
        if 'p_sym' in self.recompute_list or files_do_not_exist:
            print('* Computing p symbolic...')
            
            rule_trunc = {}
            for k in range(self.miter,self.miter+500):
                rule_trunc.update({self.eps**k:0})

            for i in range(self.N):
                logging.info('Oscillator '+str(i))
                for j in range(self.N):
                    #print(i,j,self.i[i]['vec'],self.c[i]['vec'][j])
                    #print('')
                    
                    v1 = self.i[i]['vec']; v2 = self.c[i]['vec'][j]
                    
                    tmp = eps*v1.dot(v2)

                    tmp = expand(tmp,**self._expand_kws)
                    
                    tmp = tmp.subs(rule_trunc)
                    tmp = collect(tmp,self.eps)
                    tmp = collect(expand(tmp),self.eps)

                    for k in range(self.miter):
                        #print(i,j,k,tmp.coeff(self.eps,k))
                        
                        self.p['sym'][i,j,k] += tmp.coeff(self.eps,k)

            # after saving each order k above to list, dill.
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.miter):
                        fname = self.p['sym_fnames'][i,j,k]
                        #print('fname',fname)
                        dill.dump(self.p['sym'][i,j,k],
                                  open(fname,'wb'),recurse=True)

        else:
            print('* Loading p symbolic...')
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.miter):
                        fname = self.p['sym_fnames'][i,j,k]

                        dat = lib.load_dill([fname])[0]
                        self.p['sym'][i,j,k] = dat
                
        #print('p sym',self.p['sym'][:,:,2])
        
        #self.rule_p.update({sym.Indexed('p'+str(i),k):expr})
        
        #self._evaluate_p_expansions()
        
    def _evaluate_p_expansions(self):
        """
        This function takes the above forcing function values, which
        contain p_i^{(k)} for k>1. These p_i^{(k)} for k>1 must be 
        substituted into the forcing functions before computing
        the higher-order terms.
        
        sym.Indexed('p',i,k) for p
        sym.Indexed('gg',i,j,k) for individual terms.
        """
        
        self.rule_p2g = {}
        # create substitution for p to pg
        for k in range(self.miter):
            for i in range(self.N):
                gsyms = 0
                
                
                for j in range(self.N):
                    
                    if i != j:
                        gsyms += sym.Indexed('gg',i,j,k)

                self.rule_p2g.update({sym.Indexed('p',i,k):gsyms})
                
        #print('rule p2g',self.rule_p2g)
        
        # substitute into each p sym.
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.miter):
                    self.p['sym'][i,j,k] = self.p['sym'][i,j,k].subs(self.rule_p2g)
        
        
    def load_pg(self):
        #for i in range(self.N):
        #        for j in range(self.N):
        #            for k in range(self.miter):
        
        
        
        
        
        #self.load_pg1(0,0,1)
        
        for k in range(self.miter):
            for i in range(self.N):
                for j in range(self.N):
                    if i!=j:
                        self.load_pg1(i,j,k)
                    
                    
        print('rule_pg',self.rule_pg)
        
        print('')
    
        #sys.exit()
    
    def load_pg1(self,i,j,k):
    
        if k == 0:
            #p0 is 0 (no forcing function)
            return np.zeros((self.NP,self.NP))
    
        fname = self.pg['dat_fnames'][i,j,k]
        file_dne = not(os.path.isfile(fname))
        
        if 'pg' in self.recompute_list or file_dne:
            data = self.generate_pg1(i,j,k)
            
            np.savetxt(fname,data)
            
        else:
            data = np.loadtxt(fname)
        
        #pg_interp = interp2d(self.pn,self.pn,data,k=1
        pg_interp = interp2d([0,0],[self.pn[-2],self.pn[-2]],
                             [self.dpn,self.dpn],
                             data,k=5,p=[True,True])
        
        ta = self.ths[i];tb = self.ths[j]
        name = 'gg_{}{}{}'.format(i,j,k)
        pg_imp = imp_fn(name,self.fLam2(pg_interp,i,j))
        
        print('pg_imp',pg_imp)
        
        lamtemp = lambdify([ta,tb],pg_imp(ta,tb))
        
        
        # imp, lam, subs rule here.
        self.rule_pg.update({sym.Indexed('gg',i,j,k):pg_imp(ta,tb)})
        
        xtmp = np.linspace(0,1,3)
        ytmp = np.linspace(1,2,3)
        print('gp_interp',pg_interp(xtmp,ytmp),'lamtemp',lamtemp(xtmp,ytmp))
            
        
            
    def generate_pg1(self,i,j,k):
    
        pg_data = np.zeros((self.NP,self.NP))
    
        if i==j:
            return pg_data
    
    
        rule = {**self.rule_lc['glo'],**self.rule_i['glo'],
                **self.rule_g['glo'],
                **self.rule_par,**self.rule_pg}
                
        in2 = [self.ths[i],self.ths[j]]
        
        self.p['imp'][i,j,k] = self.p['sym'][i,j,k].subs(rule)
        
        pg1_imp = self.p['imp'][i,j,k]
        pg1_lam = lambdify(in2,pg1_imp)
        
        print(i,j,k,pg1_imp)
        
        s1 = np.arange(0,self.T*self.p_iter,self.dpn)
        exponential = np.exp(s1*self.kappa_val)
        fb = fftw(exponential)
        
        a_i = np.arange(self.NP,dtype=int)
        
        #print('pg1_lam',i,j,k,pg1_lam(s1,s1-s1[0]))
        
        for i in range(self.NP):
            f1 = pg1_lam(s1,s1-s1[i])
            
            #f1_gpu = cp.array(f1)
            #fa_gpu = scipy.fft.fft(f1_gpu)
            #c_gpu = fa_gpu*fb_gpu
            #conv = np.array(conv_gpu.get())
            #conv_gpu = scipy.fft.ifft(c_gpu)*dg_noend
            
            fa = fftw(f1)
            conv = ifftw(fa*fb)*self.dpn
            
            pg_data[a_i,a_i-i] = conv[len(s1)-self.NP:len(s1)].real
        
        return pg_data
        
    def load_p(self,i,k):
        rule = self.rule_par
        
        fname = self.p['dat_fnames'][i,k]
        file_dne = not(os.path.isfile(fname))

        if 'p' in self.recompute_list or file_dne:
            print('* Computing p...')
            
            p_data = self.generate_p(i,k)
            
        else:
            print('* Loading p...')
            p_data = np.loadtxt(fname)
        
        
        p_interp0 = interp2d(self.pn,self.pn,p_data,k=1)
        
        ta = self.ths[i];tb = self.ths[j]
        name = 'p_{}{}{}'.format(i,j,k)
        
        p_imp = imp_fn(name,self.fLam_ij(p_interp0,i,j))
        lamtemp = lambdify(self.ths,p_imp(ta,tb))
        
        self.p['dat'][i,k] = p_data
        
    def generate_p(self,i,k):
        if k == 0:
            NP = self.NP
            #p0 is 0 (no forcing function)
            return np.zeros((NP,NP))
            
        

    def load_gij(self):
        """
        load or (re)calculate gij after implementing in p_sym
        """
        
        fname = '{}gij_trunc={}_NG={}_N={}.d'.format(self.dir,self.trunc_order,self.NG,self.N)

        #recompute = False
        file_does_not_exist = not(os.path.isfile(fname))

        self.p['imp'] = sym.zeros(self.N,self.miter)
        self.p['imp_check'] = sym.zeros(self.N,self.miter)
        
        self.generate_gij()
        
    def generate_gij(self):
        """
        calculate convolution terms g_{ij}
        for oscillator i and order k.
        """

        # save convolved terms here
        self.gij = {}
        
        T = self.T
        p_iter = self.p_iter
        
        exp = np.exp
        kappa = self.kappa_val
        
        
        s1 = np.arange(-T*self.p_iter,T*self.p_iter,self.dg_noend)
        
        s1_sh = shared_memory.SharedMemory(create=True, size=s1.nbytes)
        s1_b = np.ndarray(s1.shape, dtype=s1.dtype, buffer=s1_sh.buf)
        s1_b[:] = s1[:]  # Copy the original data into shared memory

        #exponential = np.heaviside(s1,0)*exp(s1*self.kappa)
        exponential = np.heaviside(s1,0)*ne.evaluate('exp(s1*kappa)')

        fb = fftw(exponential)

        # generators for polynomial        
        gens = [sym.Indexed('p'+str(i),k) for i in range(self.N) for k in range(self.miter)]
        
        self.rule_p = {}
        self.p['imp'] = copy.deepcopy(self.p['sym'])
        self.p['imp_check'] = copy.deepcopy(self.p['sym'])
        
        
        for k in range(self.miter):
            #self.rule_p.append({})
            for i in range(self.N):
                
                expr = 0
                
                poly = sym.poly(self.p['sym'][i,k],gens=gens)
                
                generators = poly.gens
                monomials = poly.monoms()
                all_coeffs = poly.coeffs()
                
                # calculate conv for each coeff
                for l in range(len(all_coeffs)):
                    c_sum = list(sym.Add.make_args(sym.expand(all_coeffs[l])))
                    
                    # split multiplied terms. (1/3)*d*x1*iy0[1]
                    for ll in range(len(c_sum)):
                        mul_terms = sym.Mul.make_args(c_sum[ll])
                        
                        subs1 = (i+1,self.N,k+1,self.miter,l+1,
                                 len(all_coeffs),ll+1,len(c_sum))
                        logging.info('i={}/{},k={}/{},'
                                     'l={}/{},ll={}/{}'.format(*subs1))
                        
                        if len(mul_terms)>1:
                            # compute g_{ij} given alpha,beta,gamma,delta
                            
                            i_idx,j_idx,fni,fnj = self._gij_sort(mul_terms)
                            
                            mul_sub = fni*fnj
                            suffix = '_'.join(str(mul_sub).split('*'))
                            suffix = suffix.replace('[','')
                            suffix = suffix.replace(']','')
                            fname = r'{}gij_{}{}_{}_NG={}.txt'
                            fname = fname.format(self.dir,i,k,suffix,self.NG)
                            
                            file_does_not_exist = not(os.path.isfile(fname))
                            
                            if 'g' in self.recompute_list or file_does_not_exist:
                                out = self._gij_conv(i_idx,j_idx,fni,fnj,fb,s1_b)
                                i_idx,j_idx,gij_data,fni,fnj = out
                                np.savetxt(fname,gij_data)
                            else:
                                gij_data = np.loadtxt(fname)
                                i_idx,j_idx,fni,fnj = self._gij_sort(mul_terms)
                            
                            t1 = self.ths[i_idx]; t2 = self.ths[j_idx]
                            
                            if i_idx == j_idx: # make trivial conv
                                #name = 'gconva{}{}_{}_{}'.format(i,k,l,ll)
                                
                                name = 'gca{}{}_{}_{}_{}'.format(i,k,l,ll,suffix)
                                
                                #g_ij_interp = interpb(self.garr_noend,gij_data,self.T)
                                g_ij_interp = interp1d(self.garr_noend[0],self.garr_noend[-2],
                                                      self.dg_noend,gij_data[:-1],k=3,p=True)
                                
                                g_ij_imp = imp_fn(name,g_ij_interp)
                                rule = {mul_sub:g_ij_imp(t1)}
                                
                                self.gij[name] = g_ij_imp(t1)
                                
                            else: # turn into 2d function
                                
                                g_ij_interp = interp2d(self.garr_noend,
                                                       self.garr_noend,
                                                       gij_data,k=1)
                                
                                
                                #g_ij_interp = interp2d([0,0],[self.garr_noend[-2],self.garr_noend[-2]],
                                #                       [self.dg_noend,self.dg_noend],
                                #                       gij_data,k=3,p=[True,True])
                                
                                # substitute back into expression.
                                #name = 'gconvb{}{}_{}_{}'.format(i,k,l,ll)
                                name = 'gcb{}{}_{}'.format(i,k,suffix)
                                
                                g_ij_imp = imp_fn(name,self.fLam_ij(g_ij_interp,i_idx,j_idx))
                                rule = {mul_sub:g_ij_imp(t1,t2)}
                                self.gij[name] = g_ij_imp(t1,t2)
                                
                                
                                # brute check
                                lamtemp = lambdify(self.ths,g_ij_imp(t1,t2))
                                #print('gij check?',name,lamtemp(0,1,1))
                                #fig,axs = plt.subplots(1,1)
                                #p = np.zeros(500)
                                #t2 = np.linspace(0,5*self.T,len(p))
                                #axs.plot(lamtemp(p,t2,t2))
                                #plt.show()
                                
                            # sub into specific term
                            c_sum[ll] = c_sum[ll].subs(rule)
                    
                    if sum(list(monomials[l])) == 0:
                            expr += sym.Add(*c_sum)
                    else:
                        for i1 in range(len(monomials[l])):
                            if monomials[l][i1] != 0:
                                expr += sym.Add(*c_sum)*generators[i1]**monomials[l][i1]

                self.p['imp'][i,k] = expr
                self.p['imp_check'][i,k] = expr
                self.rule_p.update({sym.Indexed('p'+str(i),k):expr})
                
        s1_sh.close()
        s1_sh.unlink()

        # construct p
        for k in range(1,self.miter):
            keys = list(self.rule_p.keys())
            for ll in range(len(keys)):
                self.rule_p[keys[ll]] = self.rule_p[keys[ll]].subs(self.rule_p)
        
        
        for i in range(self.N):
            tempexpr = 0
            for k in range(self.miter):
                self.p['imp'][i,k] = self.p['imp'][i,k].subs(self.rule_p)
                self.p['imp'][i,k] = self.p['imp'][i,k].subs(self.rule_par)
                tempexpr += self.p['imp'][i,k]
            
            lamtemp_g = lambdify(self.ths,tempexpr)
            in1 = np.zeros(self.N)
            in1[i] = 1

        return self.p['imp'],self.rule_p

    def _gij_sort(self,mul_terms):
        """
        sort gij variables and functions
        """
        
        fns_to_use1 = [];fns_to_use2 = []
        first_idx = -10
        idx1 = -10;idx2 = -10
        for term in mul_terms:
        
            if type(term) == sym.tensor.indexed.Indexed or\
               term in self.syms['glo']:
                
                if type(term) == sym.tensor.indexed.Indexed:
                    fn_str = str(term.base)
                    varnum = int(fn_str[-1])
                    if first_idx == -10:
                        first_idx = varnum
                        idx1 = varnum
                        fns_to_use1.append(term)
                    else:
                        idx2 = varnum
                        fns_to_use2.append(term)

                elif term in self.syms['glo']:
                    varnum = int(str(term)[-1])
                    if first_idx == -10:
                        first_idx = varnum
                        idx1 = varnum
                        fns_to_use1.append(term)
                    else:
                        idx2 = varnum
                        fns_to_use2.append(term)

        assert(idx1>=0);assert(idx2>=0)

        fn1 = sym.Mul(*fns_to_use1)
        fn2 = sym.Mul(*fns_to_use2)
        
        if idx1 == idx2:
            pass
        if idx1 < idx2: # order by index for consistency
            pass
        else:
            temp = idx2;idx2 = idx1;idx1 = temp
            tempfn = copy.deepcopy(fn2)
            fn2 = copy.deepcopy(fn1)
            fn1 = tempfn
    
        return idx1,idx2,fn1,fn2

    #@profile
    def _gij_conv(self,idx1,idx2,fn1,fn2,fb,s1):
        """
        calculate the term g(thi,thj)
        convolution of exp(k*t) and fn(thi-t)*fn(thj-t)
        so the output 
        """

        fni = fn1
        fnj = fn2
        #idx1,idx2,fnj,fni = self._gij_sort(mul_terms)
        
        # get index i,j, order k.
        #fnj_idx = idx1#self.get_index(fnj)
        #fni_idx = idx2#self.get_index(fni)
        
        i_idx = idx1#fni_idx[0]
        j_idx = idx2#fnj_idx[0]
        
        
        fni = fni.subs(self.rule_lc['glo'])
        fni = fni.subs(self.rule_g['glo'])
        fni = fni.subs(self.rule_i['glo'])
        
        fnj = fnj.subs(self.rule_lc['glo'])
        fnj = fnj.subs(self.rule_g['glo'])
        fnj = fnj.subs(self.rule_i['glo'])
        
        # given index i, j, order k, compute conv.
        # needed to avoid duplicate arguments in python
        arr = self.garr_noend
        NG = self.NG
        dg_noend = self.dg_noend
        #dxA_noend = self.dg_noend#self.dxA_noend
        
        
        if i_idx == j_idx:
            gij_data = np.zeros(NG)
            inputs = [self.ths[i_idx]]
            gij = lambdify(inputs,fnj*fni)

            f1 = np.heaviside(s1,0)*gij(s1)
            
            fa = fftw(f1)

            conv = ifftw(fa*fb)*dg_noend
            gij_data[:] = conv[len(s1)-NG:len(s1)]
            
            
        else:
            gij_data = np.zeros((NG,NG))
            inputs = [self.ths[i_idx],self.ths[j_idx]]
            
            gij = lambdify(inputs,fnj*fni)#,modules='numpy')            
            a_i = np.arange(NG,dtype=int)

            def get_conv(ll):
                f1 = gij(s1,s1-arr[ll])
                
                #f1_gpu = cp.array(f1)
                #fa_gpu = scipy.fft.fft(f1_gpu)
                #c_gpu = fa_gpu*fb_gpu
                #conv_gpu = scipy.fft.ifft(c_gpu)*dg_noend
                #conv = np.array(conv_gpu.get())
                
                fa = fftw(f1)
                conv = ifftw(fa*fb)*dg_noend
                
                return ll,conv
            
            if self.gij_parallel:
                process_g = self.process_g
                chunk_g = self.chunk_g
                pool = _ProcessPool(processes=process_g,maxtasksperchild=1)

                for x in tqdm.tqdm(pool.imap(get_conv,a_i,
                                      chunksize=chunk_g),
                               total=len(a_i)):
                               
                    ll, conv = x
                    gij_data[a_i,a_i-ll] = conv[len(s1)-NG:len(s1)].real
                    
                pool.close()
                pool.join()
                pool.terminate()
            else:
            
                for ll in range(NG):
                    ll, conv = get_conv(ll)                    
                    gij_data[a_i,a_i-ll] = conv[len(s1)-NG:len(s1)].real
            
        return (i_idx,j_idx,gij_data,fni,fnj)

    def get_index(self,fn):
        """
        fn: sympy variable either x0 or p0[0]

        returns (0,) or (0,0), respectively.
        """
        if type(fn) == sym.core.symbol.Symbol:
            idx = int(str(fn)[-1])
            return (idx,)

        elif type(fn) == sym.tensor.indexed.Indexed:
            fn_split = str(fn).split('[')
            return (int(fn_split[0][-1]),int(fn_split[-1][0]))

    def load_p_old(self,i,j,k):
    
        rule = self.rule_par
        
        fname = self.p['dat_fnames'][i,j,k]
        file_dne = not(os.path.isfile(fname))

        if 'p' in self.recompute_list or file_dne:
            print('* Computing p...')
            
            p_data = self.generate_p(i,j,k)

            #p_interp0 = interp2d([0,0],[2*np.pi,2*np.pi],
            #                 [self.dan,self.dan],
            #                 p_data,k=3,p=[True,True])
            
            

            # fix indexing (see check_isostable.ipynb)
            #X,Y = np.meshgrid(self.an,self.an,indexing='ij')
            #p_data = p_interp0(X+Y*self.om,Y)
                
            np.savetxt(fname,p_data)

        else:
            print('* Loading p...')
            p_data = np.loadtxt(fname)
            
        
        p_interp0 = interp2d(self.pn,self.pn,p_data,k=1)
        
        ta = self.ths[i];tb = self.ths[j]
        name = 'p_{}{}{}'.format(i,j,k)
        
        p_imp = imp_fn(name,self.fLam_ij(p_interp0,i,j))
        lamtemp = lambdify(self.ths,p_imp(ta,tb))
        
        self.p['dat'][i,j,k] = p_data

        if k == 0:
            imp = imp_fn('p_'+self.model_name+'_0', lambda x: 0*x)
            system1.p['imp'][k] = imp
            system1.p['lam'][k] = 0
        
        if system1.save_fig:
            
            fig,axs = plt.subplots()
            axs.imshow(p_data)
            plt.savefig('figs_temp/p_'+system1.model_name+str(k)+'.png')
            plt.close()

        # put these implemented functions into the expansion
        self.rule_p.update({sym.Indexed('p_{}{}'.format(i,k)):
                               self.p[i]['imp'][k](ta,tb)})

    def generate_p_old(self,i,k):

        NP = self.NP
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((NP,NP))

        kappa1 = self.kappa_val

        rule = {**self.rule_p,**self.rule_p,**self.rule_par,
                **self.rule_par,**self.rule_lc,**self.rule_lc,
                **self.rule_i,**self.rule_i,**self.rule_g,
                **self.rule_g}
        
        ph_imp1 = self.p[i]['sym'][k].subs(rule)
        
        print(ph_imp1)
        if k >= 1 and i == 2:
            sys.exit()

        if ph_imp1 == 0:
            return np.zeros((NP,NP))
        lam1 = lambdify(self.ths,ph_imp1)
        het1 = np.zeros((NP,int(self._m[1]*NP)))

        # get lam1 for testing
        if True:
            X,Y = np.meshgrid(self.an,self.an,indexing='ij')

            fig,axs = plt.subplots()
            im = axs.imshow(lam1(X,Y))
            plt.colorbar(im,ax=axs)
            plt.savefig('figs_temp/lam_heta_'+str(k)+'.png')
            plt.close()
                   
        an=self.an;dan=self.dan;pfactor=self.pfactor
        NP=self.NP;s=np.arange(0,self.T*pfactor*self._m[1],dan)
            
        fac = self.om*(1-system1.idx) + system1.idx
        exp1 = exp(fac*s*system1.kappa_val)

        g_in = np.fft.fft(exp1)
        a_i = np.arange(NP,dtype=int)
        
        for ll in range(len(an)):

            f_in = np.fft.fft(lam1(an[ll]+self.om*s,+s))
            conv = np.fft.ifft(f_in*g_in)            
            het1[(a_i+ll)%NP,a_i] = conv[-NP:].real

        return fac*het1*dan

    def load_p_old(self):
        """
        insert lower order p[i] into higher order terms
        """
        
        # load all p or recompute or compute new.

        rule = self.rule_par
        for i in range(self.N):
            self.p[i]['dat_check'] = []
            self.p[i]['imp_check'] = []
            self.p[i]['lam_check'] = []

        for k in range(self.miter):            
            for i in range(self.N):
                
                if k > 0:
                    rule.update({sym.Indexed('p'+str(i),k):
                                 self.p['imp_check'][i,k].subs(rule)})

        self.rule_p_check = rule

    
    def load_h_sym(self):
        """
        also compute h lam
        """

        val = 0
        for i in range(self.N):
            val += not(lib.files_exist(self.h[i]['sym_fnames']))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False

        # symbolic h terms
        dims = (self.N,self.miter)
        self.h['sym'] = sym.MutableDenseNDimArray(np.zeros(dims))

        a = self.cmat # coupling matrix
        
        if 'h_sym' in self.recompute_list or files_do_not_exist:
            print('* Computing H symbolic...')
            
            # simplify expansion for speed
            rule_trunc = {}
            for k in range(self.miter,self.miter+200):
                rule_trunc.update({self.eps**k:0})

            for i in range(self.N):
                logging.info('Oscillator '+str(i))
                for j in range(self.N):

                    v1 = self.z[i]['vec']; v2 = self.c[i]['vec'][j]
                    
                    tmp = v1.dot(v2)
            
                    tmp = sym.expand(tmp,basic=True,deep=True,
                                     power_base=False,power_exp=False,
                                     mul=True,log=False,
                                     multinomial=True)
            
                    tmp = tmp.subs(rule_trunc)
                    tmp = collect(tmp,self.eps)
                    tmp = collect(expand(tmp),self.eps)

                    for k in range(self.miter):
                        self.h['sym'][i,k] += tmp.coeff(self.eps,k)
                

            for i in range(self.N):
                for k in range(self.miter):
                    fname = self.h[i]['sym_fnames'][k]
                    dill.dump(self.h['sym'][i,k],
                              open(fname,'wb'),recurse=True)
                
        else:
            print('* Loading H symbolic...')
                              
            for i in range(self.N):
                    
                fname = self.h[i]['sym_fnames']
                dat = lib.load_dill(fname)
                self.h['sym'][i,:] = dat
        
            
    def load_h(self):
        
        if self.max_n > -1:
            self.basis = {}
            
            # precalculate mask
            self.r1 = [ll for ll in range(0,self.max_n+1)]
            self.r2 = [ll for ll in range(-self.max_n,0)]
            
            self.r = self.r1+self.r2;r = self.r

        idx = np.arange(0,self.NA,1,dtype=int)
        
        idxarr1,idxarr2 = np.meshgrid(idx,idx,indexing='ij')
        
        self.fqs = fftfreq(len(self.Aarr_noend),d=self.dxA_noend/self.T)
        
        for i in range(self.N):
            self.h[i]['dat'] = []
            self.h[i]['interp'] = []
            self.h[i]['lam'] = []
            self.h[i]['lam_all'] = []
        
        for k in range(self.miter):

            t = self.Aarr_noend
            
            for i in range(self.N):
                
                if self.max_n > -1:
                    fname = self.h[i]['lam_fnames_fourier'][k]
                    
                else:
                    fname = self.h[i]['lam_fnames'][k][:-2]
                    fname += '_NA='+str(self.NA)+'.txt'
                
                file_does_not_exist = not(os.path.isfile(fname))
                
                if 'h' in self.recompute_list or file_does_not_exist:

                    print('* Computing H i={}, order={}...'.format(i,k))
                    
                    if self.max_n > -1:
                        
                        lam,lam_all = self.generate_h_fourier(i,k)
                        
                        fn1 = self.h[i]['lam_fnames_fourier'][k]
                        fn2 = self.h[i]['lam_fnames_fourier'][k]+'all'
                        dill.dump(lam,open(fn1,'wb'),recurse=True)
                        dill.dump(lam_all,open(fn2,'wb'),recurse=True)
                        
                    else:
                        data = self.generate_h_brute(i,k,idxarr1,idxarr2)
                        np.savetxt(fname,data)

                else:
                    print('* Loading H i={}, order={}...'.format(i,k))
                    if self.max_n > -1:
                        fn1 = self.h[i]['lam_fnames_fourier'][k]
                        lam = lib.load_dill([fn1])[0]
                    else:
                    
                        data = np.loadtxt(fname)           

                #lam_avg = interp2d([0,0],[t[-2],t[-2]],
                #                   [self.dxA_noend,self.dxA_noend],
                #                   data,k=3,p=[True,True])
                lam_avg = interp2d(t,t,data,k=1)

                name = 'h{}{}'.format(i,k)
                if self.N == 2:
                    himp = imp_fn(name,self.fLam_ij(lam_avg,0,1))
                    lam = lambdify([self.ths[0],self.ths[1]],
                                    himp(self.ths[0],self.ths[1]))
                else:
                    himp = imp_fn(name,self.fLam_ij(lam_avg,1,2))
                    lam = lambdify([self.ths[1],self.ths[2]],
                                    himp(self.ths[1],self.ths[2]))
                self.h[i]['lam'].append(lam)        
                
    def generate_h_brute(self,i,k,idxarr1,idxarr2):
        """
        use this if nyquist frequency is too high 
        brute force + parallelization across CPU cores.
        """
        rule = {**self.rule_p,**self.rule_par,
                **self.rule_i['glo'],**self.rule_g['glo'],
                **self.rule_z['glo'],**self.rule_lc['glo']}
        
        del rule[0]
        assert(self.N == 3)
        
        t = self.Aarr_noend
        te = self.A_array
        dte = self.dxA
        dt = self.dxA_noend
        #r = self.r
        NA = self.NA
        N = self.N
        T = self.T

        expr = self.h['sym'][i,k].subs(rule)
        print('expr',expr)
        
        global ths,lam,t1,dt1,data,idxarr11d,idxarr21d,a_i
        ths = self.ths
        lam = lambdify(ths,expr)
        t1,dt1 = np.linspace(0,T,1000,endpoint=False,retstep=True)
        
        start = time.time()
        data = np.zeros((N-1)*(NA,)).reshape((NA**(N-1),))
        
        idxarr11d = idxarr1.reshape((NA**(N-1),))
        idxarr21d = idxarr2.reshape((NA**(N-1),))
        
        shm1 = shared_memory.SharedMemory(create=True, size=idxarr11d.nbytes)
        shm2 = shared_memory.SharedMemory(create=True, size=idxarr21d.nbytes)
        t1_sh = shared_memory.SharedMemory(create=True, size=t1.nbytes)
        t_sh = shared_memory.SharedMemory(create=True, size=t.nbytes)
        
        b1 = np.ndarray(idxarr11d.shape, dtype=idxarr11d.dtype, buffer=shm1.buf)
        b2 = np.ndarray(idxarr21d.shape, dtype=idxarr21d.dtype, buffer=shm2.buf)
        t1b = np.ndarray(t1.shape, dtype=t1.dtype, buffer=t1_sh.buf)
        tb = np.ndarray(t.shape, dtype=t.dtype, buffer=t_sh.buf)
        
        b1[:] = idxarr11d[:]  # Copy the original data into shared memory
        b2[:] = idxarr21d[:]
        t1b[:] = t1[:]
        tb[:] = t[:]
        
        a_i = np.arange(NA**(N-1),dtype=int)
        #trap = trapezoid
        
        global var_dict
        var_dict = {}
        
        #@staticmethod
        def get_avg(ll):
        
            a1 = lam(0+t1b,tb[b1[ll]]+t1b,tb[b2[ll]]+t1b)
            
            a1_gpu = cp.sum(cp.array(a1))
            del a1
            
            a1_val = a1_gpu.get()
            del a1_gpu
            
            return ll,a1_val
            #return ll,np.sum(a1)
            #return ll,0
        
        pool = _ProcessPool(processes=self.processes,maxtasksperchild=1)

        for x in tqdm.tqdm(pool.imap(get_avg,a_i,
                        chunksize=200),
                        total=len(a_i)):
                       
            ll, avg = x
            data[ll] = avg
        
        data = data.reshape((N-1)*(NA,))*dt1/T
        
        pool.close()
        pool.join()
        pool.terminate()
        
        shm1.close()
        shm2.close()
        t1_sh.close()
        t_sh.close()
        
        shm1.unlink()
        shm2.unlink()
        t1_sh.unlink()
        t_sh.unlink()
        
        return data 

    def generate_h_fourier(self,i,k):
                        
        rule = {**self.rule_p,**self.rule_par,
                **self.rule_i['glo'],**self.rule_g['glo'],
                **self.rule_z['glo'],**self.rule_lc['glo']}
        
        del rule[0]
        
        if k == 1:

            pass
        # coefficients of basis functions.
        # will use same indexing as r
        basis = np.zeros(self.N*(2*self.max_n+1,),dtype=complex)

        self.imp_list = {}

        t = self.Aarr_noend
        te = self.A_array
        dte = self.dxA
        dt = self.dxA_noend
        r = self.r

        expr = self.h['sym'][i,k].subs(rule)
        
        #expr = expr.xreplace(self.rule_p) # NEED THIS
        lamtemp = lambdify(self.ths,expr)
        
        terms = list(sym.Add.make_args(expr))
        for ll,term in enumerate(terms): # force all terms to expand
            #terms[ll] = terms[ll].subs(rule) # avoid weird subs issues
            terms[ll] = terms[ll].expand()
        expr = sym.Add(*terms)

        terms = list(sym.Add.make_args(expr)) # get terms in sum
        
        mask_f1 = _fft_mask(len(t),self.max_n,dim=1)
        
        if k == 2:
            ruletemp = {**self.rule_p}
            htemp_list = list(sym.Add.make_args(self.h['sym'][i,k]))
        
        # need this for computing 3 unique vars with overlap.
        arr_idx1 = np.outer(r,np.ones(len(r)))
        arr_idx2 = np.outer(r,np.ones(len(r))).T

        arr_idxs = arr_idx1+arr_idx2
        
        #lamtemp = lambdify(self.ths,sym.Add(*terms))

        # for each term compute the truncated Fourier series
        for q,term in enumerate(terms):
            term = term.subs(rule)
            p = np.round(100*(q+1)/(len(terms)),0);
            logging.info('{}% {}/{}'.format(p,q+1,len(terms)))

            powdict = term.as_powers_dict()
            fns = [key**powdict[key] for key in powdict.keys()
                   if not(key.is_number)]

            numbers = [key**powdict[key] for key in powdict.keys()
                       if key.is_number]
            term_coeff = float(sym.Mul(*numbers))

            var_list, fn_list = self._sort_fn_type_v2(fns)
            var_list,fn_list = unique_vf(var_list,fn_list)
            
            nvars = varcount_min(var_list)
            nvars2 = varcount(var_list)

            separable = self._is_sep(fns)
            
            if nvars == 1:
                
                lam = lambdify(var_list[0],fn_list[0])
                c1 = fft(lam(t))[self.r]/len(t)
                basis[self.N*(0,)] += term_coeff*c1[0]
            
            elif nvars == 2 and separable:
                
                v1a = _get_var_idx(fn_list[0])
                v1b = _get_var_idx(fn_list[1])

                lam1a = lambdify(var_list[0],fn_list[0])
                lam1b = lambdify(var_list[1],fn_list[1])
                
                c1a_gpu = scipy.fft.fft(cp.array(lam1a(t)))
                c1b_gpu = scipy.fft.fft(cp.array(lam1b(t)))
                
                c1a = np.array(c1a_gpu.get())[self.r]/len(t)
                c1b = np.array(c1b_gpu.get())[self.r]/len(t)
                
                d1a = augment_axes(c1a,[1],repeat=True)
                d1b = augment_axes(c1b,[1],repeat=True)
                
                if v1a[0] < v1b[0]:
                    d1b = d1b.T
                else:
                    d1a = d1a.T
                
                tot = d1a*d1b
                
                slice_dims = list(set(v1a+v1b))
                slicing = get_slice_dim(self.N,dims=slice_dims)
                basis[slicing] += term_coeff*tot
                
                
            elif nvars == 2 and not(separable):
                
                v2s = []
                for qq in range(len(var_list[0])):
                    v2s.append(_get_idx_from_var(var_list[0][qq]))
                    
                idxs = np.argsort(v2s)

                var_list_new = []
                for qq in range(len(idxs)):
                    var_list_new.append(var_list[0][idxs[qq]])
                                        
                var_list = [var_list_new]
                
                lam = lambdify(var_list[0],fn_list[0])
                data = self._evaluate_2d(lam)
                
                data_gpu = cp.asarray(data)
                coeffs = scipy.fft.fft2(data_gpu)/len(t)**2
                coeffs = np.array(coeffs.get())
                coeffs = coeffs[np.ix_(self.r,self.r)]

                tot = coeffs

                slicing = get_slice_dim(self.N,dims=list(v2s))
                basis[slicing] += term_coeff*tot
            
            elif nvars == 3:

                addsum = 0
                for var,fn in zip(var_list,fn_list):
                    if type(var) is list: # nsep
                        addsum += 1
                if addsum == 3:
                    print('got cycle, var,fn')
                    for var,fn in zip(var_list,fn_list):
                        print(var,fn)

                elif addsum == 2:
                    #print('got 2 edges')
                    pass
                        
                
                for var,fn in zip(var_list,fn_list):
                    
                    if type(var) is list: # nsep
                        
                        v2s = []
                        for qq in range(len(var)):
                            v2s.append(_get_idx_from_var(var[qq]))

                        idxs = np.argsort(v2s)
                        
                        var_new = []
                        for qq in range(len(idxs)):
                            var_new.append(var[idxs[qq]])
                            
                        var = var_new

                        lam = lambdify(var,fn)
                        #lam = sym.cos(3*np.array(var))
                        data2 = self._evaluate_2d(lam)
                        data2_gpu = cp.array(data2)
                        coeffs2_gpu = scipy.fft.fft2(data2_gpu)/len(t)**2
                        coeffs2_gpu = coeffs2_gpu[cp.ix_(self.r,self.r)]
                        #coeffs2 = fftw2(data)/len(t)**2
                        coeffs2 = np.array(coeffs2_gpu.get())
                        #coeffs2 = coeffs2[np.ix_(self.r,self.r)]
                        #coeffs2 = np.array(coeffs2_gpu.get())

                    else: #sep
                        lam = lambdify(var,fn)
                        data1 = lam(t)
                        data1_gpu = cp.array(data1)
                        coeffs1_gpu = scipy.fft.fft(data1_gpu)[self.r]/len(t)
                        coeffs1 = np.array(coeffs1_gpu.get())
                        #coeffs1 = fftw(lam(t))[self.r]/len(t)
                        v1s = _get_var_idx(fn)
                        
                max_idx = max(v1s+v2s)
                b1 = np.ones(max_idx+1,dtype=bool)
                b2 = np.ones(max_idx+1,dtype=bool)
                idxs = np.arange(max_idx+1,dtype=int)

                b1[v1s] = False
                b2[v2s] = False
                dims1 = list(idxs[b1])
                dims2 = list(idxs[b2])
                
                d1 = augment_axes(coeffs1,dims1,repeat=True)
                d2 = augment_axes(coeffs2,dims2,repeat=True)
                
                tot = d1*d2
                
                slicing = get_slice_dim(self.N,dims=list(set(dims1+dims2)))
                basis[slicing] += term_coeff*tot


            elif nvars == 4:
            
                # not uniqe vars... just keep this for now.
                
                v2a = []
                for qq in range(len(var_list[0])):
                    v2a.append(_get_idx_from_var(var_list[0][qq]))
                idxs = np.argsort(v2a)
                
                var_list_new = []
                for qq in range(len(idxs)):
                    var_list_new.append(var_list[0][idxs[qq]])
                var_list_new = [var_list_new]
                
                v2b = []
                for qq in range(len(var_list[1])):
                    v2b.append(_get_idx_from_var(var_list[1][qq]))

                idxs = np.argsort(v2b)

                var_list_new2 = []
                for qq in range(len(idxs)):
                    var_list_new2.append(var_list[1][idxs[qq]])
                var_list_new.append(var_list_new2)

                var_list = var_list_new
                
                lam2a = lambdify(var_list[0],fn_list[0])
                data1 = self._evaluate_2d(lam2a)
                coeffs2a = fft2(data1)[np.ix_(r,r)]/len(t)**2
                data1_gpu = cp.array(data1)
                coeffs2a_gpu = scipy.fft.fft2(data1_gpu)[cp.ix_(r,r)]/len(t)**2
                coeffs2a = np.array(coeffs2a_gpu.get())
                
                lam2b = lambdify(var_list[1],fn_list[1])
                data2 = self._evaluate_2d(lam2b)
                coeffs2b = fft2(data2)[np.ix_(r,r)]/len(t)**2
                data2_gpu = cp.array(data2)
                coeffs2b_gpu = scipy.fft.fft2(data2_gpu)[cp.ix_(r,r)]/len(t)**2
                coeffs2b = np.array(coeffs2b_gpu.get())


                max_idx = max(v2a+v2b)
                b1 = np.ones(max_idx+1,dtype=bool)
                b2 = np.ones(max_idx+1,dtype=bool)
                idxs = np.arange(max_idx+1,dtype=int)
                
                b1[v2a] = False
                b2[v2b] = False
                dims1 = list(idxs[b1])
                dims2 = list(idxs[b2])
                
                d1 = augment_axes(coeffs2a,dims1)
                d2 = augment_axes(coeffs2b,dims2)
                assert(len(d1.shape) == 3)
                assert(d1.shape[0] == d1.shape[1])
                assert(d1.shape[1] == d1.shape[2])
                
                # make sure coefficients are multiplied correctly
                common_idx = set(v2a).intersection(set(v2b))
                non_common_idxs = list((set(v2a).union(set(v2b))))
                non_common_idxs = non_common_idxs.difference(common_idx)
                
                non_common_idxs.sort()
                assert(len(non_common_idxs) == 2)
                assert(len(common_idx) == 1)
                
                common_idx = list(common_idx)[0]
                
                # maybe make this more readable.
                # see fourier_broadcasting notes
                slicelist = [np.nan,np.nan,np.nan]
                slicelist[common_idx] = slice(None)

                tot = np.zeros(3*(len(r),),dtype=complex)
                for i1 in range(d1.shape[0]):
                    for i2 in range(d2.shape[1]):
                        slicelist[non_common_idxs[0]] = i1
                        slicelist[non_common_idxs[1]] = i2
                        
                        
                        c0 = np.outer(d1[tuple(slicelist)],
                                      d2[tuple(slicelist)])
                        
                        e = np.zeros(len(r),dtype=complex)
                        for i3 in range(len(r)):
                            e[i3] = np.sum(c0[np.where(arr_idxs==r[i3])])
                        tot[tuple(slicelist)] = e
                slicing = get_slice_dim(self.N,dims=list(set(v2a+v2b)))
                basis[slicing] += term_coeff*tot
            
            else:
                
                logging.deubg('caught weird case?')
                logging.deubg('fn_list,var_list',fn_list,var_list)
                logging.debug('nvars',nvars)
                logging.debug('separable',separable)
                logging.debug('term',term)
                
                sys.exit()

        # now construct functions

        # mask. remove nonzero index sum
        tt = ldn(basis,self.fqs[self.r])
        tot = 0
        for qq in range(len(tt.fq_mats)):
            tot += tt.fq_mats[qq]

        basis_copy = copy.deepcopy(basis)
        lam_all = ldn(basis_copy,self.fqs[self.r])
        
        mask_basis = np.where(tot!=0)
        basis[mask_basis] = 0

        lam = ldn(basis,self.fqs[self.r])
        
        in1 = np.zeros(3)
        in1[i] = 25
        tot = np.sum(lamtemp(in1[0]+t,in1[1]+t,in1[2]+t))*dt/self.T

        return lam,lam_all



    def _get_trunc_fn1(self,var,imp,mask,fft_tol=1e-6):
        """
        truncated 1d fft in complex exponential form

        """
        t = self.Aarr_noend;dt = self.dxA_noend
        fq1s = sym.Matrix(fftfreq(len(t),d=dt/(2*np.pi))[mask])
        # in test code seems we need to divide dt by 2pi
        # independent of period
        
        lam = lambdify(var,imp)

        f1 = fft(lam(t))[mask]
        f1 /= len(t)
        f1[np.abs(f1)<fft_tol] = 0
        fft_coef = sym.Matrix(f1)

        v1 = fft_coef
        v2 = (sym.I*fq1s*var).applyfunc(sym.exp)

        reduced_fn = me(v1,v2)

        return reduced_fn

    
    def _get_trunc_fn2(self,var,imp):
        """
        truncated 1d fft in complex exponential form

        var must have 2 elements
        """
        
        out = self._generate_trunc_fn2(var,imp)

        return out
    
    def _generate_trunc_fn2(self,var,imp,fft_tol=1e-6):
        t = self.Aarr_noend;dt = self.dxA_noend

        lam = lambdify(var,imp)
        data = self._evaluate_2d(lam)
        coefs = fft2(data)/len(t)**2
        
        coefs[np.abs(coefs)<fft_tol] = 0
        #diag = np.arange(len(coefs))
        coefs = sym.Matrix(coefs[np.ix_(self.r,self.r)])
        
        #coefs = mat

        # NOTE: 2d fourier column index corresponds to x axis and row index to y axis.        
        exponential = self.exponential.xreplace({self.ths[0]:var[0],self.ths[1]:var[1]})
        
        return sym.matrices.dense.matrix_multiply_elementwise(coefs,exponential)


    def _expand_poly(self,coefs):
        """
        get polynomial expansion given tuple of coefficients
        """

        tot = 1
        for coef in coefs:
            
            tot *= coef
        
        tot = tot.expand(power_base=False,power_exp=False,log=False,
                         deep=False,mul=True)

        return tot

    def _generate_coefs1(self,name):
        i,j = symbols('i j');cc = sym.IndexedBase(name)
        p1 = sym.Sum(cc[j],(j,-self.max_n,self.max_n)).doit()
        return p1,cc

    def _generate_coefs2(self,name):
        i,j = symbols('i j');dd = sym.IndexedBase(name)
        p2 = sym.Sum(sym.Sum(dd[i,j],(i,-self.max_n,self.max_n)),(j,-self.max_n,self.max_n)).doit()
        return p2,dd
                        
    def _eliminate_coef(self,expr):
        """
        decide whether to keep or eliminate Fourier coeffs

        return polynomial of k+2 variables
        with particular coeffs removed. 
        """

        terms = sym.Add.make_args(expr)
        
        rule_x = {}
        
        for term in terms:
            coeffs = sym.Mul.make_args(term)
            idx_sum = 0
            
            for L in range(len(coeffs)):
                if str(coeffs[L])[0] == 'd':
                    idx_sum += coeffs[L].indices[-1]
                    idx_sum += coeffs[L].indices[-2]
                    
                else:
                    idx_sum += coeffs[L].indices[-1]
                    
            if idx_sum != 0:
                rule_x.update({term:0})

        # subs is slow so use xreplace.
        simplified = expr.xreplace(rule_x) 

        return simplified
        
    
    def _evaluate_2d(self,fn):
        """
        evaluate 2d function because this is dumb

        t is the time/phase array. used to construct 2d mesh
        """

        #return fn(self.T1R,self.T2R).reshape((self.NA,self.NA))
        return fn(self.T1,self.T2)
    
    def _sort_fn_type(self,fns):
        """
        fns = list(term.atoms(sym.Function))
        
        given fns get array of vars and fns.
        for k=0: [var1,var2],[fn1,fn2]
        for k=1: [var1,var2,var3],[fn1,fn2,fn3]

        for k=0 var1,var2 can be any of thi.
        for k>=1, vari can either be 1 number of list.

        """
        var_list = []
        fn_list = []
        
        for fn in fns: # get all variables
            var_str = _get_var(fn,unique=True)
            
            if type(var_str) is list:
                idx1 = int(var_str[0][-1]);idx2 = int(var_str[1][-1])
                var_list.append([self.ths[idx1],self.ths[idx2]])
            else:
                idx = int(var_str[-1])
                var_list.append(self.ths[idx])
            
            fn_list.append(fn)

        return var_list,fn_list
    
    def _sort_fn_type_v2(self,fns):
        """
        fns = list(term.atoms(sym.Function))
        
        given fns get array of vars and fns.
        for k=0: [var1,var2],[fn1,fn2]
        for k=1: [var1,var2,var3],[fn1,fn2,fn3]

        for k=0 var1,var2 can be any of thi.
        for k>=1, vari can either be 1 number of list.

        """
        
        var_list = []
        fn_list = []
                
        for fn in fns: # get all variables
            totvars = _get_var_v2(fn,unique=True)
            
            if len(totvars) == 2:
                v1 = totvars[0]; v2 = totvars[1]
                i1 = int(str(v1)[-1]);i2 = int(str(v2)[-1])
                var_list.append([self.ths[i1],self.ths[i2]])
            else:
                v1 = totvars[0]
                i1 = int(str(v1)[-1])
                var_list.append(self.ths[i1])
            
            fn_list.append(fn)

        return var_list,fn_list

        
        for fn in fns: # get all variables
            totvars = _get_var_v2(fn,unique=True)

            var_list.append(totvars)

        return var_list,fn_list


    def _xcor_1var(self,fns,vars_list):
        """
        get (trivial) cross correlation of multiple functions
        where each function depends on 1 variable
        output should be scalar
        """

        
        t = self.Aarr_noend
        dt = self.dxA_noend
        tot = 1
        for fn,v in zip(fns,vars_list):
            lam = lambdify(v,fn)

            tot *= np.sum(lam(t))*dt
        return tot

    def _is_sep(self,fns):
        """
        check if list of functions or function contains a 
        non-separable function

        if a non-separable function depends on the same variables
        then count it should count as separable.

        fns: list of functions from sympy Atom or a sympy function
        returns: bool
        """

        sep = True

        if not(type(fns) is list):
            fns = [fns]

        for fn in fns:
            fn_str = str(fn)

            if ',' in fn_str:
                # check if vars unique
                varout = _get_var(fn,unique=True)
                if type(varout) is list:
                    sep = False

        return sep
        
            
    def _dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """

        jac = self.jaclc(t)*(order > 0)
        hom = np.dot(jac-order*self.kappa_val*self.eye,z.T)
        out = hom + het_vec(t).T
    
        return out
    
    def _dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jaclc(t).T+order*self.kappa_val*self.eye,z.T)
        out = -hom - het_vec(t).T
        
        return out
    
    def _di(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jaclc(t).T+(order-1)*self.kappa_val*self.eye,z)
        out = -hom - het_vec(t).T
        
        return out
    
    def interp_lam(self,k,fn_dict,fn_type='z'):
        """
        it is too slow to call individual interpolated functions
        in the symbolic heterogeneous terms.
        since the heterogeneous terms only depend on t, just make
        and interpolated version and use that instead so only 1 function
        is called for the het. terms per iteration in numerical iteration.
        """
        
        # lambdify heterogeneous terms for use in integration
        # create lambdified heterogeneous term and interpolate
        # load kth expansion of g for k >= 1
        
        # z and i use the same heterogeneous terms
        # so they are indistinguishable in this function.
        if fn_type == 'z' or fn_type == 'i':
            fn_type = 'z'
        
        rule = {}

        for key in self.var_names:
            tmp = {sym.Indexed(fn_type+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            
            rule.update(tmp)
        
        rule = {**rule,**self.rule_lc['loc'],**self.rule_par}
        if fn_type == 'z':
            rule.update({**self.rule_g['loc']})
        
        het_imp = sym.zeros(self.dim,1)
        for i,key in enumerate(self.var_names):
            sym_fn = fn_dict['sym_'+key][k].subs(rule)
            
            lam = lambdify(self.t,sym_fn,modules='numpy')

            # evaluate
            if fn_type == 'g' and (k == 0 or k == 1):
                y = np.zeros(self.TN)
            elif fn_type == 'z' and k == 0:
                y = np.zeros(self.TN)
            elif fn_type == 'i' and k == 0:
                y = np.zeros(self.TN)
            elif sym_fn == 0:
                y= np.zeros(self.TN)
            else:
                y = lam(self.tlc)
                        
            # save as implemented fn
            fn = interp1d(self.tlc[0],self.tlc[-2],self.dtlc,y[:-1],p=True,k=5)
            imp = imp_fn(key,fn)
            
            #interp = interpb(self.lc['t'],y,self.T)
            #imp = imp_fn(key,self.fmod(interp))
            het_imp[i] = imp(self.t)
            
        het_vec = lambdify(self.t,het_imp,modules='numpy')
            
        return het_vec
    
    def fmod(self,fn):
        """
        fn has mod built-in
        
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for 
        all lambda functions.
        """
        return lambda x=self.t: fn(x)

    def fLam_ij(self,fn,i,j):
        """
        interp2db object
        """
        return lambda ti=self.ths[i],tj=self.ths[j]: fn(np.mod(ti,self.T),
                                                        np.mod(tj,self.T))



    def fLam2(self,fn,i,j):
        """
        interp2d object from fast_interp
        """
        return lambda ti=self.ths[i],tj=self.ths[j]: fn(ti,tj)
        
        
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """
        data should be (TN,dim)
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1)
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tlc,data[:,j],label=key)
            ax.legend()
            
        logging.info(fn+str(k)+' ini'+str(data[0,:]))
        logging.info(fn+str(k)+' fin'+str(data[-1,:]))
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'.png')
        plt.close()
        

def _get_var(fn,unique=False):
    """
    get independent variable from function of n variables
    return as listof strings
    """

    fn_str = str(fn).replace(' ','')

    
    if ',' in fn_str:
        a = fn_str.split(',')
        b = [a[0][-3:],a[1][:3]]

    else:
        a = fn_str.split('(')
        b = a[-1][:-1]

    if unique:
        if b is list:
            if b[0] == b[1]:
                b = b[0]
            
    return b


def _get_var_v2(fn,unique=False):
    """
    get independent variable from function of n variables
    return as tuple
    """
    
    totvars = ()
    noexp = list(fn.atoms(sym.Function))

    for j in range(len(noexp)):
        totvars += noexp[j].args

    if unique:
        totvars = tuple(set(totvars))
        
    return totvars

def _get_var_idx(fn,ret_sep=False):
    """
    get independent variable numbers from function
    return as list of ints
    """

    fn_str = str(fn).replace(' ','')
    unique_vars = _get_var(fn,unique=True)
    

    if type(unique_vars) is not list:
        unique_vars = [unique_vars]

    var_list = []
    for var in unique_vars:
        var_list.append(int(var[-1]))

    return var_list

def _get_var_idx_v2(fn,ret_sep=False):
    """
    get independent variable numbers from function
    return as list of ints
    """

    fn_str = str(fn).replace(' ','')
    unique_vars = list(_get_var_v2(fn,unique=True))

    var_list = []
    for var in unique_vars:
        var_list.append(int(str(var)[-1]))

    return var_list

def _get_idx_from_var(var):
    return int(str(var)[-1])


def _fft_mask(n,max_n,dim):
    if dim == 1:
        mask = np.zeros(n, dtype=bool)
        mask[:max_n+1] = 1
        mask[-max_n:] = 1
    elif dim == 2:
        mask = np.zeros((n,n), dtype=bool)
        mask[:max_n+1,:max_n+1] = 1
        mask[-max_n:,-max_n:] = 1

    else:
        raise ValueError('dim should be 1 or 2')

    return mask

def collect_nsep(var_list,fn_list):
    """
    extract the sep and nsep
    """
    sep_vars=[];sep_fns=[]
    nsep_vars=[];nsep_fns=[]

    for i,var in enumerate(var_list):
        if type(var) is list: # make sure nsep as 2 uniqu evars
            #nsep_vars.append(var)
            #nsep_fns.append(fn_list[i])

            if len(set(var)) == 1:
                var_list[i] = var[0]
                sep_vars.append(var[0])
                sep_fns.append(fn_list[i])
            else:
                nsep_vars.append(var)
                nsep_fns.append(fn_list[i])

        else:
            sep_vars.append(var)
            sep_fns.append(fn_list[i])

    return sep_vars,sep_fns,nsep_vars,nsep_fns

def unique_v_f1(vars_list,fns_list):

    vars_list_new = []
    fns_list_new = []
    nsep_vars = []
    nsep_fns = []

    for i in range(len(vars_list)):
        if vars_list[i] in vars_list_new:
            idx = vars_list_new.index(vars_list[i])
            fns_list_new[idx] *= fns_list[i]
        else:
            vars_list_new.append(vars_list[i])
            fns_list_new.append(fns_list[i])
        #if type(vars_list) is list:
        #    nsep_vars.append(vars_list[i])
        #    nsep_fns.append(fns_list[i])
            
        
    return vars_list_new,fns_list_new

def unique_vf(var_list,fn_list):

    
    sep_vars,sep_fns,nsep_vars,nsep_fns = collect_nsep(var_list,fn_list)
    # collect sep vars one more time
    
    if len(nsep_vars) > 0:
        for i in range(len(sep_vars)):
            
            match_not_found = True
            j = 0
            while match_not_found:
                if sep_vars[i] in nsep_vars[j]:
                    fn_list[var_list.index(nsep_vars[j])] *= sep_fns[i]
                    fn_list.remove(sep_fns[i])
                    var_list.remove(sep_vars[i])
                    match_not_found = False

                j += 1
                if j == len(nsep_vars):
                    match_not_found = False

        # once sep vars have been incorporated into nsep vars combine nsep vars.
        for i in range(len(sep_vars)):
            
            if (sep_vars[i] in sep_vars[i+1:]) and (sep_vars[i] in var_list):
                m = len(var_list)
                idx = m-var_list[::-1].index(sep_vars[i])-1 # remove from last
                fn_list[idx] = fn_list[idx]*sep_fns[i]

                try:
                    var_list.remove(sep_vars[i])
                    fn_list.remove(sep_fns[i])
                except ValueError:
                    logging.debug('wtf,var_list,fn_list,sep_vars,'
                                 'sep_fns,nsep_vars,nsep_fns',
                                 var_list,fn_list,sep_vars,sep_fns,
                                 nsep_vars,nsep_fns)

        # check if indep vars are same
        break_flag = False; i = 0
        while (i < len(var_list)) and not(break_flag):

            j = i + 1
            while (j < len(var_list)) and not(break_flag):
                if (type(var_list[i]) is list) and (type(var_list[j]) is list):

                    if equal_ignore_order(var_list[i],var_list[j]):
                        var_list.remove(var_list[i])
                        fn_list[j] *= fn_list[i]
                        fn_list.remove(fn_list[i])
                        break_flag = True
                    
                j += 1
            i += 1

        return var_list, fn_list
    
    else:
        return unique_v_f1(var_list,fn_list)

def varcount(var_list):
    """get unique vars in var_list"""
    vflat = set(flatten(var_list))
    return len(vflat)

def varcount_min(var_list):
    """count (not necessarily unique) vars in lists within var_list"""

    count = 0
    for var in var_list:
        if type(var) is list:
            count += 2
        else:
            count += 1
    return count

def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched

def get_slice_dim(n,dims):
    """
    n: dimension of h function
    dims: dimensions to update with coefficients
    """
    slicing = ()
    assert(type(dims) is list)
    assert(len(set(dims)) == len(dims) )
    assert(max(dims)<=n-1)
    
    for i in range(n):
        if i in dims:
            slicing += (slice(None),)
        else:
            #slicing += (None,)
            slicing += (0,)
    return slicing

def augment_axes(a,dims,repeat=True):
    """
    fill a with repeat elements in new axes.
    new axes are in dims.
    set repeat=True for function values
    set repeat=False for Fourier.
    """
    
    assert(type(dims) is list)
    slicing = ()
    
    b = np.zeros_like(a)
    b[:,...] = a[:,...]
    
    s = np.shape(b)
    i = 0
    while i <len(s)+len(dims):
        if i in dims:
            slicing += (None,)
        else:
            if repeat:
                slicing += (slice(None),)
            else:
                slicing += (0,)
        
        i += 1
        
    
    b = b[slicing]
    return b + np.zeros(len(slicing)*(max(s),))
