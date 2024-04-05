# view expansions for a given system. user can choose.

from nBodyCoupling import nBodyCoupling as nm

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

import cgl3
import thal3

from argparse import ArgumentDefaultsHelpFormatter as ADHF
from argparse import ArgumentParser

def get_cgl3():
    var_names = ['x','y']

    pardict = {'q_val':1,
               'eps_val':0,
               'd_val':.9,
               'sig_val':.1,
               'rho_val':.15,
               'mu_val':1}

    kwargs = {'recompute_list':[],
              
              'g_forward':False,
              'i_forward':False,
              'z_forward':False,
              'dense':True,

              'N':3,
              'coupling_mat':cgl3.coupling_mat,
              'dir':'data/dat_cgl3/',
              
              'max_n':-1,
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
              'load_all':True,
              'plot':False,
              'log_level':'INFO'}

    T_init = 2*np.pi
    lc_init = np.array([1,0,T_init])

    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = nm(cgl3.rhs,cgl3.coupling,lc_init,var_names,pardict,**kwargs)

    return a

def get_thal3():
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
    
    
    kwargs = {

              'N':3,
              'coupling_mat':thal3.coupling_mat,
              'dir':'data/dat_thalamicn/',
              'trunc_order':2,
              'max_n':25,
              
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
              'load_all':True,
              'save_fig':True,
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
    
    lc_init = np.array([-.05467127, 0.3326327, 0.3430555, 0.4488427,T_init])
    #LC_init = np.array([-.5927,0.99,0.507,.006,T_init]) # for ib=.6
    
    a = nm(thal3.rhs,thal3.coupling,lc_init,var_names,pardict,**kwargs)

    return a

def get_model(model):

    if model in ['cgl','cgl3']:
        a = get_cgl3()
    elif model in ['thal','thal3']:
        a = get_thal3()

    return a

def neat_out(d,num=1):
    # d is dictionary of sym terms
    s = ''
    for i in range(len(d)):
        print(i,len(d),d[i])
        s += '    '*num + 'Order ' +str(i)+': '+str(d[i])+'\n'

    return s


def get_pdict(pardict):
    pdict = {}
    for key in pardict.keys():
        key2 = key[:-4]
        pdict[key2] = pardict[key]

    return pdict

def main():

    d = 'View specific expansions in models'
    parser = ArgumentParser(description=d,formatter_class=ADHF)

    parser.add_argument('-t','--terms',default='all',type=str,
                        help='pick terms to display')

    parser.add_argument('-m','--model',default='cgl2',type=str,
                        help='pick model pair')

    

    args = parser.parse_args()
    print('args',args)

    a = get_model(args.model)

    args.terms = args.terms.lower()
    if args.terms == 'all':
        args.terms = 'kgp'

    for letter in args.terms:
        for i in range(a.N):
            for j in range(a.N):
                if letter == 'k':
                    print('K, i={}, j={}:'.format(i,j))
                    keys = a.k[i].keys()

                    for key in keys:
                        if 'fnames' in key or 'imp' in key:
                            pass
                        else:
                            print('  '+key+':')
                            print(neat_out(a.k[i][key][j]))
                
           
            for k in range(a.miter):
                if letter == 'p':
                    print('p, i={}:'.format(i))
                    print(a.p['sym'][i,:,k])
                    print()

            
            if letter == 'h':
                print('H:')
                print(neat_out(a.h[i]['sym']))
                print()
                
            #if letter == 'g':
            #    print('G:')
            #    keys = a.c[i].keys()
            #    for key in keys:
            #        print('  '+key+':' +str(a.c[i][key]))
            #    print()


if __name__ == "__main__":
    main()
