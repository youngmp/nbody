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

def model_cgl1():

    pd2 = {'om':1,'amp':1,'om_fix':1}
    # default period must be 2*np.pi
    system2 = rsp(var_names=[],
                  pardict=pd2,rhs=None,init=None,
                  coupling=None,
                  model_name='f1',
                  forcing_fn=np.sin,
                  idx=1,
                  TN=0)

    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system1 = rsp(var_names=['x','y','w'],
                  pardict=pd1,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl1.coupling_cgl)

    return system1,system2


def model_cgl2():
    
    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system1 = rsp(var_names=['x','y','w'],
                  pardict=pd1,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl2.coupling_cgl)

    pd2 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system2 = rsp(var_names=['x','y','w'],
                  pardict=pd2,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=1,
                  model_name='cgl1',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl2.coupling_cgl)

    return system1,system2

def model_thal1():
    
    pd1 = {'gL':0.05,'gna':3,'gk':5,
           'gt':5,'eL':-70,'ena':50,
           'ek':-90,'et':0,'esyn':-1,
           'c':1,'alpha':3,'beta':2,
           'sigmat':0.8,'vt':-20,
           'ib':8.5,'om':1,'om_fix':1}

    system1 = rsp(var_names=['v','h','r','qt'],
                  pardict=pd1,rhs=rhs_thal,
                  init=np.array([-.64,0.71,0.25,0,5]),
                  TN=2000,
                  idx=0,
                  model_name='thalf0',

                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=thal1.coupling_thal)
    
    pd2 = {'om':1,'amp':1,'om_fix':1,'esyn':0,'c':1}
    ff = lambda x: np.sin(x)+.2
    
    # default period must be 2*np.pi
    system2 = rsp(var_names=[],
                  pardict=pd2,rhs=None,init=None,
                  coupling=None,
                  model_name='thal_force1',
                  forcing_fn=ff,
                  idx=1,
                  TN=0)

    return system1,system2

def model(name):

    if name == 'cgl2':
        return model_cgl2()

    if name == 'cgl1':
        return model_cgl1()

    if name == 'thal1':
        return model_thal1()
    
def neat_out(d,num=1):
    # d is dictionary of sym terms
    s = ''
    for i in range(len(d)):
        print(i,len(d),d[i])
        s += '    '*num + 'Order ' +str(i)+': '+str(d[i])+'\n'

    return s
    
def main():

    d = 'View specific expansions in models'
    parser = ArgumentParser(description=d,formatter_class=ADHF)

    parser.add_argument('-t','--terms',default='all',type=str,
                        help='pick terms to display')

    #parser.add_argument('-m','--model',default='cgl2',type=str,
    #                    help='pick model pair')

    args = parser.parse_args()
    print('args',args)

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
                
           

            if letter == 'p':
                print('p, i={}:'.format(i))
                print(neat_out(a.p['sym'][i,:]))
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
