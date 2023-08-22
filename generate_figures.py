"""
Generate figures for strong coupling paper
"""

#from decimal import Decimal
#from matplotlib.collections import PatchCollection
import os
import matplotlib.gridspec as gridspec
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import string

from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,
                                                   mark_inset,
                                                   InsetPosition)

from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq, root

from scipy.signal import find_peaks

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from mpl_toolkits.axes_grid1 import make_axes_locatable

import CGL
import Thalamic

from nBodyCoupling import nBodyCoupling


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
preamble = (r'\usepackage{amsmath}'
            r'\usepackage{siunitx}'
            r'\usepackage{bm}'
            r'\newcommand{\ve}{\varepsilon}')

matplotlib.rcParams['text.latex.preamble'] = preamble

#import matplotlib.patches as patches
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d
#from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)

#from lib import collect_disjoint_branches

#import matplotlib as mpl

var_names_thal = ['v','h','r','w']
pardict_thal = {'gL_val':0.05,'gna_val':3,
                'gk_val':5,'gt_val':5,
                'eL_val':-70,'ena_val':50,
                'ek_val':-90,'et_val':0,
                'esyn_val':0,'c_val':1,
                'alpha_val':3,'beta_val':.2,
                'sigmat_val':0.8,'vt_val':-20,
                'ib_val':0.6}

kwargs_thal = {'N':3,'coupling_mat':Thalamic.coupling_mat,
               'dir':'home+dat_thalamicn/',
               'NA':500,'trunc_order':1,
               'TN':10000,
               'load_all':False,
               'load_test':False}


var_names_cgl = ['x','y']
pardict_cgl = {'q_val':1,
           'eps_val':0,
           'd_val':.8,
           'sig_val':.1,
           'rho_val':.2,
           'mu_val':1}

kwargs_cgl = {'recompute_LC':False,
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

          'recompute_h':False,
          'g_forward':False,
          'i_forward':False,
          'z_forward':False,
          'dense':True,

          'N':3,
          'coupling_mat':CGL.coupling_mat,
          'dir':'data/dat_cgln/',

          'trunc_order':2,
          'trunc_deriv':1,
          'NA':101,
          'p_iter':5,
          'TN':5000,
          'rtol':1e-8,
          'atol':1e-8,
          'rel_tol':1e-6,
          'method':'LSODA',
          'load_all':False}
    


# font size
size = 12

exp = np.exp
pi = np.pi

pi_label_short = [r"$0$", r"$\pi$", r"$2\pi$"]

labels = list(string.ascii_uppercase)

for i in range(len(labels)):
    labels[i] = r'\textbf{{{}}}'.format(labels[i])


def _get_appdata_path():
    import ctypes
    from ctypes import wintypes, windll
    CSIDL_APPDATA = 26
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND,
                                 ctypes.c_int,
                                 wintypes.HANDLE,
                                 wintypes.DWORD,
                                 wintypes.LPCWSTR]
    path_buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_APPDATA, 0, 0, path_buf)
    return path_buf.value

def dropbox_home():
    from platform import system
    import base64
    import os.path
    _system = system()
    if _system in ('Windows', 'cli'):
        host_db_path = os.path.join(_get_appdata_path(),
                                    'Dropbox',
                                    'host.db')
    elif _system in ('Linux', 'Darwin'):
        host_db_path = os.path.expanduser('~'
                                          '/.dropbox'
                                          '/host.db')
    else:
        raise RuntimeError('Unknown system={}'
                           .format(_system))

    
    if not os.path.exists(host_db_path):

        manualpath = os.path.expanduser('~')+'/Dropbox'
        if not os.path.exists(manualpath):
            try:
                path = "E:\\Dropbox"
            except:
                raise RuntimeError("Config path={} doesn't exist"
                               .format(host_db_path))
        else:
            path = manualpath
    
    else:
    
        with open(host_db_path, 'r') as f:
            data = f.read().split()
            path = base64.b64decode(data[1]).decode('utf-8')
    
    return path

def add_arrow_to_line2D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', arrowsize=1.5, transform=None,
        tail_space=50):

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    #if use_multicolor_lines:
    #    raise NotImplementedError("multicolor lines not supported")
    #else:
    arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n-tail_space], y[n-tail_space])
        arrow_head = (np.mean(x[n:n + 2]),
                      np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows
    

def get_phase(pt,dat_ref):
    #print(np.linalg.norm(pt-dat_ref,axis=0).shape)

    diff = pt-dat_ref
    #diff[0] /= 100
    #diff[2] *= 100
    min_idx = np.argmin(np.linalg.norm(diff,axis=1))
    max_idx = len(dat_ref)


    #print(np.linalg.norm(diff,axis=0))
    return 2*np.pi*min_idx/max_idx

def quick_plots_thalamic():

    
    #eps_vals1 = np.round(np.arange(0,0.01,0.0001)[::5][2:],5)
    #eps_vals2 = np.round(np.arange(-.01,-0.0001,0.0001)[::5][:-1],5)
    #eps_vals = np.concatenate((eps_vals2,eps_vals1))

    # for ib=.8
    #eps_vals = [0.03,0.05]
    #eps_vals = np.round(np.arange(0.01,0.06,.005),3)
    eps_vals = np.round(np.arange(0.,0.03,.001),3)[5:]
    
    het_vals = np.array([0.0])

    dhome = dropbox_home()

    data_dir = dhome+'/data/thal_full_normed/'
    ib = 0.8
    esyn = -1

    if ib == 0.6:
        fname_load_temp = data_dir+'thal3_eps={}_het={}.dat'    
        fname_save_temp = data_dir+'thal3_{}_eps={}_phases.txt'
        fname_save_temp_t = data_dir+'thal3_t_eps={}_phases.txt'
        dat_ref = np.loadtxt(data_dir+'lc_ib.6_be_.2.dat')[:,1:5]
    elif ib == 0.8:
        fname_load_temp = data_dir+'thal3_eps={}_het={}_ib={}_esyn={}.dat'
        fname_save_temp = data_dir+'thal3_{}_eps={}_ib={}_esyn={}_phases.txt'
        fname_save_temp_t = data_dir+'thal3_t_eps={}_ib={}_esyn={}_phases.txt'
        dat_ref = np.loadtxt(data_dir+'lc_ib.8_be_.2.dat')[:,1:5]
    else:
        raise Exception("what do i do? ib not known. ib=",ib)
    
    for k,eps in enumerate(eps_vals):
        
        for j,het in enumerate(het_vals):

            if ib == 0.6:
                format_tuple1 = (eps,het)
                format_tuple2 = (eps,)
            else:
                format_tuple1 = (eps,het,ib,esyn)
                format_tuple2 = (eps,ib,esyn)
                
            fname_load = fname_load_temp.format(*format_tuple1)
            fname_save0 = fname_save_temp.format(*(0,*format_tuple2))
            fname_save1 = fname_save_temp.format(*(1,*format_tuple2))
            fname_save2 = fname_save_temp.format(*(2,*format_tuple2))
            fname_save_t = fname_save_temp_t.format(*format_tuple2)
            
            file_does_not_exist = not(os.path.isfile(fname_save0))
            file_does_not_exist += not(os.path.isfile(fname_save1))
            file_does_not_exist += not(os.path.isfile(fname_save2))
            

            if file_does_not_exist or True:

                skipn = 100

                dat = np.loadtxt(fname_load)[::skipn]
                osc0 = dat[:,1:5];osc1 = dat[:,5:9];osc2 = dat[:,9:13]
                t = dat[:,0]

                th0 = np.zeros(len(t))
                th1 = np.zeros(len(t))
                th2 = np.zeros(len(t))

                for i in range(len(t)):
                    th0[i] = get_phase(osc0[i,:],dat_ref)
                    th1[i] = get_phase(osc1[i,:],dat_ref)
                    th2[i] = get_phase(osc2[i,:],dat_ref)

                np.savetxt(fname_save0,th0)
                np.savetxt(fname_save1,th1)
                np.savetxt(fname_save2,th2)
                np.savetxt(fname_save_t,t)

            else:

                th0 = np.loadtxt(fname_save0)
                th1 = np.loadtxt(fname_save1)
                th2 = np.loadtxt(fname_save1)
                t = np.loadtxt(fname_save_t)
                

            fig,axs = plt.subplots(1,2,figsize=(8,3))

            #axs.plot(th0);axs.plot(th1);axs.plot(th2)
            #axs.plot(th0);axs.plot(th1);axs.plot(th2)
            #axs.scatter(th1-th0,th2-th0,s=5,color='k')
            x = np.mod(th1-th0,2*np.pi)
            y = np.mod(th2-th0,2*np.pi)
            axs[0].scatter(x,y,s=5,label='eps='+str(eps),
                           c=np.linspace(0,1,len(x)))

            axs[0].set_xlim(0,2*np.pi)
            axs[0].set_ylim(0,2*np.pi)

            axs[1].scatter(t,x,s=5)
            axs[1].scatter(t,y,s=5)

            axs[0].legend()

            loc = 'figs_temp_normed/'
            if ib == 0.6:
                figname = loc+'thal3_sols_{}het={}_{}eps={}_phases.png'
            else:
                figname = loc+'thal3_sols_{}het={}_{}eps={}_ib={}_esyn={}_phases.png'
            fig.savefig(figname.format(string.ascii_lowercase[j+1],het,k,eps,ib,esyn))
            plt.close()

            print('eps,het',eps,het,'inits out of 1. th1-th0',
                  (th1[0]-th0[0])/(2*np.pi),'th2-th0',(th2[0]-th0[0])/(2*np.pi))

def get_h_vals(t,a,i,order=1):
    """
    meshgrid for rhs
    """
    #t = a.Aarr_noend[::skipn]
    
    tn = len(t)
    t1,t2 = np.meshgrid(t,t)
    
    lam = a.h[i]['lam'][order]
    sol = lam(t1,t2)
    
    return t1,t2,sol

def prhs(t,z,a,eps=None,order=2):
    """
    rhs for phi
    """
    tot1=0;tot2=0;tot3=0

    for j in range(order):
        
        tot1 += np.sign(eps)**(j+1)*(np.abs(eps))**(j+1)*a.h[0]['lam'][j](*z)
        tot2 += np.sign(eps)**(j+1)*(np.abs(eps))**(j+1)*a.h[1]['lam'][j](*z)
        tot3 += np.sign(eps)**(j+1)*(np.abs(eps))**(j+1)*a.h[2]['lam'][j](*z)
        
    return np.array([tot2-tot1,tot3-tot1])

def prhs_noeps(t,z,a,eps=None,order=2):
    """
    rhs for phi
    """
    tot1=0;tot2=0;tot3=0

    for j in range(order):
        
        tot1 += np.sign(eps)**(j+1)*(np.abs(eps))**(j)*a.h[0]['lam'][j](*z)
        tot2 += np.sign(eps)**(j+1)*(np.abs(eps))**(j)*a.h[1]['lam'][j](*z)
        tot3 += np.sign(eps)**(j+1)*(np.abs(eps))**(j)*a.h[2]['lam'][j](*z)
        
    return np.array([tot2-tot1,tot3-tot1])

def prhs_for_root(z,a,eps,order):
    return prhs(0,z,a,eps,order)

# combine rhs
a = CGL.coupling_mat(3)

def rhs_coupled(t,z,pd):
    z0 = z[:2];z1 = z[2:4];z2 = z[4:]

    eps = pd['eps']
    
    c0 = a[0,1]*CGL.coupling([*z0,*z1],pd)+a[0,2]*CGL.coupling([*z0,*z2],pd)
    c1 = a[1,0]*CGL.coupling([*z1,*z0],pd)+a[1,2]*CGL.coupling([*z1,*z2],pd)
    c2 = a[2,0]*CGL.coupling([*z2,*z0],pd)+a[2,1]*CGL.coupling([*z2,*z1],pd)
    
    out0 = CGL.rhs(t,z0,pd) + eps*c0
    out1 = CGL.rhs(t,z1,pd) + eps*c1
    out2 = CGL.rhs(t,z2,pd) + eps*c2

    return np.concatenate((out0,out1,out2))

def load_cgl_sol(ag,kw,recompute=False):

    home = os.path.expanduser('~')

    dir1 = home+'/dat_cgl'
    fname = dir1+'/cgl_tempsol_eps={}.txt'
    fname = fname.format(kw['args'][0]['eps'])

    msg = ('Saving some data to {} to avoid recalculating some stuff.'
           'You must delete this directory manually.'
           'Delete the directory if you get an assertion error')
    print(msg.format(fname))
    
    if not(os.path.isdir(dir1)):
        os.mkdir(dir1)
    
    file_does_not_exist = not(os.path.isfile(fname))

    if file_does_not_exist or recompute:
        sol = solve_ivp(*ag,**kw)
        y = sol.y.T
        np.savetxt(fname,y)
    else:
        y = np.loadtxt(fname)

    return y

def cgl_examples():
    var_names = var_names_cgl
    pardict = pardict_cgl
    kwargs = kwargs_cgl
    
    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])

    # co = color order
    co = ['tab:blue','tab:orange','k']

    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = nBodyCoupling(CGL.rhs,CGL.coupling,
                      LC_init,var_names,pardict,**kwargs)

    t = np.linspace(0-a.T/8,a.T+a.T/8,500)
    
    z_rhs = []
    for i in range(a.N):
        z_rhs.append([])
        for j in range(a.miter):
            X,Y,zt = get_h_vals(t,a,i,order=j)
            z_rhs[i].append(zt)

    eps_list_full = [.005,.06]
    eps_list_phase = [.005,.06]

    T_list_full = [6000,2000]
    T_list_phase = [6000,2000]

    fig,axs = plt.subplots(2,4,figsize=(8,5),
                           gridspec_kw={'width_ratios':[3,1,3,1]})

    fig_temp,axs_temp = plt.subplots(1,1)
    P = a.T

    r1 = [.05,.05] #phi2 init
    r2 = [.25,.25] # phi3 init
    
    for i in range(len(eps_list_phase)):
        
        ################ load full
        eps = eps_list_full[i];T = T_list_full[i];dt = .05;TN = int(T/dt)
        t_full = np.linspace(0,T,TN)

        pdict = {}
        for key in pardict.keys():
            key2 = key[:-4]
            pdict[key2] = pardict[key]
        pdict['eps'] = eps

        kw = {'t_eval':t_full,'method':'RK45','rtol':1e-8,'args':(pdict,)}

        z_init = [np.cos(0),np.sin(0),
                  np.cos(r1[i]),np.sin(r1[i]),
                  np.cos(r2[i]),np.sin(r2[i])]
        
        ag = [rhs_coupled,[0,t_full[-1]],z_init]

        sol = load_cgl_sol(ag,kw)

        # get angle for full model
        th0b = np.arctan2(sol[:,1],sol[:,0])
        th1b = np.arctan2(sol[:,3],sol[:,2])
        th2b = np.arctan2(sol[:,5],sol[:,4])
        tb = t_full
        
        ################ load phase
        eps = eps_list_phase[i];T = T_list_phase[i];dt = .02;TN = int(T/dt)
        t_phase = np.linspace(0,T,TN)

        kw = {'t_eval':t_phase,'method':'RK45','rtol':1e-6}
        ag = [prhs,[0,t_phase[-1]],[r1[i],r2[i]]]
        sol1 = solve_ivp(*ag,args=(a,eps,1),**kw)
        sol2 = solve_ivp(*ag,args=(a,eps,2),**kw)
        sol3 = solve_ivp(*ag,args=(a,eps,3),**kw)

        max_t_phase = T_list_phase[i]
        y1=sol1.y.T[t_phase<=max_t_phase,:]
        y2=sol2.y.T[t_phase<=max_t_phase,:]
        y3=sol3.y.T[t_phase<=max_t_phase,:]

        phi2_order1=z_rhs[1][0]-z_rhs[0][0];phi3_order1=z_rhs[2][0]-z_rhs[0][0]
        phi2_order2=z_rhs[1][1]-z_rhs[0][1];phi3_order2=z_rhs[2][1]-z_rhs[0][1]

        null_phi2 = np.abs(eps)*phi2_order2+np.sign(eps)*phi2_order1
        null_phi3 = np.abs(eps)*phi3_order2+np.sign(eps)*phi3_order1

        ############## plot full
        kw_contour = {'levels':[0],'alpha':0.5,'linewidths':1,'zorder':-1}
        axs[0,i*2].contour(X,Y,null_phi2,colors='r',**kw_contour)
        axs[0,i*2].contour(X,Y,null_phi3,colors='k',**kw_contour)

        sn = 10
        xb = np.mod(th1b-th0b,2*np.pi)
        yb = np.mod(th2b-th0b,2*np.pi)

        axs[0,i*2].scatter(xb[::sn],yb[::sn],
                           s=1,label='eps='+str(eps),
                           c=np.linspace(0,1,len(xb[::sn])))

        assert(len(xb)==len(tb))
        assert(len(yb)==len(tb))
        
        axs[0,i*2+1].scatter(xb[::sn],tb[::sn],color='k',
                             alpha=.8,label=r'$\phi_2$',s=1)
        axs[0,i*2+1].scatter(yb[::sn],tb[::sn],color='gray',
                             alpha=.8,label=r'$\phi_3$',s=1)

        ############## plot phase
        axs[1,i*2].contour(X,Y,null_phi2,colors='r',**kw_contour)
        axs[1,i*2].contour(X,Y,null_phi3,colors='k',**kw_contour)

        alpha = .75; lw = 1.5
        kw = {'alpha':0.75,'lw':1.5}
        axs[1,i*2].plot(y1[:,0],y1[:,1],color=co[0],label=r'$O(\ve)$',**kw)
        axs[1,i*2].plot(y2[:,0],y2[:,1],color=co[1],label=r'$O(\ve^2)$',**kw)
        axs[1,i*2].plot(y3[:,0],y3[:,1],color=co[2],label=r'$O(\ve^3)$',**kw)

        # dumb thing to get arrows to work
        line, = axs_temp.plot(y1[:,0],y1[:,1],color=co[0],zorder=4)
        axs_temp.clear();arrow_locs = [.25,.66]
        add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)

        if i == 0:
            line, = axs_temp.plot(y2[:,0],y2[:,1],color=co[1],zorder=4)
            axs_temp.clear();arrow_locs = [.26,.67]
            add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)

        line, = axs_temp.plot(y3[:,0],y3[:,1],color=co[2],zorder=4)
        axs_temp.clear();arrow_locs = [.28,.69]
        add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)

        kw1 = {'color':co[0],'alpha':0.75,'lw':2,
              'label':r'$\phi_2$ $(\ve)$','zorder':1}
        kw2 = kw1.copy();kw2['ls']='--';kw2['label']=r'$\phi_3$ $(\ve)$'
        axs[1,i*2+1].plot(y1[:,0],t_phase,**kw1)
        axs[1,i*2+1].plot(y1[:,1],t_phase,**kw2)

        kw1['color']=co[1];kw1['label']=r'$\phi_2$ $(\ve^2)$'
        kw2['color']=co[1];kw2['label']=r'$\phi_3$ $(\ve^2)$'
        axs[1,i*2+1].plot(y2[:,0],t_phase,**kw1)
        axs[1,i*2+1].plot(y2[:,1],t_phase,**kw2)

        kw1['color']=co[2];kw1['label']=r'$\phi_2$ $(\ve^3)$'
        kw2['color']=co[2];kw2['label']=r'$\phi_3$ $(\ve^3)$'
        axs[1,i*2+1].plot(y3[:,0],t_phase,**kw1)
        axs[1,i*2+1].plot(y3[:,1],t_phase,**kw2)

        # reverse limits
        axs[0,i*2+1].set_ylim(axs[0,i*2+1].get_ylim()[::-1])
        axs[1,i*2+1].set_ylim(axs[1,i*2+1].get_ylim()[::-1])
        
        # limits
        axs[0,i*2+1].set_ylim(T_list_full[i],0)
        axs[1,i*2+1].set_ylim(T_list_phase[i],0)

        for kk in range(2):
            
            # limits
            axs[kk,i*2+1].set_xlim(0-2*np.pi/10,2*np.pi+2*np.pi/10)
            
            # labels and ticks
            axs[kk,i*2].set_xticklabels(pi_label_short)
            axs[kk,i*2].set_xticks(np.arange(0,3,1)*pi)
            axs[kk,i*2+1].set_xticklabels(pi_label_short)
            axs[kk,i*2+1].set_xticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2].set_yticklabels(pi_label_short)
            axs[kk,i*2].set_yticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2+1].set_ylabel(r'$t$',labelpad=-.1)
            
            axs[kk,i*2].set_ylabel(r'$\phi_3$',labelpad=-.1)
            axs[kk,i*2].set_xlabel(r'$\phi_2$',labelpad=-.1)
            
            axs[kk,i*2+1].set_xlabel(r'$\phi_i$',labelpad=-.1)
    r,c = np.shape(axs)
    plt.close(fig_temp)
    # legends
    lgd = axs[0,1].legend(bbox_to_anchor=(1.5,1.1),borderpad=0.1,
                            labelspacing=0,handletextpad=-0.1,
                            framealpha=1,loc='upper right')
    for handle in lgd.legend_handles:
        handle.set_sizes([15])
        
    lgd2 = axs[1,1].legend(bbox_to_anchor=(1.8,-.09),
                        borderpad=0.1, labelspacing=0,handletextpad=0.3,
                        framealpha=1,handlelength=1,loc='lower right')
    for legobj in lgd2.legend_handles:
        legobj.set_linewidth(1.5)
        
    
    lgd3 = axs[1,0].legend(loc='lower right')
    for legobj in lgd3.legend_handles:
        legobj.set_linewidth(2)
        
    lgd4 = axs[1,2].legend(loc='lower right')
    for legobj in lgd4.legend_handles:
        legobj.set_linewidth(2)
    
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].set_title(labels[count],loc='left')
            
            count += 1
        
    for i in range(4):
        if i%2 == 0:
            t0 = axs[0,i].get_title()
            t1 = axs[1,i].get_title()
        
            t0 += r'Full $\ve={}$'.format(eps_list_full[int(i/2)])
            t1 += r'Phase $\ve={}$'.format(eps_list_phase[int(i/2)])
            
            axs[0,i].set_title(t0)
            axs[1,i].set_title(t1)
    
    plt.tight_layout()
    
    plt.subplots_adjust(hspace=.4,wspace=.5)

    return fig

def jac(fn,pt,a,eps,order=3,dx=1e-4):
    n = len(pt)
    
    out = np.zeros((2,n))
    
    for i in range(n):
        p = np.zeros(n);p[i] = dx
        out[:,i] = (fn(0,pt+p,a,eps,order)-fn(0,pt,a,eps,order))/dx

    return out

def cgl_eigs():
    var_names = var_names_cgl
    pardict = pardict_cgl
    kwargs = kwargs_cgl
    
    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])

    # co = color order
    co = ['tab:blue','tab:orange','k']

    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = nBodyCoupling(CGL.rhs,CGL.coupling,
                      LC_init,var_names,pardict,**kwargs)

    eps_list = np.linspace(0,.2,30)
    eig_o1 = np.zeros((len(eps_list),2),dtype=complex)
    eig_o2 = np.zeros((len(eps_list),2),dtype=complex)
    eig_o3 = np.zeros((len(eps_list),2),dtype=complex)

    for i in range(len(eps_list)):
        eps = eps_list[i]

        z = np.array([2.0943951, 4.1887902])
        out1 = root(prhs_for_root, z, args=(a,eps,1),method='hybr')
        out2 = root(prhs_for_root, z, args=(a,eps,2),method='hybr')
        out3 = root(prhs_for_root, z, args=(a,eps,3),method='hybr')
        eig_o1[i,:] = np.linalg.eig(jac(prhs,out1.x,a,eps,order=1))[0]
        eig_o2[i,:] = np.linalg.eig(jac(prhs,out2.x,a,eps,order=2))[0]
        eig_o3[i,:] = np.linalg.eig(jac(prhs,out3.x,a,eps,order=3))[0]

    fig,axs = plt.subplots(1,2,figsize=(8,2.5))

    axs[0].plot(eps_list,np.real(eig_o1[:,0]),color=co[0],label=r'$O(\ve)$')
    axs[0].plot(eps_list,np.real(eig_o2[:,0]),color=co[1],label=r'$O(\ve^2)$')
    axs[0].plot(eps_list,np.real(eig_o3[:,0]),color=co[2],label=r'$O(\ve^3)$')

    axs[1].plot(eps_list,np.imag(eig_o1),color=co[0])
    axs[1].plot(eps_list,np.imag(eig_o2),color=co[1])
    axs[1].plot(eps_list,np.imag(eig_o3),color=co[2])

    axs[0].plot([eps_list[0],eps_list[-1]],[0,0],color='gray',ls='--',lw=1)

    for i in range(2):
        axs[i].set_xlabel(r'$\ve$')
        axs[i].set_title(labels[i],loc='left')

        axs[i].set_xlim(eps_list[0],eps_list[-1])

    t0 = axs[0].get_title()
    axs[0].set_title(t0+'Real')
    t1 = axs[1].get_title()
    axs[1].set_title(t1+'Imaginary')
    

    lo,hi = axs[0].get_ylim()
    axs[0].set_ylim(lo,hi/2)
    axs[0].legend()
    plt.tight_layout()

    return fig

def thalamic_examples1():

    var_names = var_names_thal
    pardict = pardict_thal
    kwargs = kwargs_thal

    pardict['ib'] = 0.8
    pardict['esyn'] = -100
    kwargs['NA'] = 505
    
    T_init = 44.8
    LC_init = np.array([-.5927,0.99,0.507,.006,T_init])
    
    a = nBodyCoupling(Thalamic.rhs,Thalamic.coupling,
                      LC_init,var_names,pardict,**kwargs)
    
    t = np.linspace(0-a.T/8,a.T+a.T/8,500)

    z_rhs = []
    for i in range(a.N):
        z_rhs.append([])
        for j in range(a.miter):
            X,Y,zt = get_h_vals(t,a,i,order=j)
            z_rhs[i].append(zt)

    eps_list_full = [.005,.027]
    eps_list_phase = [.005,.016]

    T_list_full = [20000,12000]
    T_list_phase = [20000,15000]
    
    T_low_full = [0,0]

    scale = 2*np.pi/a.T
    fig,axs = plt.subplots(2,4,figsize=(8,5),
                           gridspec_kw={'width_ratios':[3,1,3,1]})

    fig_temp,axs_temp = plt.subplots(1,1)
    
    for i in range(len(eps_list_phase)):
        ################ load phase
        eps = eps_list_phase[i]#-.0014
        T = T_list_phase[i];dt = .02;TN = int(T/dt)
        t_phase = np.linspace(0,T,TN)

        if i == 0:
            r1 = .064
            r2 = .15
        else:
            r1 = .33
            r2 = .66
        sol1 = solve_ivp(prhs,[0,t_phase[-1]],[a.T*r1,a.T*r2],
                         args=(a,eps,1),t_eval=t_phase,method='RK45',rtol=1e-6)
        sol2 = solve_ivp(prhs,[0,t_phase[-1]],[a.T*r1,a.T*r2],
                         args=(a,eps,2),t_eval=t_phase,method='RK45',rtol=1e-6)
        
        max_t_phase = T_list_phase[i]
        y1=sol1.y.T[t_phase<=max_t_phase,:];y2=sol2.y.T[t_phase<=max_t_phase,:]

        phi2_order1=z_rhs[1][0]-z_rhs[0][0];phi3_order1=z_rhs[2][0]-z_rhs[0][0]
        phi2_order2=z_rhs[1][1]-z_rhs[0][1];phi3_order2=z_rhs[2][1]-z_rhs[0][1]
        
        null_phi2 = np.abs(eps)*phi2_order2+np.sign(eps)*phi2_order1
        null_phi3 = np.abs(eps)*phi3_order2+np.sign(eps)*phi3_order1

        ############## load full
        eps = eps_list_full[i]#-0.009
        ib = 0.8;esyn = -1
        
        data_dir = 'data/thal_full_normed/'
        
        fname_load_temp = data_dir+'thal3_eps={}_het={}_ib={}_esyn={}.dat'
        fname_save_temp = data_dir+'thal3_{}_eps={}_ib={}_esyn={}_phases.txt'
        fname_save_temp_t = data_dir+'thal3_t_eps={}_ib={}_esyn={}_phases.txt'
        th0 = np.loadtxt(fname_save_temp.format(0,eps,ib,esyn))
        th1 = np.loadtxt(fname_save_temp.format(1,eps,ib,esyn))
        th2 = np.loadtxt(fname_save_temp.format(2,eps,ib,esyn))
        t = np.loadtxt(fname_save_temp_t.format(eps,ib,esyn))
    
        max_t = T_list_full[i]
        min_t = T_low_full[i]
        
        th0b = th0[(t<=max_t)*(t>=min_t)]
        th1b = th1[(t<=max_t)*(t>=min_t)]
        th2b = th2[(t<=max_t)*(t>=min_t)]
        tb = t[(t<=max_t)*(t>=min_t)]
        
        ############## plot full
        axs[0,i*2].contour(X*scale,Y*scale,null_phi2,colors='r',
                        levels=[0],alpha=0.5,zorder=-1,linewidths=1)
        axs[0,i*2].contour(X*scale,Y*scale,null_phi3,colors='k',
                        levels=[0],alpha=0.5,zorder=-1,linewidths=1)

        
        
        xb = np.mod(th1b-th0b,2*np.pi)
        yb = np.mod(th2b-th0b,2*np.pi)

        if i == 1:
            low_t = 6000
            sn = 1
        else:
            low_t = 0
            sn = 2
            
        axs[0,i*2].scatter(xb[tb>low_t][::sn],yb[tb>low_t][::sn],
                           s=5,label='eps='+str(eps),
                           c=np.linspace(0,1,len(xb[tb>low_t][::sn])))
        
        axs[0,i*2+1].scatter(xb[::sn],tb[::sn],color='k',
                             alpha=.8,label=r'$\phi_2$',s=1)
        axs[0,i*2+1].scatter(yb[::sn],tb[::sn],color='gray',
                             alpha=.8,label=r'$\phi_3$',s=1)

        ############## plot phase
        kw={'levels':[0],'alpha':0.5,'linewidths':1}
        axs[1,i*2].contour(X*scale,Y*scale,null_phi2,colors='r',**kw)
        axs[1,i*2].contour(X*scale,Y*scale,null_phi3,colors='k',**kw)

        axs[1,i*2].plot(y1[:,0]*scale,y1[:,1]*scale,color='tab:blue',
                        lw=2,label=r'$O(\ve)$',zorder=3)

        # dumb thing to get arrows to work
        if i == 0:
            line, = axs_temp.plot(y1[:,0]*scale,y1[:,1]*scale,
                                    label=r'$O(\ve)$',color='tab:blue')
            axs_temp.clear()
            arrow_locs = [.33,.66]
            add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)
            
        axs[1,i*2+1].plot(y1[:,0]*scale,t_phase[t_phase<=max_t_phase],
                            color='tab:blue',alpha=.8,lw=2,
                            label=r'$\phi_2$ $(\ve)$',zorder=3)
        axs[1,i*2+1].plot(y1[:,1]*scale,t_phase[t_phase<=max_t_phase],
                        color='tab:blue',ls='--',alpha=.8,lw=2,
                        label=r'$\phi_3$ $(\ve)$',zorder=3)


        if i == 0:
            lw = 1.5;alpha = 1
        else:
            lw = 1.5; alpha = 1

        e = np.abs(eps)

        axs[1,i*2].plot(sol2.y.T[:,0]*scale,sol2.y.T[:,1]*scale,
                        color='k',alpha=alpha,
                        lw=lw,label=r'$O(\ve^2)$')

        # dumb thing to get arrows to work
        line, = axs_temp.plot(sol2.y.T[:,0]*scale,sol2.y.T[:,1]*scale,
                              label=r'$O(\ve)$',color='k')
        axs_temp.clear()
        if i == 0:
            arrow_locs = [.33,.66]
        else:
            arrow_locs = [.5,.7]
        add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)

        axs[1,i*2+1].plot(y2[:,0]*scale,t_phase[t_phase<=max_t_phase],
                          color='k',alpha=.8,lw=2,
                          label=r'$\phi_2$ $(\ve^2)$')
        axs[1,i*2+1].plot(y2[:,1]*scale,t_phase[t_phase<=max_t_phase],
                          color='gray',alpha=.8,lw=2,
                          label=r'$\phi_3$ $(\ve^2)$')

        # reverse limits
        axs[0,i*2+1].set_ylim(axs[0,i*2+1].get_ylim()[::-1])
        axs[1,i*2+1].set_ylim(axs[1,i*2+1].get_ylim()[::-1])
        
        # limits
        axs[0,i*2+1].set_ylim(T_list_full[i],0)
        axs[1,i*2+1].set_ylim(T_list_phase[i],0)
        
        for kk in range(2):
            
            # limits
            axs[kk,i*2+1].set_xlim(0-2*np.pi/10,2*np.pi+2*np.pi/10)
            
            # labels and ticks
            axs[kk,i*2].set_xticklabels(pi_label_short)
            axs[kk,i*2].set_xticks(np.arange(0,3,1)*pi)
            axs[kk,i*2+1].set_xticklabels(pi_label_short)
            axs[kk,i*2+1].set_xticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2].set_yticklabels(pi_label_short)
            axs[kk,i*2].set_yticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2+1].set_ylabel(r'$t$',labelpad=-.1)
            
            axs[kk,i*2].set_ylabel(r'$\phi_3$',labelpad=-.1)
            axs[kk,i*2].set_xlabel(r'$\phi_2$',labelpad=-.1)
            
            axs[kk,i*2+1].set_xlabel(r'$\phi_i$',labelpad=-.1)

    r,c = np.shape(axs)
    plt.close(fig_temp)
    # legends
    lgd = axs[0,1].legend(bbox_to_anchor=(1.5,1.1),borderpad=0.1,
                            labelspacing=0,handletextpad=-0.1,
                            framealpha=1,loc='upper right')
    for handle in lgd.legend_handles:
        handle.set_sizes([15])
        
    lgd2 = axs[1,1].legend(bbox_to_anchor=(1.8,-.05),
                        borderpad=0.1, labelspacing=0,handletextpad=0.3,
                        framealpha=1,handlelength=1,loc='lower right')
    for legobj in lgd2.legend_handles:
        legobj.set_linewidth(1.5)
        
    
    lgd3 = axs[1,0].legend(loc='lower right')
    for legobj in lgd3.legend_handles:
        legobj.set_linewidth(2)
        
    lgd4 = axs[1,2].legend(loc='lower right')
    for legobj in lgd4.legend_handles:
        legobj.set_linewidth(2)
    
    
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].set_title(labels[count],loc='left')
            
            count += 1
        
    for i in range(4):
        if i%2 == 0:
            t0 = axs[0,i].get_title()
            t1 = axs[1,i].get_title()
        
            t0 += r'Full $\ve={}$'.format(eps_list_full[int(i/2)])
            t1 += r'Phase $\ve={}$'.format(eps_list_phase[int(i/2)])
            
            axs[0,i].set_title(t0)
            axs[1,i].set_title(t1)
    
    plt.tight_layout()
    
    plt.subplots_adjust(hspace=.4,wspace=.5)

    return fig

def thalamic_examples2():
    
    var_names = var_names_thal
    pardict = pardict_thal
    kwargs = kwargs_thal
    
    T_init = 44.8
    LC_init = np.array([-.5927,0.99,0.507,.006,T_init])
    
    a = nBodyCoupling(Thalamic.rhs,Thalamic.coupling,
                      LC_init,var_names,pardict,**kwargs)
    
    t = np.linspace(0-a.T/8,a.T+a.T/8,500)

    z_rhs = []
    for i in range(a.N):
        z_rhs.append([])
        for j in range(a.miter):
            X,Y,zt = get_h_vals(t,a,i,order=j)
            z_rhs[i].append(zt)
    
    eps_list_phase = [-.0014,.0025]
    eps_list_full = [-.008,.0025]
    
    T_list_phase = [80000,15000]
    T_list_full = [12000,12000]
    
    scale = 2*np.pi/a.T
    fig,axs = plt.subplots(2,4,figsize=(8,5),
                           gridspec_kw={'width_ratios':[3,1,3,1]})
    
    fig_temp,axs_temp = plt.subplots(1,1)
    
    for i in range(len(eps_list_phase)):
    
        ################ load phase
        eps = eps_list_phase[i]#-.0014
        T = 100000;dt = .02;TN = int(T/dt)
        t_phase = np.linspace(0,T,TN)
        r1 = .33
        r2 = .66
        sol1 = solve_ivp(prhs,[0,t_phase[-1]],[a.T*r1,a.T*r2],
                         args=(a,eps,1),t_eval=t_phase,method='RK45',rtol=1e-6)
        sol2 = solve_ivp(prhs,[0,t_phase[-1]],[a.T*r1,a.T*r2],
                         args=(a,eps,2),t_eval=t_phase,method='RK45',rtol=1e-6)
        
        max_t_phase = T_list_phase[i]
        y1=sol1.y.T[t_phase<=max_t_phase,:];y2=sol2.y.T[t_phase<=max_t_phase,:]

        phi2_order1=z_rhs[1][0]-z_rhs[0][0];phi3_order1=z_rhs[2][0]-z_rhs[0][0]
        phi2_order2=z_rhs[1][1]-z_rhs[0][1];phi3_order2=z_rhs[2][1]-z_rhs[0][1]
        
        null_phi2 = np.abs(eps)*phi2_order2+np.sign(eps)*phi2_order1
        null_phi3 = np.abs(eps)*phi3_order2+np.sign(eps)*phi3_order1
        
        ############## load full
        eps = eps_list_full[i]#-0.009

        data_dir = 'data/thal_full_normed/'
        
        fname_load_temp = data_dir+'thal3_eps={}_het={}.dat'
        fname_save_temp = data_dir+'thal3_{}_eps={}_phases.txt'
        fname_save_temp_t = data_dir+'thal3_t_eps={}_phases.txt'
        th0 = np.loadtxt(fname_save_temp.format(0,eps))
        th1 = np.loadtxt(fname_save_temp.format(1,eps))
        th2 = np.loadtxt(fname_save_temp.format(2,eps))
        t = np.loadtxt(fname_save_temp_t.format(eps))
    
        max_t = T_list_full[i]
        th0b = th0[t<=max_t]
        th1b = th1[t<=max_t]
        th2b = th2[t<=max_t]
        tb = t[t<=max_t]
        
        ############## plot full
        axs[0,i*2].contour(X*scale,Y*scale,null_phi2,colors='r',
                        levels=[0],alpha=0.5,zorder=-1,linewidths=1)
        axs[0,i*2].contour(X*scale,Y*scale,null_phi3,colors='k',
                        levels=[0],alpha=0.5,zorder=-1,linewidths=1)
        
        sn = 2
        xb = np.mod(th1b-th0b,2*np.pi)
        yb = np.mod(th2b-th0b,2*np.pi)
        axs[0,i*2].scatter(xb[::sn],yb[::sn],s=5,label='eps='+str(eps),
                            c=np.linspace(0,1,len(xb[::sn])))
        
        sn = 10
        axs[0,i*2+1].scatter(xb[::sn],tb[::sn],color='k',
                             alpha=.8,label=r'$\phi_2$',s=1)
        axs[0,i*2+1].scatter(yb[::sn],tb[::sn],color='gray',
                             alpha=.8,label=r'$\phi_3$',s=1)
    
        ############## plot phase
        kw={'levels':[0],'alpha':0.5,'linewidths':1}
        axs[1,i*2].contour(X*scale,Y*scale,null_phi2,colors='r',**kw)
        axs[1,i*2].contour(X*scale,Y*scale,null_phi3,colors='k',**kw)

        
        e = np.abs(eps)
        
        axs[1,i*2].plot(y1[:,0]*scale,y1[:,1]*scale,color='tab:blue',
                        lw=2,label=r'$O(\ve)$',zorder=3)
                        
                        
        # dumb thing to get arrows to work
        if i == 1: 
            line, = axs_temp.plot(y1[:,0]*scale,y1[:,1]*scale,
                                    label=r'$O(\ve)$',color='tab:blue')
            axs_temp.clear()
            arrow_locs = [.33,.66]
            add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)
            
        axs[1,i*2+1].plot(y1[:,0]*scale,t_phase[t_phase<=max_t_phase],
                            color='tab:blue',alpha=.8,lw=2,
                            label=r'$\phi_2$ $O(\ve)$',zorder=3)
        axs[1,i*2+1].plot(y1[:,1]*scale,t_phase[t_phase<=max_t_phase],
                        color='tab:blue',ls='--',alpha=.8,lw=2,
                        label=r'$\phi_3$ $O(\ve)$',zorder=3)

        if i == 0:
            lw = 1.5;alpha = .6
        else:
            lw = 2; alpha = 1
            
        # dumb thing to get arrows to work
        line, = axs_temp.plot(sol2.y.T[:,0]*scale,sol2.y.T[:,1]*scale,
                                label=r'$O(\ve&2)$',color='k')
        axs_temp.clear()
        if i == 0:
            arrow_locs = [.09,.4]
        else:
            pass
        add_arrow_to_line2D(axs[1,i*2],line,arrow_locs=arrow_locs)

        kw = {'alpha':0.8,'lw':1.5}
        axs[1,i*2].plot(sol2.y.T[:,0]*scale,sol2.y.T[:,1]*scale,
                        color='k',label=r'$O(\ve^2)$',**kw)
        axs[1,i*2+1].plot(y2[:,0]*scale,t_phase[t_phase<=max_t_phase],
                          color='k',label=r'$\phi_2$ $O(\ve^2)$',**kw)
        axs[1,i*2+1].plot(y2[:,1]*scale,t_phase[t_phase<=max_t_phase],
                          color='0.4',label=r'$\phi_3$ $O(\ve^2)$',**kw)
        
        # reverse limits
        axs[0,i*2+1].set_ylim(axs[0,i*2+1].get_ylim()[::-1])
        axs[1,i*2+1].set_ylim(axs[1,i*2+1].get_ylim()[::-1])
        
        # limits
        axs[0,i*2+1].set_ylim(T_list_full[i],0)
        axs[1,i*2+1].set_ylim(T_list_phase[i],0)
      
        for kk in range(2):
            
            # limits
            axs[kk,i*2+1].set_xlim(0-2*np.pi/10,2*np.pi+2*np.pi/10)
            
            # labels and ticks
            axs[kk,i*2].set_xticklabels(pi_label_short)
            axs[kk,i*2].set_xticks(np.arange(0,3,1)*pi)
            axs[kk,i*2+1].set_xticklabels(pi_label_short)
            axs[kk,i*2+1].set_xticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2].set_yticklabels(pi_label_short)
            axs[kk,i*2].set_yticks(np.arange(0,3,1)*pi)
            
            axs[kk,i*2+1].set_ylabel(r'$t$',labelpad=-.1)
            
            axs[kk,i*2].set_ylabel(r'$\phi_3$',labelpad=-.1)
            axs[kk,i*2].set_xlabel(r'$\phi_2$',labelpad=-.1)
            
            axs[kk,i*2+1].set_xlabel(r'$\phi_i$',labelpad=-.1)
        
    r,c = np.shape(axs)
    plt.close(fig_temp)
    # legends
    lgd = axs[0,1].legend(bbox_to_anchor=(1.5,1.1),borderpad=0.1,
                            labelspacing=0,handletextpad=-0.1,
                            framealpha=1,loc='upper right')
    for handle in lgd.legend_handles:
        handle.set_sizes([15])
        
    lgd2 = axs[1,1].legend(bbox_to_anchor=(1.65,1.2),
                        borderpad=0.1, labelspacing=0,handletextpad=0.3,
                        framealpha=1,handlelength=1,loc='upper right')
    for legobj in lgd2.legend_handles:
        legobj.set_linewidth(1.5)
        
    
    lgd3 = axs[1,0].legend(loc='lower right')
    for legobj in lgd3.legend_handles:
        legobj.set_linewidth(2)
        
    lgd4 = axs[1,2].legend(loc='lower right')
    for legobj in lgd4.legend_handles:
        legobj.set_linewidth(2)
    
    
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].set_title(labels[count],loc='left')
            
            count += 1
        
    for i in range(4):
        if i%2 == 0:
            t0 = axs[0,i].get_title()
            t1 = axs[1,i].get_title()
        
            t0 += r'Full $\ve={}$'.format(eps_list_full[int(i/2)])
            t1 += r'Phase $\ve={}$'.format(eps_list_phase[int(i/2)])
            
            axs[0,i].set_title(t0)
            axs[1,i].set_title(t1)
    
    plt.tight_layout()
    
    plt.subplots_adjust(hspace=.4,wspace=.5)
    
    
    
    return fig



def thalamic_eigs(example=1):
    
    var_names = var_names_thal
    pardict = pardict_thal
    kwargs = kwargs_thal

    if example == 1:
        pardict['ib'] = 0.8
        pardict['esyn'] = -100
        kwargs['NA'] = 505

        eps_list = np.linspace(0,.02,30)
    else:
        eps_list = np.linspace(-.0015,.001,100)
        kwargs['NA'] = 500
    
    T_init = 44.8
    LC_init = np.array([-.5927,0.99,0.507,.006,T_init])
    
    a = nBodyCoupling(Thalamic.rhs,Thalamic.coupling,
                      LC_init,var_names,pardict,**kwargs)
    
    # co = color order
    co = ['tab:blue','k']

    
    eig_o1 = np.zeros((len(eps_list),2),dtype=complex)
    eig_o2 = np.zeros((len(eps_list),2),dtype=complex)

    for i in range(len(eps_list)):
        eps = eps_list[i]

        z = np.array([11.58, 23.17])
        out1 = root(prhs_for_root, z, args=(a,eps,1),method='hybr')
        out2 = root(prhs_for_root, z, args=(a,eps,2),method='hybr')
        #print(out1.x,out2.x)
        eig_o1[i,:] = np.linalg.eig(jac(prhs,out1.x,a,eps,order=1))[0]
        eig_o2[i,:] = np.linalg.eig(jac(prhs,out2.x,a,eps,order=2))[0]

    fig,axs = plt.subplots(1,2,figsize=(8,2.5))

    axs[0].plot(eps_list,np.real(eig_o1[:,0]),color=co[0],label=r'$O(\ve)$')
    axs[0].plot(eps_list,np.real(eig_o2[:,0]),color=co[1],label=r'$O(\ve^2)$')

    axs[1].plot(eps_list,np.imag(eig_o1),color=co[0])
    axs[1].plot(eps_list,np.imag(eig_o2),color=co[1])

    axs[0].plot([eps_list[0],eps_list[-1]],[0,0],color='gray',ls='--',lw=1)

    for i in range(2):
        axs[i].set_xlabel(r'$\ve$')
        axs[i].set_title(labels[i],loc='left')

        axs[i].set_xlim(eps_list[0],eps_list[-1])

    t0 = axs[0].get_title()
    axs[0].set_title(t0+'Real')
    t1 = axs[1].get_title()
    axs[1].set_title(t1+'Imaginary')
    
    
    axs[0].legend()
    plt.tight_layout()

    return fig
    
    
def generate_figure(function, args, filenames, dpi=100):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape',dpi=dpi)
            else:
                fig.savefig(name,dpi=dpi,bbox_inches='tight')
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape',dpi=dpi)
        else:
            fig.savefig(filenames,dpi=dpi)

def main():

    #quick_plots_thalamic()

    # create figs directory if it doesn't exist
    if not(os.path.isdir('figs')):
        os.mkdir('figs')
    
    # listed in order of Figures in paper
    figures = [

        #(cgl_examples,[],['figs/f_cgl_examples.pdf']),
        #(cgl_eigs,[],['figs/f_cgl_eigs.pdf']),
        
        (thalamic_examples1,[],['figs/f_thalamic_examples1.png']),
        (thalamic_examples2,[],['figs/f_thalamic_examples2.png']),
        #(thalamic_eigs,[1],['figs/f_thal_eigs1.pdf']),
        #(thalamic_eigs,[2],['figs/f_thal_eigs2.pdf']),

    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
