---
description: |
    API documentation for modules: nBodyCoupling, Thalamic, CGL.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...


    
# Module `nBodyCoupling` {#nBodyCoupling}

StrongCoupling.py computes the higher-order interaction functions from
Park and Wilson 2020 for $N=2$ models and one Floquet multiplier.
In broad strokes, this library computes functions in the following order:

* Use the equation for $\Delta x$ (15) to produce a hierarchy of
ODEs for $g^{(k)}$ and solve. (Wilson 2020)
* Do the same using (30) and (40) to generate a hierarchy of ODEs
for $Z^{(k)}$ and $I^{(k)}$, respectively. (Wilson 2020)
* Solve for $\phi$ in terms of $\theta_i$, (13), (14) (Park and Wilson 2020)
* Compute the higher-order interaction functions (15) (Park and Wilson 2020)

Assumptions:
-homogeneous vector fields
-NO SELF COUPLING

Notes:
- <code>pA</code> requires endpoint=False. make sure corresponding <code>dxA</code>s are used.




    
## Functions


    
### Function `augment_axes` {#nBodyCoupling.augment_axes}




>     def augment_axes(
>         a,
>         dims,
>         repeat=True
>     )


fill a with repeat elements in new axes.
new axes are in dims.
set repeat=True for function values
set repeat=False for Fourier.

    
### Function `collect_nsep` {#nBodyCoupling.collect_nsep}




>     def collect_nsep(
>         var_list,
>         fn_list
>     )


extract the sep and nsep

    
### Function `equal_ignore_order` {#nBodyCoupling.equal_ignore_order}




>     def equal_ignore_order(
>         a,
>         b
>     )


Use only when elements are neither hashable nor sortable!

    
### Function `flatten` {#nBodyCoupling.flatten}




>     def flatten(
>         xs
>     )




    
### Function `get_slice_dim` {#nBodyCoupling.get_slice_dim}




>     def get_slice_dim(
>         n,
>         dims
>     )


n: dimension of h function
dims: dimensions to update with coefficients

    
### Function `module_exists` {#nBodyCoupling.module_exists}




>     def module_exists(
>         module_name
>     )




    
### Function `unique_v_f1` {#nBodyCoupling.unique_v_f1}




>     def unique_v_f1(
>         vars_list,
>         fns_list
>     )




    
### Function `unique_vf` {#nBodyCoupling.unique_vf}




>     def unique_vf(
>         var_list,
>         fn_list
>     )




    
### Function `varcount` {#nBodyCoupling.varcount}




>     def varcount(
>         var_list
>     )


get unique vars in var_list

    
### Function `varcount_min` {#nBodyCoupling.varcount_min}




>     def varcount_min(
>         var_list
>     )


count (not necessarily unique) vars in lists within var_list


    
## Classes


    
### Class `nBodyCoupling` {#nBodyCoupling.nBodyCoupling}




>     class nBodyCoupling(
>         rhs,
>         coupling,
>         LC_init,
>         var_names,
>         pardict,
>         **kwargs
>     )


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
    LC_init: list or numpy array.
        initial condition of limit cycle (must be found manually).
        XPP is useful, otherwise integrate your RHS for various
        initial conditions for long times and extract an initial
        condition close to the limit cycle.

    LC_long_sim_time: float or int.
        Simulation time to compute the trjactory of an initial 
        condition near the limit cycle solution. Used to
        estimate limit cycle initial condition for use 
        in the Newton method. Default: 500
    LC_eps_time: float.
        Approximation of the time error estimate in Newton's
        method, e.g., (difference between initial conditions)/
        (LC_eps_time). Default: 1e-4
    LC_tol: float:
        Error tolerance to stop Newton's method when computing
        the limit cycle. Default: 1e-13

    var_names: list.
        list of variable names as strings
    pardict: dict.
        dictionary of parameter values. dict['par1_val'] = float.
        Make sure to use par_val format, where each parameter name is
        followed by _val.
    recompute_LC: bool.
        If True, recompute limit cycle. If false, load limit cycle if
        limit cycle data exists. Otherwise, compute. Default: False.

    recompute_monodromy: bool.
        If true, recompute kappa, the FLoquet multiplier using the
        monodromy matrix. If false, load kappa if data exists,
        otherwise compute. Default: False.
    recompute_g_sym: bool.
        If true, recompute the symbolic equations for g^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_g: bool.
        If true, recompute the ODEs for g^k. If false,
        load the data for g^k if they exist in storage.
        Otherwise, compute. Default: False.

    recompute_het_sym: bool.
        If true, recompute the symbolic equations for z^k and i^k.
        If false, load the symbolic equations if they exist in
        storage. Otherwise, compute. Default: False.
    recompute_z: bool.
        If true, recompute the ODEs for z^k. If false,
        load the data for z^k if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_i: bool.
        If true, recompute the ODEs for i^k. If false,
        load the data for i^k if they exist in storage.
        Otherwise, compute. Default: False.

    recompute_k_sym: bool.
        If true, recompute the symbolic equations for K^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_p_sym: bool.
        If true, recompute the symbolic equations for p^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_k_sym: bool.
        If true, recompute the symbolic equations for H^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.

    recompute_gij: bool
        recompute the pairwise functions for the psi expansions

    recompute_h: bool.
        If true, recompute the H functions for H^k. If false,
        load the data equations if they exist in storage.
        Otherwise, compute. Default: False.
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
        along tLC.
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







    
#### Methods


    
##### Method `fLam_ij` {#nBodyCoupling.nBodyCoupling.fLam_ij}




>     def fLam_ij(
>         self,
>         fn,
>         i,
>         j
>     )


interp2db object

    
##### Method `fmod` {#nBodyCoupling.nBodyCoupling.fmod}




>     def fmod(
>         self,
>         fn
>     )


fn has mod built-in

input function-like. usually interp1d object

needed to keep lambda input variable unique to fn.

otherwise lambda will use the same input variable for 
all lambda functions.

    
##### Method `generate_g` {#nBodyCoupling.nBodyCoupling.generate_g}




>     def generate_g(
>         self,
>         k,
>         het_vec
>     )


generate Floquet eigenfunctions g

uses Newtons method

    
##### Method `generate_gij` {#nBodyCoupling.nBodyCoupling.generate_gij}




>     def generate_gij(
>         self
>     )


calculate convolution terms g_{ij}
for oscillator i and order k.

    
##### Method `generate_h_brute` {#nBodyCoupling.nBodyCoupling.generate_h_brute}




>     def generate_h_brute(
>         self,
>         i,
>         k,
>         idxarr1,
>         idxarr2
>     )


use this if nyquist frequency is too high 
brute force + parallelization across CPU cores.

    
##### Method `generate_h_fourier` {#nBodyCoupling.nBodyCoupling.generate_h_fourier}




>     def generate_h_fourier(
>         self,
>         i,
>         k
>     )




    
##### Method `generate_het_sym` {#nBodyCoupling.nBodyCoupling.generate_het_sym}




>     def generate_het_sym(
>         self
>     )


Generate heterogeneous terms for integrating the Z_i and I_i terms.

###### Returns

None.

    
##### Method `generate_i` {#nBodyCoupling.nBodyCoupling.generate_i}




>     def generate_i(
>         self,
>         k,
>         het_vec
>     )


i0 equation is stable in forwards time
i1, i2, etc equations are stable in backwards time.

    
##### Method `generate_k_sym` {#nBodyCoupling.nBodyCoupling.generate_k_sym}




>     def generate_k_sym(
>         self
>     )


generate terms involving the coupling term (see K in paper).

    
##### Method `generate_limit_cycle` {#nBodyCoupling.nBodyCoupling.generate_limit_cycle}




>     def generate_limit_cycle(
>         self
>     )




    
##### Method `generate_z` {#nBodyCoupling.nBodyCoupling.generate_z}




>     def generate_z(
>         self,
>         k,
>         het_vec
>     )




    
##### Method `get_index` {#nBodyCoupling.nBodyCoupling.get_index}




>     def get_index(
>         self,
>         fn
>     )


fn: sympy variable either x0 or p0[0]

returns (0,) or (0,0), respectively.

    
##### Method `interp_lam` {#nBodyCoupling.nBodyCoupling.interp_lam}




>     def interp_lam(
>         self,
>         k,
>         fn_dict,
>         fn_type='z'
>     )


it is too slow to call individual interpolated functions
in the symbolic heterogeneous terms.
since the heterogeneous terms only depend on t, just make
and interpolated version and use that instead so only 1 function
is called for the het. terms per iteration in numerical iteration.

    
##### Method `load_g` {#nBodyCoupling.nBodyCoupling.load_g}




>     def load_g(
>         self
>     )


load all Floquet eigenfunctions g or recompute

    
##### Method `load_g_sym` {#nBodyCoupling.nBodyCoupling.load_g_sym}




>     def load_g_sym(
>         self
>     )




    
##### Method `load_gij` {#nBodyCoupling.nBodyCoupling.load_gij}




>     def load_gij(
>         self
>     )


load or (re)calculate gij after implementing in p_sym

    
##### Method `load_h` {#nBodyCoupling.nBodyCoupling.load_h}




>     def load_h(
>         self
>     )




    
##### Method `load_h_sym` {#nBodyCoupling.nBodyCoupling.load_h_sym}




>     def load_h_sym(
>         self
>     )


also compute h lam

    
##### Method `load_het_sym` {#nBodyCoupling.nBodyCoupling.load_het_sym}




>     def load_het_sym(
>         self
>     )




    
##### Method `load_i` {#nBodyCoupling.nBodyCoupling.load_i}




>     def load_i(
>         self
>     )


load all IRCs i or recomptue

    
##### Method `load_k_sym` {#nBodyCoupling.nBodyCoupling.load_k_sym}




>     def load_k_sym(
>         self
>     )


k[i]['type_'+key][j][k]
oscillator i, function type, oscillator j, order k, variable name 'key'.

c contains the derivatives of the coupling fn given
oscillators i,j and variable name 'key'.
c[i][key][j]
same as k just without terms collected.

    
##### Method `load_limit_cycle` {#nBodyCoupling.nBodyCoupling.load_limit_cycle}




>     def load_limit_cycle(
>         self
>     )




    
##### Method `load_monodromy` {#nBodyCoupling.nBodyCoupling.load_monodromy}




>     def load_monodromy(
>         self
>     )


if monodromy data exists, load. if DNE or 
recompute required, compute here.

    
##### Method `load_p` {#nBodyCoupling.nBodyCoupling.load_p}




>     def load_p(
>         self
>     )


insert lower order p[i] into higher order terms

    
##### Method `load_p_sym` {#nBodyCoupling.nBodyCoupling.load_p_sym}




>     def load_p_sym(
>         self
>     )


generate/load the het. terms for psi ODEs.
    
to be solved using integrating factor meothod.        
p[i]['sym'][k] is the forcing function for oscillator i of order k

    
##### Method `load_z` {#nBodyCoupling.nBodyCoupling.load_z}




>     def load_z(
>         self
>     )


load all PRCs z or recompute

    
##### Method `monodromy` {#nBodyCoupling.nBodyCoupling.monodromy}




>     def monodromy(
>         self,
>         t,
>         z
>     )


calculate right-hand side of system


$\dot \Phi = J\Phi, \Phi(0)=I$,

where $\Phi$ is a matrix solution

jacLC is the jacobian evaluated along the limit cycle

    
##### Method `numerical_jac` {#nBodyCoupling.nBodyCoupling.numerical_jac}




>     def numerical_jac(
>         self,
>         fn,
>         x,
>         eps=1e-07
>     )


return numerical Jacobian function

    
##### Method `save_temp_figure` {#nBodyCoupling.nBodyCoupling.save_temp_figure}




>     def save_temp_figure(
>         self,
>         data,
>         k,
>         fn='plot',
>         path_loc='figs_temp/'
>     )


data should be (TN,dim)



    
# Module `Thalamic` {#Thalamic}

Example: Thalamic model from Wilson and Ermentrout RSTA 2019,
Rubin and Terman JCNS 2004




    
## Functions


    
### Function `coupling` {#Thalamic.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='val'
>     )


Synaptic coupling function between Thalamic oscillators.

E.g.,this Python function is the function $G(x_i,x_j)$
in the equation
$\frac{dx_i}{dt} = F(x_i) + \varepsilon G(x_i,x_j)$

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

    
### Function `coupling_mat` {#Thalamic.coupling_mat}




>     def coupling_mat(
>         N,
>         option='val'
>     )


define coupling matrix.

    
### Function `eval_lam_arr` {#Thalamic.eval_lam_arr}




>     def eval_lam_arr(
>         pts,
>         lam=None
>     )


use this for vectorizing lamban until i figure out something better

    
### Function `main` {#Thalamic.main}




>     def main()




    
### Function `rhs` {#Thalamic.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='val'
>     )


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




    
# Module `CGL` {#CGL}

Example: Complex Ginzburgh-Landau (CGL) model from Wilson and Ermentrout RSTA
2019




    
## Functions


    
### Function `coupling` {#CGL.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='value'
>     )


Diffusive coupling function between Complex Ginzburgh Landau
(CGL) oscillators.

E.g.,this Python function is the function $G(x_i,x_j)$
in the equation
$\frac{dx_i}{dt} = F(x_i) + \varepsilon \sum_{j=1}^N G(x_i,x_j)$

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

    
### Function `coupling_mat` {#CGL.coupling_mat}




>     def coupling_mat(
>         N,
>         option='val'
>     )


define coupling matrix.

    
### Function `main` {#CGL.main}




>     def main()




    
### Function `rhs` {#CGL.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='value'
>     )


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




Generated by *pdoc* 0.10.0 (<https://pdoc3.github.io>).
