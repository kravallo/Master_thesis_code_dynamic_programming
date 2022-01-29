# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize

class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):
        
        par = self.par

        # number of periods
        par.T = 400

        # Model parameters
        par.rho_good = 0.4
        par.rho_bad = 0.5
        par.beta = 0.96
        par.R = 1.04
        par.Y = 1
        par.sigma_eta = 0.2 
        par.mandate = 0
        par.O = 0  # operating costs
        par.single_payer_regime = 0
        par.print_convergence = 0
        
        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1  # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters

        par.Nm = 100 
        par.Na = 100
        par.Nm_b = 50
        par.Npremium = 20 #20
        par.Ntax = 5
        
        # shocks
        par.Neps = 8             # number of gauss hermite nodes for transitory shock in medical expenses psi 
        par.uniform_a_bad = 0    # lower bound for uniform interval for bad types
        par.uniform_a_good = 0.5 # lower bound for uniform interval for good types
        par.uniform_b     = 1    # upper bound for uniform interval

        # simulation
        par.simT = par.T
        par.simN = 10000
        par.M_ini = 0+1e-1
        
        # technical
        par.max_iter_solve = 5000       # maximum number of iterations when solving
        par.max_iter_sim = par.simT
        par.simulate_tol = 50
        par.solve_tol = 1e-1            
        
    def create_grids(self):

        par = self.par
        
        # Grid Health type
        par.grid_theta = np.linspace(0,1,2,dtype=int) 

        # Shocks
        par.unif_densf0 = lambda x: x/(par.uniform_b-par.uniform_a_bad) # probability density function for uniform distribution
        par.unif_densf1 = lambda x: x/(par.uniform_b-par.uniform_a_good) # probability density function for uniform distribution

        par.eps0, par.eps0_w = tools.quad_gauss(par.unif_densf0,par.uniform_a_bad,par.uniform_b,deg=par.Neps)
        par.eps1, par.eps1_w = tools.quad_gauss(par.unif_densf1,par.uniform_a_good,par.uniform_b,deg=par.Neps)

        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])    # Permanent income

        # Set seed
        par.seed = 2020
        np.random.seed(par.seed)
        
        # count number of shock nodes
        par.Nshocks = par.eps0_w.size 
        
    
    def simulate(self):
    
        
        # Initialize
        par = self.par
        sol = self.sol
        
        # check attributes of class sol
        #dir(sol)
        
        # Initialize
        class sim: pass
    
        #######################################################################
        # Initialize individual randomized starting values
        #######################################################################

        shape = (par.simT, par.simN)
        sim.m = par.M_ini*np.ones(shape)       #initial values for sim.m[0,:] = par.M_ini    
        sim.c = np.nan +np.zeros(shape)
        sim.a = sim.m  - sim.c      
        sim.z = np.zeros(shape)
        sim.theta = np.nan +np.zeros(shape)    #collect type of individual (constant over time)
        sim.eps = np.nan +np.zeros(shape)
        sim.p_plus = np.nan +np.zeros(shape)
        sim.ins_cost =  np.nan +np.zeros(shape)
        sim.claims = np.nan +np.zeros(shape)
        sim.z_lag = np.nan +np.zeros(shape)
        np.random.seed(par.seed)
        
        # draw random number that is then added to sim.m and sim.p
        m_rand = np.random.uniform(low=0, high=0.1, size=(1,par.simN)) # uniform distirbuted
        
        # add random number to first row of sim. 
        sim.m[0,:] = sim.m[0,:]+m_rand
        
        #######################################################################
        # Initialize individual fixed values and random number for comparison before first period 
        #######################################################################
        
        # Random numbers
        rnd = np.random.rand(par.simT,par.simN) # uniform distirbuted
        
        # allocate values of theta
        theta_draw = np.random.choice((0,1),size=par.simN, p = (1/3,2/3))    # good and bad types are equally likely to be drawn
        sim.theta = np.tile(theta_draw, (par.simT, 1))                       # repeat drawn values T times (constant over time)
        
        #######################################################################
        # Draw one combination of the two (type specific) shocks for each period 
        #######################################################################
        # draw indexes over shocks (can be the same for both types) weights of bad type are used to draw indexes but they are identical to weights of good type by design
        
        # draw indexes between 0 and Nshocks-1, with prob =  shock_weights_bad_type
        shocki = np.random.choice(par.Nshocks,(par.simT,par.simN),replace=True,p=par.eps0_w) 
        
        # shock to bad types theta = 0
        sim.eps0 = par.eps0[shocki]  #dim (T,N)
        
        # shock for good types theta = 1
        sim.eps1 = par.eps1[shocki]
        
        # check it has a mean of 1
        assert (abs(0.75-np.mean(sim.eps1)) < 1e-2), 'The mean is not 0.75 in the simulation of eps1'
        assert (abs(0.5-np.mean(sim.eps0)) < 1e-2), 'The mean is not 0.5 in the simulation of eps0'


        #######################################################################
        # Simulate by looping over periods and individuals
        #######################################################################
        
        sim.it = 0         # Number of iteration
        diff = 1000.0
        sim.delta_m = 1000.0 # Different between distribution of m and m next
        sim.delta_z = 1000.0 # Different between distribution of z and z next
        sim.delta_a = 1000.0

        # Loop over periods 
        for t in range(par.simT): 
            
            # store previous distribution
            
            if t == 0:
                #a_dist_prev = par.M_ini*np.ones(shape)  # corresponds to initialization of sim.a
                z_dist_prev = np.mean(sim.z[t,:])
                m_dist_prev = par.M_ini*np.ones(shape)  # corresponds to initialization of sim.m
                sim.z_lag[t,:] = np.nan
            else:
                #a_dist_prev = sim.a[t-1,:]
                z_dist_prev = np.mean(sim.z[t-1,:])
                m_dist_prev = sim.m[t-1,:]
                sim.z_lag[t,:] = sim.z[t-1,:]

            #loop over indidviduals,thus interpolation over a point given rho and theta of a given individuum and not over a vector
            for n in range(par.simN):
                
                # find individual specific type
                theta = sim.theta[t,n]
                
                # check if insurance is affordable
                premium = par.lambdaa_0*(1-theta) + theta *par.lambdaa_1
                
                # if insurance is not affordable
                if premium > sim.m[t,n]:
                    
                    # Consumption given no insurance choice 
                    C0 = tools.interp_linear_1d_scalar(par.grid_m,sol.C[0,theta,:],sim.m[t,n])   # sol.c[0,theta,:] is the converged one so no dimension for t
                    sim.c[t,n] = C0 
                    sim.z[t,n] = 0
                    
                # if insurance would be affordable
                else:
                    # Values of discrete choice 
                    V0 = tools.interp_linear_1d_scalar(par.grid_m,sol.V[0,theta,:],sim.m[t,n])    # sol.v[0,theta,:] is the converged one so no dimension for t
                    V1 = tools.interp_linear_1d_scalar(par.grid_m,sol.V[1,theta,:],sim.m[t,n])    
                    
                    # Choice probabilty for each person given state
                    _ , prob = tools.logsum(V0,V1,par.sigma_eta)   
                    
                    # Consumption of discrete choice 
                    C0 = tools.interp_linear_1d_scalar(par.grid_m,sol.C[0,theta,:],sim.m[t,n])   # sol.c[0,theta,:] is the converged one so no dimension for t
                    C1 = tools.interp_linear_1d_scalar(par.grid_m,sol.C[1,theta,:],sim.m[t,n])
                    
                    # Indicator function for not taking insurance
                    I = rnd[t,n] <= prob[0] 
        
                    # Consumption for people taking insurance
                    if I == False: 
                        sim.c[t,n] = C1   
                        sim.z[t,n] = 1
                    # Consumption for people not taking insurance
                    else:
                        sim.c[t,n] = C0 
                        sim.z[t,n] = 0
                
                # calculate claims vector in benchmark
                if par.single_payer_regime == 0: #only agents with z=1 are covered
                    sim.claims[t,n] = (par.Y - (sim.eps1[t,n]*theta + sim.eps0[t,n]*(1-theta)) ) *(sim.z[t,n])
                    
                # calculate claims vector in single payer case where every agent is covered by design
                else: 
                    sim.claims[t,n] = par.Y -(sim.eps1[t,n]*theta + sim.eps0[t,n]*(1-theta))
    
                #Next period
                if t<par.simT-1:  # if not last period
                
                ##################################################################################
                # Define transition functions
                ##################################################################################
                    
                    # transition for benchmark case
                    if par.single_payer_regime == 0:
                        
                        premium = par.lambdaa_0*(1-theta) + par.lambdaa_1*(theta)
                        sim.a[t,n] = get_a(sim.c[t,n],sim.m[t,n],sim.z[t,n],theta,par) 
                        sim.p_plus[t,n] = ((par.Y*sim.eps1[t,n])*theta + (par.Y*sim.eps0[t,n])*(1-theta))*(1-sim.z[t,n]) +  par.Y*(sim.z[t,n])
                        sim.m[t+1,n] = par.R*(sim.a[t,n]) + sim.p_plus[t,n] 
                    
                    # transition for single payer case with tax only but no premia
                    else:
                        sim.a[t,n] = sim.m[t,n]-sim.c[t,n] # scalar
                        sim.p_plus[t,n] = par.Y*(1-par.tax)
                        sim.m[t+1,n] = par.R*(sim.a[t,n]) + sim.p_plus[t,n] 
                        
            #######################################################################
            # simulate until distribution of m and z are stable
            #######################################################################
        
            sim.it += 1
            sim.delta_m = np.mean(sim.m[t,:]) - np.mean(m_dist_prev)  
            sim.delta_z = np.mean(sim.z[t,:]) - z_dist_prev

            # calculate diff between distribution of m in t and m in t-1
            if sim.it > 1:
                
                #select only for range of m between 1e-6 and 5
                data_a = np.where(np.logical_and(sim.m[t+1,:]>=1e-6, sim.m[t+1,:]<=5))
                data_b = np.where(np.logical_and(sim.m[t,:]>=1e-6, sim.m[t,:]<=5))

                # calculate histograms
                a = np.histogram(sim.m[t+1,data_a[0]], bins=np.linspace(0,5,50).tolist())
                b = np.histogram(sim.m[t,data_b[0]], bins=np.linspace(0,5,50).tolist())
                
                diff = [abs(_a - _b) for _a, _b in zip(a[0], b[0])]
                #print(f'diff: {np.amax(diff)}')


            #print("-----------------------------")
            #print(f"Simulated period {t} / iteration: {sim.it} | delta avg m: {np.round(sim.delta_m,5)}| delta z: {sim.delta_z}")   
            #print("-----------------------------")
        
            #if sim.delta_m <= par.simulate_tol and sim.delta_m <= par.simulate_tol:
            if sim.it > 1 and np.amax(diff) < par.simulate_tol: 
                
                print(f"Terminated because of sucessfull convergence of simulation for sim.it = {sim.it}")
                 
                # construct data frame
                if par.single_payer_regime == 1:
                    
                    delta_m = sim.m[t,:] - m_dist_prev
                    mt = sim.m[t,:]
                    mtminus = sim.m[t-1,:]
                    simc = sim.c[t,:]
                    sim_a = sim.a[t,:]
                    simp = sim.p_plus[t-1,:]
                    claims = sim.claims[t,:]
                    theta = sim.theta[t,:]
                    ans = [delta_m,mt,mtminus,simc,sim_a,simp,claims,theta]
                    ans_ = np.transpose(np.array(ans))
                    sim.df = pd.DataFrame(ans_)
                    
                    
                else:
                    
                    delta_m = sim.m[t,:] - m_dist_prev
                    mt = sim.m[t,:]
                    mtminus = sim.m[t-1,:]
                    simc = sim.c[t,:]
                    sim_a = sim.a[t,:]
                    simp = sim.p_plus[t-1,:]
                    claims = sim.claims[t,:]
                    theta = sim.theta[t,:]
                    z = sim.z[t,:]
                    z_lag = sim.z_lag[t,:]
                    ans = [delta_m,mt,mtminus,simc,sim_a,simp,claims,theta,z,z_lag]
                    ans_ = np.transpose(np.array(ans))
                    sim.df = pd.DataFrame(ans_)
                    
                
                break
                
            elif sim.it == par.max_iter_sim:
            #if sim.it == par.max_iter_sim:
                print(f"Terminated because maximum number of iterations exceeded sim.it = {sim.it}")
                
                # construct data frame
                if par.single_payer_regime == 1:
                    
                    delta_m = sim.m[t,:] - m_dist_prev
                    mt = sim.m[t,:]
                    mtminus = sim.m[t-1,:]
                    simc = sim.c[t,:]
                    sim_a = sim.a[t,:]
                    simp = sim.p_plus[t-1,:]
                    claims = sim.claims[t,:]
                    theta = sim.theta[t,:]
                    ans = [delta_m,mt,mtminus,simc,sim_a,simp,claims,theta]
                    ans_ = np.transpose(np.array(ans))
                    sim.df = pd.DataFrame(ans_)
                    
                    
                else:
                    
                    delta_m = sim.m[t,:] - m_dist_prev
                    mt = sim.m[t,:]
                    mtminus = sim.m[t-1,:]
                    simc = sim.c[t,:]
                    sim_a = sim.a[t,:]
                    simp = sim.p_plus[t-1,:]
                    claims = sim.claims[t,:]
                    theta = sim.theta[t,:]
                    z = sim.z[t,:]
                    z_lag = sim.z_lag[t,:]
                    ans = [delta_m,mt,mtminus,simc,sim_a,simp,claims,theta,z,z_lag]
                    ans_ = np.transpose(np.array(ans))
                    sim.df = pd.DataFrame(ans_)
                    
                
                break
            
                
            else:
                pass
            
        return sim
        
    
    def solve_VFI_inf_horizon(self):

        # Initalize
        par = self.par
        sol = self.sol

        shape = (2,2,par.Nm)
        sol.C = np.nan+np.zeros(shape)
        sol.M = np.nan+np.zeros(shape)
        sol.V = np.nan+np.zeros(shape)

        # set starting V_next and C_next by looping over state variables
        for z in range(2): 

            for theta in range(2):
                
                #assign type dependend risk aversion 
                if theta == 1:
                    rho = par.rho_good
                else:
                    rho = par.rho_bad

                # assign initial consumption and value (used as first V+ in while loop)
                sol.C[z,theta,:] = par.grid_m.copy() 
                sol.M[z,theta,:] = par.grid_m.copy() 
                sol.V[z,theta,:] = util(sol.C[z,theta,:],rho) 

        sol.it = 0   #Number of iteration
        sol.delta = 1000.0 #Different between V+ and V

        while (sol.delta >= par.solve_tol and sol.it < par.max_iter_solve):

            V_next = sol.V.copy()

            # find V
            for im,m in enumerate(par.grid_m):  

                for theta in par.grid_theta:
                        
                    for z in range(2):  
                            
                        # calculate upper bound for consumption by calling get_a func with c = 0
                        upper_bound = get_a(0,m,z,theta,par)
                        
                        # call the optimizer
                        obj_fun = lambda x: - value_of_choice(x,z,m,theta,par.grid_m,V_next,par)
                        res = optimize.minimize_scalar(obj_fun, bounds=[0,upper_bound], method='bounded') 

                        sol.V[z,theta,im] = -res.fun
                        sol.C[z,theta,im] = res.x

            # opdate delta and it
            sol.it += 1
            sol.delta = np.amax(abs(sol.V - V_next))  

            # print each iteration to console
            if par.print_convergence == 1:
                print(f"Iteration: {sol.it} | delta: {sol.delta}")
            
            # print termination notification
            if sol.it == par.max_iter_solve:
                print(f"Terminated because maximum number of iterations exceeded")
        
        # only print if converged
        if sol.it < par.max_iter_solve:        
            print(f"Terminated because of sucessfull convergence for iteration = {sol.it}")
                
        return sol


def value_of_choice(x,z,m,theta,m_next,V_next,par):

    #"unpack" c
    if type(x) == np.ndarray: # vector-type: depends on the type of solver used
        c = x[0] 
    else:
        c = x

    # chose health process according to theta
    if theta == 0:
        rho = par.rho_bad
        epsgrid = par.eps0
        premium = par.lambdaa_0
        weight_health_shock = par.eps0_w
    else: 
        rho = par.rho_good
        epsgrid = par.eps1
        premium = par.lambdaa_1
        weight_health_shock = par.eps1_w
        
    #Expected Value next period given states and choice
    EV_next = 0.0 #Initialize
   
    shape = (2,)
    v_plus = np.nan+np.zeros(shape)   

    # calculate a
    a = get_a(c,m,z,theta,par)
    
    # calculate EV next
    for j,eps in enumerate(epsgrid):
        
        # can't afford insurance: P_plus is equal to the insurance case
        if premium > m:
            P_plus = par.Y*eps*(1-0) + par.Y*(0) #think of z being 0 if not affordable
            M_plus = par.R*(a) + P_plus  
            
        # could afford insurance
        else:
            P_plus = par.Y*eps*(1-z) + par.Y*(z)
            M_plus = par.R*(a) + P_plus  

        #Range over insured z=1 and not_insured z=0 next period
        for i in range(2): 

            #Interpolation for utility
            interp_result_v = tools.interp_linear_1d_scalar(m_next,V_next[i,theta,:],M_plus) 
            # Choice specific value
            v_plus[i,] = interp_result_v

        # choice probability and expected value given taste shock
        V_plus, prob = tools.logsum(v_plus[0,],v_plus[1,],par.sigma_eta)            

        # weight on the shock 
        w = weight_health_shock[j]
        EV_next +=w*V_plus 

    # Value of choice
    V_guess = util(c,rho)+par.beta*EV_next

    return V_guess
        

def get_a(c,m,z,theta,par):
    
    # chose premium according to theta
    if theta == 0:
        premium = par.lambdaa_0
    else: 
        premium = par.lambdaa_1
        
    # if insurance is unaffordable set z = 0 so that premium is not substracted
    if premium > m: 
        
        # force z to be 0
        z = 0  
       
        # can't afford to pay mandate either so set mandate to 0 as no tax penalty should apply
        a = m - c - premium*z 
    
    else:
        a = m - c - premium*z - (1-z)*par.mandate
    
    #assert (a < 0), 'a is negative for c: {0}, m: {1}, z: {2}, theta: {3}, premium: {4}'.format(c,m,z,theta, premium) 
    
    return a
        

def util(c,rho):   
    
    assert (1.0-rho) != 0 , "Divide by 0."
    
    u_fun = ((c)**(1.0-rho))/(1.0-rho) 
    return u_fun 




