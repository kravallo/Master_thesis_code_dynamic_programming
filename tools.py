import numpy as np
from numpy.linalg import eigh
from numba import njit, int64, double, boolean, int32,void
import math
#Source: Jeppe Druedahl Website & Exercises


def logsum(v1,v2,sigma):

    # setup
    V = np.array([v1, v2])

    # Maximum over the discrete choices
    mxm = V.max(0)

    # check the value of sigma
    if abs(sigma) > 1e-10:

        # numerically robust log-sum
        log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))
    
        # d. numerically robust probability
        prob = np.exp((V- log_sum) / sigma)    

    else: # No smmothing --> max-operator
        id = V.argmax(0)    #Index of maximum
        log_sum = mxm
        prob = np.zeros((v1.size*2))
        I = np.cumsum(np.ones((v1.size,1)))+id*(v1.size)-1
        I = I.astype(int)  # change type to integer
        prob[I] = 1

        prob = np.reshape(prob,(2,v1.size),'A')

    return log_sum,prob


# interpolation functions:
@njit(int64(int64,int64,double[:],double))
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(double(double[:],double[:],double))
def interp_linear_1d_scalar(grid,value,xi):
    """ raw 1D interpolation """

    # a. search
    ix = binary_search(0,grid.size,grid,xi)
    
    # b. relative positive
    rel_x = (xi - grid[ix])/(grid[ix+1]-grid[ix])
    
    # c. interpolate
    return value[ix] + rel_x * (value[ix+1]-value[ix])

@njit
def interp_linear_1d(grid,value,xi):

    yi = np.empty(xi.size)

    for ixi in range(xi.size):

        # c. interpolate
        yi[ixi] = interp_linear_1d_scalar(grid,value,xi[ixi])
    
    return yi


@njit(double(double[:],double[:],double[:,:],double,double,int32,int32),fastmath=True)
def _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded
    
    2d interpolation for one point with known location
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
    Returns:
        yi (double): output
    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right                    
            nom += nom_1*nom_2*value[j1+k1,j2+k2]

    return nom/denom


@njit(double(double[:],double[:],double[:,:],double,double),fastmath=True)
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded


    2d interpolation for one point
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
    Returns:
        yi (double): output
    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)


@njit(double[:](double[:],double[:],double[:,:],double[:],double[:]),fastmath=True)
def interp_2d_vec(grid1,grid2,value,xi1,xi2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded

    
    2d interpolation for vector of points
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector
    """
    shape = (xi1.size)
    yi = np.nan+np.zeros(shape)

    for i in range(xi1.size):
        yi[i] = interp_2d(grid1,grid2,value,xi1[i],xi2[i])

    return yi


# State space
def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between with unequal spacing
    phi = 1 -> eqaul spacing
    phi up -> more points closer to minimum
    """
    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def GaussHermite_lognorm(sigma,n):

    x, w = gauss_hermite(n)
    # adjust accordingly to the distribution X is drawn from. Here Normal Distribution
    x = np.exp(x*math.sqrt(2)*sigma - 0.5*sigma**2)    
    w = w / math.sqrt(math.pi)

    # assert a mean of one
    assert(1 - np.sum(w*x) < 1e-8 ), 'The mean in GH-lognorm is not 1'
    return x, w


def GaussHermite_norm(mu,sigma,n):

    x_gauss,w_gauss  = gauss_hermite(n)
    # adjust accordingly to the distribution X is drawn from. Here standard Gaussian
    x_gauss = np.sqrt(2)*sigma*x_gauss + mu
    w_gauss = w_gauss/np.sqrt(np.pi)

    return x_gauss, w_gauss

def gaussquad(n):

    b= np.zeros(n-1)

    for i in range(np.size(b) ):
        b[i]=(i+1)/np.sqrt(4*(i+1)*(i+1)-1)

    J = np.diag(b,-1)+np.diag(b,1)
    x,ev = eigh(J); w=1*ev[0]*ev[0]
    return(x,w)


def quad_gauss(f,a,b,deg):
    # source:
    #https://www2.math.ethz.ch/education/bachelor/lectures/fs2013/other/nm_pc/Ch07.pdf

    # getGausspointsfor[-1,1]

    [gx,w]=gaussquad(deg)

    #transform to [a,b]

    x = 0.5*(b-a)*gx+0.5*(a+b)
    
    # manually set weights to 1/n as equally distributed
    w = np.repeat(1/deg,deg)

    return x,w

@njit(double(double[:],double[:],double[:],double[:,:,:],double,double,double,int32,int32,int32),fastmath=True)
def _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3):
    """ 3d interpolation for one point with known location
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
    Returns:
        yi (double): output
    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right               
                nom += nom_1*nom_2*nom_3*value[j1+k1,j2+k2,j3+k3]

    return nom/denom

@njit(double(double[:],double[:],double[:],double[:,:,:],double,double,double),fastmath=True)
def interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3):
    """ 3d interpolation for one point
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
    Returns:
        yi (double): output
    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)

    return _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3)

@njit(void(double[:],double[:],double[:],double[:,:,:],double[:],double[:],double[:],double[:]),fastmath=True)
def interp_3d_vec(grid1,grid2,grid3,value,xi1,xi2,xi3,yi):
    """ 3d interpolation for vector of points
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        xi3 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector
    """
    #shape = (xi1.size)
    #yi = np.nan+np.zeros(shape)
    
    for i in range(yi.size):
        yi[i] = interp_3d(grid1,grid2,grid3,value,xi1[i],xi2[i],xi3[i])
        
    #return yi   


     
        
        
@njit(double(double[:],double[:],double[:],double[:],double[:,:,:,:],double,double,double,double,int32,int32,int32,int32),fastmath=True)
def _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4):
    """ 4d interpolation for one point with known location
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid
    Returns:
        yi (double): output
    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    nom_4_left = grid4[j4+1]-xi4
    nom_4_right = xi4-grid4[j4]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right  
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right  
                    nom += nom_1*nom_2*nom_3*nom_4*value[j1+k1,j2+k2,j3+k3,j4+k4]

    return nom/denom

@njit(double(double[:],double[:],double[:],double[:],double[:,:,:,:],double,double,double,double),fastmath=True)
def interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4):
    """ 4d interpolation for one point
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
    Returns:
        yi (double): output
    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)
    j4 = binary_search(0,grid4.size,grid4,xi4)

    return _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4)



@njit(void(double[:],double[:],double[:],double[:],double[:,:,:,:],double[:],double[:],double[:],double[:],double[:]),fastmath=True)
def interp_4d_vec(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,yi):
    """ 4d interpolation for vector of points
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        xi3 (numpy.ndarray): input vector
        xi4 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector
    """

    for i in range(yi.size):
        #yi[i] = interp_3d(grid1,grid2,grid3,value,xi1[i],xi2[i],xi3[i])  #from 3d
        yi[i] = interp_4d(grid1,grid2,grid3,grid4,value,xi1[i],xi2[i],xi3[i],xi4[i])
        