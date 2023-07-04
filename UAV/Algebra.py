#!/usr/bin/env
import numpy as np
# from  scipy.optimize import minimize_scalar



def EVD(DM,final_dimension):
    '''
    Eigenvalue decomposition function
    INPUT: squared distance matrix, final dimension for the index reduction
    RETURN: reduced matrix
    '''
    n = len(DM) #since it is a square matrix, no need to specify len for columns or rows

    # Centering Matrix definition
    H = np.eye(n) - np.ones((n,n))/n

    # Double centered matrix
    B = -1/2*H@DM@H

    # Eiegenvalue decomposition
    ev, EV = np.linalg.eig(B)

    LAMBDA = np.eye(final_dimension)
    U = np.zeros((n,final_dimension))

    for i in range(final_dimension):
        # Search for the heighest eigenvalue. Put it into lambda and its associated vector in U.
        # Eventually set the eigenvalue to -1000 to find the other heighest eigenvalues
        ind  = np.argmax(ev)
        LAMBDA[i,i] = ev[ind].real
        U[:,i]  = EV[:,ind].real
        ev[ind] = -1000

    S_star = np.sqrt(LAMBDA)@U.T

    if final_dimension == 2:
        S_star = np.append(S_star, np.zeros([1,S_star.shape[1]]),axis=0)

    return S_star


def remove_offset(S, S_star):
    '''
    Function to reduce the offset between the reduced matrix and the anchor's original position
    set verbose=1 to print more data
    INPUT: S (original coordinates vector), S* (new estimated coordinates vector)
    RETURN: S* without offset, vector of offset
    '''
    # Find the offset between the 2 anchors
    displX = S[0] - S_star[0,0]
    displY = S[1] - S_star[1,0]
    displZ = S[2] - S_star[2,0]


    # Generate a displacement matrix
    displacement_matrix = np.array([[displX for _ in range(len(S_star[0,:]))],
                                    [displY for _ in range(len(S_star[0,:]))],
                                    [displZ for _ in range(len(S_star[0,:]))]])
    
    
    return S_star + displacement_matrix, displacement_matrix
    

def move(DIM, N, movement='anchor'):
    DELTA = np.zeros((3,N))

    if movement == 'all':
        DELTA[:DIM,1:] = [[np.random.normal() for _ in range(N-1)] for _ in range(DIM)]
    elif movement == 'anchor':
        DELTA[:DIM,0]  = [np.random.rand() for _ in range(DIM)]

    return DELTA


def DM_from_platoon(platoon):
    '''
    Build a distance matrix from a platoon of robots. Robot objects are required
    INPUT: list of Robot objects
    RETURN: distance matrix
    '''
    distance_matrix = np.zeros((len(platoon),len(platoon)))

    for i in range(len(platoon)):
        for j in range(len(platoon)):
            distance_matrix[i,j] = platoon[i].get_distance(platoon[j])

    return distance_matrix
    

def DM_from_platoon2(platoon):
    '''
    Build a squared distance matrix from a platoon of robots. Robot objects are required
    INPUT: list of Robot objects
    RETURN: squared distance matrix
    '''
    distance_matrix = np.zeros((len(platoon),len(platoon)))

    for i in range(len(platoon)):
        for j in range(len(platoon)):
            distance_matrix[i,j] = platoon[i].get_distance(platoon[j])**2

    return distance_matrix


def DM_from_S(S):
    '''
    Build a distance matrix from a vector of coordinates. 
    INPUT: S coordinates vector
    RETURN: distance matrix
    '''
    m = DM_from_S2(S)
    return np.power(m,1/2)


def DM_from_S2(S):
    '''
    Build a squared distance matrix from a vector of coordinates. 
    INPUT: S coordinates vector
    RETURN: squared distance matrix
    '''
    e   = np.ones((1,len(S[0,:]))).T
    
    Phi_prime = np.array([np.diag(S.T@S)]).T    
    DM_prime = Phi_prime@e.T - 2*S.T@S + e@Phi_prime.T
    
    return DM_prime


def noise_matrix(type, dim, features=0):
    '''
    Build a Gaussian noise matrix.
    INPUT: Dimension of the square matrix, mean mu, variance sigma
    RETURN: nooise matrix
    '''
    m = np.zeros((dim,dim))

    if (type == 'none' or type == 0): return m
    elif (type=='gaussian'):
        if features:
            mu, sigma = features[0], features[1]
        else: mu, sigma = 0, 0.01

        for i in range(dim):
            for j in range(dim):
                if (i!=j): m[i,j] = np.random.normal(mu,sigma)

    return m


def square(matrix):
    '''
    Elevate each element of a matrix to the power of two
    INPUT: matrix
    RETURN: square matrix
    '''
    return np.power(matrix,2)
    

def expected_value(matrix,noise='gaussian'):
    '''
    Create a symmetric square matrix using the expected value operator
    INPUT: asymmetric square matrix
    RETURN: symmetrix matrix
    '''

    if (noise == 'gaussian'):
        for i in range(len(matrix[:,0])):
            for j in range(len(matrix[0,:])):
                matrix[i,j] = matrix[j,i] = 1/2*(matrix[i,j] + matrix[j,i])
        return matrix
    else:
        raise Exception("Different noise will be further implemented. Please assume Gaussian for now") 
    

def rotation_matrix(theta):
    '''
    Rotational matrix in non-homogenous coordinates
    INPUT: angle in radians
    RETURN: 2D rotational matrix
    '''
    return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0], [0,0,1]])


def get_theta(DM,DM_prime,S_star,displ,index=1,approx = 0):
    '''
    Function to get the solution to the system. The solution is unique for the no-noise case.
    The solution can be uniquely identified if n UAVs > 3.
    INPUT: Distance matrices, S* and displacement. approx parameter to get an approximated value
    RETURN: theta as unique solution of the system.
    '''
    deltaX = displ[0,0]
    deltaY = displ[1,0]

    # Definition of the system
    a2 = DM[0,index] - DM_prime[0,index] + deltaX**2 + deltaY**2
    b2 = -2*(S_star[0,index]*deltaX + S_star[1,index]*deltaY)         
    c2 =  2*(S_star[0,index]*deltaY - S_star[1,index]*deltaX)     

    a3 = DM[0,index+1] - DM_prime[0,index+1] + deltaX**2 + deltaY**2
    b3 = -2*(S_star[0,index+1]*deltaX + S_star[1,index+1]*deltaY)     
    c3 =  2*(S_star[0,index+1]*deltaY - S_star[1,index+1]*deltaX)     

    # Solution
    sinTheta = (a3*b2-a2*b3)/(b3*c2-b2*c3)
    cosTheta = (a2*c3-a3*c2)/(b3*c2-b2*c3)

    # Normalization of the solution
    mod = np.sqrt(sinTheta**2+cosTheta**2)
    sinTheta /= mod
    cosTheta /= mod

    theta = np.arctan2(sinTheta,cosTheta)
    if approx == 1:   atheta = round(theta,4)
    elif approx == 2: atheta = round(theta,3)


    if approx: return atheta
    else:      return theta


def MDS(DM,DM_prime,DM_prime2, anchor_coord, DeltaS_prime, DeltaS_prime2, DIM=2,noise=0):
    '''
    MultiDimensional Scaling algorithm, according to the paper.
    INPUT: Distance matrices and variations in anchor position (DeltaS and DeltaS_prime)
    RETURN: Estimated coordinates of UAVs
    '''
    if (noise=="none"): noise=0

    if noise:
        DM        = expected_value(DM      , noise)
        DM_prime  = expected_value(DM_prime, noise)
        DM_prime2 = expected_value(DM_prime2,noise)

 
    # Eigenvalue decomposition for a first estimation of the coordinates: S*
    S_star = EVD(DM,DIM)

    # Remove translational ambiguity
    S_star, _ = remove_offset(anchor_coord, S_star)

    
    # Estimation of the rotation angle: theta_r
    if noise: theta_r = LSE(DM,DM_prime,S_star,DeltaS_prime)
    else:     theta_r = get_theta(DM,DM_prime,S_star, DeltaS_prime)


    # New rotated coordinates: S**
    S_star2 = rotation_matrix(theta_r)@S_star

    # Estimation of the new rotation angle after another displacement
    if noise: theta_r2 = LSE(DM,DM_prime2,S_star2,DeltaS_prime2)
    else:     theta_r2 = get_theta(DM,DM_prime2,S_star2,DeltaS_prime2,approx=2)


    # Detection of flip ambiguity
    if noise:
        l = 1/2*g(-2*np.arctan2(DeltaS_prime[0,0],DeltaS_prime[1,0]) + 2*np.arctan2(DeltaS_prime2[0,0],DeltaS_prime2[1,0]))
        if np.abs(theta_r2) > np.abs(l):
            F = np.array([[-1,0,0],[0,1,0],[0,0,0]])
            theta_r3 = LSE(DM,DM_prime,F@S_star,DeltaS_prime)
            S_star2 = rotation_matrix(theta_r3)@F@S_star
        else:    print("Ritocco")

    else:
        if (theta_r2 != 0): 
            F = np.array([[-1,0,0],[0,1,0],[0,0,1]])
            theta_r3 = get_theta(DM,DM_prime,F@S_star,DeltaS_prime)
            S_star2 = rotation_matrix(theta_r3)@F@S_star       

    return S_star2


def objective_function(theta,DM,DM_prime,S_star,displ):
    '''
    Objective function for the LSE minimization
    INPUT: distance matrices, S* and displcement
    RETURN: function to be minimized
    '''
    deltaX = displ[0,0]
    deltaY = displ[1,0]

    obj = 0
    
    for index in range(len(DM)):
        a = DM[0,index] - DM_prime[0,index] + deltaX**2 + deltaY**2
        b = -2*(S_star[0,index]*deltaX + S_star[1,index]*deltaY)    
        c =  2*(S_star[0,index]*deltaY - S_star[1,index]*deltaX)

        obj += (a + b*np.cos(theta) + c*np.sin(theta))**2

    return obj


def analytical_sol(DM,DM_prime, S_star, displ):
    deltaX = displ[0,0]
    deltaY = displ[1,0]

    alpha_n = 0
    gamma_n = 0
    beta_n  = 0
    delta_n = 0
    
    for index in range(len(DM)):
        a = DM[0,index] - DM_prime[0,index] + deltaX**2 + deltaY**2
        b = -2*(S_star[0,index]*deltaX + S_star[1,index]*deltaY)    
        c =  2*(S_star[0,index]*deltaY - S_star[1,index]*deltaX)

        alpha_n += a*b
        beta_n  += a*c
        gamma_n += b*c
        delta_n += c**2-b**2
    
    p = [0, 0, 0, 0, 0]
    p[0] = 4*gamma_n**2 + delta_n**2
    p[1] = 2*(2*alpha_n*gamma_n + beta_n*delta_n)
    p[2] = alpha_n**2 + beta_n**2 - 4*gamma_n**2 - delta_n**2
    p[3] = 2*(-alpha_n*gamma_n - beta_n*delta_n)
    p[4] = - beta_n**2 + gamma_n**2

    # Get the roots
    sol = np.roots(p)

    # Get the real roots
    sol = sol[np.isreal(sol)]

    # Get feasible candidates
    sol = sol[sol>=-1]
    sol = sol[sol<=1]


    candidates = []
    
    for can in sol:
        candidates.append(np.arcsin(can))
        candidates.append(g(np.pi - np.arcsin(can)))
    
    candidates = np.array(candidates)
    
    candidates = candidates[candidates >= -np.pi]
    candidates = candidates[candidates <   np.pi]
    
    obj = [objective_function(can2,DM,DM_prime,S_star,displ) for can2 in candidates]
    return np.real(candidates[np.argmin(obj)])


def LSE(DM,DM_prime,S_star,displ):
    '''
    Least-square-error functon
    INPUT: distance matrices, S* corrdinated and displacement
    RETURN: angle that minimizes the objective function
    '''
    r = minimize_scalar(objective_function,args=(DM,DM_prime,S_star,displ))

    
    return r.x
    

def g(t):
    '''
    g-function as defined in the paper, to wrap any arbitrary angle into the [-pi, pi) range
    INPUT: angle in radians
    RETURN: wrapped angle
    '''
    t = np.real(t)
    return t - 2*np.pi*np.floor(t/(2*np.pi)+1/2)




















