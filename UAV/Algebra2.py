#!/usr/bin/env
import numpy as np

def get_distance(d1:np.array, d2:np.array = None):
    """
    Compute the distance between two points, i.e. compute vectorial difference.
    If only one vector is provided, the distance is assumed from the origin [0,0,0]^T
    Otherwise, it computes the difference via the Carnot theorem
    """
    
    # Handle the input vectors
    d1 = d1[:,0].reshape(-1,1)
    if (d2 is not None): 

        d2 = d2[:,0].reshape(-1,1)
        # Get the scalar product magnitude
        scalar_prd = np.dot(d1.T,d2)

        # Get the two individual magnitudes
        m1 = np.sqrt(np.dot(d1.T,d1))
        m2 = np.sqrt(np.dot(d2.T,d2))

        # Get the cosine between the two vectors
        cos = scalar_prd/(m1*m2)

        # Cosine theorem (Carnot)
        return (m1**2 + m2**2 - 2*m1*m2*cos)[0]
    
    else:
        return np.dot(d1.T,d1)[0]



def distance_matrix(X):
    """
    Compute the distance matrix given the coordinates matrix.
    In reality, the distances are retrieved from UWB sensors
    """

    # Unitary array
    e = np.ones(X.shape[1]).reshape(-1,1)

    Phi = np.diag(X.T @ X).reshape(-1,1)
    D = Phi @ e.T - 2* X.T @ X + e @ Phi.T

    return D


def move_anchor(elements, low = -1, high = -1):

    # Compute the randomic displacement of the anchor
    delta_anchor = np.random.uniform(low=low, high=high, size=[3,1])

    # Create a vector of zeros, i.e. the other nodes stay still
    others = np.zeros([3, elements - 1])

    return np.hstack([delta_anchor, others])


def combine_matrices(D1, D2, DELTA2, D3, DELTA3, D4, DELTA4):

    ## 1 - Start from the initial distance matrix D1
    DM = D1.copy()

    ## 2 - Add the second distance matrix information
    # Create the new column vector to add to the matrix
    col = np.hstack([get_distance(DELTA2), D2[:,-1]]).reshape(-1,1)

    # Add the vector to the symmetric matrix
    DM = np.hstack([DM, col[:-1]])
    DM = np.vstack([DM, col.reshape(1,-1)])

    ## 3 - Add the second distance matrix information
    # Create the new column vector to add to the matrix
    col = np.hstack([get_distance(DELTA3), D3[:-1,-1], get_distance(DELTA2, DELTA3), D3[-1,-1]]).reshape(-1,1)
    
    # Add the vector to the symmetric matrix
    DM = np.hstack([DM, col[:-1]])
    DM = np.vstack([DM, col.reshape(1,-1)])

    ## 4 - Add the second distance matrix information
    # Create the new column vector to add to the matrix
    col = np.hstack([get_distance(DELTA4), D4[:-1,-1], get_distance(DELTA2, DELTA4), get_distance(DELTA3, DELTA4), D4[-1,-1]]).reshape(-1,1)

    # Add the vector to the symmetric matrix
    DM = np.hstack([DM, col[:-1]])
    DM = np.vstack([DM, col.reshape(1,-1)])

    return DM


def EVD(DM, final_dimension):
    """
    Return the relative map of the nodes in the network
    """
    # Current size of the matrix. It is square, thus len = shape
    n = len(DM)

    # Centering Matrix definition
    H = np.eye(n) - np.ones((n,n))/n

    # Double centered matrix
    B = -1/2*H@DM@H

    # Eiegenvalue decomposition
    ev, EV = np.linalg.eig(B)

    # Sort according to the eigenvalues magnitude
    seq = np.argsort(ev)[::-1]

    LAMBDA = np.diag(ev[seq][:final_dimension])
    U = EV[:,seq][:, :final_dimension]

    # Relative map
    X1 = np.sqrt(LAMBDA)@U.T

    return X1.real

def solve_ambiguity(Sp, Sm, centroid):

    out = np.zeros(Sp.shape)

    for i in range(Sp.shape[1]):
        sz = Sp[-1,i]

        if (sz < centroid[-1]):
            out[:,i] = Sp[:,i]
        else:
            out[:,i] = Sm[:,i]

    return out



def MDS(distance_matrix, anchor_pos, true_pos = None):
    """
    Compute the actual coordinates of the points given the complete distance matrix.
    The matrix must contain n+1 known coordinates, i.e. n+1 anchors
    """

    # Compute the relative map of the points
    S =  EVD(distance_matrix,3)

    # Define P and Q, two sets of corresponding nodes
    # P = anchors from the relative map
    # Q = true position of the anchors
    P = anchor_pos
    P_prime = S[:,-4:]


    ## Steps for turning relative map into absolute map
    # 1. Compute the weighted centroids of both point sets
    mu, mu_prime = 1/4*np.sum(P, axis=1).reshape(-1,1), 1/4*np.sum(P_prime, axis=1).reshape(-1,1)

    # 2. Compute the centered vectors
    P_bar, P_prime_bar = P - mu, P_prime - mu_prime

    # 3. Compute the 3 × 3 covariance matrix
    H = P_bar @ P_prime_bar.T
    # H = P_prime1 @ P1.T

    # 4a. Compute the singular value decomposition
    UU, SS, VV = np.linalg.svd(H)
    Rp = UU @ VV.T

    # 4b. Compute the singular value decomposition
    SIGMA = np.diag([1,1,-1])
    Rm = UU @ SIGMA @ VV.T

    # 5.a Compute the optimal translation as
    tp = mu - Rp @ mu_prime

    # 5b. Compute the optimal translation as
    tm = mu - Rm @ mu_prime

    # 6. Apply the Euclidean transformation
    Sp = Rp @ S + tp
    Sm = Rm @ S + tm

    # 7. Solve the ambiguity
    X_hat = solve_ambiguity(Sp, Sm, mu)

    if (true_pos is not None):
        print("-----Sp-----")
        print(true_pos - Sp[:,:-3])
        print()
        print("-----Sm-----")
        print(true_pos - Sm[:,:-3])
        print()
        print("----Final choice----")
        print(true_pos - X_hat[:,:-3])

    return X_hat


# if __name__ == "__main__":

#     # Initialize UAVs coordinates, randomly
#     X = np.array([[1,2,3,4], [0,0,0,0], [0,0,0,0]])

#     # Retrieve the distances and build the distance matrix DM. In reality it comes from UWB sensors
#     DM1 = distance_matrix(X)

#     # Simulate a second virtual anchor, by moving the real one and retrieving distances
#     DELTA2 = np.hstack([np.ones([3,1]), np.zeros([3,3])])
#     DM2 = distance_matrix(X + DELTA2)

#     # Simulate a third virtual anchor, by moving the real one and retrieving distances
#     DELTA3= np.hstack([2*np.ones([3,1]), np.zeros([3,3])])
#     DM3 = distance_matrix(X + DELTA3)

#     # Simulate a fourth virtual anchor, by moving the real one and retrieving distances
#     DELTA4 = np.hstack([3*np.ones([3,1]), np.zeros([3,3])])
#     DM4 = distance_matrix(X + DELTA4)

#     # Assemble the distance information in one unique matrix
#     DM = combine_matrices(DM1, DM2, DELTA2, DM3, DELTA3, DM4, DELTA4)
#     exit(0)