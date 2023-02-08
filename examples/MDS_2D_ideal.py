import sys
sys.path.append('')
import time
import numpy as np
from UAV import *
import matplotlib.pyplot as plt

#GLOBAL PARAMETERS
N_ROBOTS  = 5
DIMENSION = 2 #2D


# Fleet of drones
platoon = []


# INITIALIZATION OF THE DRONES
for i in range(N_ROBOTS):
    if i==0: i_robot = Robot(f"op_{i}",0,0,0) #Anchor in the origin
    else:    i_robot = Robot(f"op_{i}",np.random.uniform(0, 10.0),np.random.uniform(0, 10.0),0)
    platoon.append(i_robot)

coordinates = [[],[],[]]

for drone in platoon:
    coordinates = np.append(coordinates, drone.get_coords(),axis=1)

# Vector of true coordinates S
# NOTE: IT WILL BE USED FOR PLOTTING THE ACTUAL COORDINATES
# THE ALGORITHM NEEDS ONLY THE POSITION OF THE LEADER ANCHOR/DRONE, SO LET'S CREATE A NEW VARIABLE
# TO DEMONSTRATE IT

S = coordinates[0:2,:]

plt.ion()
ii = 1
while True:

    # Simulate the movement of the anchor/leader drone
    S += move(DIMENSION,N_ROBOTS, movement='all')
    S_anc = np.copy(S)
    S_anc = S_anc[:,0]

    # Add Gaussian noise: mean: 0 | variance: 0.01
    DM = square(DM_from_S(S) + noise_matrix('gaussian',N_ROBOTS, [0,0.01]))
    
    # Simulate the movement of the anchor/leader drone
    DeltaS_prime = move(DIMENSION,N_ROBOTS, movement='anchor')
    S_prime = S + DeltaS_prime

    # Simulate a NEW communication among UAVs and get distances
    DM_prime = square(DM_from_S(S_prime) + noise_matrix('gaussian',N_ROBOTS, [0,0.01]))
    
    # Simulate a NEW movement of the anchor/leader drone to detect flip ambiguities
    DeltaS_prime2 = move(DIMENSION,N_ROBOTS, movement='anchor')
    S_prime2 = S_prime + DeltaS_prime2

    # Simulate a NEW communication among UAVs and get distances
    DM_prime2 = square(DM_from_S(S_prime2) + noise_matrix('gaussian',N_ROBOTS, [0,0.01]))
    
    S_estim = MDS(DM,DM_prime,DM_prime2, S_anc, DeltaS_prime, DeltaS_prime2,DIMENSION, noise='gaussian')
    
    plot_points(ii,plt,S=S, S_estim = S_estim)
    time.sleep(3)
    ii += 1

