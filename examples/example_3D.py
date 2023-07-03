import sys, time, ast
sys.path.append('')

import numpy as np
from UAV import *
import matplotlib.pyplot as plt



def simulation(parameters):
    
    # Initialize UAVs coordinates, randomly
    # X = np.random.uniform(low = -5, high=5, size=[3, parameters['number_uavs']])

    #while True:
    for _ in range(100):

        X = np.random.uniform(low = -5, high=5, size=[3, parameters['number_uavs']])

        d = []
        for alpha in np.linspace(0.1,2,40):
            # Retrieve the distances and build the distance matrix DM. In reality it comes from UWB sensors
            DELTA1  = alpha * np.hstack([[[0], [0], [0]], np.zeros([3,parameters['number_uavs'] - 1])])
            X = X + DELTA1
            DM1  = distance_matrix(X)

            # Simulate a second virtual anchor, by moving the real one and retrieving distances
            # DELTA2  = move_anchor(elements=parameters['number_uavs'])
            DELTA2 = alpha * np.hstack([[[1], [0], [0]], np.zeros([3,parameters['number_uavs'] - 1])])
            ANCHOR2 = X + DELTA2
            DM2 = distance_matrix(ANCHOR2)

            # Simulate a third virtual anchor, by moving the real one and retrieving distances
            # DELTA3  = move_anchor(elements=parameters['number_uavs'])
            DELTA3 = alpha * np.hstack([[[0], [1], [0]], np.zeros([3,parameters['number_uavs'] - 1])])
            ANCHOR3 = X + DELTA3
            DM3 = distance_matrix(ANCHOR3)

            # Simulate a fourth virtual anchor, by moving the real one and retrieving distances
            # DELTA4  = move_anchor(elements=parameters['number_uavs'])
            DELTA4 = alpha * np.hstack([[[0], [0], [1]], np.zeros([3,parameters['number_uavs'] - 1])])
            ANCHOR4 = X + DELTA4
            DM4 = distance_matrix(ANCHOR4)
            
            # Assemble the distance information in one unique matrix
            DM = combine_matrices(DM1, DM2, DELTA2, DM3, DELTA3, DM4, DELTA4)

            # Store the anchor and virtual anchors position into a coordinates array
            anchor_pos = np.vstack([X[:,0], ANCHOR2[:,0], ANCHOR3[:,0], ANCHOR4[:,0]]).T

            X_hat = MDS(DM, anchor_pos)

            # Compute the difference in achor positions
            anchor_pos_prime = np.hstack([X_hat[:,0].reshape(-1,1), X_hat[:,-3:]])

            #print("ALPHA: ", alpha)
            dd = 0
            for i in range(anchor_pos.shape[1]):
                #print(anchor_pos[:,i] - anchor_pos_prime[:,i], "| " , np.sqrt(np.sum((anchor_pos[:,i] - anchor_pos_prime[:,i])**2)))
                dd += np.sqrt(np.sum((anchor_pos[:,i] - anchor_pos_prime[:,i])**2))
            d.append(dd)
            #print()
        #print("MIN:", np.min(d))
        print("ALPHA:", np.linspace(0.1,2,40)[np.argmin(d)])
        #exit(0)



if __name__ == '__main__':

    # Default simulation parameters
    params = {'number_uavs' : 20, 'noise' : 'gaussian'}

    # Load parameter from terminal, if provided
    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        try:
            value = ast.literal_eval(value)

            if (key in ['number_uavs', 'noise']):
                params[key] = value
            else:
                print("Inserted paramter not recognised.")
                exit(1)
        except (SyntaxError, ValueError) as e:
            print("LOG: ", e)
            exit(1)

    # Start the simulation
    simulation(parameters=params)