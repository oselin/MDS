import sys, time, ast
sys.path.append('')

import numpy as np
from UAV import *
import matplotlib.pyplot as plt

np.random.seed(10)

def simulation(parameters):
    
    # Initialize UAVs coordinates, randomly
    X = np.random.uniform(low = -5, high=5, size=[3, parameters['number_uavs']])

    alpha = 10

    initialize_plot()
    while True:

        #
        # ANCHOR - position of the anchor, after the applied motion
        # X      - coordinates of the fleet
        #

        # Retrieve the distances and build the distance matrix DM. In reality it comes from UWB sensors
        ANCHOR1, X = move_anchor(points = X, axis = None)
        DM1  = distance_matrix(X)

        # Simulate a second virtual anchor, by moving the real one and retrieving distances
        ANCHOR2, X  = move_anchor(points = X, axis = "x", displacement=alpha)
        DM2 = distance_matrix(X)

        # Simulate a third virtual anchor, by moving the real one and retrieving distances
        ANCHOR3, X  = move_anchor(points = X, axis = "y", displacement=alpha)
        DM3 = distance_matrix(X)

        # Simulate a fourth virtual anchor, by moving the real one and retrieving distances
        # DELTA4  = move_anchor(elements=parameters['number_uavs'])
        ANCHOR4, X  = move_anchor(points = X, axis = "z", displacement=alpha)
        DM4 = distance_matrix(X)

        # Assemble the distance information in one unique matrix
        DM = combine_matrices(DM1, DM2, DM3, DM4, ANCHOR1, ANCHOR2, ANCHOR3, ANCHOR4)

        # Store the anchor and virtual anchors position into a coordinates array
        anchor_pos = np.hstack([ANCHOR1, ANCHOR2, ANCHOR3, ANCHOR4])

        X_hat = MDS(DM, anchor_pos)

        plot_uavs(true_coords=X, estimated_coords=X_hat)
        exit(0)



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