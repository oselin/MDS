import sys, time, ast
sys.path.append('')

import numpy as np
from UAV import *
import matplotlib.pyplot as plt




def main():

    # Get parameters from terminal
    argv, params = sys.argv[1:], {}

    for p in argv:
        bfr = p.split('=')
        params[bfr[0]] = bfr[1]

    # Find parameters values, otherwise assign default ones
    if 'number_uavs' in params: params['number_uavs'] = int(params['number_uavs'])
    else: params['number_uavs'] = 20
    
    if 'space' in params: params['space'] = int(params['space'][0])
    else: params['space'] = 3

    if 'noise' in params: params['noise'].lower()
    else: params['noise'] = 'gaussian'
    
    # Initialization of the UAV objects
    fleet = []
    for i in range(params['number_uavs']):
        if i==0: uav = Robot(f"op_{i}", 0, 0, 0) #Anchor in the origin
        else:    uav = Robot(f"op_{i}", np.random.uniform(0, 10.0), 
                                        np.random.uniform(0, 10.0),
                                        0)
        fleet.append(uav)

    # Vector of true coordinates S
    # NOTE: it will be used for plotting purposes. In reality the coordinates are unknonw
    S = [[],[],[]]
    for uav in fleet: S = np.append(S, uav.get_coords(),axis=1)

    #S = np.array([[0,         3.61272693, 1.86480833, 7.63564862, 7.41140807],
    #     [0,         3.87312445, 2.82227237, 4.47205364, 9.26427777],
    #     [0,         0,          0,          0,          0         ]])

    # Simulation of the motion
    plt.ion()

    idx = 1
    while True:

        # Simulate the fleet motion
        S += move(params['space'], params['number_uavs'], movement='all')
        anchor_coord = S[:,0]

        # Add noise to simulate disturbances. If none, no noise is added
        # NOTE: In reality the Distance Matrix is obtained through UltraWideBand sensors
        DM = square(DM_from_S(S) + noise_matrix(type=params['noise'], 
                                                dim=params['number_uavs']))

        # Simulate the movement of the anchor/leader drone
        DeltaS_prime = move(params['space'], params['number_uavs'], movement='anchor')
        S_prime = S + DeltaS_prime

        # Simulate a NEW communication among UAVs and get distances
        # NOTE: In reality the Distance Matrix is obtained through UltraWideBand sensors
        DM_prime = square(DM_from_S(S_prime) + noise_matrix(type=params['noise'], 
                                                            dim=params['number_uavs']))
        
        # Simulate a NEW movement of the anchor/leader drone to detect flip ambiguities
        DeltaS_prime2 = move(params['space'], params['number_uavs'], movement='anchor') + DeltaS_prime
        S_prime2 = S + DeltaS_prime2

        # Simulate a NEW communication among UAVs and get distances
        # NOTE: In reality the Distance Matrix is obtained through UltraWideBand sensors
        DM_prime2 = square(DM_from_S(S_prime2) + noise_matrix(type=params['noise'], 
                                                              dim=params['number_uavs']))
                
        S_hat = MDS(DM, DM_prime, DM_prime2, anchor_coord, DeltaS_prime, DeltaS_prime2, params['space'],params['noise'])

        plot_points(idx, plt, S=S[:2,:], S_hat=S_hat[:2,:])
        idx += 1
        if idx > 4: time.sleep(0.1)



if __name__ == '__main__':
    main()