import numpy as np
import pandas as pd

def rotate(p, origin=(0, 0), degrees=0):
    angle = degrees
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotateAndAnalyse(p_xy):

    p_xy = rotate(p_xy, degrees = -90) 

    optimalRotation = 0
    minDistance = (p_xy.max(axis=0)[0] - p_xy.min(axis=0)[0])**2

    for i in range(-180, 180): 
        p_xy = rotate(p_xy, degrees = 0.5) 
        
        currentDistance = (p_xy.max(axis=0)[0] - p_xy.min(axis=0)[0])

        if currentDistance < 0:
            currentDistance = -currentDistance

        if currentDistance < minDistance:
            minDistance = currentDistance
            optimalRotation = i

        #print(optimalRotation * 0.5)
        #print("")

    # rotate back to the most optimal rotation
    # we already did 360 rotations, go back 360-optimal number of rotations
    #return rotate(p_xy, degrees = - 0.5 * (360 - optimalRotation))

    return (0.5 * optimalRotation)


def correctlyRotateDataFrame(f):

    # PART 1: figure out the optimal rotation in XY plane
    f_xy = f[['x', 'y']].copy()

    # this gives the rotation to minimize the x interval
    optimal_xy_rotation = rotateAndAnalyse(f_xy)
    #print(optimal_xy_rotation)

    # PART 2: rotate dataframe in XY plane
    f[['x','y']] = rotate(f_xy, degrees=optimal_xy_rotation)


    # PART 3: do the same in ZY axis
    # to minimize the skewness of the floor you need to rotate in Z,Y (order is important for the function), as it minimizes the 1st parameter and we need to minize the height
    #f_zy = f[['z', 'y']].copy()
    #print(p_zy.head())
    #optimal_zy_rotation = rotateAndAnalyse(f_zy)
    #f[['z','y']] = rotate(f_zy, degrees=optimal_zy_rotation)

    return f
