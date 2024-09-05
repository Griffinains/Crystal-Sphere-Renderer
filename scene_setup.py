import math
import numpy as np

def getCameraMatrix(u, v, n, r):
    # unpack u, v, and n into model matrix variables
    u_x, u_y, u_z = u
    v_x, v_y, v_z = v
    n_x, n_y, n_z = n

    return np.matrix( 
                [[u_x,   u_y,   u_z,   np.dot(-1 * r, u)  ],
                [v_x,   v_y,   v_z,   np.dot(-1 * r, v)  ],
                [n_x,   n_y,   n_z,   np.dot(-1 * r, n)  ],
                [  0,     0,     0,       1       ]] )


def getPerspectiveProjMatrix(scene_data):

    near   = scene_data['scene']['camera']['bounds'][0]
    far    = scene_data['scene']['camera']['bounds'][1]
    right  = scene_data['scene']['camera']['bounds'][2]
    left   = scene_data['scene']['camera']['bounds'][3]
    top    = scene_data['scene']['camera']['bounds'][4]
    bottom = scene_data['scene']['camera']['bounds'][5]
    
    return np.matrix([[(2*near)/(right-left), 0, (right+left)/(right-left), 0],
                [0, (2*near)/(top-bottom), (top+bottom)/(top-bottom), 0],
                [0, 0, -1*(far+near)/(far-near), -1*(2*far*near)/(far-near)],
                [0, 0, -1, 0]] )


def getTranslationMatrix(T):

    return np.matrix( [
                [  1,   0,   0,   T[0]  ],
                [  0,   1,   0,   T[1]  ],
                [  0,   0,   1,   T[2]  ],
                [  0,   0,   0,     1   ]] )


def getX_RotationMatrix(Rx):

    return np.matrix( [
                [  1,   0,    0,   0  ],
                [  0,   math.cos(Rx),   -1 * math.sin(Rx),   0  ],
                [  0,   math.sin(Rx),        math.cos(Rx),   0  ],
                [  0,   0,   0,  1  ]] )


def getY_RotationMatrix(Ry):

    return np.matrix( [
            [  math.cos(Ry),        0,   math.sin(Ry),  0  ],
            [  0,                   1,   0,             0  ],
            [  -1 * math.sin(Ry),   0,   math.cos(Ry),  0  ],
            [  0,                   0,   0,             1  ]] )


def getZ_RotationMatrix(Rz):

    return np.matrix( [
            [  math.cos(Rz),   -1 * math.sin(Rz),   0,  0  ],
            [  math.sin(Rz),        math.cos(Rz),   0,  0  ],
            [  0,                   0,              1,  0  ],
            [  0,                   0,              0,  1  ]] )


def getScalingMatrix(S):

    return np.matrix( [
                [  S[0],     0,      0,   0  ],
                [  0,     S[1],      0,   0  ],
                [  0,        0,   S[2],   0  ],
                [  0,        0,      0,   1  ]] )


def getTBN_Matrix(tan, binorm, normal):
    
    return np.matrix( [
                [  tan[0],     binorm[0],      normal[0] ],
                [  tan[1],     binorm[1],      normal[1] ],
                [  tan[2],     binorm[2],      normal[2] ],] )