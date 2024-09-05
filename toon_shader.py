import numpy as np
import math

def toon_shading(light_dir, normal, le, Cs, z):
    intensity = max(np.dot(light_dir,normal), 0)

    if intensity > .8:
        return (1.0,0.5,0.5,1.0)
    elif intensity > .4:
        return (0.6,0.3,0.3,1.0)
    elif intensity > .15:
        return (0.4,0.2,0.2,1.0)
    else:
        return (0.2,0.1,0.1,1.0)

def toon_shading_v2(light_dir, normal, le, Cs, z):
    zmin = 0.4
    r = 4
    ###print(z)
    intensity = max(np.dot(light_dir,normal), 0)
    D = math.log(z/zmin, r)
    ###print(D)
    D = 1 - D
    #D = 1
    print(intensity)
    if intensity > .99:
        return le * Cs * D
    elif intensity > .8:
        return 0.6 * le * Cs * D
    elif intensity > .5:
        return 0.4 * le * Cs * D
    else:
        return 0.2 * le * Cs * D

def calcToonShade(lights, n0, n1, n2, alpha, beta, gamma, item, z):
    # interpolate the normal
    n_x = (alpha * n0[0]) + (beta * n1[0]) + (gamma * n2[0])
    n_y = (alpha * n0[1]) + (beta * n1[1]) + (gamma * n2[1])
    n_z = (alpha * n0[2]) + (beta * n1[2]) + (gamma * n2[2])
    
    normal = np.array([ n_x, n_y, n_z ])
    normal = np.divide(normal, np.linalg.norm(normal))

    for light in lights:
        l = light['L']
        color = toon_shading_v2(l, normal, light['le'], item['material']['Cs'], z)

    return ( int(color[0] * 255),  int(color[1] * 255), int(color[2] * 255) )

def calcToonShadeTex(lights, n0, n1, n2, alpha, beta, gamma, item, z, t_color):
    toon_color = calcToonShade(lights, n0, n1, n2, alpha, beta, gamma, item, z)
    Kt = 0.7

    toon_color += (Kt * t_color)
    color_tuple = ( int(toon_color[0]),  int(toon_color[1]), int(toon_color[2]) )

    return color_tuple    


def calcToonShadeTexNorm(lights, n0, n1, n2, alpha, beta, gamma, item, z, t_color, tan0, tan1, tan2, binorm0, binorm1, binorm2, normal_map_rgb):
    # interpolate the normal
    n_x = (alpha * n0[0]) + (beta * n1[0]) + (gamma * n2[0])
    n_y = (alpha * n0[1]) + (beta * n1[1]) + (gamma * n2[1])
    n_z = (alpha * n0[2]) + (beta * n1[2]) + (gamma * n2[2])
    
    normal = np.array([n_x, n_y, n_z])
    normal = np.divide(normal, np.linalg.norm(normal))

    # Check if normal mapping is enabled
    if normal_mapping_on:
        tan = alpha * tan0 + beta * tan1 + gamma * tan2 
        binorm = alpha * binorm0 + beta * binorm1 + gamma * binorm2 
        TBN = np.matrix([
            [tan[0], binorm[0], normal[0]],
            [tan[1], binorm[1], normal[1]],
            [tan[2], binorm[2], normal[2]],
        ])
        
        dn = np.squeeze(np.asarray((TBN @ normal_map_rgb)))
        normal = normal + dn * NORMAL_MAP_STRENGTH
        normal = np.divide(normal, np.linalg.norm(normal))

    # Calculate the toon shading color
    toon_color = toon_shading_v2(lights, normal, light['le'], item['material']['Cs'], z)

    # Apply texture color and transmission color
    Kt = 0.7
    toon_color += (Kt * t_color)
    color_tuple = (int(toon_color[0] * 255), int(toon_color[1] * 255), int(toon_color[2] * 255))

    return color_tuple