import numpy as np
import math
import noise
import scene_setup as scene
##
import sys


#######################################################################
# Helper Classes
#######################################################################

class Barycentric:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma 

class Texture:
    # shorthand class to organize texture attributes into one object   
    def __init__(self, texture, tex_arr, xres, yres):
        self.texture = texture
        self.tex_arr = tex_arr
        self.xres = xres
        self.yres = yres

class AA_Render:
    # shorthand class to organize the anti-aliasing filter values
    def __init__(self, dx, dy, weight):
        self.dx = dx
        self.dy = dy
        self.weight = weight
        self.im = None
        self.z_buffer = None

#######################################################################
# Helper Functions
#######################################################################

def clamp(min_val, max_val, num):
    if num < min_val:
        return min_val
    elif num > max_val:
        return max_val
    else:
        return num

def f01(x, y, x0, y0, x1, y1):
    ans = ((y0-y1) * x) + ((x1-x0) * y) + (x0*y1 - x1*y0)
    return ans

def f12(x, y, x1, y1, x2, y2):
    ans = ((y1-y2) * x) + ((x2-x1) * y) + (x1*y2 - x2*y1)
    return ans

def f20(x, y, x0, y0, x2, y2):
    ans = ((y2-y0) * x) + ((x0-x2) * y) + (x2*y0 - x0*y2)
    return ans

def new4x1Vec(vect):
    # take in a 3 element list and convert it into a 4x1 vector
    ret_val = np.array(vect)
    ret_val = np.append(ret_val, 1)
    ret_val.shape = (4, 1)
    
    return ret_val

#######################################################################
# Shading Functions
#######################################################################

def calcDiffuse(le, Kd, N, L):
    # calculate N.L outside of this function so we can reuse it
    # le = 3 element array [x, y, z]
    # Kd = scalar
    # NdotL = scalar
    
    # le * Kd * (N . L)
    return le * Kd * clamp(0, 1, max(np.dot(N, L), 0))

def calcSpecular(le, Ks, NdotH, n, facing):
    # calculate N.H outside of this function so we can reuse it
    # le = 3 element array [x, y, z]
    # Ks = scalar
    # NdotH = scalar
    
    # le * Ks * facing * (N . H)^n
    return le * Ks * facing * (NdotH ** n)

def perlin(u, v, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024):
    # Calculate Perlin noise value using the noise module
    perlin_value = noise.pnoise2(u * scale, v * scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeatx=repeatx,
                                  repeaty=repeaty)
    return perlin_value


def normperlin(u, v, scale=1.0, octaves=2, persistence=0.1, lacunarity=0.1, repeatx=1024, repeaty=1024):
    # Calculate Perlin noise value using the noise module
    perlin_value = noise.pnoise2(u * scale, v * scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeatx=repeatx,
                                  repeaty=repeaty)
    return perlin_value


def calcColor(le, Kd, Ks, light, n, v, spec):
    
    L = light['L']
    N = n
    E = np.array([0,0,0]) - v
    # E = np.array(light['from'])
    E = np.divide(E, np.linalg.norm(E))
    # NdotL = np.dot(N, L)
    NdotE = np.dot(N, E)
    H = (L + E)/np.linalg.norm(L + E)
    NdotH = max(np.dot(N, H), 0)
    facing = 0
    
    if (NdotH > 0):
        facing = 1

#     if (NdotH >= 0 and NdotE >= 0):
#         # R = (2 * NdotH * N ) - L      # 2 * (N . L) * N - L
#         pass

#     elif (NdotH < 0 and NdotE < 0):
#         N = -1 * N
#         # R = (2 * max(np.dot(N, H), 0) * N ) - L

#     else:
#         color_dict = { 'diffuse': [0, 0, 0], 'specular': [0, 0, 0] }
#         return color_dict
    
    # R = np.divide(R, np.linalg.norm(R))
    # R_E = clamp(0, 1, np.dot(N, E))

    v_diffuse = calcDiffuse(le, Kd, N, L)
    v_specular = calcSpecular(le, Ks, NdotH, spec, facing) # le * Ks * (N.H)^n
    color_dict = { 'diffuse': v_diffuse, 'specular': v_specular }
    return color_dict

def calcFlat(directional_lights, item, n0, v0, spec, ambient):

    Kd = item['material']['Kd']
    Ks = item['material']['Ks']
    
    for light in directional_lights:
        le = light['le']
        v0_colordict = calcColor(le, Kd, Ks, light, n0, v0, spec)
    
    Cs = item['material']['Cs']
    v0_color = Cs * (ambient + v0_colordict['diffuse']) + v0_colordict['specular']
    color_tuple = ( int(v0_color[0] * 255),  int(v0_color[1] * 255), int(v0_color[2] * 255) )

    return color_tuple

def calcGoraud(directional_lights, item, n0, n1, n2, v0, v1, v2, spec, ambient, b):
    Kd = item['material']['Kd']
    Ks = item['material']['Ks']
    alpha = b.alpha; beta = b.beta; gamma = b.gamma
    
    for light in directional_lights:
        le = light['le']
        v0_colordict = calcColor(le, Kd, Ks, light, n0, v0, spec)
        v1_colordict = calcColor(le, Kd, Ks, light, n1, v1, spec)
        v2_colordict = calcColor(le, Kd, Ks, light, n2, v2, spec)
        
    Cs = item['material']['Cs']

    v0_color = Cs * (ambient + v0_colordict['diffuse']) + v0_colordict['specular']
    v1_color = Cs * (ambient + v1_colordict['diffuse']) + v1_colordict['specular']
    v2_color = Cs * (ambient + v2_colordict['diffuse']) + v2_colordict['specular']

    # interpolate the final pixel color    
    final_r = (alpha * v0_color[0]) + (beta * v1_color[0]) + (gamma * v2_color[0])
    final_g = (alpha * v0_color[1]) + (beta * v1_color[1]) + (gamma * v2_color[1])
    final_b = (alpha * v0_color[2]) + (beta * v1_color[2]) + (gamma * v2_color[2])

    color_tuple = ( int(final_r * 255),  int(final_g * 255), int(final_b * 255) )
    
    return color_tuple

def calcPhong(directional_lights, item, n0, n1, n2, tan0, tan1, tan2, binorm0, binorm1, binorm2, v0, v1, v2, spec, normal_map_rgb, ambient, b, normal_mapping_on, NORMAL_MAP_STRENGTH,isShadow):
    Kd = item['material']['Kd']
    Ks = item['material']['Ks']
    alpha = b.alpha; beta = b.beta; gamma = b.gamma
    
    # interpolate the vertex
    v_x = (alpha * v0[0]) + (beta * v1[0]) + (gamma * v2[0])
    v_y = (alpha * v0[1]) + (beta * v1[1]) + (gamma * v2[1])
    v_z = (alpha * v0[2]) + (beta * v1[2]) + (gamma * v2[2])
    
    # interpolate the normal
    n_x = (alpha * n0[0]) + (beta * n1[0]) + (gamma * n2[0])
    n_y = (alpha * n0[1]) + (beta * n1[1]) + (gamma * n2[1])
    n_z = (alpha * n0[2]) + (beta * n1[2]) + (gamma * n2[2])
    
    vert = [ v_x, v_y, v_z ]
    normal = np.array([ n_x, n_y, n_z ])
    normal = np.divide(normal, np.linalg.norm(normal))

    # TODO: move this out of the function into its own function
    # normal_mapping_on = True
    # NORMAL_MAP_STRENGTH = 5

    if normal_mapping_on:
        tan = alpha* tan0 + beta* tan1 + gamma*tan2 
        binorm = alpha* binorm0 + beta* binorm1 + gamma*binorm2 
        TBN  = scene.getTBN_Matrix(tan, binorm, normal)
        
        dn = np.squeeze(np.asarray((TBN @ normal_map_rgb)))
        normal = normal - dn* NORMAL_MAP_STRENGTH
        normal = np.divide(normal, np.linalg.norm(normal))

    for light in directional_lights:
        le = light['le']
        v_colordict = calcColor(le, Kd, Ks, light, normal, vert, spec)
        
    Cs = item['material']['Cs']
    if isShadow:
        v_color = Cs * ambient
    else:
        v_color = Cs * (ambient + v_colordict['diffuse']) + v_colordict['specular']
    #v_color = Cs * (ambient + v_colordict['diffuse']) + v_colordict['specular']
    
    color_tuple = ( int(v_color[0] * 255),  int(v_color[1] * 255), int(v_color[2] * 255) )
    
    return color_tuple

def calcTexPhong(directional_lights, item, n0, n1, n2, tan0, tan1, tan2, binorm0, binorm1, binorm2, v0, v1, v2, spec, t_color, normal_map_rgb, ambient, b, normal_mapping_on, NORMAL_MAP_STRENGTH):
    Kt = 0.7
    Kd = item['material']['Kd']
    Ks = item['material']['Ks']
    alpha = b.alpha; beta = b.beta; gamma = b.gamma

    # interpolate the vertex
    v_x = (alpha * v0[0]) + (beta * v1[0]) + (gamma * v2[0])
    v_y = (alpha * v0[1]) + (beta * v1[1]) + (gamma * v2[1])
    v_z = (alpha * v0[2]) + (beta * v1[2]) + (gamma * v2[2])
    
    # interpolate the normal
    n_x = (alpha * n0[0]) + (beta * n1[0]) + (gamma * n2[0])
    n_y = (alpha * n0[1]) + (beta * n1[1]) + (gamma * n2[1])
    n_z = (alpha * n0[2]) + (beta * n1[2]) + (gamma * n2[2])
    
    vert = [ v_x, v_y, v_z ]
    normal = np.array([ n_x, n_y, n_z ])
    normal = np.divide(normal, np.linalg.norm(normal))

    # TODO: move this out of the funciton into its own function
    # normal_mapping_on = True
    # NORMAL_MAP_STRENGTH = 1

    if normal_mapping_on:
        tan = alpha* tan0 + beta* tan1 + gamma*tan2 
        binorm = alpha* binorm0 + beta* binorm1 + gamma*binorm2
        TBN  = np.matrix( [
                        [  tan[0],     binorm[0],      normal[0] ],
                        [  tan[1],     binorm[1],      normal[1] ],
                        [  tan[2],     binorm[2],      normal[2] ],] )
        
        dn = np.squeeze(np.asarray((TBN @ normal_map_rgb)))
        normal = normal + dn* NORMAL_MAP_STRENGTH
        normal = np.divide(normal, np.linalg.norm(normal))

    
    for light in directional_lights:
        le = light['le']
        v_colordict = calcColor(le, Kd, Ks, light, normal, vert, spec)
        
    Cs = item['material']['Cs']
    v_color = ambient + v_colordict['diffuse'] + v_colordict['specular'] 
    v_color[0] *= 255; v_color[1] *= 255; v_color[2] *= 255
    v_color += (Kt * t_color)
    
    color_tuple = ( int(v_color[0]),  int(v_color[1]), int(v_color[2]) )
    
    return color_tuple


def calcToonShadeTexNorm(lights, n0, n1, n2, alpha, beta, gamma, item, z, t_color, tan0, tan1, tan2, binorm0, binorm1, binorm2, normal_map_rgb, ambient, b, normal_mapping_on, NORMAL_MAP_STRENGTH):
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
    #toon_color = np.array([0.0, 0.0, 0.0])
    for light in lights:
        l = light['L']
        toon_color = toon_shading_v2(l, normal, light['le'], item['material']['Cs'], z)
    
    #toon_color = toon_shader.toon_shading_v2(lights, normal, light['le'], item['material']['Cs'], z)

    # Apply texture color and transmission color
    toon_color = ( int(toon_color[0] * 255),  int(toon_color[1] * 255), int(toon_color[2] * 255) )
    Kt = 0.7
    toon_color += (Kt * t_color)
    color_tuple = (int(toon_color[0]), int(toon_color[1]), int(toon_color[2]))

    return color_tuple

def textureLookup(tex_obj, u, v, tex_scaling):
    # color_tuple = ( int(v_color[0] * 255),  int(v_color[1] * 255), int(v_color[2] * 255) )
    # return color_tuple

    t_xres = tex_obj.xres
    t_yres = tex_obj.yres

    x_loc = (1 - u) * (t_xres - 1)
    y_loc = v * (t_yres - 1)
    x_loc = (tex_scaling * x_loc) % t_yres
    y_loc = (tex_scaling * y_loc) % t_yres

    f = x_loc - math.trunc(x_loc)
    g = y_loc - math.trunc(y_loc)

    texture = tex_obj.texture
    
    p00 = texture.getpixel( (     math.trunc(x_loc),     math.trunc(y_loc)) )
    p11 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,(1 + math.trunc(y_loc))%t_yres ))
    p10 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,     math.trunc(y_loc)) )
    p01 = texture.getpixel( (     math.trunc(x_loc), (1 + math.trunc(y_loc))%t_yres) )
    
    p00 = np.array(p00); p11 = np.array(p11); p10 = np.array(p10); p01 = np.array(p01)
    p00_10  = f * (p10 - p00) + p00
    p01_11  = f * (p11 - p01) + p01
    p_final = g * (p01_11 - p00_10) + p00_10
    
    if (f == 0 and g == 0):
        p_final = (0 * p01) + (1 * p00)

    return p_final

def normalMapLookup(normal_map, u, v, nm_scaling):
    # TODO: move this out of the function
    # if not normal_mapping_on:
    #     return np.array([0,0,0])
   
    nm_xres = normal_map.xres
    nm_yres = normal_map.yres

    x_loc = (1 - u) * (nm_xres - 1)
    y_loc = v * (nm_yres - 1)

    x_loc = (nm_scaling * x_loc) % nm_xres
    y_loc = (nm_scaling * y_loc) % nm_yres

    f = x_loc - math.trunc(x_loc)
    g = y_loc - math.trunc(y_loc)
    
    texture = normal_map.texture
    p00 = texture.getpixel( (    math.trunc(x_loc),     math.trunc(y_loc)) )
    p11 = texture.getpixel( ((1 + math.trunc(x_loc))%nm_xres, (1 + math.trunc(y_loc))%nm_yres) )
    p10 = texture.getpixel( ((1 + math.trunc(x_loc))%nm_yres,     math.trunc(y_loc)) )
    p01 = texture.getpixel( (    math.trunc(x_loc), (1 + math.trunc(y_loc))%nm_yres) )
    
    p00 = np.array(p00); p11 = np.array(p11); p10 = np.array(p10); p01 = np.array(p01)
    p00_10  = f * (p10 - p00) + p00
    p01_11  = f * (p11 - p01) + p01
    p_final = g * (p01_11 - p00_10) + p00_10
    
    if (f == 0 and g == 0):
        p_final = (0 * p01) + (1 * p00)
    p_final = p_final - 128

    p_final = p_final/np.linalg.norm(p_final)

    return p_final


def pnormalMapLookup(normal_map, u, v, nm_scaling):
    # Calculate Perlin noise value
    perlin_value = normperlin(u, v)
    normal = normalMapLookup(normal_map,u,v,nm_scaling)
    # Perturb the normal based on Perlin noise
    perturbed_normal = normal + (perlin_value * np.random.normal(size=3)) #* NORMAL_MAP_STRENGTH
    
    # Normalize the perturbed normal
    perturbed_normal /= np.linalg.norm(perturbed_normal)
    
    return perturbed_normal


def pTextureLookup(tex_obj, u, v, max_rgb, tex_scaling):
    # Calculate Perlin noise value
    perlin_value = perlin(u, v)

    t_xres = tex_obj.xres
    t_yres = tex_obj.yres

    # Retrieve texture color  
    x_loc = (1 - u) * (t_xres - 1)
    y_loc = v * (t_yres - 1)
    x_loc = (tex_scaling * x_loc) % t_yres
    y_loc = (tex_scaling * y_loc) % t_yres

    f = x_loc - math.trunc(x_loc)
    g = y_loc - math.trunc(y_loc)
    
    texture = tex_obj.texture
  
    p00 = texture.getpixel( (    math.trunc(x_loc),     math.trunc(y_loc)) )
    p11 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,(1 + math.trunc(y_loc))%t_yres ))
    p10 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,     math.trunc(y_loc)) )
    p01 = texture.getpixel( (    math.trunc(x_loc), (1 + math.trunc(y_loc))%t_yres) )
    
    p00 = np.array(p00); p11 = np.array(p11); p10 = np.array(p10); p01 = np.array(p01)

        # Interpolate texture colors
    p00_10 = f * (p10 - p00) + p00
    p01_11 = f * (p11 - p01) + p01
    p_final = g * (p01_11 - p00_10) + p00_10
    # Blend texture color with Perlin noise
    blended_color = (
        int((1 - perlin_value) * p_final[0] + perlin_value * max_rgb),
        int((1 - perlin_value) * p_final[1] + perlin_value * max_rgb),
        int((1 - perlin_value) * p_final[2] + perlin_value * max_rgb)
    )
    
    return np.array(blended_color)


#######################################################################
# Toon Shader.py Functions
#######################################################################

def toon_shading(light_dir, normal, le, Cs, z):
    intensity = max(np.dot(light_dir,normal), 0)

    if intensity > .8:
        return (1.0,0.5,0.5,1.0)
    elif intensity > .5:
        return (0.6,0.3,0.3,1.0)
    elif intensity > .25:
        return (0.4,0.2,0.2,1.0)
    else:
        return (0.2,0.1,0.1,1.0)

def toon_shading_v2(light_dir, normal, le, Cs, z):
    zmin = 0.4
    r = 4
    ###print(z)
    intensity = max(np.dot(light_dir,normal), 0)
    ###D = math.log(z/zmin, r)
    ###print(D)
    ###D = 1 - D
    D = 1
    
    if intensity > .8:
        return le * Cs * D
    elif intensity > .5:
        return 0.6 * le * Cs * D
    elif intensity > .25:
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