# %%prun -s cumulative
import streamlit as st
from PIL import Image
import os
import sys
import json
import math
import urllib.request
import numpy as np
import copy
import toon_shader
from urllib.error import HTTPError
import noise


max_rgb = 255
shading_mode = 2 # FLAT = 0, GORAUD = 1, PHONG = 2, TOON = 3
texture_on = 1 # Textures OFF = 0, Textures ON = 1   ## always renders with phong
anti_aliasing_on = 0 # AA OFF = 0, AA ON = 1
perlinon = 0 #0 off, 1 on
nperlin = 0
normal_mapping_on = 1

NORMAL_MAP_STRENGTH = 5
NORMAL_MAP_SCALING= 1
TEXTURE_SCALING = 1
#######################################################################
# File Handling and Input
#######################################################################

_json_file = './sphere.json'
_scene_file = './HW5_scene.json'
_texture_file = './edited.jpg'
_normal_map_file = './crystalagain.jpeg'
# _texture_file = './checker.jpg'
# _texture_file = './norm1.jpeg'
background = Image.open("./natwin.jpg")


obj_url = "https://bytes.usc.edu/cs580/s24-CGMLRen/hw/HW5/data/teapot.json"
scene_url = "https://bytes.usc.edu/cs580/s24-CGMLRen/hw/HW4/data/scene.json"

data = None
try:
    # READ ME: please upload the teapot.json file in the same directory as Final-Project.ipynb in Jupyter Lab
    print('Trying to open {} into Final-Project.ipynb...'.format(_json_file))
    with open(_json_file) as json_file:
        data = json.load(json_file)     

    print('Success! Rendering image...')
    
except OSError:
    print('Could not find {}'.format(_json_file))

try:   
    if not data:    
        # else if it fails, please fetch json data from the link professor Saty provided
        print('Fetching data from:  {}'.format(obj_url))
        with urllib.request.urlopen(obj_url) as json_file:
            data = json.load(json_file)
            
        print('Success! Rendering image...')
    
except HTTPError:
    print('Could not open/read file at: {}'.format(obj_url))
    print('Exiting program...')
    sys.exit(1)
    
scene_data = None
try:
    # READ ME: please upload the scene.json file in the same directory as Final-Project.ipynb in Jupyter Lab
    print('Trying to open {} into Final-Project.ipynb...'.format(_scene_file))
    with open(_scene_file) as scene_file:
        scene_data = json.load(scene_file)     

    print('Success!')
    
except OSError:
    print('Could not find {}'.format(_scene_file))

try:   
    if not scene_data:    
        # else if it fails, please fetch json data from the link professor Saty provided
        print('Fetching data from: {} instead'.format(scene_url))
        with urllib.request.urlopen(scene_url) as scene_file:
            scene_data = json.load(scene_file)
            
        print('Success! Rendering image...')
    
except HTTPError:
    print('Could not open/read file at: {}'.format(scene_url))
    print('Exiting program...')
    sys.exit(1)

if texture_on:
    texture = Image.open(_texture_file)
    t_xres, t_yres = texture.size

if normal_mapping_on:
    normal_map = Image.open(_normal_map_file)
    nm_xres, nm_yres = normal_map.size

    
#######################################################################
# Helper Functions
#######################################################################
def format_func(option):
    return texture_options[option]

def format_func_obj(option):
    return obj_options[option]

def format_func_shader(option):
    return shading_options[option]

obj_options = { './sphere.json' : 'Sphere'
            #    , './HW5_teapot.json' : 'Teapot' # teapot currently broken; no TNB
               }
texture_options = { './edited.jpg' : 'Voronoi Disco','./vor3.tex.jpeg' : 'Vor3tex' }
shading_options = { 2 : 'Phong', 3 : 'Toon' }

with st.sidebar.form("my_form"):

    _json_file = st.selectbox(
        'What object would you like to render?',
        options=list(obj_options.keys()), format_func=format_func_obj
    )

    anti_aliasing_on = st.toggle('Anti-Aliasing On', False) 
    perlin_on = st.toggle('Perlin Noise On', False)

    texture_on = st.toggle('Texturing On', True)
    if texture_on: 

        _texture_file = st.selectbox(
            'Which texture would you like?',
            options=list(texture_options.keys()), format_func=format_func
        )
    else:
        shading_mode = st.selectbox(
            'Which shading mode would you like?',
            options=list(shading_options.keys()), format_func=format_func_shader,
            index=2
        )

    normal_mapping_on = st.toggle('Normal Mapping On', True)
    if normal_mapping_on:
        NORMAL_MAP_STRENGTH = st.slider(
        'Select normal map strength',
        1, 50
    )
    
    NORMAL_MAP_SCALING = st.slider(
        'Select normal map scaling factor',
        1, 7
    )

    TEXTURE_SCALING = st.slider(
        'Select texture scaling factor',
        1, 7
    )
    st.form_submit_button()


if __name__ == "__main__":

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

    def calcFlat(directional_lights, item, n0, v0, spec):

        Kd = item['material']['Kd']
        Ks = item['material']['Ks']
        
        for light in directional_lights:
            le = light['le']
            v0_colordict = calcColor(le, Kd, Ks, light, n0, v0, spec)
        
        Cs = item['material']['Cs']
        v0_color = Cs * (ambient_contribution + v0_colordict['diffuse']) + v0_colordict['specular']
        color_tuple = ( int(v0_color[0] * 255),  int(v0_color[1] * 255), int(v0_color[2] * 255) )

        return color_tuple

    def calcGoraud(directional_lights, item, n0, n1, n2, v0, v1, v2, spec):
        Kd = item['material']['Kd']
        Ks = item['material']['Ks']
        
        for light in directional_lights:
            le = light['le']
            v0_colordict = calcColor(le, Kd, Ks, light, n0, v0, spec)
            v1_colordict = calcColor(le, Kd, Ks, light, n1, v1, spec)
            v2_colordict = calcColor(le, Kd, Ks, light, n2, v2, spec)
            
        Cs = item['material']['Cs']

        v0_color = Cs * (ambient_contribution + v0_colordict['diffuse']) + v0_colordict['specular']
        v1_color = Cs * (ambient_contribution + v1_colordict['diffuse']) + v1_colordict['specular']
        v2_color = Cs * (ambient_contribution + v2_colordict['diffuse']) + v2_colordict['specular']

        # interpolate the final pixel color    
        final_r = (alpha * v0_color[0]) + (beta * v1_color[0]) + (gamma * v2_color[0])
        final_g = (alpha * v0_color[1]) + (beta * v1_color[1]) + (gamma * v2_color[1])
        final_b = (alpha * v0_color[2]) + (beta * v1_color[2]) + (gamma * v2_color[2])

        color_tuple = ( int(final_r * 255),  int(final_g * 255), int(final_b * 255) )
        
        return color_tuple

    def calcPhong(directional_lights, item, n0, n1, n2, tan0, tan1, tan2, binorm0, binorm1, binorm2, v0, v1, v2, spec, normal_map_rgb):
        Kd = item['material']['Kd']
        Ks = item['material']['Ks']
        
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
        v_color = Cs * (ambient_contribution + v_colordict['diffuse']) + v_colordict['specular']
        
        color_tuple = ( int(v_color[0] * 255),  int(v_color[1] * 255), int(v_color[2] * 255) )
        
        return color_tuple

    def calcTexPhong(directional_lights, item, n0, n1, n2, tan0, tan1, tan2, binorm0, binorm1, binorm2, v0, v1, v2, spec, t_color, normal_map_rgb):
        Kt = 0.7
        Kd = item['material']['Kd']
        Ks = item['material']['Ks']
        
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
        v_color = ambient_contribution + v_colordict['diffuse'] + v_colordict['specular'] 
        v_color[0] *= 255; v_color[1] *= 255; v_color[2] *= 255
        v_color += (Kt * t_color)
        
        color_tuple = ( int(v_color[0]),  int(v_color[1]), int(v_color[2]) )
        
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
        #toon_color = np.array([0.0, 0.0, 0.0])
        for light in lights:
            l = light['L']
            toon_color = toon_shader.toon_shading_v2(l, normal, light['le'], item['material']['Cs'], z)
        
        #toon_color = toon_shader.toon_shading_v2(lights, normal, light['le'], item['material']['Cs'], z)
        color_tuple = (int(toon_color[0]* 255), int(toon_color[1]* 255), int(toon_color[2] * 255))

        # Apply texture color and transmission color
        Kt = 0.7
        toon_color += (Kt * t_color)
        #toon_color = np.clip(toon_color, 0.0, 1.0)

        return color_tuple

    def textureLookup(u, v):
        # color_tuple = ( int(v_color[0] * 255),  int(v_color[1] * 255), int(v_color[2] * 255) )
        # return color_tuple
        
        x_loc = (1 - u) * (t_xres - 1)
        y_loc = v * (t_yres - 1)
        x_loc = (TEXTURE_SCALING*x_loc)%t_xres
        y_loc = (TEXTURE_SCALING*y_loc)%t_yres
        f = x_loc - math.trunc(x_loc)
        g = y_loc - math.trunc(y_loc)
        
        p00 = texture.getpixel( (    math.trunc(x_loc),     math.trunc(y_loc)) )
        p11 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,(1 + math.trunc(y_loc))%t_yres ))
        p10 = texture.getpixel( ((1 + math.trunc(x_loc))%t_xres,     math.trunc(y_loc)) )
        p01 = texture.getpixel( (    math.trunc(x_loc), (1 + math.trunc(y_loc))%t_yres) )
        
        p00 = np.array(p00); p11 = np.array(p11); p10 = np.array(p10); p01 = np.array(p01)
        p00_10  = f * (p10 - p00) + p00
        p01_11  = f * (p11 - p01) + p01
        p_final = g * (p01_11 - p00_10) + p00_10
        
        if (f == 0 and g == 0):
            p_final = (0 * p01) + (1 * p00)

        return p_final

    def normalMapLookup(u, v):
        if not normal_mapping_on:
            return np.array([0,0,0])
        # color_tuple = ( int(v_color[0] * 255),  int(v_color[1] * 255), int(v_color[2] * 255) )
        # return color_tuple
        
        x_loc = (1 - u) * (nm_xres - 1)
        y_loc = v * (nm_yres - 1)
        x_loc = (NORMAL_MAP_SCALING*x_loc)%nm_xres
        y_loc = (NORMAL_MAP_SCALING*y_loc)%nm_yres
        f = x_loc - math.trunc(x_loc)
        g = y_loc - math.trunc(y_loc)
        
        p00 = normal_map.getpixel( (    math.trunc(x_loc),     math.trunc(y_loc)) )
        p11 = normal_map.getpixel( ((1 + math.trunc(x_loc))%nm_xres, (1 + math.trunc(y_loc))%nm_yres) )
        p10 = normal_map.getpixel( ((1 + math.trunc(x_loc))%nm_yres,     math.trunc(y_loc)) )
        p01 = normal_map.getpixel( (    math.trunc(x_loc), (1 + math.trunc(y_loc))%nm_yres) )
        
        p00 = np.array(p00); p11 = np.array(p11); p10 = np.array(p10); p01 = np.array(p01)
        p00_10  = f * (p10 - p00) + p00
        p01_11  = f * (p11 - p01) + p01
        p_final = g * (p01_11 - p00_10) + p00_10
        
        if (f == 0 and g == 0):
            p_final = (0 * p01) + (1 * p00)
        p_final = p_final - 128

        p_final = p_final/np.linalg.norm(p_final)

        return p_final

    def pnormalMapLookup(u, v):
        if not normal_mapping_on:
            return np.array([0,0,0])
        # Calculate Perlin noise value
        perlin_value = normperlin(u, v)
        normal = normalMapLookup(u,v)
        # Perturb the normal based on Perlin noise
        perturbed_normal = normal + (perlin_value * np.random.normal(size=3)) #* NORMAL_MAP_STRENGTH
        
        # Normalize the perturbed normal
        perturbed_normal /= np.linalg.norm(perturbed_normal)
        
        return perturbed_normal

    def pTextureLookup(u, v):
        # Calculate Perlin noise value
        perlin_value = perlin(u, v)
        
        # Retrieve texture color
        x_loc = (1 - u) * (t_xres - 1)
        y_loc = v * (t_yres - 1)
        x_loc = (TEXTURE_SCALING*x_loc)%t_xres
        y_loc = (TEXTURE_SCALING*y_loc)%t_yres
        f = x_loc - math.trunc(x_loc)
        g = y_loc - math.trunc(y_loc)
        
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


    def new4x1Vec(vect):
        # take in a 3 element list and convert it into a 4x1 vector
        ret_val = np.array(vect)
        ret_val = np.append(ret_val, 1)
        ret_val.shape = (4, 1)
        
        return ret_val

    #######################################################################
    # Anti-Aliasing Setup Section
    #######################################################################
    class AA_Render:
        # shorthand class to organize the anti-aliasing filter values
        
        def __init__(self, dx, dy, weight):
            self.dx = dx
            self.dy = dy
            self.weight = weight
            self.im = None
            self.z_buffer = None

    if anti_aliasing_on:
        #if(texture_on)
        _xres = t_xres-1
        _yres = t_yres-1
        
        filters = {
            "1" : AA_Render(-0.52/_xres,  0.38/_yres, 0.128),
            "2" : AA_Render( 0.41/_xres,  0.56/_yres, 0.119),
            "3" : AA_Render( 0.27/_xres,  0.08/_yres, 0.294),
            "4" : AA_Render(-0.17/_xres, -0.29/_yres, 0.249),
            "5" : AA_Render( 0.58/_xres, -0.55/_yres, 0.104),
            "6" : AA_Render(-0.31/_xres, -0.71/_yres, 0.106)
        }

    #######################################################################
    # Camera Setup Section
    #######################################################################

    up_vector = np.array([0, 1, 0])
    cam_to = np.array(scene_data['scene']['camera']['to'])
    cam_from = np.array(scene_data['scene']['camera']['from'])

    # cam_from = np.divide(cam_from, np.linalg.norm(cam_from))
    n = cam_from - cam_to
    n = np.divide(n, np.linalg.norm(n))
    # print("n: \n", n)

    u = np.array(np.cross(up_vector, n))
    u = np.divide(u, np.linalg.norm(u))

    # print("u: \n", u)

    v = np.array(np.cross(n, u))
    # print("v: \n", v)

    r = cam_from
    # print(r)

    # unpacking u, v, and n into model matrix variables
    u_x, u_y, u_z = u
    v_x, v_y, v_z = v
    n_x, n_y, n_z = n

    # define perspective matrix values from scene.json
    near   = scene_data['scene']['camera']['bounds'][0]
    far    = scene_data['scene']['camera']['bounds'][1]
    right  = scene_data['scene']['camera']['bounds'][2]
    left   = scene_data['scene']['camera']['bounds'][3]
    top    = scene_data['scene']['camera']['bounds'][4]
    bottom = scene_data['scene']['camera']['bounds'][5]

    # load matrix M, camera matrix 
    M = np.matrix( 
                [[u_x,   u_y,   u_z,   np.dot(-1 * r, u)  ],
                [v_x,   v_y,   v_z,   np.dot(-1 * r, v)  ],
                [n_x,   n_y,   n_z,   np.dot(-1 * r, n)  ],
                [  0,     0,     0,       1       ]] )

    # load matrix P, perspective projection matrix
    P = np.matrix([[(2*near)/(right-left), 0, (right+left)/(right-left), 0],
                [0, (2*near)/(top-bottom), (top+bottom)/(top-bottom), 0],
                [0, 0, -1*(far+near)/(far-near), -1*(2*far*near)/(far-near)],
                [0, 0, -1, 0]] )

    # print("\n P (perspective projection matrix): \n", P)

    # set up lighting lists -> separate ambient and directional lights
    light_list = scene_data['scene']['lights']
    directional_lights = []
    ambient = None

    for light in light_list:
        if light['type'] == 'ambient':
            ambient = light
        else:
            # make a copy of the json dictionary
            temp = copy.deepcopy(light)
            
            # calculate L as normalize(Light[from] - Light[to])
            L_world = new4x1Vec([temp['from'][0] - temp['to'][0],  temp['from'][1] - temp['to'][1],  temp['from'][2] - temp['to'][2]])
            # L_cam = M @ L_world
            L_normalized = np.divide(L_world[:-1], np.linalg.norm(L_world[:-1]))
            L_normalized = np.squeeze(np.asarray(L_normalized))
            
            # add a new "L" entry and "le" entry to the dictionary
            temp['L'] =  L_normalized
            temp['le'] = np.dot(light['intensity'],light['color'])
            
            # add the copy to the list of directional lights
            directional_lights.append(temp)

    #######################################################################
    # Initialize Scene & Image Buffer
    #######################################################################

    xres = scene_data['scene']['camera']['resolution'][0]
    yres = scene_data['scene']['camera']['resolution'][1]

    z_buffer = [[float('inf') for i in range(xres)] for j in range(yres)]
    ppm_list = [[0 for i in range(xres)] for j in range(yres)]

    im = Image.new('RGB', [xres, yres], 0x000000)
    width, height = im.size


    # set initial background color
    for y in range(height):
        for x in range(width):
            # im.putpixel( (x, y), (128, 112, 96) )
            # ppm_list[x][y] = (128, 112, 96)
            im.putpixel( (x, y), (0, 147, 250) )
            #ppm_list[x][y] = (115, 147, 179)
            # im.putpixel( (x, y), (255, 255, 255) )
            # ppm_list[x][y] = (255, 255, 255)


            
    if anti_aliasing_on:
        for key in filters:
            filters[key].z_buffer = [[float('inf') for i in range(xres)] for j in range(yres)]
            filters[key].im = Image.new('RGB', [xres, yres], 0x000000)
            filters[key].im.paste(background, (0, 0, width, height))
    else:
        im.paste(background, (0, 0, width, height))
            
    #######################################################################
    # Render Scene
    #######################################################################

    # Scale, rotate and translate each teapot in the scene
    for item in scene_data['scene']['shapes']:
        print("\n", item['id'])
        T = item['transforms'][2]['T']
        S = item['transforms'][1]['S']
        Rx = 0; Ry = 0; Rz = 0
        
        # after initializing R_x/y/z to default of 0; check teapot dict for R_x/y/z values
        if "Rx" in item['transforms'][0]:
            Rx = np.radians(item['transforms'][0]['Rx'])
            
        if "Ry" in item['transforms'][0]:
            Ry = np.radians(item['transforms'][0]['Ry'])
            
        if "Rz" in item['transforms'][0]:
            Rz = np.radians(item['transforms'][0]['Rz'])
        
        T_mat = np.matrix( [
                [  1,   0,   0,   T[0]  ],
                [  0,   1,   0,   T[1]  ],
                [  0,   0,   1,   T[2]  ],
                [  0,   0,   0,     1   ]] )

        Rx_mat = np.matrix( [
                [  1,   0,    0,   0  ],
                [  0,   math.cos(Rx),   -1 * math.sin(Rx),   0  ],
                [  0,   math.sin(Rx),        math.cos(Rx),   0  ],
                [  0,   0,   0,  1  ]] )
        
        Ry_mat = np.matrix( [
            [  math.cos(Ry),        0,   math.sin(Ry),  0  ],
            [  0,                   1,   0,             0  ],
            [  -1 * math.sin(Ry),   0,   math.cos(Ry),  0  ],
            [  0,                   0,   0,             1  ]] )
        
        Rz_mat = np.matrix( [
            [  math.cos(Rz),   -1 * math.sin(Rz),   0,  0  ],
            [  math.sin(Rz),        math.cos(Rz),   0,  0  ],
            [  0,                   0,              1,  0  ],
            [  0,                   0,              0,  1  ]] )
        
        S_mat = np.matrix( [
                [  S[0],     0,      0,   0  ],
                [  0,     S[1],      0,   0  ],
                [  0,        0,   S[2],   0  ],
                [  0,        0,      0,   1  ]] )
        
        transform_mat = T_mat @ Rx_mat @ Ry_mat @ Rz_mat @ S_mat
        # print("Combined Transformation Matrix: Translation * RotationX * RotationY * RotationZ * Scale")
        # print(transform_mat)
        
        # transform_it = Rx_it @ Ry_it @ Rz_it @ S_it
        transform_it = np.transpose(np.linalg.inv(Rx_mat @ Ry_mat @ Rz_mat @ S_mat))

        # initialize the teapot's specular coefficient (n) and calculate ambient contribution 
        specular_coeff = item['material']['n']
        ambient_contribution = item['material']['Ka'] * ambient['intensity'] * np.array(ambient['color']) # ka * la
        
        # per triangle per teapot:
        for tri in data['data']:

            v0 = tri['v0']['v']
            v1 = tri['v1']['v']
            v2 = tri['v2']['v']
            
            # per triangle, there are 3 normals
            n0 = np.array(tri['v0']['n'])
            n1 = np.array(tri['v1']['n'])
            n2 = np.array(tri['v2']['n'])

            n0 = np.divide(n0, np.linalg.norm(n0))
            n1 = np.divide(n1, np.linalg.norm(n1))
            n2 = np.divide(n2, np.linalg.norm(n2))

            tan0 = np.array(tri['v0']['tan'])
            tan1 = np.array(tri['v1']['tan'])
            tan2 = np.array(tri['v2']['tan'])

            tan0 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(tan0, np.linalg.norm(tan0))))))
            tan1 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(tan1, np.linalg.norm(tan1))))))
            tan2 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(tan2, np.linalg.norm(tan2))))))

            binorm0 = np.array(tri['v0']['binorm'])
            binorm1 = np.array(tri['v1']['binorm'])
            binorm2 = np.array(tri['v2']['binorm'])

            binorm0 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(binorm0, np.linalg.norm(binorm0))))))
            binorm1 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(binorm1, np.linalg.norm(binorm1))))))
            binorm2 = np.squeeze(np.asarray((M @ new4x1Vec(np.divide(binorm2, np.linalg.norm(binorm2))))))
            
            # print("length n0: ", np.linalg.norm(n0))
            # normal = tri['v0']['n'] # take the first vertex's normal (for color calcs)

            # World -> Camera: M dot TransformMat dot v.T
            # Camera -> NDC: P dot (M dot TransformMat dot v.T)
            # divide NDC by w

            v0_T = transform_mat @ new4x1Vec(v0)
            v0_cam = M @ v0_T
            v0_ndc = P @ v0_cam
            v0_ndc = np.divide(v0_ndc, v0_ndc[3])

            v1_T = transform_mat @ new4x1Vec(v1)
            v1_cam = M @ v1_T
            v1_ndc = P @ v1_cam
            v1_ndc = np.divide(v1_ndc, v1_ndc[3])

            v2_T = transform_mat @ new4x1Vec(v2)
            v2_cam = M @ v2_T
            v2_ndc = P @ v2_cam
            v2_ndc = np.divide(v2_ndc, v2_ndc[3])
            
            v0_ndc = np.squeeze(np.asarray(v0_ndc))
            v1_ndc = np.squeeze(np.asarray(v1_ndc))
            v2_ndc = np.squeeze(np.asarray(v2_ndc))
            
            # handle calculating new normals
            # transform normal into world space (@ transform_it), then into camera space (@ M)
            n0_it = transform_it @ new4x1Vec(n0)
            n1_it = transform_it @ new4x1Vec(n1)
            n2_it = transform_it @ new4x1Vec(n2)
            
            n0_it = np.divide(n0_it[:-1], np.linalg.norm(n0_it[:-1]))
            n1_it = np.divide(n1_it[:-1], np.linalg.norm(n1_it[:-1]))
            n2_it = np.divide(n2_it[:-1], np.linalg.norm(n2_it[:-1]))
            
            # print("length n0_it: ", np.linalg.norm(n0_it))
            
            # calc n0_new
            n0_p = new4x1Vec([v0_T.item(0) + n0_it.item(0),
                            v0_T.item(1) + n0_it.item(1),
                            v0_T.item(2) + n0_it.item(2)])
            
            n0_cam = M @ n0_p
            
            n0_new = np.array([n0_cam.item(0) - v0_cam.item(0),
                            n0_cam.item(1) - v0_cam.item(1),
                            n0_cam.item(2) - v0_cam.item(2)])
            
            # calc n1_new
            n1_p = new4x1Vec([v1_T.item(0) + n1_it.item(0),
                            v1_T.item(1) + n1_it.item(1),
                            v1_T.item(2) + n1_it.item(2)])
            
            n1_cam = M @ n1_p
            
            n1_new = np.array([n1_cam.item(0) - v1_cam.item(0),
                            n1_cam.item(1) - v1_cam.item(1),
                            n1_cam.item(2) - v1_cam.item(2)])
            
            # calc n2_new
            n2_p = new4x1Vec([v2_T.item(0) + n2_it.item(0),
                            v2_T.item(1) + n2_it.item(1),
                            v2_T.item(2) + n2_it.item(2)])
            
            n2_cam = M @ n2_p
            
            n2_new = np.array([n2_cam.item(0) - v2_cam.item(0),
                            n2_cam.item(1) - v2_cam.item(1),
                            n2_cam.item(2) - v2_cam.item(2)])

            n0 = np.squeeze(np.asarray(n0_new))
            n1 = np.squeeze(np.asarray(n1_new))
            n2 = np.squeeze(np.asarray(n2_new))
            
            v0_cam = np.divide(v0_cam[:-1], np.linalg.norm(v0_cam[:-1]))
            v1_cam = np.divide(v1_cam[:-1], np.linalg.norm(v1_cam[:-1]))
            v2_cam = np.divide(v2_cam[:-1], np.linalg.norm(v2_cam[:-1]))

            v0_cam = np.squeeze(np.asarray(v0_cam))
            v1_cam = np.squeeze(np.asarray(v1_cam))
            v2_cam = np.squeeze(np.asarray(v2_cam))
            
            if texture_on or normal_mapping_on:
                uv_0 = tri['v0']['t']
                uv_1 = tri['v1']['t']
                uv_2 = tri['v2']['t']
                
                uv_0[0] /= v0_cam[2]; uv_0[1] /= v0_cam[2]
                uv_1[0] /= v1_cam[2]; uv_1[1] /= v1_cam[2]
                uv_2[0] /= v2_cam[2]; uv_2[1] /= v2_cam[2]
                
                z0_inv = 1 / v0_cam[2]
                z1_inv = 1 / v1_cam[2]
                z2_inv = 1 / v2_cam[2]
                    
            if anti_aliasing_on: # perform 6 rendering passes, then average into 1 final img
                for n in filters:
                    dx = filters[n].dx
                    dy = filters[n].dy
                            
                    x0_w = (v0_ndc[0]+1  +dx) * ( (xres - 1)/2 )
                    y0_w = (1-(v0_ndc[1] +dy)) * ( (yres - 1)/2 )

                    x1_w = (v1_ndc[0]+1  +dx) * ( (xres - 1)/2 )
                    y1_w = (1-(v1_ndc[1] +dy)) * ( (yres - 1)/2 )

                    x2_w = (v2_ndc[0]+1  +dx) * ( (xres - 1)/2 )
                    y2_w = (1-(v2_ndc[1] +dy)) * ( (yres - 1)/2 )
                    
                    xmin = int( math.floor(min(x0_w, x1_w, x2_w) ))
                    xmax = int( math.ceil(max(x0_w, x1_w, x2_w) ))

                    ymin = int( math.floor(min(y0_w, y1_w, y2_w)) )
                    ymax = int( math.ceil(max(y0_w, y1_w, y2_w)) )

                    # begin shading the triangle, pixel by pixel
                    for y in range(max(0, ymin), min(yres, ymax)):
                        for x in range(max(0, xmin), min(xres, xmax)):
                            x0, y0 = x0_w, y0_w
                            x1, y1 = x1_w, y1_w
                            x2, y2 = x2_w, y2_w

                            _f12 = f12(x0, y0, x1, y1, x2, y2)
                            _f20 = f20(x1, y1, x0, y0, x2, y2)
                            _f01 = f01(x2, y2, x0, y0, x1, y1)

                            # if the denominator can be 0, skip this current loop
                            if (_f12 == 0 or _f20 == 0 or _f01 == 0):
                                continue

                            # calculate barycentric coordinates for this pixel
                            alpha = f12(x, y, x1, y1, x2, y2) / _f12 
                            beta  = f20(x, y, x0, y0, x2, y2) / _f20
                            gamma = f01(x, y, x0, y0, x1, y1) / _f01

                            # if pixel falls within the triangle
                            if (alpha >= 0) and (beta >= 0) and (gamma >= 0):

                                # check if it passes the z-buffer depth test                    
                                z = (alpha * v0_ndc[2]) + (beta * v1_ndc[2]) + (gamma * v2_ndc[2])

                                if (z < filters[n].z_buffer[x][y]):
                                    filters[n].z_buffer[x][y] = z

                                    total_diffuse = 0
                                    total_specular = 0

                                    if texture_on:
                                        # calculate z at that pixel (from camera space view)
                                        z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                        # interpolate u, v texture coordinates
                                        u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                        v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                        # multiply by the z_at_pixel value
                                        u_tex *= z_pix;  v_tex *= z_pix
                                        if(perlinon):
                                            tex_rgb = pTextureLookup(u_tex, v_tex)
                                        else:
                                            tex_rgb = textureLookup(u_tex, v_tex)
                                        if(nperlin):
                                            normal_map_rgb = pnormalMapLookup(u_tex, v_tex)
                                        else:
                                            normal_map_rgb = normalMapLookup(u_tex, v_tex)
                                        color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, normal_map_rgb)

                                    else:

                                        if shading_mode == 0:
                                            color_tuple = calcFlat(directional_lights, item, n0, v0_cam, specular_coeff)

                                        elif shading_mode == 1:
                                            color_tuple = calcGoraud(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff)

                                        elif shading_mode == 3:
                                            color_tuple = toon_shader.calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)
                                            
                                        else:
                                            color_tuple = calcPhong(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff)

                                    # put pixel into the display image 
                                    filters[n].im.putpixel( (x, y), color_tuple )
                                    # ppm_list[x][y] = color_tuple
            
            else: # single rendering pass

                # convert NDC vertex coordinates to Raster Space
                x0_w = (v0_ndc[0]+1) * ( (xres - 1)/2 )
                y0_w = (1-v0_ndc[1]) * ( (yres - 1)/2 )

                x1_w = (v1_ndc[0]+1) * ( (xres - 1)/2 )
                y1_w = (1-v1_ndc[1]) * ( (yres - 1)/2 )

                x2_w = (v2_ndc[0]+1) * ( (xres - 1)/2 )
                y2_w = (1-v2_ndc[1]) * ( (yres - 1)/2 )
                
                xmin = int( math.floor(min(x0_w, x1_w, x2_w) ))
                xmax = int( math.ceil(max(x0_w, x1_w, x2_w) ))

                ymin = int( math.floor(min(y0_w, y1_w, y2_w)) )
                ymax = int( math.ceil(max(y0_w, y1_w, y2_w)) )
            
                # begin shading the triangle, pixel by pixel
                for y in range(max(0, ymin), min(yres, ymax)):
                    for x in range(max(0, xmin), min(xres, xmax)):
                        x0, y0 = x0_w, y0_w
                        x1, y1 = x1_w, y1_w
                        x2, y2 = x2_w, y2_w

                        _f12 = f12(x0, y0, x1, y1, x2, y2)
                        _f20 = f20(x1, y1, x0, y0, x2, y2)
                        _f01 = f01(x2, y2, x0, y0, x1, y1)

                        # if the denominator can be 0, skip this current loop
                        if (_f12 == 0 or _f20 == 0 or _f01 == 0):
                            continue

                        # calculate barycentric coordinates for this pixel
                        alpha = f12(x, y, x1, y1, x2, y2) / _f12 
                        beta  = f20(x, y, x0, y0, x2, y2) / _f20
                        gamma = f01(x, y, x0, y0, x1, y1) / _f01

                        # if pixel falls within the triangle
                        if (alpha >= 0) and (beta >= 0) and (gamma >= 0):

                            # check if it passes the z-buffer depth test                    
                            z = (alpha * v0_ndc[2]) + (beta * v1_ndc[2]) + (gamma * v2_ndc[2])

                            if (z < z_buffer[x][y]):
                                z_buffer[x][y] = z

                                total_diffuse = 0
                                total_specular = 0

                                if texture_on:
                                    # calculate z at that pixel (from camera space view)
                                    z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                    # interpolate u, v texture coordinates
                                    u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                    v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                    # multiply by the z_at_pixel value
                                    u_tex *= z_pix;  v_tex *= z_pix
                                    if(perlinon):
                                        tex_rgb = pTextureLookup(u_tex, v_tex)
                                    else:
                                        tex_rgb = textureLookup(u_tex, v_tex)
                                    if(nperlin):
                                        normal_map_rgb = pnormalMapLookup(u_tex, v_tex)
                                    else:
                                        normal_map_rgb = normalMapLookup(u_tex, v_tex)

                                    #color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, normal_map_rgb)
                                    if(shading_mode == 3):
                                        color_tuple = calcToonShadeTexNorm(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z, tex_rgb, tan0, tan1, tan2, binorm0, binorm1, binorm2, normal_map_rgb)
                                    else:
                                        color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, normal_map_rgb)

                                else:
                                    z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                    # interpolate u, v texture coordinates
                                    u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                    v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                    # multiply by the z_at_pixel value
                                    u_tex *= z_pix;  v_tex *= z_pix
                                    
                                    normal_map_rgb = normalMapLookup(u_tex, v_tex)


                                    if shading_mode == 0:
                                        color_tuple = calcFlat(directional_lights, item, n0, v0_cam, specular_coeff)

                                    elif shading_mode == 1:
                                        color_tuple = calcGoraud(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff)

                                    elif shading_mode == 3:
                                        color_tuple = toon_shader.calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)
                                        
                                    else:
                                        color_tuple = calcPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, normal_map_rgb)

                                # put pixel into the display image and 
                                #  also into the ppm array that will save the image
                                im.putpixel( (x, y), color_tuple )
                                ppm_list[x][y] = color_tuple
        
        # uncomment this 'break' to render only the first teapot
        # break

    if anti_aliasing_on:
        # average & combine the 6 different renders into final img
        for y in range(height):
            for x in range(width):
                r = 0
                g = 0
                b = 0
                
                for n in filters:
                    weight = filters[n].weight
                    color = filters[n].im.getpixel( (x, y) )
                    r += color[0] * weight;  g += color[1] * weight;  b += color[2] * weight
                    
                r = int(r); g = int(g); b = int(b)
                ppm_list[x][y] = (r, g, b)
                #if( r > 0 or g > 0 or b > 0):
                    #blended_color = (
                        #(r + background_color[0]) // 2,
                        #(g + background_color[1]) // 2,
                        #(b + background_color[2]) // 2
                    #)
                im.putpixel( (x, y), (r, g, b) )

    im.show()
    st.image(im)

            

    