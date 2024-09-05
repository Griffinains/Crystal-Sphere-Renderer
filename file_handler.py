from PIL import Image
import sys
import json
import urllib.request
import numpy as np
from shading_lib import Texture
from shading_lib import clamp
from urllib.error import HTTPError

def openJSONFile(_json_file, obj_url):
    # attempt to open the specified json object file at a local path
    # if the attempt fails, try to open the object file via url
    data = None

    try:
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

    return data


def openSceneFile(_scene_file, scene_url):
    # attempt to open the specified scene file at a local path
    # if the attempt fails, try to open the scene file via url
    scene_data = None

    try:
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

    return scene_data


def openTextureFile(_texture_file):
    texture = Image.open(_texture_file)
    tex_arr = np.array(texture, dtype=np.int16)
    xres, yres = texture.size

    return Texture(texture, tex_arr, xres, yres)


def saveAsPPM(xres, yres, max_rgb, ppm_list, file_name):
    print('Saving output as: "FinalProject_output.ppm" in the same directory...')
    output = open(file_name,'w+')
    output.write('P3\n')
    output.write(str(xres) + ' ' + str(yres) + '\n')
    output.write(str(max_rgb) + '\n')

    for y in range(yres):
        for x in range(xres):
            rgb_tuple = ppm_list[x][y]
            _r = clamp(0, 255, rgb_tuple[0])
            _g = clamp(0, 255, rgb_tuple[1])
            _b = clamp(0, 255, rgb_tuple[2])
            write_str = '{r} {g} {b}'.format(r=_r, g=_g, b=_b)
            #print(write_str)
            output.write(write_str)
            
            if x == (xres-1):
                output.write('\n')
            else:
                output.write(' ')

    output.close()

if __name__ == "__main__":
    pass