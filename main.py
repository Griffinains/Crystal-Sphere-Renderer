import streamlit as st
import file_handler as handle
import scene_setup as scene
from shading_lib import *
import add_geometry
import shadow_map

####
import copy
from PIL import Image

max_rgb = 255
shading_mode = 2 # FLAT = 0, GORAUD = 1, PHONG = 2, TOON = 3
texture_on = 1 # Textures OFF = 0, Textures ON = 1   ## always renders with phong, will override shading
anti_aliasing_on = 0 # AA OFF = 0, AA ON = 1
perlin_on = 0 #0 off, 1 on
nperlin = 0
normal_mapping_on = 1
shadow_on = 0

NORMAL_MAP_STRENGTH = 5
NORMAL_MAP_SCALING = 4
TEXTURE_SCALING = 4

_json_file = './sphere.json'
_scene_file = './HW5_scene.json'
_texture_file = './edited.jpg'
_normal_map_file = './crystalagain.jpeg'
# _texture_file = './checker.jpg'
# _texture_file = './norm1.jpeg'
background = Image.open("./natwin.jpg")

obj_url = "https://bytes.usc.edu/cs580/s24-CGMLRen/hw/HW5/data/teapot.json"
scene_url = "https://bytes.usc.edu/cs580/s24-CGMLRen/hw/HW4/data/scene.json"

#######################################################################
# User Interface Setup
#######################################################################

def format_func(option):
    return texture_options[option]

def format_func_obj(option):
    return obj_options[option]

def format_func_shader(option):
    return shading_options[option]

presets = {
    "Default": {
        "json_file": './sphere.json',
        "_scene_file": './HW5_scene.json',
        "texture_file": './vor3.tex.jpeg',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 5
    },
    "Crystal Rainbow": {
        "json_file": './sphere.json',
        "_scene_file": './HW5_scene.json',
        "texture_file": './edited.jpg',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 5
    },
    "Constant Rainbow": {
        "json_file": './sphere.json',
        "_scene_file": './HW5_scene.json',
        "texture_file": './rainbow.jpg',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 1
    },
    "Bricks": {
        "json_file": './sphere.json',
        "texture_file": './brickstext.jpg',
        "_scene_file": './preset2.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 0,
        "SCALING_FACTOR": 1
    },
    "Wood": {
        "json_file": './sphere.json',
        "texture_file": './wood.jpg',
        "_scene_file": './preset3.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 2
    },
    "Velvet": {
        "json_file": './sphere.json',
        "texture_file": './velvet.jpg',
        "_scene_file": './preset4.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 5,
        "SCALING_FACTOR": 2
    },
    "Steel": {
        "json_file": './sphere.json',
        "texture_file": './steel.jpg',
        "_scene_file": './preset4.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 0,
        "SCALING_FACTOR": 2
    },
    "Money": {
        "json_file": './sphere.json',
        "texture_file": './money.jpg',
        "_scene_file": './preset2.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 3
    },
    "Marble": {
        "json_file": './sphere.json',
        "texture_file": './marble.jpg',
        "_scene_file": './HW5_scene.json',
        "perlin_on": True,
        "nperlin": True,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 5
    },
    "Grass": {
        "json_file": './sphere.json',
        "texture_file": './grass.jpg',
        "_scene_file": './preset2.json',
        "perlin_on": False,
        "nperlin": True,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 5
    },
    "Crazy": {
        "json_file": './sphere.json',
        "texture_file": './crazy.jpg',
        "_scene_file": './hw5_scene.json',
        "perlin_on": False,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 1,
        "SCALING_FACTOR": 3
    },
    "Clouds": {
        "json_file": './sphere.json',
        "texture_file": './clouds.jpg',
        "_scene_file": './preset2.json',
        "perlin_on": True,
        "nperlin": False,
        "texture_on": True,
        "normal_mapping_on": True,
        "NORMAL_MAP_STRENGTH": 0,
        "SCALING_FACTOR":2
    }
}
obj_options = { './sphere.json' : 'Sphere', './coke.json' : 'Saty Coke Bottle', './stanford-bunny.json' : 'Bunny'
               }
texture_options = {
    './vor3.tex.jpeg': 'Vor3tex',
    './vor3.funky.jpeg': 'Funky',
    './edited.jpg': 'Voronoi Disco',
    './rainbow.jpg': 'Rainbow',
    './brickstext.jpg': 'Bricks',
    './wood.jpg': 'Wood',
    './velvet.jpg': 'Velvet',
    './steel.jpg': 'Steel',
    './money.jpg': 'Money',
    './marble.jpg': 'Marble',
    './grass.jpg': 'Grass',
    './crazy.jpg': 'Crazy',
    './clouds.jpg': 'Clouds'
}
shading_options = {'Phong': 2, 'Toon': 3}

session_state = st.session_state
if "preset" not in session_state:
    session_state.preset = "Default"

with st.sidebar.form("my_form"):
    preset = st.selectbox("Select Preset OR Customize", options=list(presets.keys()))

    if preset != "Select Preset":
        # Set session state based on selected preset
        session_state.preset = preset

    selected_preset = presets[session_state.preset]
    _texture_file = st.selectbox(
            'Which texture would you like?',
            options=list(texture_options.keys()), format_func=format_func,
            index=list(texture_options.keys()).index(selected_preset.get("texture_file", list(texture_options.keys())[0]))
        )
    _json_file = st.selectbox(
            'Which object  would you like?',
            options=list(obj_options.keys()), format_func=format_func_obj,
            index=list(obj_options.keys()).index(selected_preset.get("json_file", list(obj_options.keys())[0]))
        )
    # Set options based on preset
    #_json_file = selected_preset["json_file"]
    #_texture_file = selected_preset["texture_file"]
    perlin_on = selected_preset["perlin_on"]
    nperlin = selected_preset["nperlin"]
    texture_on = selected_preset["texture_on"]
    _scene_file = selected_preset["_scene_file"]
    normal_mapping_on = selected_preset["normal_mapping_on"]
    NORMAL_MAP_STRENGTH = selected_preset["NORMAL_MAP_STRENGTH"]
    NORMAL_MAP_SCALING = selected_preset["SCALING_FACTOR"]

    TEXTURE_SCALING = NORMAL_MAP_SCALING


    anti_aliasing_on = st.checkbox("Anti-Aliasing On", value=anti_aliasing_on)
    perlin_on = st.checkbox("Texture Perlin Noise On", value=perlin_on)
    nperlin = st.checkbox("Normal Perlin Noise On", value=nperlin)
    texture_on = st.checkbox("Texturing On", value=texture_on)
    normal_mapping_on = st.checkbox("Normal Mapping On", value=normal_mapping_on)

    NORMAL_MAP_STRENGTH = st.slider(
        'Select normal map strength factor',
        -5, 5, value=NORMAL_MAP_STRENGTH
    )

    NORMAL_MAP_SCALING = st.slider(
        'Select scaling factor',
        1, 7, value=NORMAL_MAP_SCALING
    )
    TEXTURE_SCALING = NORMAL_MAP_SCALING
    Rx = st.slider(
        'Rx',
        -180, 180, value=-10
    )
    Ry = st.slider(
        'Ry',
        -180, 180, value=0
    )
    Rz = st.slider(
        'Rz',
        -180, 180, value=0, key="rooRz_slider"
    )
    shading_mode_label = st.selectbox(
        'What shading would you like to use?',
        options=list(shading_options.keys())
    )

    shading_mode = shading_options[shading_mode_label]
    #shadow_on = st.toggle('Shadows?"", False)
    st.form_submit_button()


if __name__ == "__main__":

#######################################################################
# File Handling and Input
#######################################################################

    json_data = handle.openJSONFile(_json_file, obj_url)
    scene_data = handle.openSceneFile(_scene_file, scene_url)
    if(shading_mode == 3): NORMAL_MAP_STRENGTH = 0

    xres = scene_data['scene']['camera']['resolution'][0]
    yres = scene_data['scene']['camera']['resolution'][1]

    texture = None
    if texture_on:
        texture = handle.openTextureFile(_texture_file)

    normal_map = None
    if normal_mapping_on:
        normal_map = handle.openTextureFile(_normal_map_file)

    filters = {}
    if anti_aliasing_on:
        _xres = xres-1
        _yres = yres-1
    
        # An AA_Render object has a dx, dy, and weight. It also stores an image and z_buffer.
        filters = {
            "1" : AA_Render(-0.52/_xres,  0.38/_yres, 0.128),
            "2" : AA_Render( 0.41/_xres,  0.56/_yres, 0.119),
            "3" : AA_Render( 0.27/_xres,  0.08/_yres, 0.294),
            "4" : AA_Render(-0.17/_xres, -0.29/_yres, 0.249),
            "5" : AA_Render( 0.58/_xres, -0.55/_yres, 0.104),
            "6" : AA_Render(-0.31/_xres, -0.71/_yres, 0.106)
        } 

#######################################################################
# Camera & Lighting Setup
#######################################################################

    up_vector = np.array([0, 1, 0])
    cam_to = np.array(scene_data['scene']['camera']['to'])
    cam_from = np.array(scene_data['scene']['camera']['from'])

    n = cam_from - cam_to
    n = np.divide(n, np.linalg.norm(n))

    u = np.array(np.cross(up_vector, n))
    u = np.divide(u, np.linalg.norm(u))

    v = np.array(np.cross(n, u))
    r = cam_from

    # load matrix M, camera matrix 
    M = scene.getCameraMatrix(u, v, n, r)

    # load matrix P, perspective projection matrix
    P = scene.getPerspectiveProjMatrix(scene_data)

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

    z_buffer = [[float('inf') for i in range(xres)] for j in range(yres)]
    ppm_list = [[0 for i in range(xres)] for j in range(yres)]

    im = Image.new('RGB', [xres, yres], 0x000000)
    shadowMap = Image.new('RGB', [xres, yres], 0x000000)
    width, height = im.size


    # set initial background color
    for y in range(height):
        for x in range(width):
            # im.putpixel( (x, y), (128, 112, 96) )
            # ppm_list[x][y] = (128, 112, 96)
            im.putpixel( (x, y), (0, 0, 0) )
            #ppm_list[x][y] = (115, 147, 179)
            # im.putpixel( (x, y), (255, 255, 255) )
            # ppm_list[x][y] = (255, 255, 255)
            
    if anti_aliasing_on:
        for key in filters:
            filters[key].z_buffer = [[float('inf') for i in range(xres)] for j in range(yres)]
            # filters[key].im = Image.new('RGB', [xres, yres], 0x000000)
            filters[key].im = Image.new('RGB', [xres, yres], (0, 147, 250)) # set the background color to your desired color 
            filters[key].im.paste(background, (0, 0, width, height))

    else:
        #Image.open(background)
        im.paste(background, (0, 0, width, height))
        # pass
            
#######################################################################
# Render Scene
#######################################################################

    # Scale, rotate and translate each teapot in the scene
    for item in scene_data['scene']['shapes']:
        print("\n", item['id'])
        T = item['transforms'][2]['T']
        S = item['transforms'][1]['S']
        #Rx = 0; Ry = 0; Rz = 0
        
        # after initializing R_x/y/z to default of 0; check teapot dict for R_x/y/z values
        if "Rx" in item['transforms'][0]:
            Rx = np.radians(Rx)
            
        if "Ry" in item['transforms'][0]:
            Ry = np.radians(Ry)
            
        if "Rz" in item['transforms'][0]:
            Rz = np.radians(Rz)
        
        T_mat  = scene.getTranslationMatrix(T)
        Rx_mat = scene.getX_RotationMatrix(Rx)
        Ry_mat = scene.getY_RotationMatrix(Ry)
        Rz_mat = scene.getZ_RotationMatrix(Rz)
        if(_json_file == './stanford-bunny.json'):
            S = [
              1,
              1,
              1
            ]
        S_mat  = scene.getScalingMatrix(S)

        
        transform_mat = T_mat @ Rx_mat @ Ry_mat @ Rz_mat @ S_mat
        
        # inverse transpose of the transformation matrix (minus translation)
        transform_it = np.transpose(np.linalg.inv(Rx_mat @ Ry_mat @ Rz_mat @ S_mat))

        # initialize the teapot's specular coefficient (n) and calculate ambient contribution 
        specular_coeff = item['material']['n']
        ambient_contribution = item['material']['Ka'] * ambient['intensity'] * np.array(ambient['color']) # ka * la
        if(shadow_on):
            add_geometry.add_floor(json_data)
            sm = shadow_map.ShadowMap(transform_mat, directional_lights[0])
            sm.calc_shadow_map(json_data)
        
        # per triangle per teapot:
        for tri in json_data['data']:

            v0 = tri['v0']['v'];  v1 = tri['v1']['v'];  v2 = tri['v2']['v']
            
            # per triangle, there are 3 normals
            n0 = np.array(tri['v0']['n']);  n1 = np.array(tri['v1']['n']);  n2 = np.array(tri['v2']['n'])

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
                                    bary_coord = Barycentric(alpha, beta, gamma)

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
                                        if perlin_on:
                                            tex_rgb = pTextureLookup(texture, u_tex, v_tex, max_rgb, TEXTURE_SCALING)
                                        else:
                                            tex_rgb = textureLookup(texture, u_tex, v_tex, TEXTURE_SCALING)
                                        # color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, ambient_contribution, bary_coord)
                                        
                                        if normal_mapping_on:
                                            if nperlin:
                                                normal_map_rgb = pnormalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                            else:
                                                normal_map_rgb = normalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                        else:
                                            normal_map_rgb = np.array([0,0,0]) 
                                        #if toon shading
                                        if shading_mode == 3:
                                            color_tuple = calcToonShadeTexNorm(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z, tex_rgb, tan0, tan1, tan2, binorm0, binorm1, binorm2, normal_map_rgb, ambient, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH)
                                        else:
                                            color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, normal_map_rgb, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH)
                                    else:

                                        if normal_mapping_on:
                                            z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                            # interpolate u, v texture coordinates
                                            u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                            v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                            # multiply by the z_at_pixel value
                                            u_tex *= z_pix;  v_tex *= z_pix
                                            normal_map_rgb = normalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)

                                            ## automatically do Phong shading
                                            if shading_mode == 3:
                                                color_tuple = calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)
                                            else:
                                                color_tuple = calcPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, normal_map_rgb, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH, isShadow)
                                            
                                        else:
                                            normal_map_rgb = np.array([0,0,0])

                                            if shading_mode == 0:
                                                color_tuple = calcFlat(directional_lights, item, n0, v0_cam, specular_coeff, ambient_contribution)

                                            elif shading_mode == 1:
                                                color_tuple = calcGoraud(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff, ambient_contribution, bary_coord)

                                            elif shading_mode == 3:
                                                color_tuple = calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)

                                            else:
                                                color_tuple = calcPhong(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH, isShadow)

                                    # put pixel into the display image 
                                    filters[n].im.putpixel( (x, y), color_tuple )
            
            else: # single rendering pass

                # convert NDC vertex coordinates to Raster Space
                x0_w = (v0_ndc[0]+1) * ( (xres - 1)/2 )
                y0_w = (1-v0_ndc[1]) * ( (yres - 1)/2 )

                x1_w = (v1_ndc[0]+1) * ( (xres - 1)/2 )
                y1_w = (1-v1_ndc[1]) * ( (yres - 1)/2 )

                x2_w = (v2_ndc[0]+1) * ( (xres - 1)/2 )
                y2_w = (1-v2_ndc[1]) * ( (yres - 1)/2 )
                # filters[key].im.paste(background, (0, 0, width, height))
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
                                bary_coord = Barycentric(alpha, beta, gamma)
                                z_buffer[x][y] = z

                                total_diffuse = 0
                                total_specular = 0
                                if(shadow_on):
                                    p_x = (alpha * v0[0]) + (beta * v1[0]) + (gamma * v2[0])
                                    p_y = (alpha * v0[1]) + (beta * v1[1]) + (gamma * v2[1])
                                    p_z = (alpha * v0[2]) + (beta * v1[2]) + (gamma * v2[2])
                                    isShadow = sm.check_shadow([p_x,p_y,p_z])
                                    isShadow = True
                                else: 
                                    isShadow = False

                                if texture_on:
                                    # calculate z at that pixel (from camera space view)
                                    z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                    # interpolate u, v texture coordinates
                                    u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                    v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                    # multiply by the z_at_pixel value
                                    u_tex *= z_pix;  v_tex *= z_pix
                                    if perlin_on:
                                        tex_rgb = pTextureLookup(texture, u_tex, v_tex, max_rgb, TEXTURE_SCALING)
                                    else:
                                        tex_rgb = textureLookup(texture, u_tex, v_tex, TEXTURE_SCALING)

                                    # if nperlin:
                                    #     normal_map_rgb = pnormalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                    # else:
                                    #     normal_map_rgb = normalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                    
                                    if normal_mapping_on:
                                        if nperlin:
                                            normal_map_rgb = pnormalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                        else:
                                            normal_map_rgb = normalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                    else:
                                        normal_map_rgb = np.array([0,0,0]) 
                                    #if toonshading
                                    if shading_mode == 3:
                                        color_tuple = calcToonShadeTexNorm(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z, tex_rgb, tan0, tan1, tan2, binorm0, binorm1, binorm2, normal_map_rgb, ambient, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH)
                                    else:
                                        color_tuple = calcTexPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, tex_rgb, normal_map_rgb, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH)

                                else:
                                    if normal_mapping_on:
                                        z_pix = 1 / ( (alpha * z0_inv) + (beta * z1_inv) + (gamma * z2_inv) )

                                        # interpolate u, v texture coordinates
                                        u_tex = (alpha * uv_0[0]) + (beta * uv_1[0]) + (gamma * uv_2[0])
                                        v_tex = (alpha * uv_0[1]) + (beta * uv_1[1]) + (gamma * uv_2[1])

                                        # multiply by the z_at_pixel value
                                        u_tex *= z_pix;  v_tex *= z_pix
                                        if nperlin:
                                            normal_map_rgb = pnormalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                        else:
                                            normal_map_rgb = normalMapLookup(normal_map, u_tex, v_tex, NORMAL_MAP_SCALING)
                                        ## automatically do Phong shading
                                        #if toon shading
                                        if shading_mode == 3:
                                            olor_tuple = calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)
                                        else:
                                            color_tuple = calcPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, normal_map_rgb, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH, isShadow)

                                    else:
                                        normal_map_rgb = np.array([0,0,0]) 

                                        if shading_mode == 0:
                                            color_tuple = calcFlat(directional_lights, item, n0, v0_cam, specular_coeff, ambient_contribution)

                                        elif shading_mode == 1:
                                            color_tuple = calcGoraud(directional_lights, item, n0, n1, n2, v0_cam, v1_cam, v2_cam, specular_coeff, ambient_contribution, bary_coord)

                                        elif shading_mode == 3:
                                            color_tuple = calcToonShade(directional_lights, n0, n1, n2, alpha, beta, gamma, item, z)

                                        else:
                                            color_tuple = calcPhong(directional_lights, item, n0, n1, n2,tan0, tan1, tan2, binorm0, binorm1, binorm2, v0_cam, v1_cam, v2_cam, specular_coeff, normal_map_rgb, ambient_contribution, bary_coord, normal_mapping_on, NORMAL_MAP_STRENGTH, isShadow)

                                # put pixel into the display image and 
                                #  also into the ppm array that will save the image
                                im.putpixel( (x, y), color_tuple )
                                ppm_list[x][y] = color_tuple
        
    if anti_aliasing_on:
        # average & combine the 6 different renders into final img
        for y in range(height):
            for x in range(width):
                r = 0; g = 0; b = 0
                
                for n in filters:
                    weight = filters[n].weight
                    color = filters[n].im.getpixel( (x, y) )
                    r += color[0] * weight;  g += color[1] * weight;  b += color[2] * weight
                    
                r = int(r); g = int(g); b = int(b)
                ppm_list[x][y] = (r, g, b)
                im.putpixel( (x, y), (r, g, b) )


    # display(im)
    # im.show()
    st.image(im)

            
#######################################################################
# Save Image as PPM
#######################################################################

    # handle.saveAsPPM(xres, yres, max_rgb, ppm_list, './FinalProject_output.ppm')