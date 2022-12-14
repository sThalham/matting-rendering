import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root', help="Path to where the final files will be saved ")
parser.add_argument('--obj', help="Path to where the final files will be saved ")
args = parser.parse_args()

#root_path = '/home/stefan/matting_rendering'
root_path = args.root
obj_name = args.obj
print('obj_name: ', obj_name)

back_dir = os.path.join(root_path, 'graycode_512_512')
out_dir = os.path.join(root_path, 'Images')
calib_dir = os.path.join(out_dir, 'Calibration')
mesh_dir = os.path.join(root_path, 'obj_000001.ply')

bproc.init()
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 10
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)


# Set intrinsics via K matrix
#bproc.camera.set_intrinsics_from_K_matrix(
#    [[537.4799, 0.0, 318.8965],
#     [0.0, 536.1447, 238.3781],
#     [0.0, 0.0, 1.0]], 640, 480
#)

img_x, img_y = 512, 512
fx, fy = 537.4799, 536.1447

bproc.camera.set_intrinsics_from_K_matrix(
    [[fx, 0.0, img_x * 0.5],
     [0.0, fy, img_y * 0.5],
     [0.0, 0.0, 1.0]], img_x, img_y
)

# compute background plane size
plane_width = ((img_x * 1000.0) / fx) * 0.0005
plane_height = ((img_y * 1000.0) / fx) * 0.0005
print(plane_width, plane_height)
room_plane = bproc.object.create_primitive('PLANE', scale=[plane_width, plane_height, 1], location=[0.0, 0.0, 1.0], rotation=[np.pi, 0, 0])
room_plane.add_uv_mapping('smart')
#light_plane_material = bproc.material.create('light_material')
#light_plane_material.make_emissive(emission_strength=10, emission_color=[1.0, 1.0, 1.0, 1.0])
#room_plane.replace_materials(light_plane_material)

# Set camera pose via cam-to-world transformation matrix
cam2world = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
bproc.camera.add_camera_pose(cam2world)

poi = [0.0, 0.0, 1.0]
rotation_matrix = bproc.camera.rotation_from_forward_vec(poi)#, inplane_rot=np.random.uniform(-0.7854, 0.7854))
cam2world_matrix = bproc.math.build_transformation_mat([0.0, 0.0, 1.0], rotation_matrix)

# sample point light on shell
#light_point = bproc.types.Light()
#light_point.set_energy(200)
#light_point.set_color(np.random.uniform([1.0,1.0,1.0],[1,1,1]))
#location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
#                            elevation_min = 5, elevation_max = 89)
#light_point.set_location([0.0, 0.0, -1.0])

# load the objects into the scene
obj = bproc.loader.load_obj(mesh_dir)[0]
obj.set_cp("category_id", 1)
obj.set_scale([0.001, 0.001, 0.001])

for mat in obj.get_materials():

    mat.map_vertex_color()

    IOR = 1.5

    # glass transparency
    glass = mat.new_node('ShaderNodeBsdfGlass')
    glass.inputs[1].default_value = 0.05  # Roughness
    glass.inputs[2].default_value = IOR  # IOR
    material_output = mat.get_the_one_node_with_type("OutputMaterial")
    link_glass_mat = mat.link(glass.outputs[0], material_output.inputs[0])

obj.set_location([0.0, 0.0, 0.3])
obj.set_rotation_euler([np.random.random() * np.pi, np.random.random() * np.pi, np.random.random() * np.pi])

obj.set_shading_mode('auto')

name_template = 'graycode_00'

data_seg = bproc.renderer.render_segmap()
img_seg = data_seg["class_segmaps"][0]
save_img = os.path.join(out_dir, obj_name + '_mask.png')
cv2.imwrite(save_img, img_seg*255)
out_img_name = name_template[:-1] + str(1) + '.png'
out_calib = os.path.join(out_dir, 'Calibration', obj_name)
save_img = os.path.join(out_calib, out_img_name)
if os.path.exists(out_calib) == False:
    os.makedirs(out_calib)
img_seg = np.where(img_seg==0, 1, 0)
cv2.imwrite(save_img, img_seg*255)

light_plane_mat = bproc.material.create('light_material')
light_plane_mat.make_emissive(emission_strength=1, emission_color=[1.0, 1.0, 1.0, 1.0])

emission = light_plane_mat.get_the_one_node_with_type('ShaderNodeEmission')
emission.inputs[1].default_value = 1.0
material_output = light_plane_mat.get_the_one_node_with_type("OutputMaterial")
link_emission_output = light_plane_mat.link(emission.outputs[0], material_output.inputs[0])
room_plane.replace_materials(light_plane_mat)

data_rho = bproc.renderer.render()
img_rho = data_rho["colors"][0]
save_img = os.path.join(out_dir, obj_name + '_rho.png')
cv2.imwrite(save_img, img_rho)
out_img_name = name_template[:-1] + str(2) + '.png'
out_calib = os.path.join(out_dir, 'Calibration', obj_name)
save_img = os.path.join(out_calib, out_img_name)
# normalize
min_img = np.nanmin(img_rho)
max_img = np.nanmax(img_rho)
img_rho = img_rho - min_img
img_rho = img_rho / (max_img - min_img) * 255.0
img_rho = img_rho.astype(np.uint8)
cv2.imwrite(save_img, img_rho)

for iidx, b_img in enumerate(os.listdir(back_dir)):

    image_path = os.path.join(back_dir, b_img)
    image = bpy.data.images.load(filepath=image_path)
    plane_mat = bproc.material.create_material_from_texture(image, 'background_image_' + str(iidx))
    plane_mat.make_emissive(emission_strength=1, emission_color=[1.0, 1.0, 1.0, 1.0])
    #texture = plane_mat.new_node('ShaderNodeTexImage')
    texture = plane_mat.get_the_one_node_with_type("ShaderNodeTexImage")
    emission = plane_mat.new_node('ShaderNodeEmission')
    emission.inputs[1].default_value = 1.0
    material_output = plane_mat.get_the_one_node_with_type("OutputMaterial")

    link_texture_emission = plane_mat.link(texture.outputs[0], emission.inputs[0])
    link_emission_output = plane_mat.link(emission.outputs[0], material_output.inputs[0])

    room_plane.replace_materials(plane_mat)

    # render the whole pipeline
    data = bproc.renderer.render()
    img = data["colors"][0]

    # normalize
    min_img = np.nanmin(img)
    max_img = np.nanmax(img)
    img = img - min_img
    img = img / (max_img - min_img) * 255.0
    img = img.astype(np.uint8)
    #img = img / (max_img - min_img) * 1.0
    #img = (np.round(img) * 255.0).astype(np.uint8)

    # binary
    #img = img / 255.0
    #img = (np.round(img) * 255.0).astype(np.uint8)

    out_img_name = b_img.split("_")[-1][:-4]
    out_enum = str(int(out_img_name)+2)
    print(out_img_name)
    print(int(out_img_name))
    print(int(out_img_name) + 2)
    print(str(int(out_img_name)+2))
    out_img_name = name_template[:-len(out_enum)] + out_enum + '.png'
    out_calib = os.path.join(out_dir, 'Calibration', obj_name)
    save_img = os.path.join(out_calib, out_img_name)

    #img = cv2.blur(img, (3,3))
    cv2.imwrite(save_img, img)

# sample light color and strenght from ceiling

#light_plane = bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, -10], rotation=[0, 0, 0])
#light_plane.set_name('light_plane')
#light_plane_material = bproc.material.create('light_material')
#light_plane_material.make_emissive(emission_strength=5, emission_color=[1.0, 1.0, 1.0, 1.0])
#light_plane.replace_materials(light_plane_material)

coco_img = os.path.join(root_path, 'COCO_train2014_000000000009.jpg')
image = bpy.data.images.load(filepath=coco_img)
plane_mat = bproc.material.create_material_from_texture(coco_img, 'coco_image')
plane_mat.make_emissive(emission_strength=10, emission_color=[1.0, 1.0, 1.0, 1.0])
texture = plane_mat.get_the_one_node_with_type("ShaderNodeTexImage")
emission = plane_mat.new_node('ShaderNodeEmission')
emission.inputs[1].default_value = 1.0
material_output = plane_mat.get_the_one_node_with_type("OutputMaterial")
link_texture_emission = plane_mat.link(texture.outputs[0], emission.inputs[0])
link_emission_output = plane_mat.link(emission.outputs[0], material_output.inputs[0])
room_plane.replace_materials(plane_mat)

data_img = bproc.renderer.render()
img_img = data_img["colors"][0]
save_img = os.path.join(out_dir, obj_name + '.png')
cv2.imwrite(save_img, img_img)
