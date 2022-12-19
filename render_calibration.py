import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
import cv2
import transforms3d as tf3d

parser = argparse.ArgumentParser()
parser.add_argument('--root', help="Path to where the final files will be saved ")
parser.add_argument('--obj', help="Path to where the final files will be saved ")
args = parser.parse_args()

#root_path = '/home/stefan/matting_rendering'
root_path = args.root
obj_name = args.obj

back_dir = os.path.join(root_path, 'graycode_512_512')
out_dir = os.path.join(root_path, 'Images')
calib_dir = os.path.join(out_dir, 'Calibration')
mesh_dir = os.path.join(root_path, 'bottle.ply')
scene_img = os.path.join(root_path, '000061.png')

# need to implement data loader for bop annotation
# poses, intrinsics, names, mesh = data_loader(bop_base_dir)
# poses: [[4x4], ....]
# intrinsics: [{"cx": 325.2611,
#   "cy": 242.04899,
#   "depth_scale": 1.0,
#   "fx": 572.4114,
#   "fy": 573.57043,
#   "height": 480,
#   "width": 640}, ....]
# name: [{"data": "train", "dir": "000000", "img":, "000035"}, ....]
# mesh: bottle.ply


#for pdx, (pose, intri, name) in enumerate(zip(poses, intrinsics, names)):

bproc.init()
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 10
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)


###################################
# keypose camera
################################
#resx: 1280
#resy: 720
#num_kp: 2
#camera {
#  fx: 675.61713
#  fy: 675.61713
#  cx: 632.1181
#  cy: 338.28537
#  baseline: 0.120007
#  resx: 1280.0
#  resy: 720.0
#}

#######################################
# example for image to crop transform
# 000000
#bbox = [578.0019607843137, 379.0019607843137, 42.99607843137255, 80.99607843137255]
#obj_x = -0.034610493479483875
#obj_y = 0.0876175923765
#obj_z = 0.7220573983027752
#obj_rot = np.array([0.9985835999418678, 0.0, -0.05320520582744299, -0.0438628583287136, -0.5659943101033659, -0.8232414533961775, -0.030113843766211186, 0.8244091465592889, -0.5651926357296327]).reshape((3,3))

# 000061
bbox =  [563.0019607843137, 583.0019607843137, 36.99607843137255, 61.99607843137255]
obj_x = -0.06191940905313875
obj_y = 0.337938141247261
obj_z = 0.8315059198991654
obj_rot = np.array([0.9999277279972268,
   0.0,
   -0.012022428303010842,
   -0.009611888642069338,
   -0.6006708255034048,
   -0.7994386599269451,
   -0.007221521933325013,
   0.7994964411359554,
   -0.600627413819839]).reshape((3,3))

# pose 000118.png
#obj_rot = np.array([0.7803792476425656,
#   0.6246155216319488,
#   -0.029388432848193122,
#   0.2972507968656275,
#   -0.4119061234725049,
#   -0.8613798866984319,
#   -0.5501365227028412,
#   0.663467252833403,
#   -0.5071104522745751]).reshape(3,3)
#obj_x, obj_y, obj_z = 0.17482042977070372, 0.34412885644715147, 0.9911807242853785
#bbox = [735.0019607843137, 544.0019607843137, 31.996078431372553, 57.99607843137255]

################
# place object
# those are placeholders, to be replaced for online computation
img_x, img_y = 960, 720
fx = 675.61713 / 1.5
fy = 675.61713 / 1.5
cx = (632.1181 - 160.0) / 1.5
cy = 338.28537 / 1.5

#bproc.camera.set_intrinsics_from_K_matrix(
#    [[fx, 0.0, img_x * 0.5],
#     [0.0, fy, img_y * 0.5],
#     [0.0, 0.0, 1.0]], img_x, img_y
#)
#######################

# load the objects into the scene
obj = bproc.loader.load_obj(mesh_dir)[0]
obj.set_cp("category_id", 1)
obj.set_scale([0.001, 0.001, 0.001])
obj.set_location([obj_x, obj_y, obj_z])
roll, pitch, yaw = tf3d.euler.mat2euler(obj_rot)
obj.set_rotation_euler([roll, pitch, yaw])

###################
# position camera facing to object
# using bounding box to comp distance
max_box = np.max([bbox[2], bbox[3]])# * 1.5
max_arg = np.argmax([bbox[2], bbox[3]])

f_temp = fx
if max_arg == 1:
    f_temp = fy

# z/f = z'/f'
#adj_f = max_box / 512.0
#adj_f = 512.0 / max_box
#f_temp = max_box / 960
f_temp = (f_temp * 512.0) / max_box

cam2world = np.array([
    [1, 0, 0, obj_x],
    [0, -1, 0, obj_y],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
bproc.camera.add_camera_pose(cam2world)

# z/f = z'/f'
bproc.camera.set_intrinsics_from_K_matrix(
    [[f_temp, 0.0, 256],
     [0.0, f_temp, 256],
     [0.0, 0.0, 1.0]], 512, 512
)

print('f: ', f_temp)

# compute background plane size
plane_width = ((512.0 * (obj_z + 0.1)) / f_temp) * 0.5
plane_height = ((512.0 * (obj_z + 0.1)) / f_temp) * 0.5

#shift_plane_factor = (poi[2] + 1.0) / poi[2]
plane_location = [obj_x, obj_y, obj_z + 0.1]

room_plane = bproc.object.create_primitive('PLANE', scale=[plane_width, plane_height, 1], location=plane_location, rotation=[np.pi, 0, 0])
room_plane.add_uv_mapping('smart')

for mat in obj.get_materials():

    mat.map_vertex_color()

    IOR = 1.5

    # glass transparency
    glass = mat.new_node('ShaderNodeBsdfGlass')
    glass.inputs[1].default_value = 0.05  # Roughness
    glass.inputs[2].default_value = IOR  # IOR
    material_output = mat.get_the_one_node_with_type("OutputMaterial")
    link_glass_mat = mat.link(glass.outputs[0], material_output.inputs[0])

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

'''
fx = 675.61713
fy = 675.61713
cx = 632.1181
cy = 338.28537

cam2world = np.array([
    [1, 0, 0, -obj_x],
    [0, -1, 0, -obj_y],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
bproc.camera.add_camera_pose(cam2world)

bproc.camera.set_intrinsics_from_K_matrix(
    [[fx, 0.0, cx],
     [0.0, fy, cy],
     [0.0, 0.0, 1.0]], 1280, 720
)

plane_width = ((1280 * (obj_z + 0.1)) / fx) * 0.5
plane_height = ((720 * (obj_z + 0.1)) / fy) * 0.5
#plane_x = ((640 - cx) / fx) * 0.5
#plane_y = ((360 - cy) / fy) * 0.5
plane_x = 0.0
plane_y = 0.0
plane_location = [plane_x, plane_y, obj_z + 0.1]
room_plane.set_location(plane_location)
room_plane.set_scale([plane_width, plane_height, 1])

plane_mat = bproc.material.create_material_from_texture(scene_img, 'coco_image')
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
'''


