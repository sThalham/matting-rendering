import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy

parser = argparse.ArgumentParser()
#parser.add_argument('output_dir', help="Path to where the final files will be saved ")
args = parser.parse_args()

out_dir = '/home/stefan/matting_rendering/images'

bproc.init()

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[537.4799, 0.0, 318.8965],
     [0.0, 536.1447, 238.3781],
     [0.0, 0.0, 1.0]], 640, 480
)
# Set camera pose via cam-to-world transformation matrix
cam2world = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
bproc.camera.add_camera_pose(cam2world)

# load the objects into the scene
obj = bproc.loader.load_obj('/home/stefan/matting_rendering/obj_000001.ply')[0]
obj.set_cp("category_id", 1)
obj.set_scale([0.001, 0.001, 0.001])
# Use vertex color for texturing
# https://dlr-rm.github.io/BlenderProc/_modules/blenderproc/python/types/MaterialUtility.html
for mat in obj.get_materials():

    mat.map_vertex_color()

    IOR = 1.5

    #general transparency
    #fresnel = mat.new_node('ShaderNodeFresnel')
    #fresnel.inputs[0].default_value = IOR
    #gloss = mat.new_node('ShaderNodeBsdfGlossy')
    ##gloss.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0] # color
    #gloss.inputs[1].default_value = 0.0  # roughness
    #trans = mat.new_node('ShaderNodeBsdfTransparent')
    #trans.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0] # color
    #mix = mat.new_node('ShaderNodeMixShader')
    #material_output = mat.get_the_one_node_with_type("OutputMaterial")

    #link_fresnel_mix = mat.link(fresnel.outputs[0], mix.inputs[0])
    #link_gloss_mix = mat.link(gloss.outputs[0], mix.inputs[1])
    #link_trans_mix = mat.link(trans.outputs[0], mix.inputs[2])
    #link_mix_mat = mat.link(mix.outputs[0], material_output.inputs[0])

    # glass transparency
    glass = mat.new_node('ShaderNodeBsdfGlass')
    glass.inputs[1].default_value = 0.2  # Roughness
    glass.inputs[2].default_value = IOR  # IOR
    material_output = mat.get_the_one_node_with_type("OutputMaterial")
    link_glass_mat = mat.link(glass.outputs[0], material_output.inputs[0])

# Set pose of object via local-to-world transformation matrix
# set object location as is, since camera==world
obj.set_location([0.0, 0.0, 0.3])
obj.set_rotation_euler([0.0, np.pi*0.5, 0.0])

obj.set_shading_mode('auto')

# create room
room_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 3.5], rotation=[np.pi, 0, 0])
image = bpy.data.images.load(filepath='/home/stefan/matting_rendering/COCO_train2014_000000000009.jpg')
plane_mat = bproc.material.create_material_from_texture(image, 'background_image_000')
room_plane.replace_materials(plane_mat)
room_plane.add_uv_mapping('smart')

room_plane1 = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[2.0, 0, 1.5], rotation=[0, np.pi*0.5, 0])
room_plane2 = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[-2.0, 0, 1.5], rotation=[0, -np.pi*0.5, 0])
room_plane3 = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 2.0, 1.5], rotation=[np.pi*0.5, 0, 0])
room_plane4 = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -2.0, 1.5], rotation=[-np.pi*0.5, 0, 0])

room_plane1.replace_materials(plane_mat)
room_plane1.add_uv_mapping('smart')
room_plane2.replace_materials(plane_mat)
room_plane2.add_uv_mapping('smart')
room_plane3.replace_materials(plane_mat)
room_plane3.add_uv_mapping('smart')
room_plane4.replace_materials(plane_mat)
room_plane4.add_uv_mapping('smart')

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, -10], rotation=[0, 0, 0])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
#light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6),
#                                    emission_color=np.random.uniform([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]))
light_plane_material.make_emissive(emission_strength=5, emission_color=[1.0, 1.0, 1.0, 1.0])
light_plane.replace_materials(light_plane_material)

# sample point light on shell
#light_point = bproc.types.Light()
#light_point.set_energy(200)
#light_point.set_color(np.random.uniform([1.0,1.0,1.0],[1,1,1]))
#location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
#                            elevation_min = 5, elevation_max = 89)
#light_point.set_location([0.0, 0.0, -1.0])

#cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
#bproc.camera.add_camera_pose(cam2world)

poi = [0.0, 0.0, 1.0]
# Compute rotation based on vector going from location towards poi
rotation_matrix = bproc.camera.rotation_from_forward_vec(poi)#, inplane_rot=np.random.uniform(-0.7854, 0.7854))
# Add homog cam pose based on location an rotation
cam2world_matrix = bproc.math.build_transformation_mat([0.0, 0.0, 1.0], rotation_matrix)
#bproc.camera.add_camera_pose(cam2world_matrix)


cam_pose = bproc.camera.get_camera_pose(frame=None)
print('cam_pose: ', cam_pose)
t_obj = bproc.object.compute_poi([obj])
print('t_obj: ', t_obj)

# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 100
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)
# render the whole pipeline
data = bproc.renderer.render()

#bproc.writer.write_bop(args.output_dir, [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)
bproc.writer.write_bop(out_dir, [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)

