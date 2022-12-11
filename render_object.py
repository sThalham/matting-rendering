import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy

parser = argparse.ArgumentParser()
#parser.add_argument('output_dir', help="Path to where the final files will be saved ")
args = parser.parse_args()

out_dir = '/home/stefan/transparent_target_rendering/images'

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
obj = bproc.loader.load_obj('/home/stefan/transparent_target_rendering/obj_000001.ply')[0]
obj.set_cp("category_id", 1)
obj.set_scale([0.001, 0.001, 0.001])
# Use vertex color for texturing
print('mat: ', obj.get_materials())
for mat in obj.get_materials():
    mat.map_vertex_color()
    #mat.set_principled_shader_value("Base Color", [1, 1, 1, 0.7])
    #mat.set_principled_shader_value("Roughness", 0.05)
    #mat.set_principled_shader_value("Metallic", 0.2)
    #mat.set_principled_shader_value("Specular", 0.0)
    #mat.set_principled_shader_value("Transmission", 1.0)
    #mat.set_principled_shader_value("Transmission Roughness", 0.0)
    #mat.set_principled_shader_value("IOR", 1.5)
    #mat.set_principled_shader_value("Subsurface IOR", 1.0)
    #mat.set_principled_shader_value("Subsurface IOR", 1.5)
    #mat.set_principled_shader_value("Alpha", 0.5)

    bpy.data.materials[mat.get_name()].use_nodes = True
    #mat.use_nodes = True
    print(mat)
    print(mat.nodes)
    nodes = mat.nodes
    scene = bpy.context.scene
    node_tree = scene.node_tree

    # Add a diffuse shader and set its location:
    layer_weight = nodes.new('ShaderNodeLayerWeight')
    rgb = nodes.new('ShaderNodeRGBCurve')
    link_fresnel = node_tree.links.new(layer_weight.outputs[1], rgb.inputs[1])

    gloss = nodes.new('ShaderNodeBsdfGlossy')
    #gloss.inputs['Strength'].default_value = 5.0
    trans = nodes.new('ShaderNodeBsdfTransparent')
    #trans.inputs['Strength'].default_value = 5.0
    mix = nodes.new('ShaderNodeBsdfGlossy')

    link_rgb_mix = node_tree.links.new(rgb.outputs[0], mix.inputs[0])
    link_gloss_mix = node_tree.links.new(gloss.outputs[0], mix.inputs[1])
    link_trans_mix = node_tree.links.new(trans.outputs[0], mix.inputs[2])

    # link emission shader to material
    link_mix_mat = mat.node_tree.links.new(material_output.inputs[0], mix.outputs[0])

# Set pose of object via local-to-world transformation matrix

#obj.set_local2world_mat(
#    [[1.0, 0.0, 0.0, 0.0],
#    [0.0, 1.0, 0.0, 0.0],
#    [0.0, 0.0, 1.0, 1.0],
#    [0, 0, 0, 1.0]]
#)
obj.set_location([0.0, 0.0, 0.3])
obj.set_rotation_euler([0.0, np.pi*0.5, 0.0])

obj.set_shading_mode('auto')

# create room
room_plane = bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0, 3.0], rotation=[np.pi, 0, 0])
image = bpy.data.images.load(filepath='/home/stefan/transparent_target_rendering/COCO_train2014_000000000009.jpg')
plane_mat = bproc.material.create_material_from_texture(image, 'background_image_000')
for mat in room_plane.get_materials():
    print('mat: ', mat)
# Set it as base color of the current material
#plane_mat.set_principled_shader_value("Base Color", image)
room_plane.replace_materials(plane_mat)
room_plane.add_uv_mapping('smart')

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

