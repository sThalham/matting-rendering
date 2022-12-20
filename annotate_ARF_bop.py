import blenderproc as bproc
import argparse
import os
import sys
import numpy as np
import bpy
import json

import cv2
import transforms3d as tf3d

sys.path.append(os.getcwd())

import render_utils as utils
from flowCalibrator import FlowCalibrator

parser = argparse.ArgumentParser()
parser.add_argument('--root', help="Path to base directory", type=str)
parser.add_argument('--split', help="Dataset split name", type=str)
parser.add_argument('--meshes', help="Meshes dir in root", type=str)
args = parser.parse_args()

#root_path = '/home/stefan/matting_rendering'
root_path = args.root
split_path = os.path.join(args.root, args.split)
meshes_path = os.path.join(args.root, args.meshes)
back_dir = os.path.join(os.getcwd(), 'graycode_512_512')
out_dir = os.path.join(root_path, 'arf_annotation')
calib_dir = os.path.join(out_dir, 'calib')
name_template = 'graycode_00'
if os.path.exists(calib_dir) == False:
    os.makedirs(calib_dir)

#camera template
#camera_name = 'camera_' + args.split.split("_")[-1] + '.json'
camera_name = 'camera.json'
gen_cam_path = os.path.join(args.root, camera_name)
with open(gen_cam_path, 'r') as stream_temp:
    cam_data = json.load(stream_temp)

img_x, img_y = cam_data["width"], cam_data["height"]
fx = cam_data["fx"]
fy = cam_data["fy"]
cx = cam_data["cx"]
cy = cam_data["cy"]

bproc.init()
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 10
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)

object_dict = {}
for odx, mesh_name in enumerate(os.listdir(meshes_path)):
    if not mesh_name.endswith('.ply'):
        continue
    obj = bproc.loader.load_obj(os.path.join(meshes_path, mesh_name))[0]
    obj.set_cp("category_id", int(mesh_name[4:-4]))
    #obj.set_scale([0.001, 0.001, 0.001])
    obj.set_scale([0.0005, 0.0005, 0.0005])
    for mat in obj.get_materials():
        mat.map_vertex_color()
        IOR = 1.5
        glass = mat.new_node('ShaderNodeBsdfGlass')
        glass.inputs[1].default_value = 0.05  # Roughness
        glass.inputs[2].default_value = IOR  # IOR
        material_output = mat.get_the_one_node_with_type("OutputMaterial")
        link_glass_mat = mat.link(glass.outputs[0], material_output.inputs[0])
    obj.set_shading_mode('auto')
    obj.hide(True)

    object_dict[str(int(mesh_name[4:-4]))] = obj


room_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 0], rotation=[np.pi, 0, 0])
room_plane.add_uv_mapping('smart')


bproc_keyframe = 0

subsets = os.listdir(split_path)
for sdx, subset in enumerate(subsets):

    training_samples = os.listdir(os.path.join(split_path, subset, 'rgb'))
    calib_path = os.path.join(split_path, subset, 'scene_camera.json')
    anno_path = os.path.join(split_path, subset, 'scene_gt.json')
    info_path = os.path.join(split_path, subset, 'scene_gt_info.json')

    if os.path.exists(calib_path):
        with open(calib_path, 'r') as streamCAM:
            camjson = json.load(streamCAM)

    with open(anno_path, 'r') as streamGT:
        scenejson = json.load(streamGT)

    with open(info_path, 'r') as streamINFO:
        gtjson = json.load(streamINFO)

    for tdx, train_sample in enumerate(training_samples):

        samp = str(int(train_sample[:-4]))

        if os.path.exists(calib_path):
            calib = camjson.get(str(samp))
            K = calib["cam_K"]
            depSca = calib["depth_scale"]
            fxca = K[0]
            fyca = K[4]
            cxca = K[2]
            cyca = K[5]

        gtPoses = scenejson.get(str(samp))
        gtBoxes = gtjson.get(str(samp))

        # initialize calibration list
        calib_images = [None] * 21

        for adx in range(len(gtBoxes)):

            bbox = gtBoxes[adx]["bbox_obj"]
            obj_id = gtPoses[adx]['obj_id']
            R = gtPoses[adx]["cam_R_m2c"]
            t = gtPoses[adx]["cam_t_m2c"]

            R = np.asarray(R, dtype=np.float32).reshape(3, 3)
            t = np.asarray(t, dtype=np.float32) * 0.001
            #t[2] = t[2] * 0.5

            obj = object_dict[str(obj_id)]
            obj.hide(False)
            obj.set_location(t)
            roll, pitch, yaw = tf3d.euler.mat2euler(R)
            obj.set_rotation_euler([roll, pitch, yaw])

            ###################
            # position camera facing to object
            # using bounding box to comp distance
            max_box = np.max([bbox[2], bbox[3]])
            max_arg = np.argmax([bbox[2], bbox[3]])
            f_temp = fx
            if max_arg == 1:
                f_temp = fy
            f_adapt = (f_temp * 512.0) / (max_box * 0.5)

            cam2world = np.array([
                [1, 0, 0, t[0]],
                [0, -1, 0, t[1]],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            bproc.utility.reset_keyframes()
            bproc.camera.add_camera_pose(cam2world)
            bproc_keyframe += 1
            #bproc.camera.add_camera_pose(cam2world)

            # z/f = z'/f'
            bproc.camera.set_intrinsics_from_K_matrix(
                [[f_adapt, 0.0, 256],
                [0.0, f_adapt, 256],
            [0.0, 0.0, 1.0]], 512, 512
            )

            # compute background plane size
            plane_width = ((512.0 * (t[2] + 0.1)) / f_adapt) * 0.5
            plane_height = ((512.0 * (t[2] + 0.1)) / f_adapt) * 0.5
            room_plane.set_scale([plane_width, plane_height, 1])

            #shift_plane_factor = (poi[2] + 1.0) / poi[2]
            plane_location = [t[0], t[1], t[2] + 0.1]
            room_plane.set_location(plane_location)

            data_seg = bproc.renderer.render_segmap()
            img_seg = data_seg["class_segmaps"][0]
            save_img = os.path.join(out_dir, subset, train_sample[:-4] + '_' + str(adx) + '_mask.png')
            if os.path.exists(os.path.join(out_dir, subset)) == False:
                os.makedirs(os.path.join(out_dir, subset))
            img_seg = np.where(img_seg==0, 1, 0)
            cv2.imwrite(save_img, img_seg*255)

            calib_images[1] = img_seg*255

            light_plane_mat = bproc.material.create('light_material')
            light_plane_mat.make_emissive(emission_strength=1, emission_color=[1.0, 1.0, 1.0, 1.0])

            emission = light_plane_mat.get_the_one_node_with_type('ShaderNodeEmission')
            emission.inputs[1].default_value = 1.0
            material_output = light_plane_mat.get_the_one_node_with_type("OutputMaterial")
            link_emission_output = light_plane_mat.link(emission.outputs[0], material_output.inputs[0])
            room_plane.replace_materials(light_plane_mat)

            data_rho = bproc.renderer.render()
            img_rho = data_rho["colors"][0]
            save_img = os.path.join(out_dir, subset, train_sample[:-4] + '_' + str(adx) + '_rho.png')
            if os.path.exists(os.path.join(out_dir, subset)) == False:
                os.makedirs(os.path.join(out_dir, subset))
            # normalize
            img_rho = np.mean(img_rho, axis=2)
            max_rho = np.nanmax(img_rho)
            img_rho = (img_rho / max_rho) * 255.0
            #img_rho = img_rho.astype(np.uint8)
            cv2.imwrite(save_img, img_rho)
            calib_images[2] = img_rho

            for iidx, b_img in enumerate(os.listdir(back_dir)):

                image_path = os.path.join(back_dir, b_img)
                image = bpy.data.images.load(filepath=image_path)
                plane_mat = bproc.material.create_material_from_texture(image, 'background_image_' + str(iidx))
                plane_mat.make_emissive(emission_strength=1, emission_color=[1.0, 1.0, 1.0, 1.0])
                #texture = plane_mat.new_node('ShaderNodeTexImage')
                texture = plane_mat.get_the_one_node_with_type("ShaderNodeTexImage")
                emission = plane_mat.new_node('ShaderNodeEmission')
                material_output = plane_mat.get_the_one_node_with_type("OutputMaterial")

                link_texture_emission = plane_mat.link(texture.outputs[0], emission.inputs[0])
                link_emission_output = plane_mat.link(emission.outputs[0], material_output.inputs[0])
                room_plane.replace_materials(plane_mat)

                # render the whole pipeline
                data = bproc.renderer.render()
                img = data["colors"][0]

                # normalize
                #img = np.mean(img, axis=2)
                min_img = np.nanmin(img)
                max_img = np.nanmax(img)
                #img = img - min_img
                #img = img / (max_img - min_img) * 255.0
                #img = img.astype(np.uint8)
                img = img / max_img
                img = img * 255.0

                # binary
                #img = img / 255.0
                #img = (np.round(img) * 255.0).astype(np.uint8)

                out_index = b_img.split("_")[-1][:-4]
                out_enum = str(int(out_index)+2)
                out_img_name = name_template[:-len(out_enum)] + out_enum + '.png'
                out_calib = os.path.join(calib_dir, out_img_name)
                cv2.imwrite(out_calib, img)
                calib_images[int(out_index)+2] = img

            imgs = calib_images[1:]
            utils.listRgb2Gray(imgs)

            calibrator = FlowCalibrator(imgs, os.path.join(out_dir, subset), train_sample[:-4] + '_' + str(adx) + '_flow')
            calibrator.findCorrespondence()

            obj.hide(True)



