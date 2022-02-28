from hashlib import new
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy
        
# Helper function to draw registrations (reccomended)
def draw_registrations(source, target, transformation = None, recolor = False):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if(recolor):
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
        if(transformation is not None):
            source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

images = [x for x in range(400)]
images = ['{:>03}'.format(number) for number in images]
# print(images)
# exit()
recon = None

# Read in images. We have images 000000 - 0000400
color_raw0 = o3d.io.read_image(f"RGBD/color/000{images[0]}.jpg")
depth_raw0 = o3d.io.read_image(f"RGBD/depth/000{images[0]}.png")
color_raw1 = o3d.io.read_image(f"RGBD/color/000{images[1]}.jpg")
depth_raw1 = o3d.io.read_image(f"RGBD/depth/000{images[1]}.png")

rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw0, 
    depth_raw0, 
    convert_rgb_to_intensity = True)

rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw1, 
    depth_raw1, 
    convert_rgb_to_intensity = True)

# Source pointcloud
camera = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

source = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image0, camera)

# Target pointcloud
target = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, camera)

# Flip it, otherwise the pointcloud will be upside down
source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Parameters
threshold = 0.02
trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

#Evaluate registration
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)

voxel_size = 0.05
source = source.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                            max_nn=30))
source_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(source,
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,max_nn=100))

target = target.voxel_down_sample(voxel_size)
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                            max_nn=30))
target_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(target,
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))

###
# ICP code here
###
loss = o3d.pipelines.registration.TukeyLoss(k=.1)
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init, p2l)

# transform source
mesh_t = copy.deepcopy(source).transform(reg_p2l.transformation)
# these should match now
recon = target + mesh_t
recon = recon.voxel_down_sample(voxel_size)

for i,img in enumerate(images):
    print(i)
    color_raw0 = o3d.io.read_image(f"RGBD/color/000{img}.jpg")
    depth_raw0 = o3d.io.read_image(f"RGBD/depth/000{img}.png")

    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw0, 
        depth_raw0, 
        convert_rgb_to_intensity = True)

    # Source pointcloud
    camera = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    source = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image0, camera)

    # Flip it, otherwise the pointcloud will be upside down
    source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Parameters
    threshold = 1.0
    trans_init = reg_p2l.transformation

    voxel_size = 0.05
    source = source.voxel_down_sample(voxel_size)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                max_nn=30))
    recon.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                max_nn=30))

    reg_p2l = o3d.pipelines.registration.registration_icp(source, recon, threshold, trans_init, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    mesh_t = source.transform(reg_p2l.transformation)
    recon += mesh_t
    recon = recon.voxel_down_sample(voxel_size)
    # break

recon.paint_uniform_color([1, 0.706, 0])
source.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([recon,source])
