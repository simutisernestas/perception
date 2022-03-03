import open3d as o3d
import numpy as np
import copy
import scipy
from scipy.spatial import procrustes
import warnings
warnings.filterwarnings("error")

def draw_registrations(source, target, transformation = None, recolor = False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(recolor):
        source_temp.paint_uniform_color([1, .1, 0])
        target_temp.paint_uniform_color([0, .1, 1])
    if(transformation is not None):
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

use_car = True
use_scipy_rot = False

if use_car:
    mesh = o3d.io.read_triangle_mesh("global_registration/car1.ply")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    mesh = o3d.io.read_triangle_mesh("global_registration/car2.ply")
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
else:
    source = o3d.io.read_point_cloud("global_registration/r1.pcd")
    target = o3d.io.read_point_cloud("global_registration/r2.pcd")

voxel_size = 0.04
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

target_fpfh_features.data.shape, source_fpfh_features.data.shape, target.dimension, source.dimension

# matches by features
kd1 = scipy.spatial.KDTree(target_fpfh_features.data.T)
kd2 = scipy.spatial.KDTree(source_fpfh_features.data.T)
indexes = kd1.query_ball_tree(kd2, r=30)
# target => source
matches = {}
for i in range(len(indexes)):
    if indexes[i]:
        matches[i] = indexes[i][0]
n_matched = len(matches)
print(f"Match N: {n_matched}")

# center all the points!
target_points = np.asarray(target.points)
target_centroid = np.mean(target_points, axis=0)
target_points -= target_centroid
source_points = np.asarray(source.points)
source_centroid = np.mean(source_points, axis=0)
source_points -= source_centroid

# all the matched points in centered xyz
match_targets = target_points[list(matches.keys())]
match_sources = source_points[list(matches.values())]

optimal_R = np.zeros((3,3))
min_norm = np.inf
for i in range(1000):
    target_random_sample = np.random.choice(list(matches.keys()), (3,), replace=False)
    source_random_sample = [matches[target_random_sample[x]] for x in range(3)]

    if use_scipy_rot:
        try:
            (estimated_rotation,rmsd) = scipy.spatial.transform.Rotation.align_vectors(
                target_points[target_random_sample], source_points[source_random_sample])
        except:
            continue
    else:
        Cov = target_points[target_random_sample].T @ source_points[source_random_sample]
        U, S, Vt = np.linalg.svd(Cov)
        # lecture_rot = U @ Vt !WRONG
        # https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
        v = Vt.T
        d = np.linalg.det(v @ U.T)
        e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
        r = v @ e @ U.T
        estimated_rotation = r.T

    if use_scipy_rot:
        norm = np.linalg.norm(estimated_rotation.apply(match_sources) - match_targets)
    else:
        norm = np.linalg.norm((estimated_rotation @ match_sources.T).T - match_targets)

    if min_norm > norm:
        min_norm = norm
        optimal_R = estimated_rotation

if use_scipy_rot:
    source_transformed = optimal_R.apply(source_points)
else:
    source_transformed = (optimal_R @ source_points.T).T
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(source_transformed)
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(target_points)
draw_registrations(pcd1, pcd2, recolor=True)