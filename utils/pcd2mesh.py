import os
import numpy as np
import open3d as o3d

def create_sphere_at_xyz(xyz):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015, resolution=20)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.8, 0.1, 0.1])
    sphere = sphere.translate(xyz)
    return sphere

INPUT_DIR = 'kitti_car/input'
OUTPUT_DIR = 'kitti_car/input_mesh'
for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        data = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename))
        np_data = np.array(data.points)
        mesh = create_sphere_at_xyz(np_data[0])
        for i in range(1, np_data.shape[0]):
            mesh += create_sphere_at_xyz(np_data[i])
        o3d.io.write_triangle_mesh(os.path.join(OUTPUT_DIR, filename.split('.')[0]+'.obj'), mesh)
