import numpy as np
import open3d as o3d

def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip().split()
    if first_line[0] == 'OFF':
        if len(first_line) == 1:
            lines = lines[1:]
            counts = lines[0].strip().split()
            lines = lines[1:]
        else:
            counts = first_line[1:]
            lines = lines[1:]
    else:
        raise ValueError("Invalid OFF file format")

    n_vertices = int(counts[0])
    vertices = np.array([list(map(float, lines[i].strip().split())) for i in range(n_vertices)])
    if len(vertices) == 0:
        raise ValueError(f"No vertices loaded from: {file_path}")
    return vertices

def preprocess_pointcloud(filepath, num_points=1024):
    if filepath.endswith('.off'):
        vertices = read_off(filepath)
    else:
        pcd = o3d.io.read_point_cloud(filepath)
        vertices = np.asarray(pcd.points)

    if len(vertices) == 0:
        raise ValueError(f"No points found in file: {filepath}")

    idx = np.random.choice(len(vertices), num_points, replace=len(vertices) < num_points)
    sampled = vertices[idx]
    sampled = sampled - sampled.mean(axis=0)
    return sampled
