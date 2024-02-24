import numpy as np
from mayavi import mlab
import pyvista as pv


def visualize_as_mesh(points):
    clouds = pv.PolyData(points)
    surf = clouds.delaunay_3d(alpha=2.0)
    pv.plot(surf)


def visualize(points):
    x, y, z = points.T
    mlab.points3d(x, y, z, mode='point')
    mlab.show()


def read_stl_file(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(80)
        face_count = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(face_count * 50), dtype=np.float32)
    vertices = np.reshape(data[0::12], (-1, 3))
    return vertices


def write_points_to_file(file_path, points):
    np.savetxt(file_path, points)

