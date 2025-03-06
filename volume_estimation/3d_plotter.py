import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pyvista as pv


def load_data(file_path):
    points = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Assume each line is x y z format
                coords = line.strip().split()
                if len(coords) == 3:
                    x, y, z = map(float, coords)
                    points.append([x, y, z])
        return np.array(points)
    except Exception as e:
        print(f"Error loading point cloud file: {e}")
        return None
    
def plot_point_cloud(points):
    """Plots the point cloud in 3D using matplotlib."""
    if points is None or points.shape[0] == 0:
        print("No points to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization')

    plt.show()


def generate_mesh_from_points(points):
    # Perform Delaunay triangulation
    delaunay = Delaunay(points[:, :2])  # Use x and y for 2D triangulation

    # Generate faces for the mesh
    faces = []
    for simplex in delaunay.simplices:
        faces.append([3, simplex[0], simplex[1], simplex[2]])  # Each face is a triangle (3 vertices)

    faces = np.array(faces)

    # Create the PyVista mesh
    mesh = pv.PolyData(points)
    mesh.faces = faces
    return mesh


def main():
    parser = argparse.ArgumentParser(description="Visualize a point cloud in 3D.")
    parser.add_argument('file_path', type=str, help="The file path to the point cloud data.")
    args = parser.parse_args()

    # Load point cloud data
    points = load_data(args.file_path)

    # Visualize the point cloud in 3D
    plot_point_cloud(points)
    # Generate mesh from point cloud data
    # mesh = generate_mesh_from_points(points)

    # Visualize the mesh
    # mesh.plot(show_edges=True, color="white")

if __name__ == '__main__':
    main()