import open3d as o3d

# Path to your .ply file
file_path = "/Users/mars/Desktop/seda.ply"

# Load point cloud data
point_cloud = o3d.io.read_point_cloud(file_path)

# Visualize point cloud
o3d.visualization.draw_geometries([point_cloud])
