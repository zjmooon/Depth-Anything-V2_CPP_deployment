import open3d as o3d

pcd = o3d.io.read_point_cloud("../build/cloud_cluster_0001.pcd")

print(pcd)
print(f"point cloud include:  {len(pcd.points)} points")

o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=800, height=600)
