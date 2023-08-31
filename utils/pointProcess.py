import torch
import numpy as np
import open3d as o3d
from copy import copy, deepcopy


def pc_normalize(pc):
        centriod = np.mean(pc, axis=0)
        pc = pc - centriod
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


def pcd_downsample(initPcd, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """
        Downsample pointcloud to 4096 points
        Modify based on the version from https://blog.csdn.net/SJTUzhou/article/details/122927787
    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert len(initPcd.points) > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given point cloud."
    if len(initPcd.points) == desiredNumOfPoint:
        return initPcd
    
    pcd = deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd = deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."
    
    pcd = deepcopy(initPcd)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy(midVoxelSize)
        else:
            rightVoxelSize = copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd = deepcopy(initPcd)
        pcd = pcd.voxel_down_sample(midVoxelSize)
    
    return pcd


class LaserProjection(object):
    """Class that contains LaserScan with x,y,z,r"""

    def __init__(self, device, top_size=[64, 64], z_range=[-100.0, 100.0], sph_size=[64, 64], fov_range=[-90, 90], max_dis=30, sph_ocu=16, visib_thresh=3.0, visib_radius=25):
        #! For top down view
        self.proj_H = (int)(top_size[0]/2)
        self.proj_W = (int)(top_size[1]/2)
        self.proj_Z_min = z_range[0]
        self.proj_Z_max = z_range[1]

        self.device = device

        #! For spherical view
        self.sph_H = sph_size[0]
        self.sph_W = sph_size[1]
        self.sph_down = fov_range[0]
        self.sph_up = fov_range[1]

        #! Set sph occlusion factor
        self.sph_ocu = sph_ocu

        #! For activate range
        self.max_dis = max_dis

        self.visib_thresh = visib_thresh
        self.visib_radius = visib_radius

    def proj(self, pt):
        sph_proj = self.do_sph_projection(pt)
        sph_proj = sph_proj.reshape([1, self.sph_H, self.sph_W])
        sph_proj = torch.from_numpy(sph_proj).to(
            self.device, dtype=torch.float)
        return sph_proj

    def proj_img(self, pt):
        sph_proj = self.do_sph_projection(pt)
        sph_proj = sph_proj.reshape([self.sph_H, self.sph_W, 1])
        return sph_proj

    def do_top_projection(self, points):
        """ Project a pointcloud into a BEV picture
        """

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get projections in image coords
        proj_x = scan_x/self.max_dis
        proj_y = scan_y/self.max_dis

        # scale to image size using angular resolution
        proj_x = (proj_x + 1.0)*self.proj_W                # in [0.0, 2W]
        proj_y = (proj_y + 1.0)*self.proj_H                # in [0.0, 2H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(2*self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,2W-]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(2*self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,2H-1]

        data_grid = np.zeros((2*self.proj_H, 2*self.proj_W), dtype='float64')
        data_grid[proj_y, proj_x] = scan_z

        data_norm = (data_grid - data_grid.min()) / \
            (data_grid.max() - data_grid.min())

        return data_norm

    def do_sph_projection(self, points):
        """ Project a pointcloud into a spherical projection image.projection.
                Function takes no arguments because it can be also called externally
                if the value of the constructor was not set (in case you change your
                mind about wanting the projection)
        """

        # laser parameters
        fov_up = self.sph_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.sph_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)                  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        # sph_H = self.sph_H*self.sph_ocu
        # sph_W = self.sph_W*self.sph_ocu
        sph_H = self.sph_H
        sph_W = self.sph_W
        proj_x *= sph_W                              # in [0.0, W] 128
        proj_y *= sph_H                              # in [0.0, H] 64

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(sph_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(sph_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

        data_grid = np.zeros((sph_H, sph_W), dtype='float32')

        # NOTE Maybe the constraints come from the far building, instead of neighbor objects
        # NOTE For indoor datasets, constraints from near objects will reduce the accuracy

        indices = np.argsort(depth)[::-1]
        data_grid[proj_y[indices], proj_x[indices]] = depth[indices]
        # data_grid[proj_y, proj_x] = depth

        # sph_grid = skimage.measure.block_reduce(data_grid, (self.sph_ocu, self.sph_ocu), np.max)
        # sph_grid = 1./(sph_grid + 1)
        sph_norm = (data_grid - data_grid.min()) / \
            (data_grid.max() - data_grid.min())

        # depth_img = torch.from_numpy(data_grid).cuda()
        # projected_points = visibility.visibility2(depth_img, torch.FloatTensor([self.sph_down, self.sph_up]).cuda(), torch.zeros_like(depth_img, device='cuda'), depth_img.shape[1], depth_img.shape[0], self.visib_thresh, self.visib_radius)
        # data_out = projected_points.cpu().numpy()
        # data_out[data_out ==0] = 100
        # data_out = skimage.measure.block_reduce(data_out, (self.sph_ocu, self.sph_ocu), np.min)
        # data_index = (data_out ==100)
        # data_out[data_index] = 0
        # # data_out= 1./(data_out+ 1)
        # data_norm = (data_out - data_out.min()) / (data_out.max() - data_out.min())
        # ## NOTE define -1 for all negative points
        # data_norm[data_index] = -1
        return sph_norm
    
    def do_multi_sph_projection(self, points):
        depth = np.linalg.norm(points, 2, axis=1)
        index_r20 = np.where(depth<=20)
        index_r35 = np.intersect1d(np.where(depth>20), np.where(depth<=35)) 
        index_r50 = np.where(depth>35)
        
        sphere_r20 = self.do_sph_projection(points[index_r20])
        sphere_r35 = self.do_sph_projection(points[index_r35])
        sphere_r50 = self.do_sph_projection(points[index_r50])
        
        return np.stack((sphere_r20,sphere_r35,sphere_r50), axis=0)