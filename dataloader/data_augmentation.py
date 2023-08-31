'''
Author: Shiqi Zhao
data_augmentation

Codes of Data Augmentation for Point Cloud is adopted from MinkLoc++
Original Code: https://github.com/jac99/MinkLocMultimodal
'''

import random
import math
import numpy as np
from scipy.linalg import expm, norm
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.geometry import get_projection_grid, rand_rotation_matrix, rotate_grid, project_2d_on_sphere


class Augment_Point_Data():
    def __init__(self, mode=0, is_train=False):
        if is_train:
            if mode == 0:
                transform = [transforms.ToTensor()]
            elif mode == 1:
                transform = [transforms.ToTensor(), 
                             JitterPoints(sigma=0.001, clip=0.002), 
                             RemoveRandomPoints(r=(0.0, 0.1)),
                             RandomTranslation(max_delta=0.01), 
                             RemoveRandomBlock(p=0.4),
                             RandomRotation(axis=np.array([0,0,1]),
                                            max_theta=15,
                                            max_theta2=None)]
            else:
                raise NotImplementedError(f'Uncognized data augmentation mode')
        else:
            transform = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform)
    
    def __call__(self, input):
        output = self.transform(input)
        return output


class Augment_RGB_Data():
    def __init__(self, mode=0, is_train=False):
        if is_train:
            if mode == 0:
                transform = [transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            elif mode == 1:
                transform = [transforms.ToTensor(), 
                             transforms.Resize([224, 224]),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                             transforms.RandomErasing(scale=(0.1, 0.4)),
                             transforms.RandomRotation(degrees=5),
                             transforms.RandomHorizontalFlip(p=0.5)]
            else:
                raise NotImplementedError(f'Uncognized data augmentation mode')
        else:
            transform = [transforms.ToTensor(),
                         transforms.Resize([224, 224]),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(transform)
    
    def __call__(self, input):
        output = self.transform(input)
        return output


class Augment_SPH_Data():
    def __init__(self, img_size, mode=0, is_train=False):
        if is_train:
            if mode == 0:
                transform = [transforms.ToTensor(), 
                             transforms.Normalize([0.5], [0.5])]
            elif mode == 1:
                transform = [SphRandomRotate(p=1, img_size=img_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])]
            else:
                raise NotImplementedError(f'Uncognized data augmentation mode')
        else:
            transform = [transforms.ToTensor(),
                         transforms.Resize(img_size, Image.BICUBIC),
                         transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(transform)
    
    def __call__(self, input):
        output = self.transform(input)
        return output


class SphRandomRotate:
    '''Random rotate spherical projection of point cloud'''
    def __init__(self, p, img_size):
        assert 0 < p <= 1, 'probability must be in (0, 1] range)!'
        self.p = p
        self.grid = get_projection_grid(b=(int)(img_size[0]/2.0))
    
    def __call__(self, sph_img):
        #* conver from PIL.Image into numpy.array
        signals = np.expand_dims(np.array(sph_img), axis=0)
        #* generate random rotation along all three axis
        rot = rand_rotation_matrix(deflection=1)
        #* generate random rotation along z-axis
        # rot_angle = 2*np.pi*random.uniform(1)
        # cosval = np.cos(rot_angle)
        # sinval = np.sin(rot_angle)
        # rot = np.array([[cosval, -sinval, 0],
        #                 [sinval, cosval, 0],
        #                 [0, 0, 1]])
        rotated_grid = rotate_grid(rot, self.grid)
        rot_sph_img = project_2d_on_sphere(signals, rotated_grid, projection_origin=[0,0,0.00001])
        rot_sph_img = np.squeeze(rot_sph_img)
        return rot_sph_img


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords