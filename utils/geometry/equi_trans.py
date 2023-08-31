import numpy as np
from equilib import equi2pers, equi2equi

class EquiTrans(object):
    """Class that contains Equi Trans with x,y,z,r"""

    def __init__(self, pers, fov_x, rot):
        self.w_pers = pers[0]
        self.h_pers = pers[1]
        self.fov_x = fov_x
        self.rot = rot #{'roll': 0.,'pitch': 0, 'yaw': 0.}

    def trans(self, equi_img):
        equi_img = np.asarray(equi_img)
        equi_img = np.transpose(equi_img, (2, 0, 1))
        pers_img = equi2pers(
            equi=equi_img,
            rot=self.rot,
            w_pers=self.w_pers,
            h_pers=self.h_pers,
            fov_x=self.fov_x,
            skew=0.0,
            sampling_method="default",
            mode="bilinear",)
        pers_img = np.transpose(pers_img, (1, 2, 0))
        return pers_img

class Equi2Equi(object):

    def __init__(self, pers, rot):
        self.w_out = pers[0]
        self.h_out = pers[1]
        # self.rot={'roll': 0.,'pitch': 0, 'yaw': 0.}
        self.rot = rot

    def trans(self, equi_img):
        equi_img = np.asarray(equi_img)
        equi_img = np.transpose(equi_img, (2, 0, 1))
        img_out = equi2equi(equi_img, self.rot, w_out=self.w_out, h_out=self.h_out)
        img_out = np.transpose(img_out, (1, 2, 0))
        return img_out
        
