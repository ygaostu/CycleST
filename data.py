import numpy as np
from PIL import Image
from glob import glob
from scipy.ndimage import shift

def load_lf_3d(lf_folder, angu_res_gt, im_h, im_w):
    lf_volume = np.zeros((angu_res_gt, im_h, im_w, 3), np.uint8) 
    im_names = sorted(glob(f"{lf_folder}/*png"))
    assert len(im_names) == angu_res_gt
    for i, im_name in enumerate(im_names):
        lf_volume[i] = np.asarray(Image.open(im_name))

    return lf_volume

def load_lf_4d(lf_folder, angu_res_gt_rows, angu_res_gt_cols, im_h, im_w):
    lf_volume = np.zeros((angu_res_gt_rows, angu_res_gt_cols, im_h, im_w, 3), np.uint8) 
    im_names = sorted(glob(f"{lf_folder}/*png"))
    assert len(im_names) == angu_res_gt_rows * angu_res_gt_cols
    for i, im_name in enumerate(im_names):
        lf_volume[i // angu_res_gt_cols, i % angu_res_gt_cols, :] = np.asarray(Image.open(im_name))
    
    return lf_volume

class lfShear:

    def __init__(self, interp_rate, samp_interval=32, angu_res_in=3, angu_res_dense=65, 
                im_h=512, im_w=512, epi_w=672, border_mode="nearest"):
        self.interp_rate = interp_rate
        self.samp_interval = samp_interval
        self.angu_res_in = angu_res_in
        self.angu_res_dense = angu_res_dense
        assert self.angu_res_dense == ((self.angu_res_in - 1)*self.samp_interval + 1) 
        self.im_h = im_h
        self.im_w = im_w
        self.epi_w = epi_w
        self.border_mode = border_mode
        self.pad_left = 0
        self.pad_right = 0

    def padding_estimate(self, dmin):
        self.dmin = dmin * self.interp_rate # dmin of the input SSLF
        width_real = self.dmin * (self.angu_res_in - 1) + self.im_w
        pad_left = int((self.epi_w - width_real) // 2)
        self.pad_left = pad_left
        self.pad_right = self.epi_w - self.im_w - self.pad_left

    def pre_shear(self, lf_volume_in, dmin):
        self.padding_estimate(dmin)
        lf_volume_shear = np.pad(lf_volume_in, ( (0, 0), (0, 0), (self.pad_left, self.pad_right), (0, 0) ), mode="edge" )
        for i in range(len(lf_volume_in)):
            lf_volume_shear[i] = shift(lf_volume_shear[i], [0, i*self.dmin, 0], order=0, mode=self.border_mode)
        return lf_volume_shear 

    def back_shear(self, lf_volume):
        s = self.dmin / self.samp_interval
        for i in range(self.angu_res_dense):
            lf_volume[i] = shift(lf_volume[i], [0, -i*s, 0], order=0, mode=self.border_mode) 
        return lf_volume[:, :, self.pad_left:(self.pad_left+self.im_w), :]