import cv2
import numpy as np
from PIL import Image
from glob import glob

class lfShear:

    def __init__(self, interp_rate, samp_interval=32, angu_res_in=9, angu_res_dense=65, 
                im_h=512, im_w=512, epi_w=672, border_mode=cv2.BORDER_REPLICATE):
        self.interp_rate = interp_rate
        self.samp_interval = samp_interval
        self.angu_res_in = angu_res_in
        self.angu_res_dense = angu_res_dense
        assert(self.angu_res_dense == ((self.angu_res_in - 1)*self.samp_interval + 1) )
        
        # build a 2D grid
        self.im_h = im_h
        self.im_w = im_w
        self.epi_w = epi_w
        x = np.arange(0, self.epi_w)
        y = np.arange(0, self.im_h)
        gx, gy = np.meshgrid(x, y)
        self.gx, self.gy = gx.astype(np.float32), gy.astype(np.float32)

        self.border_mode = border_mode

    def load_shear(self, lf_folder, dmin):
        # load all images
        im_names = sorted(glob(f"{lf_folder}/*png"))
        im_names = im_names[::self.interp_rate]
        assert(len(im_names) == self.angu_res_in)

        dmin = dmin * self.interp_rate
        shift = self.shiftCal(dmin)

        lf_volume = np.zeros((self.angu_res_in, self.im_h, self.epi_w, 3), np.uint8) # 9, 512, 672, 3
        for i, im_name in enumerate(im_names):
            im = np.asarray(Image.open(im_name))
            lf_volume[i] = cv2.remap(im, self.gx - i * dmin - shift, self.gy, 
                            interpolation=cv2.INTER_CUBIC, borderMode=self.border_mode)
        return lf_volume, (dmin/self.samp_interval, shift)

    def back_shear(self, lf_volume, shifts):
        s, shift = shifts
        for i in range(self.angu_res_dense):
            lf_volume[i] = cv2.remap(lf_volume[i], self.gx + i * s + shift, self.gy, 
                            interpolation=cv2.INTER_CUBIC, borderMode=self.border_mode)
        return lf_volume[:, :, :self.im_w, :]

    def shiftCal(self, dmin):
        width_real = dmin * (self.angu_res_in - 1) + self.im_w
        return (self.epi_w - width_real) // 2
