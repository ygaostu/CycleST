import tensorflow as tf
import numpy as np
import shutil, os, argparse, time
from PIL import Image
from os.path import join, exists
from data import lfShear
from model import Model

def create_path(path):
    if exists(path):
        shutil.rmtree(path) 
    os.makedirs(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_base", type=str, default="./demo")
    parser.add_argument("--path_shearlet_system", type=str, default="./shearlets")
    parser.add_argument("--path_model", type=str, default="./ckpt")
    parser.add_argument("--interp_rate", default=4, type=int)
    parser.add_argument("--samp_interval", default=32, type=int) # Fixed, don't change if you want to use the pre-trained model
    parser.add_argument("--im_h", default=512, type=int)
    parser.add_argument("--im_w", default=512, type=int)
    parser.add_argument("--epi_w", default=672, type=int)
    parser.add_argument("--epi_h", default=256, type=int)
    parser.add_argument("--angu_res_gt", default=9, type=int)
    parser.add_argument("--name_lf", type=str, default="tower_r_5")
    parser.add_argument("--dmin", default=-3.6, type=float)
    parser.add_argument("--dmax", default=3.5, type=float)
    args = parser.parse_args()

    path_base = args.path_base
    path_shearlet_system = args.path_shearlet_system
    path_model = args.path_model
    interp_rate = args.interp_rate
    samp_interval = args.samp_interval
    im_h = args.im_h
    im_w = args.im_w
    epi_w = args.epi_w
    epi_h = args.epi_h
    angu_res_gt = args.angu_res_gt
    name_lf = args.name_lf
    dmin = args.dmin
    dmax = args.dmax
    drange = dmax - dmin
    assert drange*interp_rate <= samp_interval

    # Other parameters
    angu_res_in = (angu_res_gt - 1) // interp_rate + 1
    angu_res_full = (angu_res_in - 1) * samp_interval + 1
    assert epi_h == max(angu_res_full+127, 256) # Required, set `epi_h` by following this equation
    assert epi_w == ((int(im_w + abs(dmin) * (angu_res_gt - 1) + 128) >> 4) + 1) << 4 # Required, set `epi_w` by following this equation
    row_start = (epi_h - angu_res_full) // 2
    row_end = row_start + angu_res_full
    path_shearlets = join(path_shearlet_system, f"st_{epi_h}_{epi_w}_5.mat")
    path_save_lf = join(path_base, name_lf + "_lf_rec")
    create_path(path_save_lf)
    path_save_epi = join(path_base, name_lf + "_epi_rec")
    create_path(path_save_epi)

    # Load CycleST model
    model = Model(path_shearlets, epi_h, epi_w)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), generator=model)
    checkpoint.restore(tf.train.latest_checkpoint(path_model))
    print(f"Best checkpoint loaded at step: {checkpoint.step.numpy()}")

    # Initialize shearing operation
    op_shear = lfShear(interp_rate, samp_interval, angu_res_in=angu_res_in, angu_res_dense=angu_res_full,
                        im_h=im_h, im_w=im_w, epi_w=epi_w)

    start_time = time.time() 
    print(f"\n Start reconstructing {name_lf} dmin={dmin*interp_rate} drange={drange*interp_rate} \n")

    # Load and pre-shear the given light field
    lf_volume, shifts = op_shear.load_shear(join(path_base, name_lf), dmin)

    lf_vol_pad = np.zeros((epi_h, im_h, epi_w, 3), np.float32) 
    lf_vol_pad[row_start:row_end:samp_interval] = lf_volume
    lf_vol_pad = np.transpose(lf_vol_pad, (1, 3, 0, 2)) / 255.  # im_h, 3, epi_h, epi_w
    lf_vol_pad = lf_vol_pad[np.newaxis, ...]
    lf_vol_rec = np.zeros((im_h, 3, epi_h, epi_w))

    for i in range(im_h):
        lf_vol_rec[i] = model(tf.convert_to_tensor(lf_vol_pad[:, i], tf.float32) ).numpy()
        if (i+1) % 128 == 0:
            print(f"finish recovering {i+1} rows")
    print(f"Reconstruction time: {(time.time()-start_time):.3f} s")

    lf_vol_rec = np.transpose(lf_vol_rec, (2, 0, 3, 1)) * 255. # epi_h, im_h, im_w, 3
    lf_vol_rec = np.clip(np.round(lf_vol_rec), 0, 255).astype(np.uint8)
    
    # Save intermediate EPI reconstruction results
    for i in range(im_h):
        Image.fromarray(lf_vol_rec[:, i]).save(join(path_save_epi, f"{(i+1):04d}.png"))

    # Post-shear and tailor the reconstructed light field 
    lf_vol_rec = op_shear.back_shear(lf_vol_rec[row_start:row_end], shifts)
    lf_vol_rec = lf_vol_rec[::(samp_interval//interp_rate)]
   
    for i in range(len(lf_vol_rec)):
        Image.fromarray(lf_vol_rec[i]).save(join(path_save_lf, f"{(i+1):04d}.png"))   

    print(f"Total time: {(time.time()-start_time):.3f} s")
    
