import tensorflow as tf
import numpy as np
import shutil, os, argparse, time
from PIL import Image
from os.path import join, exists
from data import lfShear, load_lf_3d, load_lf_4d
from model import Model

def create_path(path):
    if exists(path):
        shutil.rmtree(path) 
    os.makedirs(path)

def rec_lf_3d(model, lf_volume_in, op_shear, epi_h, dmin, row_start, row_end, save_res=False, path_save_epi=None, path_save_lf=None):
    start_time = time.time() 
    
    # Load and pre-shear the given light field
    lf_volume = op_shear.pre_shear(lf_volume_in, dmin)

    lf_vol_pad = np.zeros((epi_h, op_shear.im_h, op_shear.epi_w, 3), np.float32) 
    lf_vol_pad[row_start:row_end:op_shear.samp_interval] = lf_volume
    lf_vol_pad = np.transpose(lf_vol_pad, (1, 3, 0, 2)) / 255.  # im_h, 3, epi_h, epi_w
    lf_vol_pad = lf_vol_pad[np.newaxis, ...]
    lf_vol_rec = np.zeros((op_shear.im_h, 3, epi_h, op_shear.epi_w))

    for i in range(op_shear.im_h):
        lf_vol_rec[i] = model(tf.convert_to_tensor(lf_vol_pad[:, i], tf.float32) ).numpy()
        if (i+1) % 128 == 0:
            print(f"finish recovering {i+1} rows / EPIs")
    print(f"Reconstruction time: {(time.time()-start_time):.3f} s")

    lf_vol_rec = np.transpose(lf_vol_rec, (2, 0, 3, 1)) * 255. # epi_h, im_h, im_w, 3
    lf_vol_rec = np.clip(np.round(lf_vol_rec), 0, 255).astype(np.uint8)
    
    if save_res:
        # Save intermediate EPI reconstruction results
        for i in range(op_shear.im_h):
            Image.fromarray(lf_vol_rec[:, i]).save(join(path_save_epi, f"{(i+1):04d}.png"))

    # Post-shear and tailor the reconstructed light field 
    lf_vol_rec = op_shear.back_shear(lf_vol_rec[row_start:row_end])
    lf_vol_rec = lf_vol_rec[::(op_shear.samp_interval//op_shear.interp_rate)]
    lf_vol_rec[::op_shear.interp_rate] = lf_volume_in
   
    if save_res:
        for i in range(len(lf_vol_rec)):
            Image.fromarray(lf_vol_rec[i]).save(join(path_save_lf, f"{(i+1):04d}.png"))   

    print(f"Total time: {(time.time()-start_time):.3f} s")

    return lf_vol_rec

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_base", type=str, default="./demo")
    parser.add_argument("--path_shearlet_system", type=str, default="./shearlets/st_255_255_5.mat")
    parser.add_argument("--path_model", type=str, default="./ckpt")
    parser.add_argument("--interp_rate", default=4, type=int)
    parser.add_argument("--samp_interval", default=32, type=int) # Fixed, don't change it if you want to use the pre-trained model
    parser.add_argument("--im_h", default=512, type=int)
    parser.add_argument("--im_w", default=512, type=int)
    parser.add_argument("--angu_res_gt", default=9, type=int)
    parser.add_argument("--name_lf", type=str, default="tower_r_5")
    parser.add_argument("--dmin", default=-3.6, type=float)
    parser.add_argument("--dmax", default=3.5, type=float)
    parser.add_argument("--full_parallax", action='store_true')
    args = parser.parse_args()

    path_base = args.path_base
    path_shearlet_system = args.path_shearlet_system
    path_model = args.path_model
    interp_rate = args.interp_rate
    samp_interval = args.samp_interval
    im_h = args.im_h
    im_w = args.im_w
    angu_res_gt = args.angu_res_gt
    name_lf = args.name_lf
    dmin = args.dmin
    dmax = args.dmax
    drange = dmax - dmin
    assert drange*interp_rate <= samp_interval
    full_parallax = args.full_parallax

    # Other parameters
    angu_res_in = (angu_res_gt - 1) // interp_rate + 1
    angu_res_full = (angu_res_in - 1) * samp_interval + 1
    epi_h = max(angu_res_full+127, 256) 
    epi_w = ((int(im_w + abs(dmin) * (angu_res_gt - 1) + 128) >> 4) + 1) << 4
    row_start = (epi_h - angu_res_full) // 2
    row_end = row_start + angu_res_full
    path_save_lf = join(path_base, name_lf + "_lf_rec")
    create_path(path_save_lf)
    if not full_parallax:
        path_save_epi = join(path_base, name_lf + "_epi_rec")
        create_path(path_save_epi)

    # Load CycleST model
    model = Model(path_shearlet_system, epi_h, epi_w)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), generator=model)
    checkpoint.restore(tf.train.latest_checkpoint(path_model))
    print(f"Best checkpoint loaded at step: {checkpoint.step.numpy()}")

    # Initialize shearing operation
    op_shear = lfShear(interp_rate, samp_interval, angu_res_in=angu_res_in, angu_res_dense=angu_res_full,
                        im_h=im_h, im_w=im_w, epi_w=epi_w)

    if not full_parallax:
        lf_volume_gt = load_lf_3d(join(path_base, name_lf), angu_res_gt, im_h, im_w)
        lf_volume_in = lf_volume_gt[::interp_rate]
        print(f"\n Start reconstructing {name_lf} (dmin={dmin*interp_rate} drange={drange*interp_rate}) \n")
        rec_lf_3d(model, lf_volume_in, op_shear, epi_h, dmin, row_start, row_end, True, path_save_epi, path_save_lf)
    else:
        angu_res_gt_rows, angu_res_gt_cols = angu_res_gt, angu_res_gt
        lf_volume_gt = load_lf_4d(join(path_base, name_lf), angu_res_gt_rows, angu_res_gt_cols, im_h, im_w)
        lf_volume_rec = np.zeros((angu_res_gt_rows, angu_res_gt_cols, im_h, im_w, 3), np.uint8)
        
        for row in range(0, angu_res_gt_rows, interp_rate):
            lf_volume_in = lf_volume_gt[row, ::interp_rate] 
            print(f"\n Start reconstructing {name_lf} row {row+1} (dmin={dmin*interp_rate} drange={drange*interp_rate}) \n")
            lf_volume_rec[row] = rec_lf_3d(model, lf_volume_in, op_shear, epi_h, dmin, row_start, row_end)

        if im_h != im_w:
            # Load CycleST model
            tf.keras.backend.clear_session()
            epi_w_new = ((int(im_h + abs(dmin) * (angu_res_gt - 1) + 128) >> 4) + 1) << 4
            model = Model(path_shearlet_system, epi_h, epi_w_new)
            checkpoint = tf.train.Checkpoint(step=tf.Variable(0), generator=model)
            checkpoint.restore(tf.train.latest_checkpoint(path_model))
            print(f"Best checkpoint loaded at step: {checkpoint.step.numpy()}")

            # Initialize shearing operation
            op_shear = lfShear(interp_rate, samp_interval, angu_res_in=angu_res_in, angu_res_dense=angu_res_full,
                                im_h=im_w, im_w=im_h, epi_w=epi_w_new)

        for col in range(angu_res_gt_cols):
            lf_volume_in = np.rot90(lf_volume_rec[::interp_rate, col], 1, (1, 2)).copy()
            print(f"\n Start reconstructing {name_lf} col {col+1} (dmin={dmin*interp_rate} drange={drange*interp_rate}) \n")
            lf_volume_rec[:, col] = np.rot90(rec_lf_3d(model, lf_volume_in, op_shear, epi_h, dmin, row_start, row_end), 3, (1, 2))

        # Save the reconstructed full-parallax (4D) light field
        for row in range(angu_res_gt_rows):
            for col in range(angu_res_gt_cols):
                Image.fromarray(lf_volume_rec[row, col]).save(join(path_save_lf, f"output_Cam{(row*angu_res_gt_cols + col + 1):03d}.png"))   