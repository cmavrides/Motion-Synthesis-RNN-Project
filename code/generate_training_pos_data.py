import read_bvh
import numpy as np
import os
from os import listdir


def generate_pos_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    bvh_dances_names = listdir(src_bvh_folder)

    for bvh_dance_name in bvh_dances_names:
        if bvh_dance_name.endswith(".bvh"):
            full_path = os.path.join(src_bvh_folder, bvh_dance_name)
            dance = read_bvh.get_train_data(full_path)

            output_filename = bvh_dance_name + ".npy"
            output_path = os.path.join(tar_traindata_folder, output_filename)
            np.save(output_path, dance)

            print(f"Saved: {output_path}")


def generate_pos_bvh_from_traindata(src_train_folder, tar_bvh_folder):
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    dances_names = listdir(src_train_folder)

    for dance_name in dances_names:
        if dance_name.endswith(".npy"):
            input_path = os.path.join(src_train_folder, dance_name)
            dance = np.load(input_path)

            dance2 = [dance[i] for i in range(int(dance.shape[0]))]

            output_filename = dance_name + ".bvh"
            output_path = os.path.join(tar_bvh_folder, output_filename)
            read_bvh.write_traindata_to_bvh(output_path, np.array(dance2))

            print(f"Reconstructed: {output_path}")


# Example usage:
bvh_dir_path = "train_data_bvh/martial/"
pos_enc_dir_path = "train_data_pos/martial/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_pos/martial/"

#comment the following to run test_generate_data.py
generate_pos_traindata_from_bvh(bvh_dir_path, pos_enc_dir_path)
generate_pos_bvh_from_traindata(pos_enc_dir_path, bvh_reconstructed_dir_path)
