import read_bvh
import numpy as np
from os import listdir
import os


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    for filename in listdir(src_bvh_folder):
        if not filename.endswith(".bvh"):
            continue
        filepath = os.path.join(src_bvh_folder, filename)
        train_data = read_bvh.get_train_data(filepath)

        output_path = os.path.join(tar_traindata_folder, filename.replace(".bvh", ".npy"))
        np.save(output_path, train_data)
        print(f"Saved training data to {output_path}")


def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    for filename in listdir(src_train_folder):
        if not filename.endswith(".npy"):
            continue
        filepath = os.path.join(src_train_folder, filename)
        train_data = np.load(filepath)

        output_path = os.path.join(tar_bvh_folder, filename.replace(".npy", ".bvh"))
        read_bvh.write_traindata_to_bvh(output_path, train_data)
        print(f"Reconstructed BVH written to {output_path}")


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

print('skeleton: ', skeleton)

bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\martial"
euler_enc_dir_path = r"train_data_euler\martial"
bvh_reconstructed_dir_path = r"reconstructed_euler_bvh\martial"

#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)

bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\salsa"
euler_enc_dir_path = r"train_data_euler\salsa"
bvh_reconstructed_dir_path = r"reconstructed_euler_bvh\salsa"

#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)

bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\indian"
euler_enc_dir_path = r"train_data_euler\indian"
bvh_reconstructed_dir_path = r"reconstructed_euler_bvh\indian"

#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)


