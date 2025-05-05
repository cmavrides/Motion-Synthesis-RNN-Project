import read_bvh
import read_bvh
import numpy as np
from os import listdir
import os
from scipy.spatial.transform import Rotation as R
from read_bvh import joint_index

# Load reference skeleton
standard_bvh_file = "train_data_bvh/standard.bvh"
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    for filename in os.listdir(src_bvh_folder):
        if filename.endswith(".bvh"):
            bvh_path = os.path.join(src_bvh_folder, filename)
            train_data = read_bvh.get_train_data(bvh_path)  # shape: (frames, joints*3)
            # Assume first 3 values are root translation, rest are Euler angles (in groups of 3)
            quat_data = []
            for frame in train_data:
                root = frame[:3]
                eulers = frame[3:].reshape(-1, 3)  # shape: (num_joints, 3)
                # Convert each joint's Euler angles to quaternion (BVH is usually ZXY order)
                quats = R.from_euler('zxy', eulers, degrees=True).as_quat()  # shape: (num_joints, 4)
                # Store as [root, q1, q2, ...]
                quat_frame = np.concatenate([root, quats.flatten()])
                quat_data.append(quat_frame)
            quat_data = np.array(quat_data)
            out_path = os.path.join(tar_traindata_folder, filename.replace(".bvh", ".npy"))
            np.save(out_path, quat_data)
            print(f"Saved quaternion training data: {out_path}")

def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    for filename in os.listdir(src_train_folder):
        if filename.endswith(".npy"):
            train_data_path = os.path.join(src_train_folder, filename)
            quat_data = np.load(train_data_path)
            # Convert quaternions back to Euler angles for BVH
            euler_data = []
            for frame in quat_data:
                root = frame[:3]
                quats = frame[3:].reshape(-1, 4)  # shape: (num_joints, 4)
                # Convert each joint's quaternion to Euler angles (BVH is usually ZXY order)
                eulers = R.from_quat(quats).as_euler('zxy', degrees=True)  # shape: (num_joints, 3)
                euler_frame = np.concatenate([root, eulers.flatten()])
                euler_data.append(euler_frame)
            euler_data = np.array(euler_data)
            bvh_out_path = os.path.join(tar_bvh_folder, filename.replace(".npy", ".bvh"))
            read_bvh.write_traindata_to_bvh(bvh_out_path, euler_data)
            print(f"Reconstructed BVH from quaternion data saved: {bvh_out_path}")

#bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\indian"
#quad_enc_dir_path = r"train_data_quad\indian"
#bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\indian"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
#generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
#generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)



#bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\salsa"
#quad_enc_dir_path = r"train_data_quad\salsa"
bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\salsa"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
#generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
#generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)


bvh_dir_path = r"C:\Users\alexa\Desktop\MAI645_Team_04\train_data_bvh\martial"
quad_enc_dir_path = r"train_data_quad\martial"
bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\martial"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)
