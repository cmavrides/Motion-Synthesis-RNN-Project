import read_bvh
import read_bvh
import numpy as np
from os import listdir
import os
from scipy.spatial.transform import Rotation as R

# Load reference skeleton
standard_bvh_file = "train_data_bvh/standard.bvh"
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

def euler_frame_to_quat_vector(euler_frame):
    quat_vector = []

    # Hip translation
    hip_pos = euler_frame[:3]
    quat_vector.extend(hip_pos)

    # Hip rotation
    hip_angles = euler_frame[3:6]
    hip_r = R.from_euler('zxy', hip_angles, degrees=True)
    quat_vector.extend(hip_r.as_quat())

    # All other joints (from frame data length)
    total_angle_values = len(euler_frame) - 6
    num_joints = total_angle_values // 3

    for i in range(num_joints):
        idx = 6 + i * 3
        angles = euler_frame[idx:idx+3]
        if len(angles) != 3:
            continue  # skip if not full rotation data
        r = R.from_euler('zxy', angles, degrees=True)
        quat_vector.extend(r.as_quat())

    return np.array(quat_vector)


def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    for filename in listdir(src_bvh_folder):
        if not filename.endswith(".bvh"):
            continue

        filepath = os.path.join(src_bvh_folder, filename)
        raw_data = read_bvh.parse_frames(filepath)

        quat_data = np.array([euler_frame_to_quat_vector(frame) for frame in raw_data])
        print(f"[DEBUG] {filename} vector length per frame: {len(quat_data[0])}")
        quat_data = quat_data / 100.0  # normalization

        output_path = os.path.join(tar_traindata_folder, filename.replace(".bvh", ".npy"))
        np.save(output_path, quat_data)
        print(f"✅ Saved quaternion training data to {output_path}")


def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    from decode_helpers import decode_quat_to_bvh

    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    for filename in listdir(src_train_folder):
        if not filename.endswith(".npy"):
            continue

        filepath = os.path.join(src_train_folder, filename)
        train_data = np.load(filepath)
        print(f"[DEBUG] Loaded {filename} with shape {train_data.shape}")

        # Re-scale data back (reverse normalization)
        train_data = train_data * 100.0

        # Get the same skeleton that was used for encoding
        global skeleton, non_end_bones

        output_path = os.path.join(tar_bvh_folder, filename.replace(".npy", ".bvh"))
        try:
            # Pass the skeleton to the decode function
            decode_quat_to_bvh(output_path, train_data)
            print(f"✅ Reconstructed BVH written to {output_path}")
        except ValueError as e:
            print(f"❌ Error decoding {filename}: {e}")
            print(f"Data has shape {train_data.shape} with vector length {train_data.shape[1]}")
            print(f"Expected quaternion vector length: {(len(train_data[0])-3)//4 + 1} joints")
            print(f"Check decode_helpers.py for expected number of joints")


bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\indian"
quad_enc_dir_path = r"train_data_quad\indian"
bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\indian"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)



bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\salsa"
quad_enc_dir_path = r"train_data_quad\salsa"
bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\salsa"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)


bvh_dir_path = r"C:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\train_data_bvh\martial"
quad_enc_dir_path = r"train_data_quad\martial"
bvh_reconstructed_dir_path = r"reconstructed_quad_bvh\martial"


#comment the following to run test_generate_data.py
# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)
