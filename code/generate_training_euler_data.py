import read_bvh
import numpy as np
import os

# Desired fixed feature dimension
FEATURE_DIM = 171
# Constants for normalization
WEIGHT_TRANSLATION = 0.01  # scale for translation channels
ANGLE_SCALE = 180.0        # maximum absolute Euler angle in degrees

# Standard BVH template (for write_frames) and its param count
STANDARD_BVH_FILE = "../train_data_bvh/standard.bvh"
# Load once to get expected raw parameter count
_template = read_bvh.parse_frames(STANDARD_BVH_FILE)
EXPECTED_PARAMS = _template.shape[1]


def _pad_or_truncate(array, target_dim):
    """Pad with zeros or truncate each feature vector to target_dim columns."""
    F, D = array.shape
    if D == target_dim:
        return array
    elif D < target_dim:
        pad_width = target_dim - D
        padding = np.zeros((F, pad_width), dtype=array.dtype)
        return np.concatenate([array, padding], axis=1)
    else:
        # truncate extra dims to the first target_dim
        return array[:, :target_dim]


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    """
    Reads BVH files, normalizes translation+Euler angles, pads/truncates to FEATURE_DIM,
    and saves as .npy arrays of shape (frames, FEATURE_DIM).
    """
    os.makedirs(tar_traindata_folder, exist_ok=True)

    for fname in os.listdir(src_bvh_folder):
        if not fname.lower().endswith('.bvh'):
            continue
        bvh_path = os.path.join(src_bvh_folder, fname)
        raw = read_bvh.parse_frames(bvh_path)  # (F, P)

        # Split raw into translation and angles
        trans = raw[:, :3]
        angles = raw[:, 3:]

        # Normalize
        trans_norm = trans * WEIGHT_TRANSLATION
        angles_norm = angles / ANGLE_SCALE

        # Combine and enforce FEATURE_DIM
        data = np.concatenate([trans_norm, angles_norm], axis=1)
        data_fixed = _pad_or_truncate(data, FEATURE_DIM)

        out_file = os.path.join(tar_traindata_folder, fname.replace('.bvh', '.npy'))
        np.save(out_file, data_fixed)
        print(f"Saved Euler train data: {out_file} (shape: {data_fixed.shape})")


def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    """
    Loads .npy of shape (F, FEATURE_DIM), denormalizes, reconstructs raw channels,
    pads/truncates to EXPECTED_PARAMS, and writes BVH using STANDARD_BVH_FILE.
    """
    os.makedirs(tar_bvh_folder, exist_ok=True)

    for fname in os.listdir(src_train_folder):
        if not fname.lower().endswith('.npy'):
            continue
        npy_path = os.path.join(src_train_folder, fname)
        data_norm = np.load(npy_path)

        # Enforce FEATURE_DIM
        data_fixed = _pad_or_truncate(data_norm, FEATURE_DIM)

        # Split back
        trans_norm = data_fixed[:, :3]
        angles_norm = data_fixed[:, 3:FEATURE_DIM]

        # Denormalize
        trans = trans_norm / WEIGHT_TRANSLATION
        angles = angles_norm * ANGLE_SCALE

        # Recombine raw channels and enforce EXPECTED_PARAMS
        raw = np.concatenate([trans, angles], axis=1)
        raw_fixed = _pad_or_truncate(raw, EXPECTED_PARAMS)

        out_bvh = os.path.join(tar_bvh_folder, fname.replace('.npy', '.bvh'))
        read_bvh.write_frames(STANDARD_BVH_FILE, out_bvh, raw_fixed)
        print(f"Reconstructed BVH: {out_bvh} (params: {raw_fixed.shape[1]})")


if __name__ == '__main__':
    # Example usage (update paths accordingly)
    bvh_dir_path = r"C:\Users\alexa\Desktop\MAI645_Team_04\train_data_bvh\martial"
    euler_enc_dir_path = r"train_data_euler\martial"
    bvh_reconstructed_dir_path = r"reconstructed_euler_bvh\martial"

    generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)
    generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)
