# decode_helpers.py

import read_bvh
import rotation2xyz
import numpy as np

# Load shared skeleton + joint index
standard_bvh_file = "train_data_bvh/standard.bvh"
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

print(f"[INFO] Detected joints: {len(non_end_bones)}")
print(f"Skeleton: {len(skeleton)} joints")

# Joint index for converting data back to XYZ
sample_data = read_bvh.parse_frames(standard_bvh_file)
joint_index = read_bvh.get_pos_joints_index(sample_data[0], non_end_bones, skeleton)

# ========== POS DECODING ========== #
def decode_pos_to_bvh(output_file, train_data):
    read_bvh.write_traindata_to_bvh(output_file, train_data)

# ========== EULER DECODING ========== #
def decode_euler_to_bvh(output_file, train_data):
    xyz_motion = []
    for frame in train_data:
        data = frame * 100.0  # undo normalization
        hip_pos = data[joint_index['hip'] * 3: joint_index['hip'] * 3 + 3]
        positions = {}
        for joint in joint_index:
            pos = data[joint_index[joint] * 3: joint_index[joint] * 3 + 3]
            if joint != 'hip':
                pos += hip_pos
            positions[joint] = pos
        xyz_motion.append(positions)

    read_bvh.write_xyz_to_bvh(
        xyz_motion=xyz_motion,
        skeleton=skeleton,
        non_end_bones=non_end_bones,
        format_filename=standard_bvh_file,
        output_filename=output_file
    )

# ========== QUATERNION DECODING ========== #
def decode_quat_to_bvh(output_file, train_data):
    xyz_motion = []
    success_count = 0
    try:
        for i, frame in enumerate(train_data):
            try:
                position_dict = quaternion_vector_to_position_dict(frame, skeleton, joint_index)
                xyz_motion.append(position_dict)
                success_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to decode frame {i}: {e}")
                # If we have at least one successful frame, use it as a template
                if success_count > 0:
                    xyz_motion.append(xyz_motion[0])  # Use first frame as fallback

        if success_count == 0:
            print("[ERROR] Could not decode any frames successfully")
            return

        read_bvh.write_xyz_to_bvh(
            xyz_motion=xyz_motion,
            skeleton=skeleton,
            non_end_bones=non_end_bones,
            format_filename=standard_bvh_file,
            output_filename=output_file
        )
        print(f"[SUCCESS] Decoded {success_count}/{len(train_data)} frames successfully")
    except Exception as e:
        print(f"[ERROR] Failed to decode animation: {e}")
        raise

def quaternion_vector_to_position_dict(data, skeleton, joint_index):
    from scipy.spatial.transform import Rotation as R

    # Extract hip position
    hip_pos = data[0:3]
    positions = {'hip': hip_pos}

    # Calculate expected data size
    num_joints = len(joint_index)
    expected_length = 3 + 4 * (num_joints - 1)
    actual_length = len(data)

    if actual_length != expected_length:
        print(f"[WARNING] Data length mismatch - have {actual_length}, expected {expected_length}")
        actual_num_joints = (actual_length - 3) // 4 + 1
        print(f"[INFO] Will decode {actual_num_joints} joints instead of {num_joints}")

    # Extract all quaternions from the data
    quaternions = {}
    offset = 3

    # Extract hip quaternion
    if offset + 4 <= len(data):
        quaternions['hip'] = data[offset:offset+4]
        offset += 4

    # Extract other joint quaternions
    for joint in joint_index:
        if joint == 'hip' or offset + 4 > len(data):
            continue

        quat = data[offset:offset+4]
        quaternions[joint] = quat
        offset += 4

    # Function to recursively calculate positions for all skeleton joints
    def calculate_joint_positions(joint_name, parent_pos=None):
        # Skip joints not in skeleton
        if joint_name not in skeleton:
            return

        # If it's the root joint
        if parent_pos is None:
            positions[joint_name] = hip_pos
        else:
            # Get parent's position
            parent = skeleton[joint_name]['parent']

            # If joint has quaternion data, apply rotation
            if joint_name in quaternions:
                try:
                    # Get parent orientation (if available)
                    parent_quat = quaternions.get(parent, [0, 0, 0, 1])
                    parent_r = R.from_quat(parent_quat)

                    # Get this joint's quaternion and compute orientation
                    joint_quat = quaternions[joint_name]
                    joint_r = R.from_quat(joint_quat)

                    # Get offset from skeleton
                    offset = np.array(skeleton[joint_name]['offsets'])

                    # Apply rotations
                    rotated_offset = parent_r.apply(offset)
                    positions[joint_name] = positions[parent] + rotated_offset
                except Exception as e:
                    print(f"[WARNING] Error processing joint {joint_name}: {e}")
                    # Fallback: just use offset
                    offset = np.array(skeleton[joint_name]['offsets'])
                    positions[joint_name] = positions[parent] + offset
            else:
                # No quaternion - just use offset
                offset = np.array(skeleton[joint_name]['offsets'])
                positions[joint_name] = positions[parent] + offset

        # Process children recursively
        for child in skeleton[joint_name].get('children', []):
            calculate_joint_positions(child)

    # Ensure positions exist for ALL joints in the skeleton
    for joint_name in skeleton:
        if joint_name not in positions:
            parent = skeleton[joint_name].get('parent')
            if parent and parent in positions:
                # Use parent's position plus offset
                offset = np.array(skeleton[joint_name]['offsets'])
                positions[joint_name] = positions[parent] + offset

    # Start processing from the root 'hip' joint
    calculate_joint_positions('hip')

    # Final check - ensure ALL joints have positions
    missing_joints = []
    for joint in skeleton:
        if joint not in positions:
            missing_joints.append(joint)
            # Use hip position as fallback
            positions[joint] = hip_pos

    if missing_joints:
        print(f"[WARNING] Filled in missing positions for joints: {missing_joints}")

    return positions
