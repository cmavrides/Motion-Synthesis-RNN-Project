# test_decode_helpers.py

import numpy as np
from decode_helpers import decode_pos_to_bvh, decode_euler_to_bvh, decode_quat_to_bvh

# This should point to the .npy files created by generate_quad_traindata_from_bvh
quat_input = "debug_output/quat/01.npy"

quat_data = np.load(quat_input)
print("Quaternion shape:", quat_data.shape)
print("Example vector length:", len(quat_data[0]))


# Sample paths
euler_input = "debug_output/euler/01.npy"
euler_output_bvh = "debug_output/euler/decoded_01.bvh"

quat_input = "debug_output/quat/01.npy"
quat_output_bvh = "debug_output/quat/decoded_01.bvh"

# Euler decode test
euler_data = np.load(euler_input)
decode_euler_to_bvh(euler_output_bvh, euler_data)

# Quat decode test
quat_data = np.load(quat_input)
decode_quat_to_bvh(quat_output_bvh, quat_data)

print("✅ Decoding test complete!")
