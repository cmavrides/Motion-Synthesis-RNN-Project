import os
import generate_training_euler_data
import generate_training_quad_data
import generate_training_pos_data

# ========== CONFIG ========== #
# Use a small set of BVH files for fast debugging
bvh_dir = "train_data_bvh/indian"
euler_out_dir = "debug_output/euler"
euler_recon_dir = "debug_output/euler_reconstructed"

quat_out_dir = "debug_output/quat"
quat_recon_dir = "debug_output/quat_reconstructed"

pos_out_dir = "debug_output/pos"
pos_recon_dir = "debug_output/pos_reconstructed"

# ============================ #

print("Running Euler encoding/decoding...")
generate_training_euler_data.generate_euler_traindata_from_bvh(bvh_dir, euler_out_dir)
generate_training_euler_data.generate_bvh_from_euler_traindata(euler_out_dir, euler_recon_dir)

print("\nRunning Quaternion encoding/decoding...")
generate_training_quad_data.generate_quad_traindata_from_bvh(bvh_dir, quat_out_dir)
generate_training_quad_data.generate_bvh_from_quad_traindata(quat_out_dir, quat_recon_dir)

print("\nRunning Positional encoding/decoding...")
generate_training_pos_data.generate_pos_traindata_from_bvh(bvh_dir, pos_out_dir)
generate_training_pos_data.generate_pos_bvh_from_traindata(pos_out_dir, pos_recon_dir)
print("\n✅ Done! Check 'debug_output/' for results.")
