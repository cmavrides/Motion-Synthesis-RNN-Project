import os
import torch
import torch.nn as nn
import numpy as np
import random
import read_bvh
from scipy.spatial.transform import Rotation as R

Hip_index = read_bvh.joint_index['hip']

Joints_num = 56  # Should match training (227 = 3 + 56*4)
In_frame_size = 227
Hidden_size = 1024

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=227, hidden_size=1024, out_frame_size=227):
        super(acLSTM, self).__init__()
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, out_frame_size)

    def init_hidden(self, batch):
        return [torch.zeros(batch, self.hidden_size).cuda() for _ in range(3)], \
               [torch.zeros(batch, self.hidden_size).cuda() for _ in range(3)]

    def forward_lstm(self, in_frame, vec_h, vec_c):
        h0, c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        h1, c1 = self.lstm2(h0, (vec_h[1], vec_c[1]))
        h2, c2 = self.lstm3(h1, (vec_h[2], vec_c[2]))
        out = self.decoder(h2)
        return out, [h0, h1, h2], [c0, c1, c2]

    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size(0)
        vec_h, vec_c = self.init_hidden(batch)
        out_seq = torch.zeros(batch, 0).cuda()
        out_frame = torch.zeros(batch, self.out_frame_size).cuda()

        for i in range(initial_seq.size(1)):
            in_frame = initial_seq[:, i]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), dim=1)

        for _ in range(generate_frames_number):
            in_frame = out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)

            # Normalize quaternion part of the output
            # This ensures model predictions always produce valid unit quaternions
            B = out_frame.size(0)
            F = self.out_frame_size
            root_dim = 3
            quat_dim = 4
            J = (F - root_dim) // quat_dim

            out_frame_view = out_frame.view(B, F)
            quats = out_frame_view[:, root_dim:].view(B, J, quat_dim)
            quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
            out_frame_view[:, root_dim:] = quats.view(B, J * quat_dim)

            out_seq = torch.cat((out_seq, out_frame), dim=1)

        return out_seq

def load_dances(folder):
    print("Loading quad .npy sequences...")
    return [np.load(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".npy")]

def get_dance_len_lst(dances):
    return [i for i in range(len(dances)) for _ in range(10)]

def generate_seq(initial_seq_np, generate_frames_number, model, save_folder):
    initial_seq_tensor = torch.tensor(initial_seq_np, dtype=torch.float32).cuda()
    out_tensor = model(initial_seq_tensor, generate_frames_number)

    for b in range(out_tensor.size(0)):
        seq = out_tensor[b].detach().cpu().numpy().reshape(-1, In_frame_size)

        # Diagnostic information - helpful for debugging
        print(f"First frame root: {seq[0, :3]}")
        print(f"First frame quaternion norms: {np.linalg.norm(seq[0, 3:].reshape(-1, 4), axis=1)}")

        # Clamp root translation to reasonable values
        root = np.clip(seq[:, :3], -2, 2)

        # Handle quaternions with extra care
        quats = seq[:, 3:].reshape(seq.shape[0], -1, 4)

        # Check for and handle near-zero quaternions
        quats_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        min_norm = 1e-2  # Minimum acceptable norm

        # Create a mask for quaternions with small norms
        small_norm_mask = quats_norm < min_norm

        # Replace small norm quaternions with identity quaternion [1,0,0,0]
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                if quats_norm[i, j, 0] < min_norm:
                    quats[i, j] = identity_quat

        # Normalize all quaternions to unit length
        quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

        # Reshape for conversion
        num_frames = quats.shape[0]
        num_joints = quats.shape[1]
        quats_flat = quats.reshape(-1, 4)

        # Convert to Euler angles for BVH
        eulers_flat = R.from_quat(quats_flat).as_euler('zxy', degrees=True)
        eulers = eulers_flat.reshape(num_frames, num_joints * 3)

        # Combine root and joint rotations
        bvh_data = np.concatenate([root, eulers], axis=1)

        # Write to BVH file
        read_bvh.write_traindata_to_bvh(os.path.join(save_folder, f"out{b:02d}.bvh"), bvh_data)

def synthesize_motion(dances, args):
    model = acLSTM(In_frame_size, Hidden_size, In_frame_size).cuda()
    model.load_state_dict(torch.load(args["model_weights"]))
    model.eval()

    len_lst = get_dance_len_lst(dances)
    speed = args["frame_rate"] / 30
    batch_data = []

    for _ in range(args["batch"]):
        idx = random.choice(len_lst)
        dance = dances[idx]
        start = random.randint(10, int(len(dance) - args["initial_seq_len"] * speed - 10))
        sequence = [dance[int(start + i * speed)] for i in range(args["initial_seq_len"])]
        batch_data.append(sequence)

    batch_np = np.array(batch_data)
    generate_seq(batch_np, args["generate_frames"], model, args["output_dir"])

# === Settings ===
args = {
    "model_weights": "C:/Users/alexa/Desktop/MAI645_Team_04/results_quad_weights_v4/0007000.weight",
    "dances_folder": "C:/Users/alexa/Desktop/MAI645_Team_04/train_data_quad/martial/",
    "output_dir": "C:/Users/alexa/Desktop/result_quat_fourth/",
    "frame_rate": 60,
    "batch": 5,
    "initial_seq_len": 15,
    "generate_frames": 400
}

if __name__ == "__main__":
    os.makedirs(args["output_dir"], exist_ok=True)
    dances = load_dances(args["dances_folder"])
    synthesize_motion(dances, args)
