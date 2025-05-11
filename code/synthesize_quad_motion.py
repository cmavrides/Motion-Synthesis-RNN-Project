import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import read_bvh
from generate_training_quad_data import generate_bvh_from_quad_traindata

# ——— constants ———
Hip_index       = read_bvh.joint_index['hip']
Seq_len         = 100               # your DEFAULT_SEQ_LEN
Hidden_size     = 1024
Joints_num      = 43
Condition_num   = 5
Groundtruth_num = 5
In_frame_size   = 3 + 4 * Joints_num
out_frame_size = 175

def get_condition_list(condition_num: int, groundtruth_num: int, seq_len: int) -> np.ndarray:
    ones = np.ones((seq_len, groundtruth_num), dtype=np.int32)
    zeros = np.zeros((seq_len, condition_num), dtype=np.int32)
    pattern = np.concatenate((ones, zeros), axis=1).reshape(-1)
    return pattern[:seq_len]


class acLSTM(nn.Module):
    def __init__(self, in_frame_size, hidden_size, out_frame_size):
        super().__init__()
        self.lstm1   = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2   = nn.LSTMCell(hidden_size,    hidden_size)
        self.lstm3   = nn.LSTMCell(hidden_size,    hidden_size)
        self.decoder = nn.Linear(hidden_size, out_frame_size)
        self.hidden_size = hidden_size

    def init_hidden(self, batch, device):
        zeros = torch.zeros(batch, self.hidden_size, device=device)
        return ([zeros.clone() for _ in range(3)],
                [zeros.clone() for _ in range(3)])

    def forward_lstm(self, inp, h, c):
        h0, c0 = self.lstm1(inp, (h[0], c[0]))
        h1, c1 = self.lstm2(h0,  (h[1], c[1]))
        h2, c2 = self.lstm3(h1,  (h[2], c[2]))
        out = self.decoder(h2)
        return out, [h0, h1, h2], [c0, c1, c2]


def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
    """
    initial_seq_np: np array (B, 103, D) of absolute (scaled) hip + quats
    """
    B, raw_S, D = initial_seq_np.shape
    # raw_S=103 → seed_delta_steps=102 → model inputs=101
    seed_delta_steps = raw_S - 1
    model_input_len  = seed_delta_steps - 1  # 101
    M = (D - 3) // 4
    dev = next(model.parameters()).device

    # 1) Build the Δ-hip seed inputs exactly as in training
    dif       = initial_seq_np[:, 1:, :3] - initial_seq_np[:, :-1, :3]
    seed_diff = initial_seq_np[:, :-1].copy()      # shape (B,102,D)
    seed_diff[:, :, 0] = dif[:, :, 0]
    seed_diff[:, :, 2] = dif[:, :, 2]
    seed_input = torch.from_numpy(
        seed_diff[:, :model_input_len]
    ).float().to(dev)                               # (B,101,D)

    # 2) teacher-forcing mask over those 101 inputs
    cond = get_condition_list(Condition_num, Groundtruth_num, model_input_len)

    # 3) warm up the LSTM exactly as in training
    h, c        = model.init_hidden(B, dev)
    out_frame   = torch.zeros(B, D, device=dev)

    # initialize prev_q from the *last* seed frame’s quaternions
    # (absolute, already normalized in load_dances)
    prev_q = torch.from_numpy(
        initial_seq_np[:, -1, 3:].reshape(B, M, 4)
    ).float().to(dev)

    for t in range(model_input_len):
        inp = seed_input[:, t] if cond[t] == 1 else out_frame
        raw, h, c = model.forward_lstm(inp, h, c)

        hip = raw[:, :3]
        q   = raw[:, 3:].view(B, M, 4)
        q   = F.normalize(q, p=2, dim=-1, eps=1e-6)

        # --- enforce quaternion sign continuity ---
        # dot product between each predicted q and prev_q
        dotsign = torch.sign((prev_q * q).sum(-1, keepdim=True))  # shape (B, M, 1)
        q = q * dotsign                                            # flip if needed
        prev_q = q                                                 # update for next step

        out_frame = torch.cat([hip, q.view(B, M*4)], dim=1)

    # 4) now pure free-run for generate_frames_number steps
    frames = []
    for _ in range(generate_frames_number):
        raw, h, c = model.forward_lstm(out_frame, h, c)

        hip = raw[:, :3]
        q   = raw[:, 3:].view(B, M, 4)
        q   = F.normalize(q, p=2, dim=-1, eps=1e-6)

        # continue quaternion continuity
        dotsign = torch.sign((prev_q * q).sum(-1, keepdim=True))
        q = q * dotsign
        prev_q = q

        out_frame = torch.cat([hip, q.view(B, M*4)], dim=1)
        frames.append(out_frame.unsqueeze(1))

    pred = torch.cat(frames, dim=1).detach().cpu().numpy()  # (B, T, D)

    # 5) stitch seed + generated, integrate hip, renormalize, write BVH
    for b in range(B):
        last_seed = initial_seq_np[b, -1, :3]    # (3,)
        gen       = pred[b]                      # (T, D)

        # integrate X/Z deltas → absolute + seed offset
        gen[:, 0] = np.cumsum(gen[:, 0]) + last_seed[0]
        gen[:, 2] = np.cumsum(gen[:, 2]) + last_seed[2]

        # renormalize quaternions (double check after numpy ops)
        flat_q = gen[:, 3:].reshape(-1, 4)
        flat_q /= np.linalg.norm(flat_q, axis=1, keepdims=True)
        gen[:, 3:] = flat_q.reshape(gen[:, 3:].shape)

        combined = np.vstack([initial_seq_np[b], gen])  # (103+T, D)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fn  = tmp / f"{b:02d}.bvh.npy"
            np.save(str(fn), combined)
            generate_bvh_from_quad_traindata(str(tmp), save_dance_folder)

    return np.stack([np.vstack([initial_seq_np[b], pred[b]]) for b in range(B)], axis=0)

def get_dance_len_lst(dances):
    lst = []
    for i, d in enumerate(dances):
        length = max(1, int(len(d)/100))
        lst += [i]*length
    return lst


def load_dances(dance_folder):
    files = os.listdir(dance_folder)
    dances = []
    for f in files:
        arr = np.load(os.path.join(dance_folder, f))
        # renormalize quats
        flatq = arr[:,3:].reshape(-1,4)
        flatq /= np.linalg.norm(flatq, axis=1, keepdims=True)
        arr[:,3:] = flatq.reshape(arr[:,3:].shape)
        dances.append(arr)
    return dances


def test(dances,
         frame_rate: int,
         batch: int,
         initial_seq_len: int,            # stays in signature
         generate_frames_number: int,
         read_weight_path: str,
         write_bvh_motion_folder: str,
         in_frame_size: int,
         hidden_size: int,
         out_frame_size: int):

    torch.cuda.set_device(0)
    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    model.cuda()

    speed = frame_rate / 30.0

    # — seed with (initial_seq_len + 3) frames to get 101 Δ-hip inputs —
    seed_frames = initial_seq_len + 3  # e.g. 100 + 3 = 103

    dance_len_lst = get_dance_len_lst(dances)
    dance_batch = []
    for _ in range(batch):
        dance_id = random.choice(dance_len_lst)
        dance    = dances[dance_id]
        F        = dance.shape[0]

        # ensure we can grab seed_frames before the end
        start_id = random.randint(10, int(F - seed_frames*speed - 10))
        sample_seq = [
            dance[int(i*speed + start_id)]
            for i in range(seed_frames)
        ]
        dance_batch.append(sample_seq)

    batch_np = np.array(dance_batch)  # shape (B, 103, D)
    generate_seq(batch_np, generate_frames_number, model, write_bvh_motion_folder)


read_weight_path=r"C:/Users/alexa/Desktop/MAI645_Team_04/results_quad_weights_last_updated_train/0017000.weight"
write_bvh_motion_folder = "./synth_bvh_quad_latest/"
dances_folder = r"C:/Users/alexa/Desktop/MAI645_Team_04/train_data_quad/martial/"
dance_frame_rate = 60
batch = 5
initial_seq_len = 100
generate_frames_number = 400

if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

dances = load_dances(dances_folder)



test(dances,
         dance_frame_rate,
         batch,
         initial_seq_len,
         generate_frames_number,
         read_weight_path,
         write_bvh_motion_folder,
         In_frame_size,
         Hidden_size,
         In_frame_size)

