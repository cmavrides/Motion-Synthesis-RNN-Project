import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import read_bvh
import argparse

# ——— Hyper-parameters you already had ———
Hip_index = read_bvh.joint_index['hip']
Seq_len = 100
Hidden_size = 1024
Joints_num = 57
Condition_num = 5
Groundtruth_num = 5
In_frame_size = Joints_num * 3

# ——— Matching your generate_euler_training_data.py ———
FEATURE_DIM = 171
WEIGHT_TRANSLATION = 0.01
ANGLE_SCALE = 180.0
STANDARD_BVH_FILE = "..//train_data_bvh/standard.bvh"
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
        return array[:, :target_dim]


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=FEATURE_DIM, hidden_size=Hidden_size, out_frame_size=FEATURE_DIM):
        super(acLSTM, self).__init__()
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch):
        zeros = lambda: torch.zeros(batch, self.hidden_size).cuda()
        h = [Variable(zeros()) for _ in range(3)]
        c = [Variable(zeros()) for _ in range(3)]
        return h, c

    def forward_lstm(self, in_frame, h, c):
        h0, c0 = self.lstm1(in_frame, (h[0], c[0]))
        h1, c1 = self.lstm2(h0, (h[1], c[1]))
        h2, c2 = self.lstm3(h1, (h[2], c[2]))
        out_frame = self.decoder(h2)
        return out_frame, [h0, h1, h2], [c0, c1, c2]

    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        gt = np.ones((seq_len, groundtruth_num))
        con = np.zeros((seq_len, condition_num))
        lst = np.concatenate([gt, con], axis=1).reshape(-1)
        return lst[:seq_len]

    def forward(self, real_seq, condition_num=Condition_num, groundtruth_num=Groundtruth_num):
        batch, seq_len, _ = real_seq.size()
        cond = self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        h, c = self.init_hidden(batch)

        out_seq = Variable(torch.zeros(batch, 1).cuda())
        out_frame = Variable(torch.zeros(batch, self.out_frame_size).cuda())

        for i in range(seq_len):
            if cond[i] == 1:
                in_frame = real_seq[:, i]
            else:
                in_frame = out_frame
            out_frame, h, c = self.forward_lstm(in_frame, h, c)
            out_seq = torch.cat([out_seq, out_frame], dim=1)

        return out_seq[:, 1:]

    def calculate_loss(self, out_seq, groundtruth_seq):
        """
        Angle-distance loss:
          cos_error = cos(Δθ)
          AD = sum_i (1 - cos_error_i)
        and Δθ must be in radians.
        """
        # Δ in normalized units
        diff = out_seq - groundtruth_seq
        # convert normalized diff to radians: normalized angles ([-1,1]) × π → [−π,π]
        rad_diff = diff * math.pi
        cos_error = torch.cos(rad_diff)
        # sum over all elements in the batch
        loss = torch.mean(1.0 - cos_error)
        return loss


Hip = read_bvh.joint_index['hip']

def train_one_iteraton(real_seq_np,
                       model,
                       optimizer,
                       iteration,
                       save_dance_folder,
                       print_loss=False,
                       save_bvh_motion=True):
    """
    real_seq_np: (B, T, F) numpy, in normalized space [trans*WEIGHT, angles/ANGLE_SCALE]
    model: acLSTM with your new calculate_loss
    """

    # 1) build “difference” version of hip‐x/z
    dif = real_seq_np[:, 1:] - real_seq_np[:, :-1]
    seq_np = real_seq_np[:, :-1].copy()
    seq_np[:, :, Hip*3    ] = dif[:, :, Hip*3    ]
    seq_np[:, :, Hip*3 + 2] = dif[:, :, Hip*3 + 2]

    # 2) to torch
    real_seq = torch.FloatTensor(seq_np).cuda()
    T = real_seq.size(1)
    in_seq  = real_seq[:, :T-1]
    gt_seq  = real_seq[:, 1:T].contiguous().view(seq_np.shape[0], -1)

    # 3) forward + loss + backward
    pred_seq = model(in_seq)
    optimizer.zero_grad()
    loss = model.calculate_loss(pred_seq, gt_seq)
    loss.backward()
    optimizer.step()

    if print_loss:
        print(f"iter {iteration:07d}  loss {loss.item():.6f}")

    # 4) optionally write out GT & pred as BVH
    if save_bvh_motion:
        # grab first batch element
        gt_arr  = gt_seq[0].detach().cpu().numpy().reshape(-1, seq_np.shape[2])
        out_arr = pred_seq[0].detach().cpu().numpy().reshape(-1, seq_np.shape[2])

        # unroll hip diffs back to absolute
        def unroll_hip(arr):
            last_x = last_z = 0.0
            out = arr.copy()
            for f in range(out.shape[0]):
                out[f, Hip*3    ] += last_x
                out[f, Hip*3 + 2] += last_z
                last_x = out[f, Hip*3    ]
                last_z = out[f, Hip*3 + 2]
            return out

        gt_arr  = unroll_hip(gt_arr)
        out_arr = unroll_hip(out_arr)

        # decode & write exactly as in generate_euler_training_data.py
        for tag, seq in (("gt", gt_arr), ("out", out_arr)):
            trans_norm  = seq[:, :3]
            angles_norm = seq[:, 3:]
            trans  = trans_norm  / WEIGHT_TRANSLATION
            angles = angles_norm * ANGLE_SCALE
            raw    = np.concatenate([trans, angles], axis=1)
            raw_fixed = _pad_or_truncate(raw, EXPECTED_PARAMS)

            out_bvh = os.path.join(
                save_dance_folder,
                f"{iteration:07d}_{tag}.bvh"
            )
            read_bvh.write_frames(STANDARD_BVH_FILE, out_bvh, raw_fixed)

    return loss.item()

def get_dance_len_lst(dances):
    len_lst = []
    for d in dances:
        length = max(1, int(d.shape[0] / 10))
        len_lst.append(length)
    idxs = []
    for i, l in enumerate(len_lst):
        idxs += [i] * l
    return idxs


def load_dances(dance_folder):
    files = os.listdir(dance_folder)
    dances = []
    print("Loading motion files...")
    for f in files:
        dances.append(np.load(os.path.join(dance_folder, f)))
    print(len(dances), "motion files loaded")
    return dances


def train(dances, frame_rate, batch, seq_len, read_weight_path,
          write_weight_folder, write_bvh_motion_folder,
          in_frame, out_frame, hidden_size=Hidden_size, total_iter=500000):

    seq_len = seq_len + 2
    torch.cuda.set_device(0)

    model = acLSTM(in_frame_size=in_frame,
                   hidden_size=hidden_size,
                   out_frame_size=out_frame)
    if read_weight_path:
        model.load_state_dict(torch.load(read_weight_path))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    # Pre‐compute speed factor
    speed = frame_rate / 30.0

    dance_idxs = get_dance_len_lst(dances)
    for iteration in range(total_iter):
        batch_seqs = []

        for _ in range(batch):
            # Pick a dance proportional to its length
            did = random.choice(dance_idxs)
            dance = dances[did]
            dance_len = dance.shape[0]

            # Compute maximum valid start so that:
            #   max_i int(i*speed + start) < dance_len
            max_start = int(dance_len - seq_len * speed - 10)
            start = random.randint(10, max_start)

            # Sample seq_len frames at interval 'speed'
            sample = [
                dance[int(i * speed + start)]
                for i in range(seq_len)
            ]

            # Augment
            T = [0.1 * (random.random() - 0.5), 0.0, 0.1 * (random.random() - 0.5)]
            R = [0, 1, 0, (random.random() - 0.5) * math.pi * 2]
            batch_seqs.append(read_bvh.augment_train_data(sample, T, R))

        batch_np = np.array(batch_seqs)

        print_loss  = (iteration % 20  == 0)
        save_motion = (iteration % 1000 == 0)
        if save_motion:
            ckpt = os.path.join(write_weight_folder, f"{iteration:07d}.weight")
            torch.save(model.state_dict(), ckpt)

        train_one_iteraton(batch_np,
                           model,
                           optimizer,
                           iteration,
                           write_bvh_motion_folder,
                           print_loss,
                           save_motion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dances_folder',    required=True)
    parser.add_argument('--write_weight_folder', required=True)
    parser.add_argument('--write_bvh_motion_folder', required=True)
    parser.add_argument('--read_weight_path', default="")
    parser.add_argument('--dance_frame_rate', type=int, default=60)
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--in_frame',        type=int, required=True)
    parser.add_argument('--out_frame',       type=int, required=True)
    parser.add_argument('--hidden_size',     type=int, default=1024)
    parser.add_argument('--seq_len',         type=int, default=100)
    parser.add_argument('--total_iterations',type=int, default=100000)
    args = parser.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(args.dances_folder)
    train(dances, args.dance_frame_rate, args.batch_size, args.seq_len,
          args.read_weight_path, args.write_weight_folder,
          args.write_bvh_motion_folder, args.in_frame, args.out_frame,
          args.hidden_size, total_iter=args.total_iterations)


if __name__ == '__main__':
    main()
