#!/usr/bin/env python3

"""
acLSTM trainer for Euler-angle motion sequences with unified loss.
"""

from __future__ import annotations

import os
import random
import csv
import atexit
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import read_bvh

# global parameters updated at runtime
weight_translation = 0.01  # 1 m → scaled units (≈ tens of cm)
FRAME_DIM: int = -1
HIP_IDX: int = -1
TEMPLATE_BVH: str = "../train_data_bvh/standard.bvh"

SEQ_LEN = 100
HIDDEN_SIZE = 1024
COND_STEPS = 5
GT_STEPS = 5

# CSV loss log
_loss_log_f = open("loss_log_euler.csv", "w", newline="")
csv.writer(_loss_log_f).writerow(["iter", "loss"])
atexit.register(_loss_log_f.close)


class acLSTM(nn.Module):
    def __init__(self, frame_dim: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm1 = nn.LSTMCell(frame_dim, hidden)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        self.lstm3 = nn.LSTMCell(hidden, hidden)
        self.dec = nn.Linear(hidden, frame_dim)

    @staticmethod
    def _init_state(batch: int, device):
        h = [torch.zeros(batch, HIDDEN_SIZE, device=device) for _ in range(3)]
        c = [torch.zeros(batch, HIDDEN_SIZE, device=device) for _ in range(3)]
        return h, c

    def _forward_step(self, x: torch.Tensor, hc):
        h, c = hc
        h[0], c[0] = self.lstm1(x, (h[0], c[0]))
        h[1], c[1] = self.lstm2(h[0], (h[1], c[1]))
        h[2], c[2] = self.lstm3(h[1], (h[2], c[2]))
        return self.dec(h[2]), (h, c)

    def forward(self, seq: torch.Tensor, *, cond: int = COND_STEPS, gt: int = GT_STEPS):
        B, T, C = seq.shape
        device = seq.device
        pattern = torch.tensor(([1] * gt + [0] * cond) * 100,
                               dtype=torch.bool, device=device)[:T]
        hc = self._init_state(B, device)
        xo = torch.zeros(B, C, device=device)
        out = []
        for t in range(T):
            xi = seq[:, t] if pattern[t] else xo
            xo, hc = self._forward_step(xi, hc)
            out.append(xo)
        return torch.stack(out, dim=1)


def compute_loss(pred_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
    """
    Unified loss combining MSE on translations and angular difference on rotations.
    Expects motion sequences with first 3 dims = translation, rest = Euler angles (rad).
    """
    trans_pred, rot_pred = pred_seq[..., :3], pred_seq[..., 3:]
    trans_tgt, rot_tgt = tgt_seq[..., :3], tgt_seq[..., 3:]
    loss_trans = F.mse_loss(trans_pred, trans_tgt)
    loss_rot = torch.mean(1 - torch.cos(rot_pred - rot_tgt))
    return loss_trans + loss_rot


def hip_differential(seq_np: np.ndarray, hip_idx: int) -> np.ndarray:
    diff = seq_np[:, 1:] - seq_np[:, :-1]
    base = seq_np[:, :-1].copy()
    base[:, :, hip_idx * 3 + 0] = diff[:, :, hip_idx * 3 + 0]
    base[:, :, hip_idx * 3 + 2] = diff[:, :, hip_idx * 3 + 2]
    return base


def train_one_iter(
    batch_np: np.ndarray,
    model: acLSTM,
    optim,
    iteration: int,
    save_dir: str,
    print_loss: bool = False,
    save_bvh: bool = False
) -> None:
    global HIP_IDX, TEMPLATE_BVH

    batch_np[:, :, :3] *= weight_translation
    batch_np = hip_differential(batch_np, HIP_IDX)

    seq = torch.tensor(batch_np, dtype=torch.float32, device="cuda")
    target = seq[:, 1:].clone()
    inp = seq[:, :-1]

    pred = model(inp)
    loss = compute_loss(pred, target)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if iteration % 100 == 0:
        csv.writer(_loss_log_f).writerow([iteration, loss.item()])
    if print_loss:
        print(f"[{iteration:06d}] loss = {loss.item():.6f}")

    if save_bvh:
        gt_seq = target[0].cpu().numpy().copy()
        out_seq = pred[0].detach().cpu().numpy().copy()

        for seq_arr in (gt_seq, out_seq):
            last_x = last_z = 0.0
            for f in seq_arr:
                f[HIP_IDX*3 + 0] += last_x; last_x = f[HIP_IDX*3 + 0]
                f[HIP_IDX*3 + 2] += last_z; last_z = f[HIP_IDX*3 + 2]

        for seq_arr in (gt_seq, out_seq):
            seq_arr[:, :3] /= weight_translation
            seq_arr[:, 3:] = np.rad2deg(seq_arr[:, 3:])

        fname_gt = os.path.join(save_dir, f"{iteration:07d}_gt.bvh")
        fname_out = os.path.join(save_dir, f"{iteration:07d}_out.bvh")
        read_bvh.write_frames(TEMPLATE_BVH, fname_gt, gt_seq)
        read_bvh.write_frames(TEMPLATE_BVH, fname_out, out_seq)

    best = train_one_iter.__dict__.get("_best", float('inf'))
    stagnant = train_one_iter.__dict__.get("_stagnant", 0)
    if loss.item() < best - 1e-6:
        train_one_iter._best, train_one_iter._stagnant = loss.item(), 0
    else:
        train_one_iter._stagnant = stagnant + 1
        train_one_iter._best = best
    if train_one_iter._stagnant > 1000 and iteration % 100 == 0:
        print("-- convergence detected; terminating.")
        raise SystemExit


def load_dances(folder: str) -> List[np.ndarray]:
    print("Loading motion files …")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".npy")]
    dances = [np.load(path) for path in files]
    print(f"{len(dances)} clips loaded.")
    return dances


def get_weighted_index_pool(dances: List[np.ndarray]) -> list[int]:
    pool = []
    for idx, d in enumerate(dances):
        pool.extend([idx] * max(1, d.shape[0] // 10))
    return pool


def main():
    global FRAME_DIM, HIP_IDX, TEMPLATE_BVH

    ap = argparse.ArgumentParser()
    ap.add_argument("--dances_folder", required=True)
    ap.add_argument("--write_weight_folder", required=True)
    ap.add_argument("--write_bvh_motion_folder", required=True)
    ap.add_argument("--read_weight_path", default="")
    ap.add_argument("--template_bvh", default="",
                    help="Optional custom hierarchy for BVH dumps")
    ap.add_argument("--dance_frame_rate", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--total_iterations", type=int, default=100_000)
    args = ap.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(args.dances_folder)
    if not dances:
        raise SystemExit("No .npy clips found in --dances_folder")

    FRAME_DIM = dances[0].shape[1]
    HIP_IDX = read_bvh.joint_index['hip']
    print(f"✔  Frame dim = {FRAME_DIM}  (hip index = {HIP_IDX})")

    if args.template_bvh and os.path.isfile(args.template_bvh):
        TEMPLATE_BVH = args.template_bvh
    else:
        raw_folder = args.dances_folder.replace("train_data_euler", "train_data_bvh")
        try:
            TEMPLATE_BVH = next(
                os.path.join(raw_folder, f)
                for f in os.listdir(raw_folder)
                if f.lower().endswith(".bvh")
            )
        except StopIteration:
            TEMPLATE_BVH = os.path.join(os.path.dirname(__file__), "..",
                                        "train_data_bvh", "standard.bvh")
    print(f"✔  BVH template = {TEMPLATE_BVH}")

    model = acLSTM(frame_dim=FRAME_DIM).cuda()
    if args.read_weight_path:
        model.load_state_dict(torch.load(args.read_weight_path))
        print(f"✔  resumed weights from {args.read_weight_path}")
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    pool = get_weighted_index_pool(dances)
    speed = args.dance_frame_rate / 30.0
    seq_len_plus1 = args.seq_len + 1

    for it in range(args.total_iterations):
        batch = []
        for _ in range(args.batch_size):
            d = dances[random.choice(pool)]
            start = random.randint(10, int(d.shape[0] - seq_len_plus1 * speed - 10))
            frames = [d[int(start + i * speed)] for i in range(seq_len_plus1)]
            batch.append(frames)
        batch_np = np.asarray(batch, dtype=np.float32)

        print_loss = (it % 20 == 0)
        save_bvh = (it % 1000 == 0)
        if save_bvh:
            ckpt = os.path.join(
                args.write_weight_folder, f"{it:07d}.weight"
            )
            torch.save(model.state_dict(), ckpt)
            print(f"🔖  checkpoint → {ckpt}")

        try:
            train_one_iter(
                batch_np,
                model,
                optim,
                it,
                args.write_bvh_motion_folder,
                print_loss,
                save_bvh
            )
        except SystemExit:
            final_ckpt = os.path.join(
                args.write_weight_folder, f"{it:07d}_final.weight"
            )
            torch.save(model.state_dict(), final_ckpt)
            print(f"✔  convergence at iter {it}; final weights → {final_ckpt}")
            break
    else:
        final_ckpt = os.path.join(
            args.write_weight_folder, "final.weight"
        )
        torch.save(model.state_dict(), final_ckpt)
        print(f"✔  training finished; final weights → {final_ckpt}")


if __name__ == "__main__":
    main()
