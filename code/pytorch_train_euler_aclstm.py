#!/usr/bin/env python3

import os
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import read_bvh

# ——— Hyper-parameters ———
WEIGHT_TRANSLATION = 0.01       # scale for meters → unitless
ANGLE_SCALE = 180.0             # degrees per unit (unused here)
STANDARD_BVH_FILE = "../train_data_bvh/standard.bvh"

# sequencing / conditioning
SEQ_LEN = 100
COND_STEPS = 5
GT_STEPS = 5

# these will be set at runtime
FEATURE_DIM = -1   # full frame size (translation+angles)
HIP_IDX     = -1   # index of hip joint in Euler array


# ——— Model + Loss from your first script ———
class acLSTM(nn.Module):
    def __init__(self, frame_dim: int, hidden: int = 1024):
        super().__init__()
        self.lstm1 = nn.LSTMCell(frame_dim, hidden)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        self.lstm3 = nn.LSTMCell(hidden, hidden)
        self.dec   = nn.Linear(hidden, frame_dim)

    @staticmethod
    def _init_state(self, batch: int, device):
        h = [torch.zeros(batch, self.dec.in_features, device=device) for _ in range(3)]
        c = [torch.zeros(batch, self.dec.in_features, device=device) for _ in range(3)]
        return h, c

    def _forward_step(self, x: torch.Tensor, hc):
        h, c = hc
        h[0], c[0] = self.lstm1(x, (h[0], c[0]))
        h[1], c[1] = self.lstm2(h[0], (h[1], c[1]))
        h[2], c[2] = self.lstm3(h[1], (h[2], c[2]))
        return self.dec(h[2]), (h, c)

    def forward(self,
                seq: torch.Tensor,
                *,
                cond: int = COND_STEPS,
                gt:   int = GT_STEPS) -> torch.Tensor:
        B, T, C = seq.shape
        device  = seq.device

        # build [1 * GT + 0 * COND] repeating mask
        pattern = torch.tensor(([1]*gt + [0]*cond)*100,
                               dtype=torch.bool, device=device)[:T]

        hc = self._init_state(self, B, device)
        xo = torch.zeros(B, C, device=device)
        out = []
        for t in range(T):
            inp = seq[:,t] if pattern[t] else xo
            xo, hc = self._forward_step(inp, hc)
            out.append(xo)
        return torch.stack(out, dim=1)


def compute_loss(pred_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
    """
    pred_seq, tgt_seq: (B, T, F) with first 3 dims translation (in WEIGHT-scaled units),
    rest = Euler angles in radians.
    """
    # translation: mean squared error
    loss_t = F.mse_loss(pred_seq[..., :3], tgt_seq[..., :3])
    # rotation: mean (1 - cos Δθ)
    loss_r = torch.mean(1.0 - torch.cos(pred_seq[..., 3:] - tgt_seq[..., 3:]))
    return loss_t + loss_r


def hip_differential(seq_np: np.ndarray, hip_idx: int) -> np.ndarray:
    """
    Converts absolute hip x/z to frame‐to‐frame deltas.
    """
    diff = seq_np[:,1:] - seq_np[:,:-1]
    base = seq_np[:, :-1].copy()
    base[:,:,hip_idx*3 + 0] = diff[:,:,hip_idx*3 + 0]
    base[:,:,hip_idx*3 + 2] = diff[:,:,hip_idx*3 + 2]
    return base


# ——— Training iteration ———
def train_one_iteration(real_np: np.ndarray,
                        model: acLSTM,
                        optimizer,
                        iteration: int,
                        save_folder: str,
                        print_loss: bool=False,
                        save_bvh: bool=True) -> float:
    global HIP_IDX

    # 1) scale translations & do hip differential
    real_np[...,:3] *= WEIGHT_TRANSLATION
    seq_np = hip_differential(real_np, HIP_IDX)

    # 2) to torch
    seq = torch.tensor(seq_np, dtype=torch.float32, device="cuda")
    inp = seq[:, :-1]      # feed up to T-1
    tgt = seq[:, 1:]       # predict 1..T

    # 3) forward + loss + backward
    pred = model(inp)      # out shape B x (T-1) x F
    loss = compute_loss(pred, tgt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if print_loss:
        print(f"[{iteration:07d}] loss = {loss.item():.6f}")

    # 4) optionally dump BVH (first example only)
    if save_bvh:
        gt_arr  = tgt[0].cpu().numpy().copy()
        out_arr = pred[0].detach().cpu().numpy().copy()

        # un‐difference hip
        def unroll(x):
            lx = lz = 0.0
            out = x.copy()
            for f in range(out.shape[0]):
                out[f, HIP_IDX*3   ] += lx
                out[f, HIP_IDX*3+2 ] += lz
                lx = out[f, HIP_IDX*3]
                lz = out[f, HIP_IDX*3+2]
            return out

        gt_arr  = unroll(gt_arr)
        out_arr = unroll(out_arr)

        for tag, arr in (("gt", gt_arr), ("out", out_arr)):
            # convert back to bvh units
            trans = arr[:,:3] / WEIGHT_TRANSLATION
            angles = arr[:,3:]         # already radians
            # bvh wants degrees:
            angles = (angles * 180.0/math.pi)
            raw = np.concatenate([trans, angles], axis=1)

            # pad/truncate to template size
            F,D = raw.shape
            if D < EXPECTED_PARAMS:
                pad = np.zeros((F, EXPECTED_PARAMS-D))
                raw = np.concatenate([raw, pad],axis=1)
            else:
                raw = raw[:,:EXPECTED_PARAMS]

            out_file = os.path.join(save_folder, f"{iteration:07d}_{tag}.bvh")
            read_bvh.write_frames(STANDARD_BVH_FILE, out_file, raw)

    return loss.item()


# ——— Data loading & main loop ———
def get_dance_pool(dances):
    pool = []
    for i,d in enumerate(dances):
        pool += [i]*max(1, d.shape[0]//10)
    return pool

def load_dances(folder):
    print("Loading motion files…")
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".npy")]
    dances = [np.load(f) for f in files]
    print(f"{len(dances)} clips loaded.")
    return dances

def main():
    global FEATURE_DIM, HIP_IDX, EXPECTED_PARAMS, STANDARD_BVH_FILE

    ap = argparse.ArgumentParser()
    ap.add_argument("--dances_folder",    required=True)
    ap.add_argument("--write_weight_folder", required=True)
    ap.add_argument("--write_bvh_motion_folder", required=True)
    ap.add_argument("--read_weight_path", default="")
    ap.add_argument("--template_bvh",     default="")
    ap.add_argument("--dance_frame_rate", type=int, default=60)
    ap.add_argument("--batch_size",       type=int, default=32)
    ap.add_argument("--seq_len",          type=int, default=SEQ_LEN)
    ap.add_argument("--total_iterations",type=int, default=100000)
    args = ap.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(args.dances_folder)
    if not dances:
        raise SystemExit("No .npy found in --dances_folder")

    # infer dims & hip idx
    FEATURE_DIM = dances[0].shape[1]
    HIP_IDX     = read_bvh.joint_index['hip']
    EXPECTED_PARAMS = read_bvh.parse_frames(
        args.template_bvh or STANDARD_BVH_FILE
    ).shape[1]
    STANDARD_BVH_FILE = args.template_bvh if args.template_bvh else STANDARD_BVH_FILE

    # build model
    model = acLSTM(frame_dim=FEATURE_DIM).cuda()
    if args.read_weight_path:
        model.load_state_dict(torch.load(args.read_weight_path))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    pool  = get_dance_pool(dances)
    speed = args.dance_frame_rate / 30.0
    Lp1   = args.seq_len + 1

    for it in range(args.total_iterations):
        batch_seqs = []
        for _ in range(args.batch_size):
            d = dances[random.choice(pool)]
            maxstart = int(d.shape[0] - Lp1*speed - 10)
            s = random.randint(10, maxstart)
            frames = [d[int(s + i*speed)] for i in range(Lp1)]
            batch_seqs.append(frames)
        batch_np = np.array(batch_seqs, dtype=np.float32)

        print_loss = (it % 20  == 0)
        save_bvh   = (it % 1000 == 0)
        if save_bvh:
            ckpt = os.path.join(args.write_weight_folder, f"{it:07d}.weight")
            torch.save(model.state_dict(), ckpt)

        train_one_iteration(batch_np,
                            model,
                            optim,
                            it,
                            args.write_bvh_motion_folder,
                            print_loss,
                            save_bvh)

    # final save
    torch.save(model.state_dict(),
               os.path.join(args.write_weight_folder, "final.weight"))

if __name__ == "__main__":
    main()
