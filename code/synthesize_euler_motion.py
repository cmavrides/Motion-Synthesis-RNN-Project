#!/usr/bin/env python3
"""
Euler-angle Motion Synthesis Refactored

Loads encoded .npy clips, selects random seed segments,
and generates continuation via an acLSTM model.
Outputs synthesized BVH files.
"""
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import read_bvh

# --- Constants ---
HIDDEN_SIZE = 1024
WEIGHT_TRANSLATION = 0.01
HIP_INDEX = read_bvh.joint_index['hip']

# --- Model Definition ---
class acLSTM(nn.Module):
    def __init__(self, frame_dim: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm1 = nn.LSTMCell(frame_dim, hidden)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        self.lstm3 = nn.LSTMCell(hidden, hidden)
        self.dec = nn.Linear(hidden, frame_dim)

    @staticmethod
    def init_state(batch: int, device: torch.device):
        h = [torch.zeros(batch, HIDDEN_SIZE, device=device) for _ in range(3)]
        c = [torch.zeros(batch, HIDDEN_SIZE, device=device) for _ in range(3)]
        return h, c

    def forward_step(self, x: torch.Tensor, state):
        h, c = state
        h[0], c[0] = self.lstm1(x, (h[0], c[0]))
        h[1], c[1] = self.lstm2(h[0], (h[1], c[1]))
        h[2], c[2] = self.lstm3(h[1], (h[2], c[2]))
        out = self.dec(h[2])
        return out, (h, c)

    def forward(self, seq: torch.Tensor, *, gt: int, cond: int) -> torch.Tensor:
        B, T, C = seq.shape
        device = seq.device
        pattern = torch.tensor(([1]*gt + [0]*cond) * ((T // (gt+cond)) + 1),
                               device=device, dtype=torch.bool)[:T]
        h, c = acLSTM.init_state(B, device)
        xo = torch.zeros(B, C, device=device)
        outputs = []
        for t in range(T):
            inp = seq[:, t] if pattern[t] else xo
            xo, (h, c) = self.forward_step(inp, (h, c))
            outputs.append(xo)
        return torch.stack(outputs, dim=1)

# --- Data Utilities ---
def load_and_diff_clips(folder: Path) -> list[np.ndarray]:
    """
    Load .npy clips and apply hip X/Z frame differencing in-place.
    """
    clips = []
    for file in sorted(folder.glob('*.npy')):
        arr = np.load(file).astype(np.float32)
        # subtract previous frame hip x,z
        arr[1:, 0] -= arr[:-1, 0]
        arr[1:, 2] -= arr[:-1, 2]
        clips.append(arr)
    return clips

# --- Synthesis Pipeline ---
def select_seeds(clips: list[np.ndarray], batch: int, seed_len: int) -> np.ndarray:
    """
    Randomly sample batch of seed sequences of length `seed_len`.
    """
    seeds = []
    for _ in range(batch):
        clip = random.choice(clips)
        if len(clip) < seed_len:
            raise ValueError("Clip shorter than seed length")
        start = random.randint(0, len(clip) - seed_len)
        seeds.append(clip[start:start + seed_len])
    return np.stack(seeds, axis=0)  # shape (B, seed_len, D)


def generate_continuation(
    model: acLSTM,
    seed_np: np.ndarray,
    gen_len: int,
    gt: int,
    cond: int,
    device: torch.device
) -> np.ndarray:
    """
    Given seed_np (B, T, D), generate `gen_len` additional frames.
    """
    model.eval()
    seq = torch.from_numpy(seed_np).to(device)
    out, (h, c) = None, acLSTM.init_state(seq.size(0), device)
    # feed all seed frames
    with torch.no_grad():
        for t in range(seq.size(1)):
            out, (h, c) = model.forward_step(seq[:, t], (h, c))
        # generate new frames
        generated = []
        for _ in range(gen_len):
            out, (h, c) = model.forward_step(out, (h, c))
            generated.append(out.cpu().numpy())
    gen_np = np.stack(generated, axis=1)
    return np.concatenate([seed_np, gen_np], axis=1)

# --- Output ---
def write_to_bvh(sequence: np.ndarray, out_path: Path):
    """
    Reconstruct absolute translations, decode units, and write BVH.
    """
    data = sequence.copy()
    last_x = last_z = 0.0
    for t in range(data.shape[0]):
        data[t,0] += last_x; last_x = data[t,0]
        data[t,2] += last_z; last_z = data[t,2]
    # undo encoding
    data[:,:3] /= WEIGHT_TRANSLATION
    data[:,3:] *= 180.0 / math.pi
    data = np.round(data, 6)
    template = getattr(read_bvh, 'standard_bvh_file', 'train_data_bvh/standard.bvh')
    read_bvh.write_frames(template, str(out_path), data)

# --- Main Execution ---
def main():
    # Input paths
    read_weight_path = "/Users/tooulas/Downloads/0011200final.weight"
    dances_folder = Path("../train_data_euler/martial/")
    write_folder = Path("./out_euler_new/")

    # Parameters
    dance_frame_rate = 60
    batch_size = 5
    seed_len = 15      # ground-truth frames
    gen_frames = 400
    cond = seed_len    # reuse seed_len as cond steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prep data
    os.makedirs(write_folder, exist_ok=True)
    clips = load_and_diff_clips(dances_folder)

    # Prepare model
    frame_dim = clips[0].shape[1]
    model = acLSTM(frame_dim).to(device)
    model.load_state_dict(torch.load(read_weight_path, map_location=device))

    # Sample seeds and generate
    seed_np = select_seeds(clips, batch_size, seed_len)
    full_seq = generate_continuation(model, seed_np, gen_frames, gt=seed_len, cond=cond, device=device)

    # Write outputs
    for i, seq in enumerate(full_seq):
        out_path = write_folder / f"euler_synth_out_{i:02d}.bvh"
        write_to_bvh(seq, out_path)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
