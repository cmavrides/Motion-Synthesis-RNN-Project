import argparse
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import read_bvh
from generate_training_quad_data import generate_bvh_from_quad_traindata, weight_translation

# Index of hip joint in BVH data
Hip_index = read_bvh.joint_index['hip']

# Default hyperparameters
DEFAULT_SEQ_LEN = 100
DEFAULT_HIDDEN_SIZE = 1024
Joints_num = 43
In_frame_size = 3 + 4 * Joints_num
Condition_num = 5
Groundtruth_num = 5


def get_condition_list(condition_num: int, groundtruth_num: int, seq_len: int) -> np.ndarray:
    """
    Produce a repeating pattern of groundtruth (1) and condition (0) flags of length seq_len.
    """
    ones = np.ones((seq_len, groundtruth_num), dtype=np.int32)
    zeros = np.zeros((seq_len, condition_num), dtype=np.int32)
    pattern = np.concatenate((ones, zeros), axis=1).reshape(-1)
    return pattern[:seq_len]


class acLSTM(nn.Module):
    def __init__(self,
                 in_frame_size: int,
                 hidden_size: int,
                 out_frame_size: int):
        super().__init__()
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, out_frame_size)

    def init_hidden(self, batch: int, device: torch.device):
        """
        Initialize hidden and cell states for 3-layer LSTM.
        """
        zeros = torch.zeros(batch, self.hidden_size, device=device)
        return ([zeros.clone() for _ in range(3)],
                [zeros.clone() for _ in range(3)])

    def forward_lstm(self,
                     in_frame: torch.Tensor,
                     h: list[torch.Tensor],
                     c: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h0, c0 = self.lstm1(in_frame, (h[0], c[0]))
        h1, c1 = self.lstm2(h0, (h[1], c[1]))
        h2, c2 = self.lstm3(h1, (h[2], c[2]))
        out_frame = self.decoder(h2)
        return out_frame, [h0, h1, h2], [c0, c1, c2]

    def forward(self,
                real_seq: torch.Tensor,
                condition_num: int = Condition_num,
                groundtruth_num: int = Groundtruth_num) -> torch.Tensor:
        batch, seq_len, _ = real_seq.shape
        D = self.out_frame_size
        M = (D - 3) // 4
        device = real_seq.device

        cond = get_condition_list(condition_num, groundtruth_num, seq_len)
        h, c = self.init_hidden(batch, device)

        out_seq = []
        out_frame = torch.zeros(batch, D, device=device)

        for t in range(seq_len):
            inp = real_seq[:, t] if cond[t] == 1 else out_frame
            raw, h, c = self.forward_lstm(inp, h, c)

            hip = raw[:, :3]
            quats = raw[:, 3:].view(batch, M, 4)
            quats = F.normalize(quats, p=2, dim=-1, eps=1e-6)
            out_frame = torch.cat([hip, quats.view(batch, M * 4)], dim=1)
            out_seq.append(out_frame.unsqueeze(1))

        # concatenate and flatten: [batch, seq_len, D] -> [batch, seq_len*D]
        return torch.cat(out_seq, dim=1).reshape(batch, -1)

    def calculate_loss(self,
                       out_flat: torch.Tensor,
                       gt_flat: torch.Tensor) -> torch.Tensor:
        B, flat = out_flat.size()
        D = self.out_frame_size
        L = flat // D

        out = out_flat.view(B, L, D)
        gt = gt_flat.view(B, L, D)

        # hip translation MSE
        hip_loss = F.mse_loss(out[:, :, :3], gt[:, :, :3])

        # quaternion angle loss
        M = (D - 3) // 4
        pred_q = out[:, :, 3:].view(B, L, M, 4)
        gt_q = gt[:, :, 3:].view(B, L, M, 4)
        pred_q = F.normalize(pred_q, p=2, dim=-1, eps=1e-6)

        dot = torch.clamp((pred_q * gt_q).sum(-1), -1.0 + 1e-6, 1.0 - 1e-6)
        angle = 2 * torch.acos(torch.abs(dot))
        quat_loss = angle.mean()

        return hip_loss + quat_loss


import torch
import torch.nn.functional as F
import numpy as np
import tempfile
from pathlib import Path
from generate_training_quad_data import generate_bvh_from_quad_traindata

def train_one_iteration(
    real_seq_np: np.ndarray,
    model: acLSTM,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    total_iterations: int,
    save_dance_folder: Path,
    print_loss: bool = False,
    save_bvh_motion: bool = True
) -> None:
    """
    One training step with scheduled sampling and a warm‐up period:
      - real_seq_np: (B, seq_len+1, D) absolute (scaled hip+quats)
      - seq_len+1 = 100+2 = 102 deltas → 101 inputs
    """
    # 1) Build Δ-hip inputs
    dif = real_seq_np[:, 1:, :3] - real_seq_np[:, :-1, :3]
    real_seq_diff = real_seq_np[:, :-1].copy()        # [B, 102, D]
    real_seq_diff[:, :, 0] = dif[:, :, 0]
    real_seq_diff[:, :, 2] = dif[:, :, 2]

    # 2) To tensors: inp=(B,101,D), gt_seq=(B,101,D)
    device = next(model.parameters()).device
    inp    = torch.from_numpy(real_seq_diff[:, :-1]).float().to(device)
    gt_seq = torch.from_numpy(real_seq_diff[:,  1: ]).float().to(device)
    B, L, D = inp.shape
    M = (D - 3) // 4

    # 3) Compute teacher-forcing ratio with warmup
    warmup_iters = 10000
    if iteration < warmup_iters:
        teacher_ratio = 1.0
    else:
        teacher_ratio = max(
            0.0,
            1.0 - (iteration - warmup_iters) / float(total_iterations - warmup_iters)
        )

    # 4) Unroll LSTM with per-step sampling; always force t=0
    h, c      = model.init_hidden(B, device)
    out_frame = torch.zeros(B, D, device=device)
    outputs   = []

    for t in range(L):
        if t == 0:
            inp_t = inp[:, 0, :]
        else:
            mask = (torch.rand(B, device=device) < teacher_ratio).float().unsqueeze(1)
            inp_t = mask * inp[:, t, :] + (1.0 - mask) * out_frame

        raw, h, c = model.forward_lstm(inp_t, h, c)
        hip   = raw[:, :3]
        quats = raw[:, 3:].view(B, M, 4)
        quats = F.normalize(quats, p=2, dim=-1, eps=1e-6)
        out_frame = torch.cat([hip, quats.view(B, M*4)], dim=1)
        outputs.append(out_frame.unsqueeze(1))

    # 5) Flatten predictions & ground truth for loss
    pred_flat = torch.cat(outputs, dim=1).reshape(B, -1)  # [B,101*D]
    gt_flat   = gt_seq.reshape(B, -1)                    # [B,101*D]

    # 6) Backprop
    optimizer.zero_grad()
    loss = model.calculate_loss(pred_flat, gt_flat)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if print_loss:
        print(f"Iter {iteration:07d}  Loss={loss.item():.6f}")

    # 7) Optional BVH snapshot
    if save_bvh_motion:
        D0 = D
        L0 = pred_flat.size(1) // D0
        pred_enc = pred_flat[0].detach().cpu().numpy().reshape(L0, D0)

        # integrate hip deltas → absolute
        pred_enc[:, 0] = np.cumsum(pred_enc[:, 0])
        pred_enc[:, 2] = np.cumsum(pred_enc[:, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp     = Path(tmpdir)
            gt_npy  = tmp / f"{iteration:07d}_gt.bvh.npy"
            out_npy = tmp / f"{iteration:07d}_out.bvh.npy"
            np.save(str(gt_npy), real_seq_np[0, 1:1+L0, :])
            np.save(str(out_npy), pred_enc)
            generate_bvh_from_quad_traindata(str(tmp), str(save_dance_folder))



def get_dance_length_list(dances: list[np.ndarray]) -> list[int]:
    """Build a list of dance indices weighted by dance duration."""
    lengths = [max(1, int(len(dance) / 100)) for dance in dances]
    indices = [i for i, l in enumerate(lengths) for _ in range(l)]
    return indices


def load_dances(dance_folder: Path) -> list[np.ndarray]:
    """Load and renormalize quaternion dances from a folder of .npy files."""
    dances = []
    print("Loading motion files...")
    for f in dance_folder.iterdir():
        if f.suffix.lower() != '.npy':
            continue
        arr = np.load(str(f))
        # renormalize quaternions
        flat_q = arr[:, 3:].reshape(-1, 4)
        norms = np.linalg.norm(flat_q, axis=1, keepdims=True)
        flat_q /= norms
        arr[:, 3:] = flat_q.reshape(arr[:, 3:].shape)
        dances.append(arr)
    print(f"{len(dances)} motion files loaded.")
    return dances


def train(dances: list[np.ndarray],
          frame_rate: float,
          batch_size: int,
          seq_len: int,
          read_weight_path: str,
          write_weight_folder: Path,
          write_bvh_motion_folder: Path,
          in_frame: int,
          out_frame: int,
          hidden_size: int,
          total_iter: int) -> None:
    """Full training loop."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = acLSTM(in_frame, hidden_size, out_frame).to(device)
    if read_weight_path:
        model.load_state_dict(torch.load(read_weight_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()

    dance_indices = get_dance_length_list(dances)
    speed = frame_rate / 30.0
    seq_len += 2

    for iteration in range(total_iter):
        batch = []
        for _ in range(batch_size):
            idx = random.choice(dance_indices)
            dance = dances[idx]
            F = len(dance)
            start = random.randint(10, max(10, int(F - seq_len*speed - 10)))
            seq = [dance[int(i*speed + start)] for i in range(seq_len + 1)]
            batch.append(seq)
        batch_np = np.array(batch)

        print_loss = (iteration % 20 == 0)
        save_bvh = (iteration % 1000 == 0)
        if save_bvh:
            ckpt = write_weight_folder / f"{iteration:07d}.weight"
            torch.save(model.state_dict(), str(ckpt))

        train_one_iteration(batch_np, model, optimizer,
                    iteration, total_iter,
                    write_bvh_motion_folder,
                    print_loss, save_bvh)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dances_folder', type=Path, required=True)
    parser.add_argument('--write_weight_folder', type=Path, required=True)
    parser.add_argument('--write_bvh_motion_folder', type=Path, required=True)
    parser.add_argument('--read_weight_path', type=str, default="")
    parser.add_argument('--dance_frame_rate', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_frame', type=int, required=True)
    parser.add_argument('--out_frame', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument('--total_iterations', type=int, default=100000)
    args = parser.parse_args()

    args.write_weight_folder.mkdir(parents=True, exist_ok=True)
    args.write_bvh_motion_folder.mkdir(parents=True, exist_ok=True)

    dances = load_dances(args.dances_folder)
    train(
        dances,
        args.dance_frame_rate,
        args.batch_size,
        args.seq_len,
        args.read_weight_path,
        args.write_weight_folder,
        args.write_bvh_motion_folder,
        args.in_frame,
        args.out_frame,
        args.hidden_size,
        total_iter=args.total_iterations,
    )


if __name__ == '__main__':
    main()
