import os
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

import read_bvh

# =============================================================================
#  HARD‑CODED CONFIGURATION ----------------------------------------------------
# =============================================================================

MODEL_WEIGHTS   = r"C:/Users/alexa/Desktop/MAI645_Team_04/results_quad_weights_v7/0024000.weight"
DANCES_FOLDER   = r"C:/Users/alexa/Desktop/MAI645_Team_04/train_data_quad/martial/"
OUTPUT_FOLDER   = r"C:/Users/alexa/Desktop/result_quat_v7/"
FRAME_RATE      = 60      # source data fps
BATCH_SIZE      = 5
WARMUP_LEN      = 15      # initial ground‑truth frames
GENERATE_FRAMES = 400     # frames to generate after warm‑up

# =============================================================================
#  CONSTANTS ------------------------------------------------------------------
# =============================================================================

HIP_INDEX   = read_bvh.joint_index["hip"]   # index of hip joint
FRAME_SIZE  = 227                            # 3 root xyz + 56 × 4‑dim quats
HIDDEN_SIZE = 1024

# =============================================================================
#  Quaternion helpers ---------------------------------------------------------
# =============================================================================

def normalize_quaternions_numpy(arr: np.ndarray, root_dim: int = 3) -> np.ndarray:
    """Normalise every quaternion in *arr* (x, y, z, w ordering)."""
    out = arr.copy()
    quat_flat = out[..., root_dim:]
    joint_cnt = quat_flat.shape[-1] // 4
    quats = quat_flat.reshape(-1, joint_cnt, 4)
    norms = np.linalg.norm(quats, axis=2, keepdims=True) + 1e-8
    quats /= norms
    out[..., root_dim:] = quats.reshape(quat_flat.shape)
    return out

# =============================================================================
#  BVH ⇆ Quaternion conversion ------------------------------------------------
# =============================================================================

def generate_bvh_from_quad_traindata(npy_path: str, bvh_path: str):
    """Decode quaternion motion (227‑D) back to BVH using the reference skeleton."""
    quat_data = np.load(npy_path)
    euler_frames = []
    for frame in quat_data:
        root = frame[:3]
        quats = frame[3:].reshape(-1, 4)
        quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
        eulers = R.from_quat(quats).as_euler("zxy", degrees=True)
        euler_frames.append(np.concatenate([root, eulers.flatten()]))
    read_bvh.write_traindata_to_bvh(bvh_path, np.asarray(euler_frames))

# =============================================================================
#  acLSTM model ---------------------------------------------------------------
# =============================================================================

class acLSTM(nn.Module):
    """Three‑layer LSTM decoder, matching the training architecture."""

    def __init__(self, in_frame_size: int = FRAME_SIZE, hidden_size: int = HIDDEN_SIZE,
                 out_frame_size: int = FRAME_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, out_frame_size)

    # ------------------------------------------------------------------
    def init_hidden(self, batch: int, device: torch.device):
        z = torch.zeros(batch, self.hidden_size, device=device)
        return ([z.clone(), z.clone(), z.clone()], [z.clone(), z.clone(), z.clone()])

    # ------------------------------------------------------------------
    def forward_lstm(self, x, h, c):
        h0, c0 = self.lstm1(x, (h[0], c[0]))
        h1, c1 = self.lstm2(h0, (h[1], c[1]))
        h2, c2 = self.lstm3(h1, (h[2], c[2]))
        return self.decoder(h2), [h0, h1, h2], [c0, c1, c2]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, warmup: torch.Tensor, gen_frames: int):
        """Autoregressively continue *warmup* for *gen_frames* steps."""
        device = warmup.device
        batch = warmup.shape[0]
        h, c = self.init_hidden(batch, device)
        outputs = []
        cur = None
        for t in range(warmup.shape[1]):
            cur, h, c = self.forward_lstm(warmup[:, t], h, c)
            outputs.append(cur)
        for _ in range(gen_frames):
            cur, h, c = self.forward_lstm(cur, h, c)
            outputs.append(cur)
        return torch.stack(outputs, dim=1)   # (B, W-1+G, 227)

# =============================================================================
#  Data helpers ---------------------------------------------------------------
# =============================================================================

def load_dances(folder: str):
    dances = [np.load(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".npy")]
    if not dances:
        raise RuntimeError("No .npy files in " + folder)
    print(f"Loaded {len(dances)} motion files from {folder}")
    return dances

def sampling_pool(dances):
    pool = []
    for idx, d in enumerate(dances):
        pool.extend([idx] * max(int(len(d) / 100), 1))
    return pool

# =============================================================================
#  Generation -----------------------------------------------------------------
# =============================================================================

def generate_batch(model: acLSTM, dances: list[np.ndarray], batch_size: int,
                   warmup_len: int, gen_frames: int, frame_rate: int,
                   save_dir: str, device: torch.device):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pool  = sampling_pool(dances)
    speed = frame_rate / 30.0  # network operates at 30 fps

    # ------------------------------------------------------------------
    #  Build warm‑up batch (absolute coords)
    # ------------------------------------------------------------------
    warmups_abs = []
    for _ in range(batch_size):
        dance = dances[random.choice(pool)]
        start = random.randint(10, int(len(dance) - warmup_len * speed - 10))
        warmups_abs.append([dance[int(start + i * speed)] for i in range(warmup_len)])
    warmups_abs = np.asarray(warmups_abs)                      # (B, W, 227)

    # ------------------------------------------------------------------
    #  Derivative coding on root X/Z only (mirror of training)
    # ------------------------------------------------------------------
    dxz = warmups_abs[:, 1:, :3] - warmups_abs[:, :-1, :3]
    warmups_dif = warmups_abs[:, :-1].copy()
    warmups_dif[:, :, 0] = dxz[:, :, 0]
    warmups_dif[:, :, 2] = dxz[:, :, 2]

    warmups_t = torch.tensor(warmups_dif, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    model.eval()
    pred = model.sample(warmups_t, gen_frames)                 # (B, W-1+G, 227)
    pred_np = pred.cpu().numpy()

    # ------------------------------------------------------------------
    #  Re‑integrate root translation
    # ------------------------------------------------------------------
    for b in range(batch_size):
        seq = pred_np[b]
        # Start accumulation from *first* warm‑up frame so there is no jump
        last_x = warmups_abs[b, 0, 0]
        last_z = warmups_abs[b, 0, 2]
        for f in range(seq.shape[0]):
            seq[f, 0] += last_x; last_x = seq[f, 0]
            seq[f, 2] += last_z; last_z = seq[f, 2]
        seq = normalize_quaternions_numpy(seq)

        tmp_npy = os.path.join(save_dir, f"sample{b:02d}.npy")
        out_bvh = os.path.join(save_dir, f"sample{b:02d}.bvh")
        np.save(tmp_npy, seq)
        generate_bvh_from_quad_traindata(tmp_npy, out_bvh)
        os.remove(tmp_npy)
    print(f"Saved {batch_size} BVH files to {save_dir}")

# =============================================================================
#  Main -----------------------------------------------------------------------
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dances = load_dances(DANCES_FOLDER)
    model = acLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    generate_batch(model, dances, BATCH_SIZE, WARMUP_LEN, GENERATE_FRAMES,
                   FRAME_RATE, OUTPUT_FOLDER, device)

if __name__ == "__main__":
    main()
