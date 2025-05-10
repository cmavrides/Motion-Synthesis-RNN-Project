# euler_synthesize.py
import os
import numpy as np
import torch
import read_bvh

#— import your normalization constants & helpers from your generation script —
from generate_training_euler_data import (
    WEIGHT_TRANSLATION,
    ANGLE_SCALE,
    STANDARD_BVH_FILE,
    EXPECTED_PARAMS,
    _pad_or_truncate
)

# import your trained model definition
from pytorch_train_euler_aclstm import acLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Hip = read_bvh.joint_index['hip']


def generate_euler_seq(initial_seq_np, generate_frames, model, save_folder):
    """
    initial_seq_np: np.array, shape (B, L0, F)  — already normalized (trans*W, angles/A)
    generate_frames: how many new frames to synthesize
    model: acLSTM loaded with your weights
    save_folder: where to write the .bvh outputs
    """
    B, L0, F = initial_seq_np.shape
    L = L0 + generate_frames

    # — build the “teacher-forcing” input tensor —
    # we need a full length L, but only first L0 steps are real:
    full_np = np.zeros((B, L, F), dtype=np.float32)
    # apply hip-diff on the initial L0 frames
    dif = initial_seq_np[:, 1:] - initial_seq_np[:, :-1]
    init_tf = initial_seq_np[:, :-1].copy()
    init_tf[:, :, Hip*3    ] = dif[:, :, Hip*3    ]
    init_tf[:, :, Hip*3 + 2] = dif[:, :, Hip*3 + 2]
    full_np[:, :L0-1, :] = init_tf

    inp = torch.from_numpy(full_np).to(device)
    # forward: feed L0 ground-truth steps, then let it predict the rest
    pred_flat = model.forward(
        inp,
        condition_num=generate_frames,
        groundtruth_num=L0
    )  # → (B, L*F) flattened
    pred = pred_flat.view(B, L, F).detach().cpu().numpy()

    #— stitch GT + generated
    result = np.concatenate([initial_seq_np, pred[:, L0:, :]], axis=1)  # (B, L, F)

    #— undo hip-diff and decode back to raw BVH channels —
    def unroll_hip(arr):
        last_x = last_z = 0.0
        out = arr.copy()
        for t in range(out.shape[0]):
            out[t, Hip*3    ] += last_x
            out[t, Hip*3 + 2] += last_z
            last_x = out[t, Hip*3    ]
            last_z = out[t, Hip*3 + 2]
        return out

    os.makedirs(save_folder, exist_ok=True)
    for b in range(B):
        seq_norm = unroll_hip(result[b])

        trans_norm  = seq_norm[:, :3]
        angles_norm = seq_norm[:, 3:]
        trans  = trans_norm  / WEIGHT_TRANSLATION
        angles = angles_norm * ANGLE_SCALE
        raw    = np.concatenate([trans, angles], axis=1)
        raw_fixed = _pad_or_truncate(raw, EXPECTED_PARAMS)

        out_path = os.path.join(save_folder, f"out{b:02d}_euler.bvh")
        read_bvh.write_frames(STANDARD_BVH_FILE, out_path, raw_fixed)
        print(f"Wrote {out_path}  (frames={raw_fixed.shape[0]})")

    return result



if __name__ == "__main__":
    # 1) load your model & weights
    model = acLSTM(in_frame_size=171, hidden_size=1024, out_frame_size=171)
    ckpt = torch.load("/Users/tooulas/Downloads/0000000.weight", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # 2) prepare a small batch of initial sequences:
    #    here we load from your numpy‐encoded Euler data
    initial_folder = "../train_data_euler/martial/"
    files = sorted(f for f in os.listdir(initial_folder) if f.endswith(".npy"))
    batch_np = np.stack([np.load(initial_folder+f)[:15] for f in files[:5]])
    #    ⇒ (B=5, L0=15, F=171)

    # 3) run synthesis
    out_dir = "./euler_out_bvh/"
    generate_euler_seq(batch_np, generate_frames=400, model=model, save_folder=out_dir)
