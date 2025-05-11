import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import transforms3d.euler as euler

import read_bvh

# Constants
HIP_IDX = read_bvh.joint_index.get('hip', 0)
TRANSLATION_SCALE = 0.01
ANGLE_SCALE = 180.0 / math.pi
DEFAULT_HIDDEN = 1024

class MotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size: int = DEFAULT_HIDDEN):
        super(MotionLSTM, self).__init__()
        self.cell1 = nn.LSTMCell(input_size, hidden_size)
        self.cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.cell3 = nn.LSTMCell(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, input_size)

    @staticmethod
    def init_states(batch_size: int, device):
        h_states = [torch.zeros(batch_size, DEFAULT_HIDDEN, device=device) for _ in range(3)]
        c_states = [torch.zeros(batch_size, DEFAULT_HIDDEN, device=device) for _ in range(3)]
        return h_states, c_states

    def step(self, inp, states):
        h_list, c_list = states
        h_list[0], c_list[0] = self.cell1(inp, (h_list[0], c_list[0]))
        h_list[1], c_list[1] = self.cell2(h_list[0], (h_list[1], c_list[1]))
        h_list[2], c_list[2] = self.cell3(h_list[1], (h_list[2], c_list[2]))
        return self.output_layer(h_list[2]), (h_list, c_list)

    def forward(self, sequence, warmup: int = 5, predict: int = 5):
        batch, seq_len, feat_dim = sequence.size()
        device = sequence.device
        # Build boolean mask: warmup True then predict False repeating
        mask = ([True] * warmup + [False] * predict) * ((seq_len // (warmup + predict)) + 1)
        mask = torch.tensor(mask[:seq_len], dtype=torch.bool, device=device)

        h_states, c_states = MotionLSTM.init_states(batch, device)
        previous_output = torch.zeros(batch, feat_dim, device=device)
        outputs = []

        idx = 0
        while idx < seq_len:
            current_input = sequence[:, idx] if mask[idx] else previous_output
            previous_output, (h_states, c_states) = self.step(current_input, (h_states, c_states))
            outputs.append(previous_output)
            idx += 1

        return torch.stack(outputs, dim=1)


def collect_motion_arrays(source_dir):
    arrays = []
    filenames = [f for f in os.listdir(source_dir) if f.lower().endswith('.npy')]
    idx = 0
    while idx < len(filenames):
        filepath = os.path.join(source_dir, filenames[idx])
        arr = np.load(filepath).astype(np.float32)
        row = arr.shape[0] - 1
        while row > 0:
            arr[row, 0] -= arr[row-1, 0]
            arr[row, 2] -= arr[row-1, 2]
            row -= 1
        arrays.append(arr)
        idx += 1
    return arrays


def export_bvh_euler_sequence(frames: np.ndarray, output_path: str):
    data = frames.copy()
    t = 0
    last_x, last_z = 0.0, 0.0
    total = data.shape[0]
    while t < total:
        data[t, 0] += last_x
        last_x = data[t, 0]
        data[t, 2] += last_z
        last_z = data[t, 2]
        t += 1

    data[:, :3] /= TRANSLATION_SCALE
    data[:, 3:] *= ANGLE_SCALE
    data = np.round(data, 6)
    template = getattr(read_bvh, 'standard_bvh_file', 'train_data_bvh/standard.bvh')
    read_bvh.write_frames(template, output_path, data)


def synthesize_sequence(seed_seq: np.ndarray,
                        num_generate: int,
                        net: MotionLSTM,
                        out_dir: str,
                        dev):
    batch_size, seed_len, feat_dim = seed_seq.shape
    tensor_seq = torch.tensor(seed_seq, device=dev)
    h_states, c_states = MotionLSTM.init_states(batch_size, dev)

    # Warm-up phase
    index = 0
    last_out = None
    while index < seed_len:
        last_out, (h_states, c_states) = net.step(tensor_seq[:, index], (h_states, c_states))
        index += 1

    # Generation phase
    generated = []
    count = 0
    while count < num_generate:
        last_out, (h_states, c_states) = net.step(last_out, (h_states, c_states))
        generated.append(last_out.detach().cpu().numpy())
        count += 1

    gen_arr = np.stack(generated, axis=1)
    full_seq = np.concatenate([seed_seq, gen_arr], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while i < batch_size:
        filename = f"euler_synth_out_{i:02d}.bvh"
        export_bvh_euler_sequence(full_seq[i], os.path.join(out_dir, filename))
        i += 1

    return full_seq


def run_generation_pipeline(motion_list,
                            rate: int,
                            batch_count: int,
                            seed_length: int,
                            gen_length: int,
                            weight_file: str,
                            output_folder: str,
                            dev):
    feature_dim = motion_list[0].shape[1]
    model = MotionLSTM(feature_dim).to(dev)
    raw_state = torch.load(weight_file, map_location=dev)
    mapped_state = {}
    for key, val in raw_state.items():
        if key.startswith('lstm1.'):
            mapped_state['cell1.' + key[len('lstm1.'):]] = val
        elif key.startswith('lstm2.'):
            mapped_state['cell2.' + key[len('lstm2.'):]] = val
        elif key.startswith('lstm3.'):
            mapped_state['cell3.' + key[len('lstm3.'):]] = val
        elif key.startswith('dec.') or key.startswith('decoder.'):
            mapped_state['output_layer.' + key.split('.', 1)[1]] = val
        else:
            mapped_state[key] = val
    model.load_state_dict(mapped_state)
    model.eval()

    # Prepare seeds
    seeds = []
    counter = 0
    while counter < batch_count:
        clip = random.choice(motion_list)
        if clip.shape[0] < seed_length:
            raise RuntimeError("Seed clip shorter than required length")
        start_idx = random.randint(0, clip.shape[0] - seed_length)
        seeds.append(clip[start_idx:start_idx + seed_length])
        counter += 1

    seeds_np = np.stack(seeds, axis=0)

    # Generate and save
    return synthesize_sequence(seeds_np,
                               gen_length,
                               model,
                               output_folder,
                               dev)

if __name__ == "__main__":
    weights_path = "./output_weights_euler_martial/0011200final.weight"
    weave_folder = "./out_euler_new/"
    source_folder = "../train_data_euler/martial/"
    os.makedirs(weave_folder, exist_ok=True)

    dances_data = collect_motion_arrays(source_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_sequences = run_generation_pipeline(
        dances_data,
        rate=60,
        batch_count=5,
        seed_length=15,
        gen_length=400,
        weight_file=weights_path,
        output_folder=weave_folder,
        dev=device
    )
    print(f"Generated synthesized .bvh files")
