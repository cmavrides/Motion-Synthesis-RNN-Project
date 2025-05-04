import os
import torch
import torch.nn as nn
import numpy as np
import random
import read_bvh
import argparse

Hip_index = read_bvh.joint_index['hip']
Seq_len = 100
Hidden_size = 1024
Joints_num = 57
Condition_num = 5
Groundtruth_num = 5
In_frame_size = Joints_num * 3

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
        super(acLSTM, self).__init__()
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, out_frame_size)

    def init_hidden(self, batch):
        zeros = lambda: torch.zeros(batch, self.hidden_size).cuda()
        return [zeros() for _ in range(3)], [zeros() for _ in range(3)]

    def forward_lstm(self, in_frame, vec_h, vec_c):
        h0, c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        h1, c1 = self.lstm2(h0, (vec_h[1], vec_c[1]))
        h2, c2 = self.lstm3(h1, (vec_h[2], vec_c[2]))
        out = self.decoder(h2)
        return out, [h0, h1, h2], [c0, c1, c2]

    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        gt = np.ones((100, groundtruth_num))
        cond = np.zeros((100, condition_num))
        mask = np.concatenate((gt, cond), axis=1).reshape(-1)
        return mask[:seq_len]

    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        batch = real_seq.size(0)
        seq_len = real_seq.size(1)
        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        vec_h, vec_c = self.init_hidden(batch)
        out_seq = torch.zeros(batch, 0).cuda()
        out_frame = torch.zeros(batch, self.out_frame_size).cuda()

        for i in range(seq_len):
            in_frame = real_seq[:, i] if condition_lst[i] == 1 else out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), dim=1)

        return out_seq

    def calculate_loss(self, out_seq, groundtruth_seq):
        return nn.MSELoss()(out_seq, groundtruth_seq)

def load_dances(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    print(f"Loading {len(files)} quad motion files...")
    return [np.load(os.path.join(folder, f)) for f in files]

def get_dance_len_lst(dances):
    return [i for i, dance in enumerate(dances) for _ in range(max(1, 10))]

def train_one_iteration(batch_np, model, optimizer, iteration, save_folder, print_loss=False, save_bvh=False):
    real_seq = torch.tensor(batch_np, dtype=torch.float32).cuda()
    in_seq = real_seq[:, :-1]
    target_seq = real_seq[:, 1:].contiguous().view(real_seq.size(0), -1)

    output_seq = model(in_seq, Condition_num, Groundtruth_num)
    loss = model.calculate_loss(output_seq, target_seq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if print_loss:
        print(f"### Iteration {iteration:07d} ###\nLoss: {loss.item():.6f}")

    if save_bvh:
        out_np = output_seq[0].detach().cpu().numpy().reshape(-1, In_frame_size)
        gt_np = target_seq[0].detach().cpu().numpy().reshape(-1, In_frame_size)
        read_bvh.write_traindata_to_bvh(os.path.join(save_folder, f"{iteration:07d}_gt.bvh"), gt_np)
        read_bvh.write_traindata_to_bvh(os.path.join(save_folder, f"{iteration:07d}_out.bvh"), out_np)

def train(dances, args):
    model = acLSTM(in_frame_size=args.in_frame, hidden_size=args.hidden_size, out_frame_size=args.out_frame).cuda()

    if args.read_weight_path:
        model.load_state_dict(torch.load(args.read_weight_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    dance_len_lst = get_dance_len_lst(dances)
    speed = args.dance_frame_rate / 30

    for iteration in range(args.total_iterations):
        batch = []
        for _ in range(args.batch_size):
            dance_id = random.choice(dance_len_lst)
            dance = dances[dance_id]
            start = random.randint(10, len(dance) - (args.seq_len + 2) * int(speed) - 10)
            sample = [dance[int(start + i * speed)] for i in range(args.seq_len + 2)]

            T = [0.1 * (random.random() - 0.5), 0.0, 0.1 * (random.random() - 0.5)]
            R = [0, 1, 0, (random.random() - 0.5) * np.pi * 2]
            aug = read_bvh.augment_train_data(sample, T, R)
            batch.append(aug)

        batch_np = np.array(batch)

        print_loss = (iteration % 20 == 0)
        save_bvh = (iteration % 1000 == 0)

        if save_bvh:
            torch.save(model.state_dict(), os.path.join(args.write_weight_folder, f"{iteration:07d}.weight"))

        train_one_iteration(batch_np, model, optimizer, iteration, args.write_bvh_motion_folder, print_loss, save_bvh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dances_folder', type=str, required=True)
    parser.add_argument('--write_weight_folder', type=str, required=True)
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True)
    parser.add_argument('--read_weight_path', type=str, default="")
    parser.add_argument('--dance_frame_rate', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_frame', type=int, default=In_frame_size)
    parser.add_argument('--out_frame', type=int, default=In_frame_size)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--total_iterations', type=int, default=100000)

    args = parser.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(args.dances_folder)
    train(dances, args)

if __name__ == '__main__':
    main()
