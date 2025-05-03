import os
import torch
import torch.nn as nn
import numpy as np
import random
import read_bvh

# index of the hip joint in your BVH data
Hip_index = read_bvh.joint_index['hip']

# model hyperparameters
Seq_len        = 100
Hidden_size    = 1024
Joints_num     = 57
Extra_dims     = 4

# adjust frame size to match your checkpoint (57 joints * 3 + 4 extra dims = 175)
In_frame_size  = Joints_num * 3 + Extra_dims   # 175
Out_frame_size = In_frame_size                 # 175

Condition_num   = 5
Groundtruth_num = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
        super(acLSTM, self).__init__()
        self.in_frame_size  = in_frame_size
        self.hidden_size    = hidden_size
        self.out_frame_size = out_frame_size

        # three-layer LSTM
        self.lstm1   = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2   = nn.LSTMCell(self.hidden_size,   self.hidden_size)
        self.lstm3   = nn.LSTMCell(self.hidden_size,   self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size,     self.out_frame_size)

    def init_hidden(self, batch):
        # initialize h and c to zero
        h0 = torch.zeros(batch, self.hidden_size, device=device)
        h1 = torch.zeros(batch, self.hidden_size, device=device)
        h2 = torch.zeros(batch, self.hidden_size, device=device)
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        c1 = torch.zeros(batch, self.hidden_size, device=device)
        c2 = torch.zeros(batch, self.hidden_size, device=device)
        return ([h0, h1, h2], [c0, c1, c2])

    def forward_lstm(self, in_frame, vec_h, vec_c):
        h0, c0 = self.lstm1(in_frame,         (vec_h[0], vec_c[0]))
        h1, c1 = self.lstm2(h0,               (vec_h[1], vec_c[1]))
        h2, c2 = self.lstm3(h1,               (vec_h[2], vec_c[2]))
        out_frame = self.decoder(h2)
        return out_frame, [h0, h1, h2], [c0, c1, c2]

    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size(0)
        vec_h, vec_c = self.init_hidden(batch)

        # prepare output buffer (we'll drop the very first zero-frame later)
        out_seq   = torch.zeros(batch, 1, self.out_frame_size, device=device)
        out_frame = torch.zeros(batch,     self.out_frame_size, device=device)

        # feed in the seed sequence
        for i in range(initial_seq.size(1)):
            in_frame     = initial_seq[:, i, :]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq      = torch.cat([out_seq, out_frame.unsqueeze(1)], dim=1)

        # generate new frames autoregressively
        for _ in range(generate_frames_number):
            in_frame     = out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq      = torch.cat([out_seq, out_frame.unsqueeze(1)], dim=1)

        # drop the initial zero frame
        return out_seq[:, 1:, :]

    def calculate_loss(self, out_seq, groundtruth_seq):
        return nn.MSELoss()(out_seq, groundtruth_seq)


def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
    """
    initial_seq_np: numpy array of shape [batch, seq_len, frame_size]
    """
    batch, seq_len, _ = initial_seq_np.shape

    # compute frame-to-frame diff
    dif     = initial_seq_np[:, 1:] - initial_seq_np[:, :-1]
    # copy all but last frame
    seq_dif = initial_seq_np[:, :-1].copy()

    # only use diffs for hip X and Z; zero out hip Y diff
    seq_dif[:, :, Hip_index*3    ] = dif[:, :, Hip_index*3]
    seq_dif[:, :, Hip_index*3 + 1] = 0.0
    seq_dif[:, :, Hip_index*3 + 2] = dif[:, :, Hip_index*3 + 2]

    # send into the network
    initial_seq  = torch.FloatTensor(seq_dif).to(device)
    predict_seq  = model(initial_seq, generate_frames_number)

    for b in range(batch):
        out_seq   = predict_seq[b].cpu().detach().numpy().reshape(-1, In_frame_size)
        init_hip_y = initial_seq_np[b, -1, Hip_index*3 + 1]

        last_x = 0.0
        last_z = 0.0
        for f in range(out_seq.shape[0]):
            # accumulate X & Z
            out_seq[f, Hip_index*3    ] += last_x
            last_x = out_seq[f, Hip_index*3    ]
            out_seq[f, Hip_index*3 + 2] += last_z
            last_z = out_seq[f, Hip_index*3 + 2]

            # restore Y to the last seed height
            out_seq[f, Hip_index*3 + 1] = init_hip_y

        fn = os.path.join(save_dance_folder, f"out{b:02d}.bvh")
        read_bvh.write_traindata_to_bvh(fn, out_seq)

    return predict_seq.cpu().detach().numpy().reshape(batch, -1, In_frame_size)


def get_dance_len_lst(dances):
    len_lst = []
    for d in dances:
        length = max(int(len(d) / 100), 1)
        len_lst.append(length)
    index_lst = []
    for idx, length in enumerate(len_lst):
        index_lst += [idx] * length
    return index_lst


def load_dances(dance_folder):
    files  = os.listdir(dance_folder)
    dances = []
    for fn in files:
        print(f"load {fn}")
        arr = np.load(os.path.join(dance_folder, fn))
        print(f"frame number: {arr.shape[0]}")
        dances.append(arr)
    return dances


def test(dances, frame_rate, batch, initial_seq_len, generate_frames_number,
         read_weight_path, write_bvh_motion_folder,
         in_frame_size=In_frame_size, hidden_size=Hidden_size, out_frame_size=Out_frame_size):

    # initialize model & load pretrained weights
    model = acLSTM(in_frame_size, hidden_size, out_frame_size).to(device)
    model.load_state_dict(torch.load(read_weight_path, map_location=device))

    speed         = frame_rate / 30.0
    dance_len_lst = get_dance_len_lst(dances)
    random_range  = len(dance_len_lst)

    dance_batch = []
    for _ in range(batch):
        dance_id = dance_len_lst[random.randint(0, random_range - 1)]
        dance    = dances[dance_id]
        L        = dance.shape[0]
        start_id = random.randint(10, int(L - initial_seq_len*speed - 10))
        seq      = [dance[int(i*speed + start_id)] for i in range(initial_seq_len)]
        dance_batch.append(seq)

    dance_batch_np = np.array(dance_batch)
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)


if __name__ == "__main__":
    read_weight_path        = "../results_quad/0060000.weight"
    write_bvh_motion_folder = "./quad_out_bvh/"
    dances_folder           = "../train_data_quad/martial/"
    dance_frame_rate        = 60
    batch                   = 5
    initial_seq_len         = 15
    generate_frames_number  = 400

    os.makedirs(write_bvh_motion_folder, exist_ok=True)
    dances = load_dances(dances_folder)

    test(
        dances,
        dance_frame_rate,
        batch,
        initial_seq_len,
        generate_frames_number,
        read_weight_path,
        write_bvh_motion_folder
    )
