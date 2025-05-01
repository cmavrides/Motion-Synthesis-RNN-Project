# filepath: c:\Users\alexa\OneDrive - University of Cyprus\Masters in Artificial Intelligence\2nd semester\ML For Graphics - 645\FINAL PROJECT\MAI645_Team_04\code\synthesize_quat_motion.py
import os
import torch
import torch.nn as nn
import numpy as np
import random
import read_bvh

# Define constants
Hip_index = read_bvh.joint_index['hip']
Seq_len = 100
Hidden_size = 1024
Condition_num = 5
Groundtruth_num = 5

Joints_num = 57  # Keep this as is
In_frame_size = 175  # Match the size from trained weights
Out_frame_size = 175  # Match the size from trained weights



class acLSTMQuat(nn.Module):
    def __init__(self, in_frame_size=175, hidden_size=1024, out_frame_size=175):
        super(acLSTMQuat, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        # LSTM layers
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch):
        c0 = torch.zeros(batch, self.hidden_size).cuda()
        c1 = torch.zeros(batch, self.hidden_size).cuda()
        c2 = torch.zeros(batch, self.hidden_size).cuda()
        h0 = torch.zeros(batch, self.hidden_size).cuda()
        h1 = torch.zeros(batch, self.hidden_size).cuda()
        h2 = torch.zeros(batch, self.hidden_size).cuda()
        return ([h0, h1, h2], [c0, c1, c2])

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]

        return (out_frame, vec_h_new, vec_c_new)

    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size()[0]
        (vec_h, vec_c) = self.init_hidden(batch)

        out_seq = torch.zeros(batch, 1).cuda()
        out_frame = torch.zeros(batch, self.out_frame_size).cuda()

        for i in range(initial_seq.size()[1]):
            in_frame = initial_seq[:, i]
            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        for i in range(generate_frames_number):
            in_frame = out_frame
            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        return out_seq[:, 1: out_seq.size()[1]]

    def calculate_loss(self, out_seq, groundtruth_seq):
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss

def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
    initial_seq = torch.FloatTensor(initial_seq_np).cuda()
    predict_seq = model.forward(initial_seq, generate_frames_number)

    batch = initial_seq_np.shape[0]
    for b in range(batch):
        out_seq = np.array(predict_seq[b].data.tolist()).reshape(-1, In_frame_size)
        # Here you would need to convert quaternion representation back to the desired format
        # For example, you might want to convert quaternions to Euler angles or another format
        # This part is left as a placeholder for your specific needs

        # Save the output sequence to a file
        read_bvh.write_traindata_to_bvh(save_dance_folder + "out" + "%02d" % b + ".bvh", out_seq)

    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)

def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []
    for dance_file in dance_files:
        print("load " + dance_file)
        dance = np.load(dance_folder + dance_file)
        print("frame number: " + str(dance.shape[0]))
        dances.append(dance)
    return dances

def test(dance_batch_np, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
         write_bvh_motion_folder, in_frame_size=175, hidden_size=1024, out_frame_size=175):
    torch.cuda.set_device(0)

    # Initialize and load pretrained weights
    model = acLSTMQuat(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    model.cuda()

    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)

# Define paths and parameters
read_weight_path = "C:/Users/alexa/Desktop/MAI645_Team_04/results_quad/0060000.weight"
write_bvh_motion_folder = "C:/Users/alexa/Desktop/result_quat/"
dances_folder = "C:/Users/alexa/Desktop/MAI645_Team_04/train_data_quad/martial/"
dance_frame_rate = 60
batch = 5
initial_seq_len = 15
generate_frames_number = 400

if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

dances = load_dances(dances_folder)

# Replace the batch loading code with this:
def prepare_dance_batch(dances, batch_size, initial_seq_len):
    batch_data = []
    for i in range(batch_size):
        if i >= len(dances):
            break
        dance = dances[i]
        # Take only the initial sequence length we need
        if len(dance) > initial_seq_len:
            # Randomly select a starting point
            start_idx = random.randint(0, len(dance) - initial_seq_len)
            sequence = dance[start_idx:start_idx + initial_seq_len]
        else:
            # If sequence is too short, pad with zeros
            sequence = np.pad(dance, ((0, initial_seq_len - len(dance)), (0, 0)), 'constant')
        batch_data.append(sequence)

    return np.array(batch_data)

# Replace the batch loading line with:
dance_batch_np = prepare_dance_batch(dances, batch, initial_seq_len)
test(dance_batch_np, dance_frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
     write_bvh_motion_folder, In_frame_size, Hidden_size, Out_frame_size)
