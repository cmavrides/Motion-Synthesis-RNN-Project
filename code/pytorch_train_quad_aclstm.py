import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse

Hip_index = read_bvh.joint_index['hip']

Seq_len = 100
Hidden_size = 1024
Joints_num = 56  # 56 joints for quaternion data
Condition_num = 5
Groundtruth_num = 5
In_frame_size = 227  # 3 root + 56*4 quaternion values


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=227, hidden_size=1024, out_frame_size=227):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)


    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])

    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))

        out_frame = self.decoder(vec_h[2]) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]


        return (out_frame,  vec_h_new, vec_c_new)

    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]


    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):

        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]

        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)

        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())


        for i in range(seq_len):

            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame

            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)

            out_seq = torch.cat((out_seq, out_frame),1)

        return out_seq[:, 1: out_seq.size()[1]]

    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size)
    def calculate_loss(self, out_seq, groundtruth_seq):
        B, TF = out_seq.shape
        F = self.out_frame_size
        T = TF // F
        root_dim = 3
        quat_dim = 4
        J = (F - root_dim) // quat_dim

        # Root translation loss (MSE)
        pred_root = out_seq.view(B, T, F)[:, :, :root_dim]
        true_root = groundtruth_seq.view(B, T, F)[:, :, :root_dim]
        root_loss = torch.nn.functional.mse_loss(pred_root, true_root)

        # Quaternion loss (geodesic)
        pred_q = out_seq.view(B, T, F)[:, :, root_dim:].contiguous().view(B, T, J, quat_dim)
        true_q = groundtruth_seq.view(B, T, F)[:, :, root_dim:].contiguous().view(B, T, J, quat_dim)

        # Normalize quaternions
        pred_q = pred_q / (pred_q.norm(dim=-1, keepdim=True) + 1e-8)
        true_q = true_q / (true_q.norm(dim=-1, keepdim=True) + 1e-8)

        # Geodesic loss (angle between quaternions)
        dot = (pred_q * true_q).sum(dim=-1).abs().clamp(1e-6, 1.0 - 1e-6)
        quat_loss = 2.0 * torch.acos(dot)
        quat_loss = quat_loss.mean()

        # Add small regularization for quaternion norm
        norm_loss = ((pred_q.norm(dim=-1) - 1.0) ** 2).mean() * 0.1

        return root_loss + quat_loss + norm_loss


#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):
    real_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_np).cuda())
    seq_len = real_seq.size()[1]-1
    in_real_seq = real_seq[:, 0:seq_len]
    predict_groundtruth_seq = real_seq[:, 1:seq_len+1].contiguous().view(real_seq_np.shape[0], -1)
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)

    # Normalize quaternion part of output
    with torch.no_grad():
        B, TF = predict_seq.shape
        F = In_frame_size
        T = TF // F
        root_dim = 3
        quat_dim = 4
        J = (F - root_dim) // quat_dim
        pred_seq_reshaped = predict_seq.view(B, T, F)
        quats = pred_seq_reshaped[:, :, root_dim:].contiguous().view(B, T, J, quat_dim)
        quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
        pred_seq_reshaped[:, :, root_dim:] = quats.view(B, T, J * quat_dim)
        predict_seq = pred_seq_reshaped.view(B, T * F)

    optimizer.zero_grad()
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq)
    loss.backward()
    optimizer.step()

    if print_loss:
        print("########### iter %07d ######################" % iteration)
        print("loss: " + str(loss.detach().cpu().numpy()))

    if save_bvh_motion:
        gt_seq = predict_groundtruth_seq[0].detach().cpu().numpy().reshape(-1, In_frame_size)
        out_seq = predict_seq[0].detach().cpu().numpy().reshape(-1, In_frame_size)
        read_bvh.write_traindata_to_bvh(os.path.join(save_dance_folder, "%07d_gt.bvh" % iteration), gt_seq)
        read_bvh.write_traindata_to_bvh(os.path.join(save_dance_folder, "%07d_out.bvh" % iteration), out_seq)

# Rest of the code unchanged
def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        length = 10
        if length < 1:
            length = 1
        len_lst = len_lst + [length]
    index_lst = []
    index = 0
    for length in len_lst:
        for i in range(length):
            index_lst = index_lst + [index]
        index = index + 1
    return index_lst

def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []
    print('Loading motion files...')
    for dance_file in dance_files:
        dance = np.load(os.path.join(dance_folder, dance_file))
        dances = dances + [dance]
    print(len(dances), ' Motion files loaded')
    return dances

def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame, out_frame, hidden_size=1024, total_iter=500000):

    seq_len = seq_len+2
    torch.cuda.set_device(0)

    model = acLSTM(in_frame_size=in_frame, hidden_size=hidden_size, out_frame_size=out_frame)

    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path))

    model.cuda()
    #model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

    model.train()

    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)

    speed=frame_rate/30 # we train the network with frame rate of 30

    for iteration in range(total_iter):
        #get a batch of dances
        dance_batch=[]
        for b in range(batch):
            # randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0,random_range)]
            dance=dances[dance_id].copy()
            dance_len = dance.shape[0]

            start_id = random.randint(10, int(dance_len-seq_len*speed-10))
            sample_seq=[]
            for i in range(seq_len):
                sample_seq=sample_seq+[dance[int(i*speed+start_id)]]

            # augment the direction and position of the dance, helps the model to not overfeed
            T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            R=[0,1,0,(random.random()-0.5)*np.pi*2]
            sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R)
            dance_batch=dance_batch+[sample_seq_augmented]
        dance_batch_np=np.array(dance_batch)


        print_loss=False
        save_bvh_motion=False
        if(iteration % 20==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)

        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss, save_bvh_motion)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dances_folder', type=str, required=True, help='Path for the training data')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store test generated bvh')
    parser.add_argument('--read_weight_path', type=str, default="", help='Checkpoint model path')
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Dance frame rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_frame', type=int, required=True, help='Input channel')
    parser.add_argument('--out_frame', type=int, required=True, help='Output channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Checkpoint model path')
    parser.add_argument('--seq_len', type=int, default=100, help='Checkpoint model path')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Checkpoint model path')

    args = parser.parse_args()


    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances= load_dances(args.dances_folder)

    train(dances, args.dance_frame_rate, args.batch_size, args.seq_len, args.read_weight_path, args.write_weight_folder,
          args.write_bvh_motion_folder, args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations)

if __name__ == '__main__':
    main()
