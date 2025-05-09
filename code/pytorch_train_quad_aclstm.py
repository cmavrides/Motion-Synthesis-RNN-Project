import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import math
from scipy.spatial.transform import Rotation as R

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num = 19  # Assuming 19 joints for quaternion data (57/3 = 19 joints in positional)
Condition_num=5
Groundtruth_num=5
In_frame_size = 227  # Updated for quaternion data

def normalize_quaternions_torch(seq, root_dim=3):
    """
    seq shape: [batch, seq_len, frame_size]
    Normalize each quaternion [w, x, y, z] without doing in-place slicing,
    preserving gradient flows.
    """
    batch, seq_len, frame_size = seq.shape
    quat_dim = frame_size - root_dim
    joint_count = quat_dim // 4

    # Flatten [batch, seq_len, frame_size], using reshape instead of view
    seq_flat = seq.reshape(batch * seq_len, frame_size)

    quats = seq_flat[:, root_dim:]  # shape: [batch*seq_len, quat_dim]
    quats_reshaped = quats.reshape(-1, joint_count, 4)

    # Compute norms and normalize
    norms = quats_reshaped.norm(dim=2, keepdim=True) + 1e-8
    quats_normalized = quats_reshaped / norms

    # Rebuild a safe copy of the full sequence
    out = seq.clone().reshape(batch * seq_len, frame_size)
    out[:, root_dim:] = quats_normalized.reshape(batch * seq_len, quat_dim)

    return out.reshape(batch, seq_len, frame_size)



def calculate_quaternion_loss(self, out_seq, groundtruth_seq):
            batch, total_dim = out_seq.shape
            frame_size = 227
            seq_len = total_dim // frame_size

            out_seq_reshaped = out_seq.view(batch, seq_len, frame_size)
            gt_seq_reshaped = groundtruth_seq.view(batch, seq_len, frame_size)

            # Normalize quaternions in torch to maintain gradient flow
            out_seq_norm = normalize_quaternions_torch(out_seq_reshaped)

            loss_function = nn.MSELoss()
            loss = loss_function(out_seq_norm, gt_seq_reshaped)
            return loss



# Update the BVH generation function to handle quaternions properly
def generate_bvh_from_quad_traindata(npy_file_path, bvh_output_path):
    """Convert quaternion data (.npy) to BVH format"""
    quat_data = np.load(npy_file_path)
    # Convert quaternions back to Euler angles for BVH
    euler_data = []
    for frame in quat_data:
        root = frame[:3]  # First 3 values are root translation
        quats = frame[3:].reshape(-1, 4)  # Rest are quaternions, reshape to (num_joints, 4)

        # Normalize quaternions
        quats_norm = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

        # Convert each joint's quaternion to Euler angles (BVH is usually ZXY order)
        eulers = R.from_quat(quats_norm).as_euler('zxy', degrees=True)  # shape: (num_joints, 3)
        euler_frame = np.concatenate([root, eulers.flatten()])
        euler_data.append(euler_frame)
    euler_data = np.array(euler_data)
    read_bvh.write_traindata_to_bvh(bvh_output_path, euler_data)
    print(f"Reconstructed BVH from quaternion data saved: {bvh_output_path}")

def generate_quad_traindata_from_bvh(bvh_file_path, npy_output_path):
    """Convert BVH format to quaternion data (.npy)"""
    train_data = read_bvh.get_train_data(bvh_file_path)  # shape: (frames, joints*3)
    # Assume first 3 values are root translation, rest are Euler angles (in groups of 3)
    quat_data = []
    for frame in train_data:
        root = frame[:3]
        eulers = frame[3:].reshape(-1, 3)  # shape: (num_joints, 3)
        # Convert each joint's Euler angles to quaternion (BVH is usually ZXY order)
        quats = R.from_euler('zxy', eulers, degrees=True).as_quat()  # shape: (num_joints, 4)
        # Store as [root, q1, q2, ...]
        quat_frame = np.concatenate([root, quats.flatten()])
        quat_data.append(quat_frame)
    quat_data = np.array(quat_data)
    np.save(npy_output_path, quat_data)
    print(f"Saved quaternion training data: {npy_output_path}")


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=227, hidden_size=1024, out_frame_size=227):
        super(acLSTM, self).__init__()

        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size

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

        out_frame = self.decoder(vec_h2) #out b*227
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

    # Quaternion Loss function
    # q_pred and q_gt are batches of quaternions (b*seq_len*frame_size)
    # Replace the acLSTM.calculate_loss method with this improved version
    def calculate_loss(self, out_seq, groundtruth_seq):
        batch, total_dim = out_seq.shape
        frame_size = self.out_frame_size
        seq_len = total_dim // frame_size

        out_seq_reshaped = out_seq.view(batch, seq_len, frame_size)
        gt_seq_reshaped = groundtruth_seq.view(batch, seq_len, frame_size)

        # Normalize quaternions in torch to maintain gradient flow
        out_seq_norm = normalize_quaternions_torch(out_seq_reshaped)
        gt_seq_norm = normalize_quaternions_torch(gt_seq_reshaped)  # Also normalize ground truth

        # Root position loss (MSE on first 3 values)
        root_loss = nn.MSELoss()(out_seq_norm[:,:,:3], gt_seq_norm[:,:,:3])

        # Quaternion loss (MSE on normalized quaternions)
        quat_loss = nn.MSELoss()(out_seq_norm[:,:,3:], gt_seq_norm[:,:,3:])

        # Combine losses - weight can be adjusted as needed
        total_loss = root_loss + quat_loss

        return total_loss

#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):

    # For quaternion data, we only need to calculate dx and dz for root translation (first 3 values)
    # Subtract post (t+1) - t. Then, you will have the difference between each pose timestamp
    dif = real_seq_np[:, 1:real_seq_np.shape[1], :3] - real_seq_np[:, 0:real_seq_np.shape[1]-1, :3]
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()

    # Replace the root x and z with difference values
    real_seq_dif_hip_x_z_np[:,:,0] = dif[:,:,0]  # x component
    real_seq_dif_hip_x_z_np[:,:,2] = dif[:,:,2]  # z component

    real_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda())

    seq_len=real_seq.size()[1]-1
    in_real_seq=real_seq[:, 0:seq_len]

    predict_groundtruth_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np[:, 1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0], -1)

    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)

    optimizer.zero_grad()

    # The loss function for quaternion representation
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq)

    loss.backward()

    optimizer.step()

    if(print_loss==True):
        print ("###########"+"iter %07d"%iteration +"######################")
        print ("loss: "+str(loss.detach().cpu().numpy()))


    if(save_bvh_motion==True):
        ##save the first motion sequence in the batch
        gt_seq = np.array(predict_groundtruth_seq[0].data.tolist()).reshape(-1, In_frame_size)
        last_x = 0.0
        last_z = 0.0

        # Change hip xyz for ground truth sequence
        for frame in range(gt_seq.shape[0]):
            gt_seq[frame, 0] = gt_seq[frame, 0] + last_x
            last_x = gt_seq[frame, 0]

            gt_seq[frame, 2] = gt_seq[frame, 2] + last_z
            last_z = gt_seq[frame, 2]

        out_seq = np.array(predict_seq[0].data.tolist()).reshape(-1, In_frame_size)
        last_x = 0.0
        last_z = 0.0

        # Change hip xyz for output sequence
        for frame in range(out_seq.shape[0]):
            out_seq[frame, 0] = out_seq[frame, 0] + last_x
            last_x = out_seq[frame, 0]

            out_seq[frame, 2] = out_seq[frame, 2] + last_z
            last_z = out_seq[frame, 2]

        # Normalize the quaternions for both sequences
        for frame in range(gt_seq.shape[0]):
            for j in range((In_frame_size - 3) // 4):
                idx = 3 + j * 4
                quat = gt_seq[frame, idx:idx+4]
                gt_seq[frame, idx:idx+4] = quat / (np.linalg.norm(quat) + 1e-8)

        for frame in range(out_seq.shape[0]):
            for j in range((In_frame_size - 3) // 4):
                idx = 3 + j * 4
                quat = out_seq[frame, idx:idx+4]
                out_seq[frame, idx:idx+4] = quat / (np.linalg.norm(quat) + 1e-8)

        # Save ground truth and output sequences to temporary npy files
        gt_npy_path = save_dance_folder + "%07d"%iteration + "_gt.npy"
        out_npy_path = save_dance_folder + "%07d"%iteration + "_out.npy"

        np.save(gt_npy_path, gt_seq)
        np.save(out_npy_path, out_seq)

        # Convert quaternion data back to BVH format
        gt_bvh_path = save_dance_folder + "%07d"%iteration + "_gt.bvh"
        out_bvh_path = save_dance_folder + "%07d"%iteration + "_out.bvh"

        # Use the decoder function to convert npy to bvh
        generate_bvh_from_quad_traindata(gt_npy_path, gt_bvh_path)
        generate_bvh_from_quad_traindata(out_npy_path, out_bvh_path)

        # Clean up temporary npy files
        os.remove(gt_npy_path)
        os.remove(out_npy_path)

#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=len(dance)/100
        length = 10
        if(length<1):
            length=1
        len_lst=len_lst+[length]

    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    print('Loading motion files...')
    for dance_file in dance_files:
        if dance_file.endswith(".npy"):  # Only load .npy files for quaternion data
            # print ("load "+dance_file)
            dance=np.load(dance_folder+dance_file)
            dances=dances+[dance]
    print(len(dances), ' Motion files loaded')

    return dances

# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame, out_frame, hidden_size=1024, total_iter=500000):

    seq_len=seq_len+2
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

            # For quaternion data, augmentation is more complex - need rotation composition
            # Basic position augmentation for now
            T=[0.1*(random.random()-0.5), 0.0, 0.1*(random.random()-0.5)]
            R=[0, 1, 0, (random.random()-0.5)*np.pi*2]

            # Apply augmentation - for quaternion data
            # For now, just translate the root position
            sample_seq_augmented = np.array(sample_seq)
            for i in range(len(sample_seq)):
                # Apply translation to root position (first 3 values)
                sample_seq_augmented[i, 0] += T[0]
                sample_seq_augmented[i, 1] += T[1]
                sample_seq_augmented[i, 2] += T[2]

                # Note: proper quaternion rotation composition would be more complex
                # and would need to be implemented to handle R rotation properly

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
    parser.add_argument('--in_frame', type=int, default=227, help='Input channel (227 for quaternion data)')
    parser.add_argument('--out_frame', type=int, default=227, help='Output channels (227 for quaternion data)')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size for LSTM')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Total training iterations')

    args = parser.parse_args()


    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances = load_dances(args.dances_folder)

    train(dances, args.dance_frame_rate, args.batch_size, args.seq_len, args.read_weight_path, args.write_weight_folder,
          args.write_bvh_motion_folder, args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations)

if __name__ == '__main__':
    main()
