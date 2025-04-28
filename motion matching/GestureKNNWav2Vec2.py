# Import spatial distance calculation functions for similarity measures
from scipy import spatial
import numpy as np
import os
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R   


class GestureKNN(object):
    def __init__(self, feat_train, motn_train, n_aud_feat=28, n_motion_feat=228, step_sz=4, random_init = False):
        super(GestureKNN, self).__init__()
        
        # feat_train shape    : (num_seq, num_frames, (n_aud_feat + n_body_feat))
        # control_mask shape  : (num_seq, num_frames)
        # motn_train shape    : (num_seq, num_frames, n_joints)

        self.n_aud_feat = n_aud_feat
        self.n_motion_feat = n_motion_feat
        self.step_sz = step_sz
        self.random_init = random_init
        self.features_train = feat_train
        self.motion_train = motn_train
        
        self.db_sequences = feat_train.shape[0]
        self.db_frames = feat_train.shape[1]

    def init_body(self):
        # Pick a Random Index
        init_seq_idx = np.random.randint(0, self.db_sequences)
        init_frm_idx = np.random.randint(0, self.db_frames) 

        return init_seq_idx, init_frm_idx
    
    def search_audio_init(self, ft):
        audio_feature = ft[:self.n_aud_feat, 0] #28

        aud_dist_cands = []
        best_match = None
        best_score = float('inf')

        num_seqs, num_frames, _ = self.features_train.shape

        for seqs in range(num_seqs):
            for frame in range(num_frames):
                frame_audio = self.features_train[seqs, frame, :self.n_aud_feat]
                audio_sim_score = spatial.distance.cosine(audio_feature, frame_audio)

                if audio_sim_score < best_score:
                    best_score = audio_sim_score
                    best_match = (seqs, frame)

                aud_dist_cands.append((audio_sim_score, seqs, frame))
        

        return best_match[0], best_match[1]

    def search_motion(self, feat_test, desired_k):

        # feat_test shape    : (self.n_aud_feat, num_frames))

        n_frames = feat_test.shape[-1]  # 478 Frames

        feat_test = np.concatenate((feat_test[:, 0:1], feat_test), axis=1) # 28, 478

        pose_feat = np.zeros((self.n_motion_feat, feat_test.shape[1])) # 228, 478

        feat_test = np.concatenate((feat_test, pose_feat), axis=0)  # 256, 478

        if self.random_init:
            init_seq, init_frm = self.init_body()

        else:
            init_seq, init_frm = self.search_audio_init(feat_test)

        # For the First Frame, add a pick a random body feature for the db and add here
        feat_test[self.n_aud_feat:, 0] = self.features_train[init_seq, init_frm, self.n_aud_feat:]

        pred_motion = np.zeros((self.n_motion_feat, n_frames + 1)) #228, 478

        # Start processing from frame 1 (frame 0 is just a duplicate for initialization)
        j = 1

        # Process all frames in the test data
        while j < (self.step_sz * (n_frames // self.step_sz)):
            # print(feat_test[self.n_aud_feat:, j-1].shape)
            
            # Search for pose candidates based on previous frame's pose
            pos_dist_cands, pose_cands, feat_cands = self.search_pose_cands(feat_test[self.n_aud_feat:, j-1])
            
            # Get number of retained candidates
            n_retained = pos_dist_cands.shape[0]

            # Get current audio test features
            audio_test_feat = feat_test[:self.n_aud_feat, j]
            
            # Calculate audio distance for each candidate
            aud_dist_cands = []
            # print(f"Pose Dist Cand: {pos_dist_cands.shape}")

            # print(pos_dist_cands.shape, pose_cands.shape, feat_cands.shape)

            for k in range(n_retained):
                # Calculate cosine distance between audio features
                # feat_cands shape : (num_seqs, feature_points, step_size)
                # print(audio_test_feat.shape)
                # print(feat_cands[k, :self.n_aud_feat, 0].shape)

                audio_sim_score = spatial.distance.cosine(audio_test_feat, feat_cands[k, :self.n_aud_feat, -1])
                aud_dist_cands.append(audio_sim_score)

                # print(audio_sim_score.shape)

            # Convert distances to rankings for pose and audio
            pos_score = np.array(pos_dist_cands).argsort().argsort()
            aud_score = np.array(aud_dist_cands).argsort().argsort()
            
            # print(pos_score)
            # print(aud_score)

            # Combine pose and audio rankings to get final score
            combined_score = pos_score + aud_score

            # Sort candidates by combined score
            combined_sorted_idx = np.argsort(combined_score).tolist()

            # Reorder candidates based on combined score
            feat_cands = feat_cands[combined_sorted_idx]
            pose_cands = pose_cands[combined_sorted_idx]
            
            # Update features and motion with the selected candidate (based on desired_k)
            feat_test[self.n_aud_feat:, j:(j+self.step_sz)] = feat_cands[desired_k, self.n_aud_feat:, :self.step_sz]
            pred_motion[:, j:(j+self.step_sz)] = pose_cands[desired_k, :, :self.step_sz]
            
            # Move to the next frame by step size
            j += self.step_sz

            # Return predicted motion (excluding the initialization frame)
            # pred_motion shape    : (self.n_joints, num_frames))

        return pred_motion[:, 1:]
        
    def search_pose_cands(self, body_test_feat):
        # Initialize lists to store candidates
        pos_dist_cands = []
        pose_cands = []
        feat_cands = []

        # print(body_test_feat.shape) ## 228

        # Iterate through all sequences in the training database
        for k in range(self.features_train.shape[0]):

            # Calculate distances between test body features and all training frames
            body_dist_list = []
            body_train_feat = self.features_train[k, :, self.n_aud_feat:]

            # Calculate Euclidean distance for each frame in the sequence
            for l in range(body_train_feat.shape[0]):

                # # print("Shapes Needed")
                # print(body_test_feat.shape, body_train_feat[l].shape)

                body_dist = np.linalg.norm(body_test_feat - body_train_feat[l])
                body_dist_list.append(body_dist)

            # Sort frames by distance (closest first)
            sorted_idx_list = np.argsort(body_dist_list)

            # Initialize variables for candidate search
            pose_cand_ctr = 0
            pose_cand_found = False

            # Search for a valid candidate frame
            while pose_cand_ctr < len(sorted_idx_list) - 1:
                # Get frame index and distance
                f = sorted_idx_list[pose_cand_ctr]
                d = body_dist_list[f]

                # Move to next candidate
                pose_cand_ctr += 1
                
                # Skip identical frames (distance = 0)
                if d == 0:
                    continue
                
                # Skip frames near the end of sequence where full step size can't be used
                if f >= self.db_sequences - self.step_sz or f >= self.db_frames - self.step_sz:
                    continue
                    
                # Valid candidate found
                pose_cand_found = True
                break

            # Skip if no valid candidate found
            if pose_cand_found == False:
                continue
            
            # Extract features and poses for the selected candidate
            # feat_cand shape: (num_feat_dim, step_sz)
            # pose_cand shape: (num_feat_dim, step_sz)

            feat_cand = self.features_train[k, f:(f+self.step_sz), :].transpose()
            pose_cand = self.motion_train[k, f:(f+self.step_sz), :].transpose()

            # Store candidate information
            pos_dist_cands.append(d)
            pose_cands.append(pose_cand)
            feat_cands.append(feat_cand)
        
        # Convert lists to numpy arrays
        pos_dist_cands = np.array(pos_dist_cands)

        pose_cands = np.array(pose_cands)
        feat_cands = np.array(feat_cands)
        
        # Return all candidate information
        return pos_dist_cands, pose_cands, feat_cands

def get_wav_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            files.append(os.path.join(folder, file))
    return files

def get_bvh_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".bvh"):
            files.append(os.path.join(folder, file))
    return files

def extract_file_name(file_path):
    base_name = file_path.split('/')[-1].split('.')[0]
    return '_'.join(base_name)

def match_files(wav_files, bvh_files):
    matching_tuples = []
    for wav in wav_files:
        wav_numbers = extract_file_name(wav)
        for bvh in bvh_files:
            bvh_numbers = extract_file_name(bvh)
            if wav_numbers == bvh_numbers:
                matching_tuples.append((wav, bvh))
    return matching_tuples

def extract_motion_data(bvh_file):
    with open(bvh_file, 'r') as f:
        lines = f.readlines()

    motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION") + 3
    motion_data = [list(map(float, line.strip().split())) for line in lines[motion_index:]]

    return np.array(motion_data)

def convert_wav_to_vec(dataset):
    wav2vec = []
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    encoder.eval()

    for w, _ in dataset:  # Iterate over each sequence

        waveform, sample_rate = torchaudio.load(w)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Process audio
        wav_inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            output = encoder(wav_inputs["input_values"])

        x = output.extract_features # 199, 512
        x = x.transpose(1, 2)                  # (1, 512, 199)  --  channels first
        x = F.interpolate(x, size=480, mode='linear', align_corners=False)
        x = x.transpose(1, 2) 

        wav2vec.append(x.squeeze(0))

    return wav2vec

def extract_bvh_hierarchy(bvh_file):
    with open(bvh_file, 'r') as f:
        lines = f.readlines()

    motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    hierarchy = "".join(lines[:motion_index])

    return hierarchy

def write_bvh(output_file, predicted_motion, template_hierarchy, frame_time=0.00833):
    with open(output_file, 'w') as f:
        f.write(template_hierarchy)
        f.write("MOTION\n")
        f.write(f"Frames: {len(predicted_motion)}\n")
        f.write(f"Frame Time: {frame_time}\n")

        for frame in predicted_motion:
            f.write(' '.join(map(str, frame)) + '\n')


wav_folder = "../data/wav2clips"
bvh_folder = "../data/bvh2clips"

wavFiles = get_wav_files(wav_folder)
bvhFiles = get_bvh_files(bvh_folder)

dataset = match_files(wavFiles, bvhFiles)

min_frames = 480

audio_features, motion_features = [], []

wav2vec = convert_wav_to_vec(dataset)
audio_features.extend(wav2vec)

print(len(audio_features))

for _, bf in dataset:
    motion_data = extract_motion_data(bf)
    motion_data = motion_data[:min_frames]

    motion_features.append(motion_data) 

audio_features, motion_features = np.array(audio_features), np.array(motion_features)

print(audio_features.shape)
print(motion_features.shape)

features = np.concatenate((audio_features, motion_features), axis=2)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_features, train_motion_features = features[:train_size], motion_features[:train_size]

print(f"Dataset Size: {len(dataset)}")
print(f'Train_Size: {train_size}')
print(f"Train Dataset Shape: {train_features.shape}")
print(f"Test Set Size: {test_size}")

# Model
gesture_knn = GestureKNN(train_features, train_motion_features, n_aud_feat = 512, step_sz=4, random_init=True)

for idx in range(train_size, len(dataset)):
    test_audio = audio_features[idx].transpose(1,0)

    pred_seqs = gesture_knn.search_motion(test_audio, 0)
    print(pred_seqs.shape)
    pred_seqs = pred_seqs.transpose(1, 0)

    test_bvh_file = dataset[idx][1]
    template_hierarchy = extract_bvh_hierarchy(test_bvh_file) 
    new_file_name = test_bvh_file.split("/")[-1]
    new_file_name = f"./preds/GKKWV/{new_file_name}"
    write_bvh(new_file_name, pred_seqs, template_hierarchy)
