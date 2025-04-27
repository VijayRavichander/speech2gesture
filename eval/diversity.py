import numpy as np
from bvh import Bvh
from math import ceil

def extract_motion_data(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        mocap = Bvh(f.read())
    
    joint_names = mocap.get_joints_names()
    n_frames = mocap.nframes
    motion_data = []

    for frame_idx in range(n_frames):
        frame_data = []
        for joint in joint_names:
            channels = mocap.joint_channels(joint)
            for channel in channels:
                value = mocap.frame_joint_channel(frame_idx, joint, channel)
                frame_data.append(float(value))
        motion_data.append(frame_data)
    
    return np.array(motion_data)

def compute_diversity_metric(motion_data, clip_length=50):
    n_frames = motion_data.shape[0]
    n_clips = n_frames // clip_length
    clips = np.array_split(motion_data[:n_clips * clip_length], n_clips)
    
    total_distance = 0
    for i in range(n_clips):
        for j in range(i + 1, n_clips):
            distance = np.sum(np.abs(clips[i] - clips[j]))
            total_distance += distance
    
    normalization_factor = n_clips * ceil(n_clips / 2)
    diversity = total_distance / normalization_factor if normalization_factor > 0 else 0
    return diversity

# Example usage
bvh_file_path = '../data/bvh2clips_test/Copy of 1_wayne_0_1_1_part_000.bvh'
motion_data = extract_motion_data(bvh_file_path)
diversity = compute_diversity_metric(motion_data, clip_length=50)
print(f'Diversity Metric: {diversity}')


# bvh_file_path = '../vq-vae/preds/test.bvh'
# motion_data = extract_motion_data(bvh_file_path)
# diversity = compute_diversity_metric(motion_data, clip_length=50)
# print(f'Diversity Metric: {diversity}')