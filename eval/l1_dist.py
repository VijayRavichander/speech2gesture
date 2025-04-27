import numpy as np
from bvh import Bvh

def parse_bvh_file(file_path):
    with open(file_path, 'r') as f:
        bvh_data = Bvh(f.read())
    frames = np.array(bvh_data.frames, dtype=float)
    return frames

def compute_l1_distance(frames1, frames2):
    if frames1.shape != frames2.shape:
        raise ValueError("BVH files have different numbers of frames or channels.")
    l1_distances = np.abs(frames1 - frames2)
    mean_l1_per_frame = np.mean(l1_distances, axis=1)
    overall_mean_l1 = np.mean(mean_l1_per_frame)
    return overall_mean_l1

# Example usage
bvh_file_1 = '../data/bvh2clips_test/Copy of 1_wayne_0_1_1_part_000.bvh'
bvh_file_2 = '../vq-vae/preds/test.bvh'

frames1 = parse_bvh_file(bvh_file_1)
frames2 = parse_bvh_file(bvh_file_2)

min_frames = min(frames1.shape[0], frames2.shape[0])
frames1, frames2 = frames1[:min_frames], frames2[:min_frames]

l1_distance = compute_l1_distance(frames1, frames2)

print(f"Average L1 Distance: {l1_distance}")
