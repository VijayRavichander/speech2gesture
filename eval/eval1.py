import numpy as np
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load BVH data
# -----------------------------
def load_bvh_data(file_path):
    bvh = BVH()
    bvh.load(file_path)
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    return local_rotations, local_positions, parents, offsets

# -----------------------------
# 2. Compute Global Positions
# -----------------------------
def compute_global_positions(local_rotations, local_positions, parents, offsets):
    global_positions, _ = fk(local_rotations, local_positions[:, 0, :], offsets, parents)
    return global_positions  # shape (frames, joints, 3)

def draw_2d_skeleton(global_positions, parents, frame_idx=0, joint_names=None):
    """
    Draws a 2D skeleton for a given frame.

    Parameters:
    - global_positions: np.ndarray of shape (frames, joints, 3)
    - parents: list or np.ndarray of parent indices for each joint
    - frame_idx: index of the frame to visualize
    - joint_names: list of joint names (optional)
    """
    joints = global_positions[frame_idx]
    
    plt.figure(figsize=(8, 8))
    for i, parent in enumerate(parents):
        if parent == -1:
            continue  # Skip the root joint
        x = [joints[i][0], joints[parent][0]]
        y = [joints[i][1], joints[parent][1]]
        plt.plot(x, y, 'ro-')
        if joint_names:
            plt.text(joints[i][0], joints[i][1], joint_names[i], fontsize=8)
    
    plt.title(f"2D Skeleton at Frame {frame_idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# -----------------------------
# 3. Metrics
# -----------------------------

# Mean Per Joint Position Error
def compute_mpjpe(predicted, ground_truth):
    error = np.linalg.norm(predicted - ground_truth, axis=-1)  # (frames, joints)
    mpjpe = np.mean(error)
    return mpjpe

# Mean Per Joint Angle Error
def compute_mpjae(predicted_rotations, ground_truth_rotations):
    # predicted_rotations, ground_truth_rotations are (frames, joints, 3) in degrees
    diff = np.abs(predicted_rotations - ground_truth_rotations)  # (frames, joints, 3)
    mpjae = np.mean(diff)
    return mpjae

# Percentage of Correct Keypoints (PCK)
def compute_pck(predicted, ground_truth, threshold):
    error = np.linalg.norm(predicted - ground_truth, axis=-1)  # (frames, joints)
    correct = error < threshold
    pck = np.mean(correct)
    return pck

# -----------------------------
# 4. Main Evaluation
# -----------------------------
def evaluate(gt_path, pred_path, pck_threshold=150):
    # Load both BVH files
    gt_rot, gt_pos, gt_parents, gt_offsets = load_bvh_data(gt_path)
    pred_rot, pred_pos, pred_parents, pred_offsets = load_bvh_data(pred_path)

    assert gt_parents.tolist() == pred_parents.tolist(), "Skeleton hierarchy mismatch!"

    # Compute global joint positions
    gt_global_pos = compute_global_positions(gt_rot, gt_pos, gt_parents, gt_offsets)
    pred_global_pos = compute_global_positions(pred_rot, pred_pos, pred_parents, pred_offsets)

    # draw_2d_skeleton(pred_global_pos, gt_parents)
    # draw_2d_skeleton(pred_global_pos, pred_parents)

    # Compute metrics
    mpjpe = compute_mpjpe(pred_global_pos, gt_global_pos)
    mpjae = compute_mpjae(pred_rot, gt_rot)
    pck = compute_pck(pred_global_pos, gt_global_pos, pck_threshold)

    print(f"MPJPE (cm): {mpjpe:.3f}")
    print(f"MPJAE (degrees): {mpjae:.3f}")
    print(f"PCK @ {pck_threshold}cm: {pck*100:.2f}%")

# -----------------------------
# 5. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Provide your ground truth and predicted BVH file paths
    gt_bvh_path = '../data/bvh2clips_test/Copy of 1_wayne_0_1_1_part_000.bvh'
    pred_bvh_path = '../vq-vae/preds/test copy.bvh'

    evaluate(gt_bvh_path, pred_bvh_path, pck_threshold=10)
