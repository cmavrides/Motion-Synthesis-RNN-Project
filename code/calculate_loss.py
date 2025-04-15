import torch
import torch.nn.functional as F


def regularize_angles(predicted, target):
    """
    Wrap angle differences to [-180, 180] to avoid circular jump issues.
    """
    diff = predicted - target
    diff = (diff + 180.0) % 360.0 - 180.0
    return diff


def quaternion_loss(predicted, target, eps=1e-8):
    """
    Loss = 1 - cosine similarity between unit quaternions
    Handles batch and frame dimensions.
    """
    pred_norm = F.normalize(predicted, dim=-1, eps=eps)
    target_norm = F.normalize(target, dim=-1, eps=eps)
    dot_product = torch.sum(pred_norm * target_norm, dim=-1)
    loss = 1.0 - dot_product ** 2  # squared cosine similarity
    return loss.mean()


def calculate_loss(predicted, target, mode="pos"):
    """
    Calculate loss based on representation mode.
    Args:
        predicted: [batch, seq_len, dim]
        target: [batch, seq_len, dim]
        mode: 'pos' | 'euler' | 'quat'
    """
    if mode == "pos":
        return F.mse_loss(predicted, target)

    elif mode == "euler":
        angle_diff = regularize_angles(predicted, target)
        return F.smooth_l1_loss(angle_diff, torch.zeros_like(angle_diff))

    elif mode == "quat":
        # assumes [batch, seq_len, dim] where dim is 4 * num_joints
        B, T, D = predicted.shape
        predicted = predicted.view(B * T, -1, 4)
        target = target.view(B * T, -1, 4)
        return quaternion_loss(predicted, target)

    else:
        raise ValueError(f"Unsupported loss mode: {mode}")
