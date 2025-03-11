import torch
import math
import torch.nn as nn
import numpy as np
from geotransformer.modules.ops import pairwise_distance
class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """

    def __init__(self, ratio=1.0):
        super().__init__()
        self.ratio = ratio

    def __call__(self, corr_gt, corr_es, log_var_mask):
        m, n = corr_gt.shape
        indices = np.sum(np.array(corr_es.cpu().detach()) > 0)
        indices = max(indices, 1)
        indice_ratio = 1. /  (indices / (n * m))

        ref_mask = log_var_mask[:m]
        src_mask = log_var_mask[m:n+m]
        laplace_mask = torch.ger(ref_mask, src_mask)

        loss1 = math.sqrt(2) * indice_ratio * torch.exp(-0.5 * laplace_mask) * \
             torch.abs(corr_gt - corr_es)
        loss1 = loss1.mean()
        # each dimension is multiplied
        loss2 = 0.5 * laplace_mask.mean()
        loss = loss1 + loss2
        return loss, loss1, loss2


 
def get_indices_from_loc(full_points_loc, corr_loc):
    distances = torch.cdist(corr_loc, full_points_loc, p=2)  # [M, N]
    min_distances, indices = torch.min(distances, dim=1)
    return indices


def apply_transform(points, transform):
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
    points_t = torch.matmul(transform, points_h.t()).t()[:, :3]
    return points_t

def get_correspondences(ref_points, src_points, transform, matching_radius):
    src_points_transformed = apply_transform(src_points, transform)
    dists = torch.cdist(ref_points, src_points_transformed, p=2)
    corr_indices = (dists <= matching_radius).nonzero(as_tuple=False)

    return corr_indices

class LaplaceLoss(nn.Module):
    def __init__(self, matching_radius=0.1,max_points=256, stage=1):
        super(LaplaceLoss, self).__init__()
        self.loss = NLLLaplace()
        self.matching_radius = matching_radius
        self.stage = stage
        self.max_points = max_points
        
    def forward(self, output_dict, data_dict):
        device = output_dict['src_points'].device
        indices_src = get_indices_from_loc(output_dict['src_points'], output_dict['src_points_c'] ).to(device)#(N)
        indices_ref = get_indices_from_loc(output_dict['ref_points'], output_dict['ref_points_c'] ).to(device)#(M)

        m = len(indices_ref)
        n = len(indices_src)
        src_node_corr_indices = output_dict['src_node_corr_indices'].to(device)#(n1)
        ref_node_corr_indices = output_dict['ref_node_corr_indices'].to(device)#（m1）
        
        corr_es = torch.zeros((m, n), device=device)
        for x, y in zip(ref_node_corr_indices, src_node_corr_indices):
            corr_es[x, y] = 1

        src_back_indices = data_dict['src_back_indices'].to(device)#(M)
        ref_back_indices = data_dict['ref_back_indices'].to(device)#(M)
        set_src_indices = set(indices_src.tolist())
        set_src_back_indices = set(src_back_indices.tolist())

        set_ref_indices = set(indices_ref.tolist())
        set_ref_back_indices = set(ref_back_indices.tolist())

        intersection_src = set_src_indices.intersection(set_src_back_indices)
        intersection_src = torch.tensor(list(intersection_src), device=device)
        intersection_indices_src = torch.nonzero(indices_src.unsqueeze(1) == intersection_src, as_tuple=True)[0]

        intersection_ref = set_ref_indices.intersection(set_ref_back_indices)
        intersection_ref = torch.tensor(list(intersection_ref), device=device)
        intersection_indices_ref = torch.nonzero(indices_ref.unsqueeze(1) == intersection_ref, as_tuple=True)[0]

        corr_gt_indices = output_dict['gt_node_corr_indices'].to(device)
        corr_gt_overlap = output_dict['gt_node_corr_overlaps'].to(device)
        corr_gt = torch.zeros((m, n), device=device)
        corr_overlap = torch.zeros((m, n), device=device)
        for idx, (x, y) in enumerate(corr_gt_indices):
            corr_gt[x, y] = 1
            corr_overlap[x, y] = corr_gt_overlap[idx]
        mask_ref = torch.zeros(corr_gt.size(0), dtype=torch.bool, device=device)
        mask_ref[intersection_indices_ref] = True

        mask_src = torch.zeros(corr_gt.size(1), dtype=torch.bool, device=device)
        mask_src[intersection_indices_src] = True
        mask = mask_ref.unsqueeze(1) & mask_src
        corr_gt[~mask] = 0

        gt_mask = torch.cat((mask_ref, mask_src),dim=0).float().to(device)
        indices_back = np.sum(np.array(corr_gt.cpu().detach()) > 0)

        if indices_back > self.max_points:
            ones_indices = torch.nonzero(corr_gt, as_tuple=False)
            scores = corr_overlap[ones_indices[:, 0], ones_indices[:, 1]]
            sorted_indices = torch.argsort(scores, descending=True)
            top_indices = sorted_indices[:self.max_points]
            new_corr_gt = torch.zeros_like(corr_gt)
            top_ones_indices = ones_indices[top_indices]
            new_corr_gt[top_ones_indices[:, 0], top_ones_indices[:, 1]] = 1
            corr_gt = new_corr_gt

        if self.stage == 1:
            corr_sp_mask = torch.ones(corr_gt.shape[0] + corr_gt.shape[1], device=device)
        else:
            corr_sp_mask = output_dict['corr_sp_mask']
        loss = self.loss(corr_gt, corr_es, (1 - corr_sp_mask))
        return loss, 1-gt_mask

