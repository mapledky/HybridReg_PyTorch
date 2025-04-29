import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None, , laplace_mask=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).
            laplace_mask (BoolTensor=None): mask of the deforming foregrounds in the source and target points.

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        n, _ = src_feats.shape
        m, _ = ref_feats.shape
        
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        
        if not (laplace_mask is None):
            laplace_mask = torch.squeeze(laplace_mask)
            laplace_mask_src = laplace_mask[m:][src_indices]
            laplace_mask_ref = laplace_mask[:m][ref_indices]
            
            laplace_mask_src = laplace_mask_src.unsqueeze(0).expand_as(matching_scores)
            laplace_mask_ref = laplace_mask_ref.unsqueeze(1).expand_as(matching_scores)
            # Combine masks
            combined_mask = (laplace_mask_ref == 1) & (laplace_mask_src == 1)
            probability_mask = torch.rand_like(matching_scores) < 0.5
            combined_mask = combined_mask & probability_mask
            # Apply the mask to the matching scores
            filtered_scores = matching_scores.clone()
            filtered_scores[combined_mask] = float(-1)
        else:
            filtered_scores = matching_scores
        
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores
