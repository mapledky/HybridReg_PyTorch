import argparse
import os
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.registration import compute_registration_error
from config import make_cfg
from model import create_model
from tqdm import tqdm
import json


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="directory containing the point cloud data pairs")
    parser.add_argument("--weights", default='assets/weights/HybridMatch/best.pth.tar', help="model weights file")
    parser.add_argument("--way", default='lgr')
    return parser


def load_data(src_file, ref_file, gt_file, src_back_indices=[], ref_back_indices=[]):
    src_points = np.load(src_file)  # n,3
    ref_points = np.load(ref_file)
    if len(src_back_indices) == 0:
        src_back_indices = np.arange(0, len(src_points))
    if len(ref_back_indices) == 0:
        ref_back_indices = np.arange(0, len(ref_points))
    if len(src_points) > 20000:
        src_points, src_back_indices = point_cut(src_points, src_back_indices)
    
    if len(ref_points) > 20000:
        ref_points, ref_back_indices = point_cut(ref_points, ref_back_indices)

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "src_back_indices": src_back_indices,
        "ref_back_indices": ref_back_indices
    }

    if gt_file is not None:
        transform = np.load(gt_file)  # 4*4
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def point_cut(points, indices, max_points=20000):
    keep_indices = np.random.choice(len(points), max_points, replace=False)
    points = points[keep_indices]
    new_indices = []
    for i, idx in enumerate(indices):
        if idx in keep_indices:
            new_idx = np.where(keep_indices == idx)[0][0]
            new_indices.append(new_idx)
    return points, np.array(new_indices)


def process_pair(model, data_dict, cfg):
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)
    ref_corr_points = output_dict['ref_corr_points']
    src_corr_points = output_dict['src_corr_points']
    corr = np.hstack((src_corr_points, ref_corr_points))
    # get results
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]
    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    return rre, rte, estimated_transform, corr

def compute_RMSE(src_pcd_back, gt, estimate_transform):
    gt_np = np.array(gt)
    estimate_transform_np = np.array(estimate_transform)
    
    realignment_transform = np.linalg.inv(gt_np) @ estimate_transform_np
    
    transformed_points = np.dot(src_pcd_back, realignment_transform[:3,:3].T) + realignment_transform[:3,3]
    
    rmse = np.sqrt(np.mean(np.linalg.norm(transformed_points - src_pcd_back, axis=1) ** 2))
    
    return rmse

def batch_test(data_dir, weights):
    print(data_dir)
    cfg = make_cfg()

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(weights)['model']
    # Initialize missing keys with random values
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - set(state_dict.keys())
    for key in missing_keys:
        if 'weight' in key:
            state_dict[key] = torch.randn_like(model.state_dict()[key])
        elif 'bias' in key:
            state_dict[key] = torch.zeros_like(model.state_dict()[key])
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    rre_anim = []
    rte_anim = []
    rre_anim_all = 0.
    rte_anim_all = 0.
    
    num_pairs_anim = 0
    num_pairs_succ_anim = 0

    rre_wo_anim = [] 
    rte_wo_anim = []

    rre_wo_anim_all = 0.
    rte_wo_anim_all = 0.

    num_pairs_wo_anim = 0
    num_pairs_succ_wo_anim = 0

    subdirs = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_dir) for d in dn]    

    total_subdirs = len(subdirs)

    with tqdm(total=total_subdirs, desc='Processing subdirectories') as pbar:
        for subdir in subdirs:
            subdir_path = subdir
            if not os.path.isdir(subdir_path):
                print(subdir_path)
                continue
            src_wo_anim_file = os.path.join(subdir_path, 'src.npy')
            ref_wo_anim_file = os.path.join(subdir_path, 'ref_wo_anim.npy')

            src_true_file = os.path.join(subdir_path, 'src.npy')
            ref_true_file = os.path.join(subdir_path, 'ref.npy')
          
            gt_file = os.path.join(subdir_path, 'relative_transform.npy')

            src_back_indices_json = os.path.join(subdir_path, 'src_back_indices.json')
            ref_back_indices_json = os.path.join(subdir_path, 'ref_back_indices.json')
            with open(src_back_indices_json , 'r') as file:
                data = json.load(file)
                src_back_indices = np.array(data['back_indices'])
            with open(ref_back_indices_json , 'r') as file:
                data = json.load(file)
                ref_back_indices = np.array(data['back_indices'])
            data_dict_true = None
            data_dict_wo_anim = None

            if os.path.exists(src_true_file) and os.path.exists(ref_true_file) and os.path.exists(gt_file):
                data_dict_true = load_data(src_true_file, ref_true_file, gt_file, src_back_indices)
                print(len(data_dict_true.get('src_points')) , len(data_dict_true.get('ref_points')))
                rre, rte, estimate_rt,  corr = process_pair(model, data_dict_true, cfg)

                rmse = compute_RMSE(data_dict_true.get('src_points'), data_dict_true.get('transform'), estimate_rt)
                print('rmse_true ' , rmse)
                rre_anim_all += rre
                rte_anim_all += rte
                if rmse <= 0.2:       
                    num_pairs_succ_anim += 1
                    rre_anim.append(rre)
                    rte_anim.append(rte)

                num_pairs_anim += 1

            if os.path.exists(src_wo_anim_file) and os.path.exists(ref_wo_anim_file) and os.path.exists(gt_file):
                data_dict_wo_anim = load_data(src_wo_anim_file, ref_wo_anim_file, gt_file)
                print(len(data_dict_true.get('src_points')) , len(data_dict_true.get('ref_points')))
                rre, rte, estimate_rt,  corr = process_pair(model, data_dict_wo_anim, cfg)
                rmse = compute_RMSE(data_dict_wo_anim.get('src_points'),  data_dict_wo_anim.get('transform'), estimate_rt)
                print('rmse_wo_anim ' , rmse)
                rre_wo_anim_all += rre
                rte_wo_anim_all += rte
                if rmse <= 0.2:
                    num_pairs_succ_wo_anim += 1
                    rre_wo_anim.append(rre)
                    rte_wo_anim.append(rte)
                num_pairs_wo_anim += 1
            pbar.update(1)

    if num_pairs_anim != 0:
        rr = num_pairs_succ_anim / num_pairs_anim
        median_rre = np.median(np.array(rre_anim))
        median_rte = np.median(np.array(rte_anim))
        print(f"median RRE_anim(deg): {median_rre:.3f}, median RTE_anim(m): {median_rte:.3f}")
        print(f"RR_anim: {rr:.3f}")
        print(f"avg RRE_anim(deg): {rre_anim_all / num_pairs_anim:.3f}, avg RTE_anim(m): {rte_anim_all / num_pairs_anim:.3f}")

    if num_pairs_wo_anim != 0:
        rr = num_pairs_succ_wo_anim / num_pairs_wo_anim
        median_rre = np.median(np.array(rre_wo_anim))
        median_rte = np.median(np.array(rte_wo_anim))
        print(f"median RRE_wo_anim(deg): {median_rre:.3f}, median RTE_wo_anim(m): {median_rte:.3f}")
        print(f"RR_wo_anim: {rr:.3f}")
        print(f"avg RRE_wo_anim(deg): {rre_wo_anim_all / num_pairs_wo_anim:.3f}, avg RTE_wo_anim(m): {rte_wo_anim_all / num_pairs_wo_anim:.3f}")


def main():
    parser = make_parser()
    cfg = make_cfg()
    args = parser.parse_args()
    dataset = os.path.join(cfg.dataset.root, args.data_dir)
    if args.way == 'lgr':
        batch_test(dataset, args.weights)

if __name__ == "__main__":
    main()
