import os
import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from utils.geometry_util import laplacian_decomposition, get_operators, calculate_dist_no_boundary_mask,calculate_advanced_dist_mask
from utils.shape_util import read_shape, compute_geodesic_distmat, write_off

def preprocess_mask(dist_data_root,data_root,shape_name = None, is_no_boundary = False):
    off_files = sorted(glob(os.path.join(data_root, 'partial_off', '*.off')))
    if len(off_files) == 0:
        off_files = sorted(glob(os.path.join(data_root, 'shapes', '*.off')))
    if len(off_files) == 0:
        off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))    
    assert len(off_files) != 0
    dist_files = sorted(glob(os.path.join(dist_data_root, '*.mat')))
    assert len(dist_files) == len(off_files)
    for off_file, dist_file in tqdm(zip(off_files,dist_files)):
        if shape_name is not None and not shape_name in off_file:
            continue
        torch.cuda.empty_cache()
        if not is_no_boundary and os.path.exists(os.path.join(dist_data_root,os.path.basename(dist_file)[:-4]+'_max_dist.npy')):
            continue
        if not is_no_boundary  and os.path.exists(os.path.join(dist_data_root,os.path.basename(dist_file)[:-4]+'_max_dist_fps.npy')):
            continue
        print(off_file,dist_file)
        dist_mat = sio.loadmat(dist_file)['D']
        verts, faces = read_shape(off_file)
        if is_no_boundary:
            max_dist = calculate_dist_no_boundary_mask(dist_mat,faces)
        else:
            max_dist = calculate_advanced_dist_mask(dist_mat,verts,faces)
        if is_no_boundary:
            np.save(os.path.join(dist_data_root,os.path.basename(dist_file)[:-4]+'_no_boundary.npy'),max_dist)
        else:
            np.save(os.path.join(dist_data_root,os.path.basename(dist_file)[:-4]+'_max_dist.npy'),max_dist)

def preprocess(data_root,n_eig):
    
    spectral_dir = os.path.join(data_root, 'diffusion')
    os.makedirs(spectral_dir, exist_ok=True)
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    if len(off_files) == 0:
        off_files = sorted(glob(os.path.join(data_root, 'shapes', '*.off')))
    assert len(off_files) != 0

    partial_off_files = sorted(glob(os.path.join(data_root, 'partial_off', '*.off')))
    assert len(off_files) != 0
    
    print("processing full shapes")
    for off_file in tqdm(off_files):
        verts, faces = read_shape(off_file)
        filename = os.path.basename(off_file)

    
        # recompute laplacian decomposition
        get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                        k=n_eig, cache_dir=spectral_dir)

        
    print("processing partial shapes")
    for off_file in tqdm(partial_off_files):
        verts, faces = read_shape(off_file)
        filename = os.path.basename(off_file)
        

        # recompute laplacian decomposition
        get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                        k=n_eig, cache_dir=spectral_dir)

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--dist_data_root', required=True, help='data root contains the distances matrix.')
    parser.add_argument('--n_eig', type=int, default=200, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')
    parser.add_argument('--shape_name', required=False, help='shape name')
    parser.add_argument('--no_boundary', required=False, action='store_true', help='is no_boundary')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    dist_data_root = args.dist_data_root
    shape_name = args.shape_name
    n_eig = args.n_eig
    no_eig = args.no_eig
    is_no_boundary = args.no_boundary
    fps_samples = args.fps_samples
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    preprocess(data_root,n_eig)
    # Uncomment in case you want to calculate the masks - EXTREMELY SLOW, use the precompute ones
    # preprocess_mask(dist_data_root,data_root, shape_name, is_no_boundary)
    
    