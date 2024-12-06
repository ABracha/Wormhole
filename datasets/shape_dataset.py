import os, re
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.shape_util import read_shape
from utils.geometry_util import get_operators, sparse_torch_to_np_v2
from utils.registry import DATASET_REGISTRY
from utils.misc import find_existing_pairs, read_words_from_file


def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    # item['L_ind'], item['L_vals'], item['L_shape'] = sparse_torch_to_np_v2(L)
    # item['L'] =  sparse_torch_to_np(L)#L.to_dense()

    return item


class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        self.off_files = []
        self.corr_files = [] if self.return_corr else None
        self.dist_files = [] if self.return_dist else None

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        if self.return_dist:
            assert self._size == len(self.dist_files)

        if self.return_corr:
            assert self._size == len(self.corr_files)

    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*0.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*0.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*0.mat'))

    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=os.path.join(self.data_root, 'diffusion'))

        # get geodesic distance matrix
        if self.return_dist:
            mat = sio.loadmat(self.dist_files[index])
            item['dist'] = torch.from_numpy(mat['D']).float()

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['corr'] = torch.from_numpy(corr).long()

        return item

    def __len__(self):
        return self._size

class SinglePartialShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, 
                 return_mask = False, fps = False):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.return_mask = return_mask
        self.fps = fps
        self.num_evecs = num_evecs

        self.off_files = []
        self.corr_files = [] if self.return_corr else None
        self.mask_files = [] if self.return_corr else None
        self.dist_files = [] if self.return_dist else None
        self.dist_mask_files = [] if self.return_mask else None

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        if self.return_dist:
            assert self._size == len(self.dist_files)
        
        if self.return_mask:
            assert self._size == len(self.dist_mask_files)

        if self.return_corr:
            assert self._size == len(self.corr_files)

    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'partial_off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'partial_corr')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

            mask_path = os.path.join(self.data_root, 'masks')
            assert os.path.isdir(mask_path), f'Invalid path {mask_path} not containing .vts files'
            self.mask_files = sort_list(glob(f'{mask_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'partial_dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))
        
        if self.return_mask:
            dist_mask_path = os.path.join(self.data_root, 'partial_dist')
            assert os.path.isdir(dist_mask_path), f'Invalid path {dist_mask_path} not containing .npy files'
            if not self.fps:
                self.dist_mask_files = sort_list(glob(f'{dist_mask_path}/*_max_dist.npy'))
            else:
                self.dist_mask_files = sort_list(glob(f'{dist_mask_path}/*_max_dist_fps.npy'))
    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=os.path.join(self.data_root, 'diffusion'))

        # get geodesic distance matrix
        if self.return_dist:
            mat = sio.loadmat(self.dist_files[index])
            item['dist'] = torch.from_numpy(mat['D']).float()
        
        if self.return_mask:
            mask = np.load(self.dist_mask_files[index])
            item['mask'] = torch.from_numpy(mask).float()


        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['corr'] = torch.from_numpy(corr).long()

            mask = np.loadtxt(self.mask_files[index], dtype=np.int32)  # minus 1 to start from 0
            item['mask'] = torch.from_numpy(mask).long().bool()
        return item

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False):
        super(SingleFaustDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs,
                                                 return_corr, return_dist)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        # assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:8]
            if self.corr_files:
                self.corr_files = self.corr_files[:8]
            if self.dist_files:
                self.dist_files = self.dist_files[:8]
            self._size = 8
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[8:]
            if self.corr_files:
                self.corr_files = self.corr_files[8:]
            if self.dist_files:
                self.dist_files = self.dist_files[8:]
            self._size = 2

class SinglePartialFaustDataset(SinglePartialShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, 
                 return_mask = False, fps = False):
        super(SinglePartialFaustDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs,
                                                 return_corr, return_dist, return_mask, fps)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
                self.mask_files = self.mask_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            if self.dist_mask_files:
                self.dist_mask_files = self.dist_mask_files[:80]
            self._size = 80
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
                self.mask_files = self.mask_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            if self.dist_mask_files:
                self.dist_mask_files = self.dist_mask_files[80:]
            self._size = 20


class PairPartialShapeDataset(Dataset):
    def __init__(self, full_dataset, part_dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(full_dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(full_dataset)}'
        assert isinstance(part_dataset, SinglePartialShapeDataset), f'Invalid input data type of dataset: {type(full_dataset)}'
        self.full_dataset = full_dataset
        self.part_dataset = part_dataset
        self.combinations = list(product(range(len(full_dataset)),range(len(part_dataset))))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]
        item = dict()
        item['first'] = self.full_dataset[first_index]
        item['second'] = self.part_dataset[second_index]
        if self.part_dataset.return_corr:
            mask = item['second']['mask']
            evecs_1, evecs_2 = item['first']['evecs'], item['second']['evecs']
            evecs_2_a = evecs_2[item['second']['corr']]
            item['first']['corr'] = item['first']['corr'][mask]
            evecs_1_a = evecs_1[item['first']['corr']]
            item['first']['c_gt'] = torch.linalg.lstsq(evecs_2_a, evecs_1_a)[0][:evecs_1_a.size(-1)]
        return item


    def __len__(self):
        return len(self.combinations)



@DATASET_REGISTRY.register()
class PairPartialFaustDataset(PairPartialShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, 
                 return_mask = False, fps = False):
        full_dataset = SingleFaustDataset(data_root, phase, return_faces,
                                     return_evecs, num_evecs,
                                     return_corr, return_dist)
        part_dataset = SinglePartialFaustDataset(data_root, phase, return_faces,
                                     return_evecs, num_evecs,
                                     return_corr, return_dist, 
                                     return_mask, fps)
        super(PairPartialFaustDataset, self).__init__(full_dataset, part_dataset)


@DATASET_REGISTRY.register()
class PairShrec16geoDataset(Dataset):
    """
    Pair SHREC16 Dataset
    """
    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 data_root,
                 categories=None,
                 cut_type='cuts', return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, return_dist=False, 
                 is_train= True, distance_folder = "/home/amit/Data_GeomFmaps/TOSCA",
                 return_mask= False, fps = False, no_boundary=False):
        assert cut_type in ['cuts', 'holes'], f'Unrecognized cut type: {cut_type}'
        self.distance_folder = distance_folder
        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)
        self.cut_type = cut_type

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs
        self.return_mask = return_mask
        self.fps = fps
        self.no_boundary = no_boundary

        # full shape files
        self.full_off_files = dict()
        self.full_dist_files = dict()

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()
        self.partial_dist_files = dict()
        self.mask_files = dict()

        # load full shape files
        off_path = os.path.join(data_root, 'null', 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files'
        for cat in self.categories:
            off_file = os.path.join(off_path, f'{cat}.off')
            assert os.path.isfile(off_file)
            self.full_off_files[cat] = off_file



        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, cut_type, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files.'
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f'*{cat}*.off')))
            assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files
            self._size += len(partial_off_files)
        
        if return_dist:
            null_dist_path = os.path.join(self.distance_folder, f'dist_null')
            assert os.path.isdir(null_dist_path), f'Invalid path {null_dist_path} without .mat files'
            for cat in self.categories:
                dist_file = os.path.join(null_dist_path, f'cuts-{cat}.mat')
                assert os.path.isfile(dist_file)
                self.full_dist_files[cat] = dist_file

            dist_path = os.path.join(self.distance_folder, f'train_{cut_type}_shapes','distance_matrix')
            if not is_train:
                dist_path = os.path.join(self.distance_folder, f'test_{cut_type}_shapes','distance_matrix')
            for cat in self.categories:
                partial_dist_files = []
                mask_files = []
                for file_name in self.partial_off_files[cat]:
                    partial_dist_files.append(os.path.join(dist_path,f'{Path(file_name).stem.replace("_", "-")}.mat'))
                    assert os.path.exists(partial_dist_files[-1])
                    if self.return_mask:
                        if self.fps:
                            mask_files.append(os.path.join(dist_path,f'{Path(file_name).stem.replace("_", "-")}_max_dist_fps.npy'))
                        elif self.no_boundary:
                            mask_files.append(os.path.join(dist_path,f'{Path(file_name).stem.replace("_", "-")}_no_boundary.npy'))
                        else:
                            mask_files.append(os.path.join(dist_path,f'{Path(file_name).stem.replace("_", "-")}_max_dist.npy'))
                        assert os.path.exists(mask_files[-1])
                self.partial_dist_files[cat] = partial_dist_files
                if self.return_mask:
                    self.mask_files[cat] = mask_files
        
        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, cut_type, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} without .vts files.'
            for cat in self.categories:
                partial_corr_files = sorted(glob(os.path.join(corr_path, f'*{cat}*.vts')))
                assert len(partial_corr_files) == len(self.partial_off_files[cat])
                self.partial_corr_files[cat] = partial_corr_files

    def _get_category(self, index):
        assert index < len(self)
        size = 0
        for cat in self.categories:
            if index < size + len(self.partial_off_files[cat]):
                return cat, index - size
            else:
                size += len(self.partial_off_files[cat])

    def __getitem__(self, index):
        # get category
        cat, index = self._get_category(index)

        # get full shape
        full_data = dict()
        # get vertices
        off_file = self.full_off_files[cat]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        full_data['name'] = basename
        verts, faces = read_shape(off_file)
        full_data['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            full_data['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            full_data = get_spectral_ops(full_data, self.num_evecs, cache_dir=os.path.join(self.data_root, 'null',
                                                                                           'diffusion'))

        # get geodesic distance matrix
        if self.return_dist:
            dist_file = self.full_dist_files[cat]
            mat = sio.loadmat(dist_file)
            full_data['dist'] = torch.from_numpy(mat['D']).float()

        # get partial shape
        partial_data = dict()
        # get vertices
        off_file = self.partial_off_files[cat][index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data['name'] = basename
        partial_data['full_name'] = off_file
        verts, faces = read_shape(off_file)
        partial_data['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            partial_data['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            partial_data = get_spectral_ops(partial_data, self.num_evecs,
                                            cache_dir=os.path.join(self.data_root, self.cut_type, 'diffusion'))
        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.partial_corr_files[cat][index], dtype=np.int32) - 1
            full_data['corr'] = torch.from_numpy(corr).long()
            partial_data['corr'] = torch.arange(0, len(corr)).long()
        
        if self.return_dist:
            dist_file = self.partial_dist_files[cat][index]
            mat = sio.loadmat(dist_file)
            partial_data['dist'] = torch.from_numpy(mat['D']).float()
            if self.return_mask:
                mask_file = self.mask_files[cat][index]
                mask = np.load(mask_file,allow_pickle=True)
                partial_data['mask'] = torch.from_numpy(mask).float()

        return {'first': full_data, 'second': partial_data}

    def __len__(self):
        return self._size
