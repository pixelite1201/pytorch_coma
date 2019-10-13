import os
import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from psbody.mesh import Mesh
from utils import get_vert_connectivity

class ScanDataset(Dataset):

    def __init__(self, root_dir, train=True, split='sliced', split_term='sliced',
                 transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.pre_transform = pre_transform
        self.scan_dir = os.path.join(self.root_dir, 'scan_raw')
        self.mesh_dir = self.scan_dir.replace('scan', 'mesh')
        self.all_scans = glob.glob(self.scan_dir + '/*/*/*.obj')
        self.train_scan_files = []
        self.test_scan_files = []
        for idx, scan_path in enumerate(self.all_scans):
            mesh_path = scan_path.replace('scan', 'mesh').replace('obj', 'ply')
            if os.path.exists(mesh_path):
                if split == 'sliced':
                    if idx % 100 < 10:
                        self.test_scan_files.append(scan_path)
                    else:
                        self.train_scan_files.append(scan_path)
                elif split == 'expression':
                    if mesh_path.split('/')[-2] == split_term:
                            self.test_scan_files.append(scan_path)
                    else:
                            self.train_scan_files.append(scan_path)

                elif split == 'identity':
                    if mesh_path.split('/')[-3]== split_term:
                            self.test_scan_files.append(scan_path)
                    else:
                            self.train_scan_files.append(scan_path)
                else:
                    raise Exception('sliced, expression and identity are the only supported split terms')

        self.train_processed_files = [sf.replace('scan', 'processed_scan_'+split_term+'_train').replace('.obj', '.pt')
                                      for sf in self.train_scan_files]
        self.test_processed_files = [sf.replace('scan', 'processed_scan_'+split_term+'_test').replace('.obj', '.pt')
                                     for sf in self.test_scan_files]
        self.norm_file = os.path.join(self.root_dir, split_term + '_norm.pt')

        super(ScanDataset, self).__init__(root_dir, transform, pre_transform)
        norm_path = self.processed_paths[-1]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']

    @property
    def raw_file_names(self):
        return self.all_scans

    @property
    def processed_file_names(self):
        return self.train_processed_files + self.test_processed_files + [self.norm_file]

    def process_helper(self, data_files, processed_files):
        mesh_vertices=[]
        for idx, scans in tqdm(enumerate(data_files)):
            if os.path.exists(processed_files[idx]):
                continue
            scan = Mesh(filename=data_files[idx])
            mesh_file = data_files[idx].replace('scan', 'mesh').replace('.obj', '.ply')
            mesh = Mesh(filename=mesh_file)
            #Todo Instead of storing vertices for calculating mean and std, calculate it on the go
            mesh_vertices.append(mesh.v)
            scan_verts = torch.Tensor(scan.v)
            adjacency = get_vert_connectivity(scan.v, scan.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            mesh_verts = torch.Tensor(mesh.v)
            data = Data(x=scan_verts, y=mesh_verts, edge_index=edge_index)
            if not os.path.exists(os.path.dirname(processed_files[idx])):
                os.makedirs(os.path.dirname(processed_files[idx]))
            torch.save(data, processed_files[idx])
        return mesh_vertices

    def process(self):
        train_mesh_vertices = self.process_helper(self.train_scan_files, self.train_processed_files)
        test_mesh_vertices = self.process_helper(self.test_scan_files, self.test_processed_files)
        # Store the mean and std for the meshes
        mean_train = torch.from_numpy(np.mean(train_mesh_vertices, axis=0)).float()
        std_train = torch.from_numpy(np.std(train_mesh_vertices, axis=0)).float()
        norm_dict = {'mean': mean_train, 'std': std_train}
        torch.save(norm_dict, self.processed_paths[-1])

    def __len__(self):
        if self.train:
            return len(self.train_scan_files)
        else:
            return len(self.test_scan_files)

    def __getitem__(self, idx):
        if self.train:
            data = torch.load(self.train_processed_files[idx])
        else:
            data = torch.load(self.test_processed_files[idx])

        if self.transform:
            data = self.transform(data)
        return data


def prepare_sliced_dataset(path):

    print("Preparing sliced dataset")
    ScanDataset(path)


def prepare_expression_dataset(path):

    print("Preparing expression dataset")
    test_exps = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up', 'mouth_down',
                 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
    for exp in test_exps:
        ScanDataset(path, split='expression', split_term=exp)

def prepare_identity_dataset(path):

    print("Preparing identity dataset")
    test_ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170811_03274_TA',
                'FaceTalk_170904_00128_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170913_03279_TA',
                'FaceTalk_170728_03272_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170811_03275_TA',
                'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170915_00223_TA']

    for ids in test_ids:
        ScanDataset(path, split='identity', split_term=ids)


if __name__ == '__main__':
    prepare_identity_dataset('/is/ps2/ppatel/py_coma_data/alignment_alldata/')
