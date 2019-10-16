import argparse
import os
import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from psbody.mesh import Mesh
from utils import get_vert_connectivity


class AmassDataset(Dataset):

    def __init__(self, root_dir, raw_data_dir, bm_path, template_path, num_betas=16,
                 permute=False, train=True, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_tranform = pre_transform
        self.num_betas = num_betas
        self.train = train
        self.permute = permute
        self.template_mesh = Mesh(filename=template_path)
        self.bm = BodyModel(bm_path=bm_path, num_betas=self.num_betas, batch_size=1, model_type='smplh')
        self.faces = c2c(self.bm.f)
        self.data_file = glob.glob(os.path.join(raw_data_dir, '*.pt'))
        self.ds = {}
        for data_fname in self.data_file:
            k = os.path.basename(data_fname).replace('.pt', '')
            self.ds[k] = torch.load(data_fname)
        self.train_processed_files = []
        self.test_processed_files = []
        for n in range(len(self.ds['trans'])):
            if n % 100 < 10:
                self.test_processed_files.append(os.path.join(self.root_dir, 'test_pose.'+format(n, '05')+'.pt'))
            else:
                self.train_processed_files.append(os.path.join(self.root_dir, 'train_pose.'+format(n, '05')+'.pt'))

        super(AmassDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        return self.train_processed_files + self.test_processed_files


    def get_pose_shape(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]
        return data

    def permute_mesh(self, input_vertices):
        idx = np.random.permutation(len(input_vertices))
        mapping = {a: b for a, b in zip(idx, np.arange(len(input_vertices)))}
        output_vertices = input_vertices[idx]
        output_faces = self.faces.copy()
        for i, f in enumerate(self.faces):
            for j, v in enumerate(f):
                output_faces[i][j] = mapping[v]
        return output_vertices, output_faces

    def process(self):
        test_idx = 0
        train_idx = 0
        for idx in tqdm(range(len(self.ds['trans']))):
            bdata = self.get_pose_shape(idx)
            body_vertices = self.bm.forward(pose_body=bdata['pose_body'].unsqueeze(0), betas=bdata['betas'].unsqueeze(0)).v
            body_vertices_numpy = body_vertices.detach().numpy()[0]
            mesh_verts_y = torch.Tensor(body_vertices_numpy)
            if self.permute:
                vertices, faces = self.permute_mesh(body_vertices_numpy)
            else:
                vertices, faces = body_vertices_numpy, self.faces

            mesh_verts_x = torch.Tensor(vertices)
            adjacency = get_vert_connectivity(mesh_verts_x, faces).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts_x, y=mesh_verts_y, edge_index=edge_index)

            #Since processed_files are named in a way that every 100th data point will be stored as
            if idx % 100 < 10:
                torch.save(data, self.test_processed_files[test_idx])
                test_idx = test_idx+1
            else:
                torch.save(data, self.train_processed_files[train_idx])
                train_idx = train_idx+1

    def __len__(self):
        if self.train:
            return len(self.train_processed_files)
        else:
            return len(self.test_processed_files)

    def __getitem__(self, idx):
        if self.train:
            data = torch.load(self.train_processed_files[idx])
        else:
            data = torch.load(self.test_processed_files[idx])

        if self.transform:
            data = self.transform(data)

        return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Amass Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-r', '--root_dir', default='sliced', help='output files will be stored here')
    parser.add_argument('-d', '--data_dir', help='path where the amass raw data is stored')
    parser.add_argument('-bm', '--bm_path', help='path where the smpl body model is stored')
    parser.add_argument('-t', '--template', default='./template/template_body.obj',
                        help='path where the body template file is stored')

    args = parser.parse_args()
    dataset = AmassDataset(args.root_dir, args.data_dir, args.bm_path, args.template)

    # Uncomment following lines to visualize the dataset

    # from psbody.mesh import Mesh
    # from torch_geometric.data import DataLoader
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    # template_mesh = Mesh(filename=args.template)
    # for data in train_loader:
    #     Mesh(v=data.x, f=template_mesh.f).show()
