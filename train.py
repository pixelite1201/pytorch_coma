import os
import logging
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch import tensor

from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model import Coma
from transform import Normalize


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def main(config_fname):
    if not os.path.exists(config_fname):
        print('Config not found' + config_fname)

    config = read_config(config_fname)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    output_dir = config['visual_output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['eval']
    visualize = config['visualize']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    data_dir = config['data_dir']
    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir, dtype='train', split='expression', split_term='cheeks_in', pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split='expression', split_term='cheeks_in', pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.to(device)

    if eval_flag:
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        train_loss = train(coma, train_loader, len(dataset), optimizer, device)
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device, visualize=visualize)

        print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    count=0
    for data in train_loader:
        count+=1
        data = data.to(device)
        optimizer.zero_grad()
        out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset


def evaluate(coma, output_dir, test_loader, dataset, template_mesh, device, visualize=False):
    coma.eval()
    total_loss = 0
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()

        if visualize and i % 100 == 0:
            meshviewer = MeshViewers(shape=(1, 2))
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            result_mesh = Mesh(v=save_out, f=template_mesh.f)
            expected_mesh = Mesh(v=expected_out, f=template_mesh.f)
            meshviewer[0][0].set_static_meshes([result_mesh])
            meshviewer[0][1].set_static_meshes([expected_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(output_dir, 'file'+str(i)+'.png'), blocking=True)

    return total_loss/len(dataset)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pkg_path, _ = os.path.split(os.path.realpath(__file__))
        config_fname = os.path.join(pkg_path, 'default.cfg')
    else:
        config_fname = str(sys.argv[1])
    main(config_fname)
