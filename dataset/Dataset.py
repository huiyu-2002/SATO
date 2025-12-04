import os
import vtk
import pickle
import torch
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from dataset.serialize import Point

from torch.utils.data import Dataset

def pos_to_order_inverse_index(pos, tensor=False):
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    data_dict = dict(coord=pos, grid_size = torch.tensor([0.05,0.05,0.05]))
    B,N,C = data_dict['coord'].shape
    data_dict['batch'] = torch.arange(B).repeat_interleave(N).cuda()
    data_dict['coord'] = data_dict['coord'].view(B*N,C).cuda()
    point = Point(data_dict)
    point.serialization(order=["z", "z-trans", "hilbert", "hilbert-trans"], shuffle_orders=False)
    if torch == False:
        order = point['serialized_order'][None, ...].cpu().numpy()
        inverse = point['serialized_inverse'][None, ...].cpu().numpy()
    else:
        order = point['serialized_order'][None, ...]
        inverse = point['serialized_inverse'][None, ...]      
    return order, inverse


def sato_collate_fn(batch):
    lengths = [item['x'].shape[0] for item in batch]
    min_len = min(lengths)
    
    batch_x = []
    batch_y = []
    
    for item in batch:
        x = item['x']
        y = item['y']
        
        if x.shape[0] > min_len:
            # Randomly sample min_len indices
            idx = torch.randperm(x.shape[0])[:min_len]
            x = x[idx]
            y = y[idx]
            
        batch_x.append(x)
        batch_y.append(y)
        
    return {
        'x': torch.stack(batch_x),
        'y': torch.stack(batch_y)
    }


class SATO_Dataset(Dataset):
    def __init__(self, data_list, config=None, is_train=True):
        self.data_list = data_list
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        x = data['Surface_data']['Surface_points']
        y = data['Surface_data']['Surface_pressure']

        # Move downsample logic here to allow parallel processing by DataLoader workers
        if self.config is not None and hasattr(self.config.model, 'down_sample'):
            # Use numpy random generation which is process-safe in workers
            num_points = x.shape[0]
            sample_size = int(num_points * self.config.model.down_sample)
            
            # Generate indices (on CPU)
            sampled_indices = np.random.choice(num_points, sample_size, replace=False)
            
            x = x[sampled_indices]
            y = y[sampled_indices]

        return {'x': x, 'y': y}


class VTKDataset():
    def __init__(self):
        pass

    def get_all_file_paths(self, directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    # generate data dictionary
    def get_data_dict(self, directory):
        # read all SurfacePressure file names
        SurfacePressure_file_paths = self.get_all_file_paths(os.path.join(directory, 'SurfacePressure', 'VTK'))

        # load train/test/val index
        with open(os.path.join(directory, 'train_val_test_splits/train_design_ids.txt'), 'r') as file:
            train_index = [line.strip()[-4:] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/test_design_ids.txt'), 'r') as file:
            test_index = [line.strip()[-4:] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/val_design_ids.txt'), 'r') as file:
            val_index = [line.strip()[-4:] for line in file]

        with open(os.path.join(directory, 'norm', 'mean.pkl'), 'rb') as f:
            mean_data = pickle.load(f)
        with open(os.path.join(directory, 'norm', 'std.pkl'), 'rb') as f:
            std_data = pickle.load(f)

        train_data_lst, test_data_lst, val_data_lst = [], [], []
        for file_path in SurfacePressure_file_paths:
            index = file_path[-8:-4]
            Surface_points = np.load(os.path.join(directory, 'SurfacePressure', 'points', f'points_{index}.npy'))
            Surface_pressure = np.load(os.path.join(directory, 'SurfacePressure', 'pressure', f'pressure_{index}.npy'))

            Surface_points = torch.Tensor(Surface_points).float()
            Surface_pressure = torch.Tensor(Surface_pressure).float()

            Surface_data = {
                'Surface_points': Surface_points,
                'Surface_pressure': Surface_pressure
            }

            data = {'index': index, 'Surface_data': Surface_data}

            if index in train_index:
                train_data_lst.append(data)
            elif index in test_index:
                test_data_lst.append(data)
            else:
                val_data_lst.append(data)

        return train_data_lst, test_data_lst, val_data_lst, mean_data, std_data