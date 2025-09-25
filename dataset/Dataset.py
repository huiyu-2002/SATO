import os
import vtk
import pickle
import torch
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from dataset.serialize import Point

def pos_to_order_inverse_index(pos, tensor=False):
    data_dict = dict(coord=pos[None, ...], grid_size = torch.tensor([0.05,0.05,0.05]))
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