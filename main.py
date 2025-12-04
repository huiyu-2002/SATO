import yaml
import shutil
import os
from datetime import datetime
from torch.utils.data import DataLoader
from dataset.Dataset import VTKDataset, SATO_Dataset, sato_collate_fn
from models.SATO import Model
from Train import train
from utils import *

if __name__ == '__main__':
    # load config file
    with open('configs/SATO.yml', 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create results directory and save config file
    now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    results_dir = os.path.join('results', now_str)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2('configs/SATO.yml', os.path.join(results_dir, 'SATO.yml'))

    # get data
    Dataset = VTKDataset()
    train_data_lst, test_data_lst, val_data_lst, mean_data, std_data = Dataset.get_data_dict(config.data.directory)

    # Create DataLoaders
    train_dataset = SATO_Dataset(train_data_lst, config, is_train=True)
    test_dataset = SATO_Dataset(test_data_lst, config, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sato_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=sato_collate_fn)

    # build model
    model = Model(space_dim=config.model.input_dim,
                  n_layers=config.model.depth, 
                  n_hidden=config.model.hidden_dim,
                  fun_dim=0,
                  n_head=config.model.num_heads,
                  mlp_ratio=config.model.mlp_ratio,
                  slice_num=config.model.num_slices,
                  patch_size=config.model.patch_size,
                  shift=config.model.shift,
                  n_iter=config.model.n_iter).to(device)
    print(f'count_params: {count_params(model)}')

    # train model
    train(config, model, train_loader, test_loader, mean_data, std_data, device, results_dir)
