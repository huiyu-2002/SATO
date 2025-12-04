import os
import torch
import numpy as np
from utils import LpLoss
from timeit import default_timer
from dataset.Dataset import pos_to_order_inverse_index

def train(config, model, train_loader, test_loader, mean_data, std_data, device, results_dir):
    # set loss function, optimizer and scheduler
    L2_fn = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.scheduler_step, gamma=config.training.scheduler_gamma)
    
    for key in mean_data:
        for tensor in mean_data[f'{key}']:
            mean_data[f'{key}'][f'{tensor}'] = mean_data[f'{key}'][f'{tensor}'].cuda()
    for key in std_data:
        for tensor in std_data[f'{key}']:        
            std_data[f'{key}'][f'{tensor}'] = std_data[f'{key}'][f'{tensor}'].cuda()

    t0 = default_timer()
    for ep in range(config.training.epochs):
        model.train()
        t1 = default_timer()
        # Changed from enumerate(train_data) to enumerate(train_loader)
        for n_iter, data in enumerate(train_loader):
            x = data['x'].to(device)
            y = data['y'].to(device)
            
            order, inverse = pos_to_order_inverse_index(x, tensor=True)
            
            # Normalize
            x = (x - mean_data['Surface_mean']['Surface_points']) / std_data['Surface_std']['Surface_points']
            
            # Model expects [B, N, C], which x already is.
            y_hat = model(x, order, inverse).reshape(-1)
            y_hat = y_hat * std_data['Surface_std']['Surface_pressure'] + mean_data['Surface_mean']['Surface_pressure']
            
            train_loss = L2_fn(y_hat[None, ...], y[None, ...].reshape(1, -1)) # y shape match

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            if n_iter % 100 == 0:
                t2 = default_timer()
                log_str = f'{ep} {n_iter} {(t2-t1):.2f} train_loss:{train_loss.item():.4f}'
                print(log_str)
                with open(os.path.join(results_dir, 'train_iter_log.txt'), 'a', encoding='utf-8') as f:
                    f.write(log_str + '\n')
        
        scheduler.step()

        with torch.no_grad():
            torch.cuda.empty_cache()
            test_losses = []
            for data in test_loader:
                x = data['x'].to(device)
                y = data['y'].to(device)
                
                # x is already sampled by Dataset logic if configured
                
                order, inverse = pos_to_order_inverse_index(x, tensor=True)

                x = (x - mean_data['Surface_mean']['Surface_points']) / std_data['Surface_std']['Surface_points']
                y_hat = model(x, order, inverse).reshape(-1)
                y_hat = y_hat * std_data['Surface_std']['Surface_pressure'] + mean_data['Surface_mean']['Surface_pressure']
                
                test_loss = L2_fn(y_hat[None, ...], y[None, ...].reshape(1, -1))
                test_losses.append(test_loss.item())

        test_loss = np.mean(test_losses)
        t2 = default_timer()
        log_str = f'{ep} {(t2-t1):.2f} train_loss:{train_loss.item():.4f} test_loss:{test_loss.item():.4f}'
        print(log_str)
        with open(os.path.join(results_dir, 'train_epoch_log.txt'), 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')

        torch.save(model.state_dict(), os.path.join(results_dir, 'checkpoint_latest.pth'))

    torch.save(model.state_dict(), os.path.join(results_dir, 'checkpoint_latest.pth'))
    
    t2 = default_timer()
    time = t2 - t0
    hour = int(time // (60**2))
    time = time - (hour*(60**2))
    minute = int(time // 60)
    second = int(time - (minute*60))
    print(f'total_time: {hour}h_{minute}m_{second}s')
    with open(os.path.join(results_dir, 'train_epoch_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f'total_time: {hour}h_{minute}m_{second}s\n')