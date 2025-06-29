"""
This script demonstrates how to evalute the latent motion model for echocardiography recosntruction 
and save the latent vectors for further trajectory analysis.
Author: Yingyu Yang 
"""
import numpy as np
from time import time 
from models import MotionAnatomy2DAE
from EchoDynamicDataset import EntireEcho 
import pandas as pd
from torch.utils.data import DataLoader
import torch 
import os 
from tqdm import tqdm
import lightning as L
L.seed_everything(666)


if __name__ == '__main__':
    # dataset setup 
    #===============================================modify path here=====================================================
    data_dir = '/data/Echonet-Dynamic/Videos' # modify path 
    csv_path = '/data/Echonet-Dynamic/FileList.csv' # modify path 
    model_folder = '' # modify model folder here 
    #===============================================================================================================
    df = pd.read_csv(csv_path)
    df_train = df[df['Split']=='TRAIN']
    df_val = df[df['Split']=='VAL']
    df_test = df[df['Split']=='TEST']
    
    p = 1 # time step between frames 
    dataset_valid = EntireEcho(data_dir, df_val, size=128, period=p)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    dataset_test = EntireEcho(data_dir, df_test, size=128, period=p)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # load model 
    datasplit = 'validation' # or 'test'
    if datasplit == 'validation':
        current_loader = dataloader_valid
    else:
        current_loader = dataloader_test

    save_folder = os.path.join(model_folder, f'echonet_{datasplit}_p{p}')
    os.makedirs(save_folder, exist_ok=True)

    model = MotionAnatomy2DAE(zdim=512, motion_dim=2, lr=0.001)
    cpkt = model_folder + '/checkpoints/best_model.ckpt'
    device = torch.device('cuda')
    state_dict = torch.load(cpkt, weights_only=True, map_location=device)['state_dict']
    model.to(device)
    model.load_state_dict(state_dict)

    # save all information for further analysis
    with torch.no_grad():
        model.eval()
        k = 0
        for batch in tqdm(current_loader):
            batch = batch.to(device)
            latent, recon, alpha = model(batch)
            meanz, framez = latent 
            meanrec, framerec = recon 
            output_data = {}  
            # save latent trajectory and reconstruction for evaluation 
            output_data = {
                'alphas': alpha.detach().cpu().numpy(),
                'meanrec': meanrec[0,0,0].detach().cpu().numpy(),
                'framerec': framerec[0,:,0].detach().cpu().numpy(),
                'meanz': meanz.detach().cpu().numpy(),
                'framez': framez.detach().cpu().numpy()
            }
            np.savez(os.path.join(save_folder, f'{datasplit}{k}_output.npz'), **output_data)  
            k+=1   
            break  


    