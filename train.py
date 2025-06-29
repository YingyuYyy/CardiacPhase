"""
This script demonstrates how to train the latent motion model for echocardiography recosntruction.
Author: Yingyu Yang 
"""
from models import MotionAnatomy2DAE
from EchoDynamicDataset import EchoDynamicDatasetFromCache 
import pandas as pd
import kornia.augmentation as K
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
import torch
import yaml
import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
L.seed_everything(666)


def parse_args():
    parser = argparse.ArgumentParser(description="Load config from YAML")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    # Accessing elements from the config
    dmotion = config['dmotion']
    zdim = config['zdim']
    lr = config['lr']
    epochs = config['epochs']
    frames = config['frames']
    t_step = config['t_step']
    augment = config['augment']
    transformation = config['transformation']
    savefolder = config['savefolder']
    name = config['name']
    data_dir = config['data_dir']
    cache_dir = config['cache_dir']
    csv_path = config['csv_path']

    # augmentation 
    if augment:
        device = torch.device('cuda:0')
    else:
        device = None
    # Define augmentations using Kornia
    batch_same = True
    augmentation = nn.Sequential(
        K.RandomAffine(60, translate=0.1, scale=(0.85,1.15), same_on_batch=batch_same, align_corners=False, 
                       padding_mode=2, p=0.5), # Spatial transformation 
        K.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=batch_same),
        K.RandomBrightness(brightness=(0.8, 1.2), same_on_batch=batch_same),  # Brightness adjustment
        K.RandomContrast(contrast=(0.8, 1.2), same_on_batch=batch_same),  # Contrast adjustment
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.25, 1.5), same_on_batch=batch_same),  # Blurring
        K.RandomGaussianNoise(mean=0.0, std=0.01, same_on_batch=batch_same),  # Noise addition
    ).to(device)  # Move augmentations to GPU

    # log folder setting 
    version = 'lr{}_e{}_tstep{}_dmotion{}_zdim{}'.format(str(lr).split('.')[-1], epochs, t_step, dmotion, zdim) 
    if augment:
        version += '_aug'
        transformation = augmentation
    logger = TensorBoardLogger(save_dir = savefolder, name=name, version=version)

    # set data dataframe and cache 
    df = pd.read_csv(csv_path) 
    idx = np.where(df['NumberOfFrames'].values - frames * t_step > 0)[0]
    df = df.iloc[idx]
    df_train = df[df['Split']=='TRAIN']
    df_val = df[df['Split']=='VAL']

    # only perform augmentation for training dataset 
    dataset_train = EchoDynamicDatasetFromCache(data_dir, df_train, split='train', frames=frames, t_step=t_step, 
                                       cache_dir=cache_dir,
                                       device = device, augment=transformation
                                       )
    dataset_valid = EchoDynamicDatasetFromCache(data_dir, df_val, split='val', frames=frames, t_step=t_step, 
                                       cache_dir=cache_dir,
                                       device = device
                                       )
    
    print('Number of training examples: {}'.format(len(dataset_train)))
    print('Number of validation examples: {}'.format(len(dataset_valid)))

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, drop_last=False)
    dataloader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=False, drop_last=False)


    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=savefolder + "/"  + name + "/" + version + "/checkpoints",  # Directory to save the checkpoints
        filename="best_model",    # Filename for the checkpoint
        save_top_k=3,             # Save only the top-k models based on monitor
        monitor="valid_MSE",       # Metric to monitor
        mode="min"                # Save the model with the minimum validation loss
    )

    regular_checkpoint = ModelCheckpoint(
        dirpath=savefolder + "/"  + name + "/" + version + "/checkpoints", 
        filename="epoch-{epoch:02d}",
        every_n_epochs=50,  # Save every 50 epochs
        save_top_k=-1,  # Keep all checkpoints
        verbose=True
    )

    # set model and start training 
    model = MotionAnatomy2DAE(zdim=zdim, motion_dim=dmotion, lr=lr)
    trainer = L.Trainer(accelerator='gpu', devices=[0], max_epochs=epochs, default_root_dir=savefolder, logger=logger,
                        log_every_n_steps=5, callbacks=[checkpoint_callback, regular_checkpoint])
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders = dataloader_valid)
