"""
This script contains the latent motion model for cardiac reconstruction. 
Author: Yingyu Yang 
"""
import torch 
import lightning as L
from .encoder import AEEncoder
from .decoder import AEDecoder
import os 

class MotionAnatomy2DAE(L.LightningModule):
    def __init__(self, zdim=512, motion_dim=2, lr=0.001, save_every_n_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AEEncoder(zdim, motion_dim)
        self.decoder = AEDecoder(zdim)
        self.learning_rate = lr
        self.rec_loss = torch.nn.MSELoss(reduction='none')
        self.save_every_n_epochs=save_every_n_epochs

    def forward(self, batch):
        B,T,C,H,W = batch.shape
        mean_z, frame_z, alpha = self.encoder(batch)
        mean_recon = self.decoder(mean_z).reshape(B,T,C,H,W)
        frame_recon = self.decoder(frame_z).reshape(B,T,C,H,W)
        return [mean_z, frame_z], [mean_recon, frame_recon], alpha

    def training_step(self, batch, batch_idx):
        # batch (B,T,C,H,W)
        B,T,C,H,W = batch.shape
        mean_z, frame_z, _ = self.encoder(batch)
        mean_recon = self.decoder(mean_z).reshape(B,T,C,H,W)
        frame_recon = self.decoder(frame_z).reshape(B,T,C,H,W)
        mean_loss = self.rec_loss(mean_recon, batch)
        mean_loss = mean_loss.mean(dim=(2,3,4)).sum()/T
        frame_loss = self.rec_loss(frame_recon, batch)
        frame_loss = frame_loss.mean(dim=(2,3,4)).sum()
        loss = mean_loss + frame_loss

        self.log('train_MSE', loss, prog_bar=False) # total loss 
        self.log('train_mean', mean_loss, prog_bar=False) # loss for reconstruction from mean vector 
        self.log('train_frame', frame_loss, prog_bar=False) # loss for reconstruction from frame vector 

        if batch_idx == 0:
            with torch.no_grad():
                video_recs = torch.cat((batch, mean_recon, frame_recon), dim=-2)  
                video_recs = video_recs.repeat(1,1,3,1,1)
                self.logger.experiment.add_video('train_recs', video_recs[:8], global_step=self.current_epoch,
                                                 fps=12)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        B,T,C,H,W = batch.shape
        mean_z, frame_z, _ = self.encoder(batch)
        mean_recon = self.decoder(mean_z).reshape(B,T,C,H,W)
        frame_recon = self.decoder(frame_z).reshape(B,T,C,H,W)
        mean_loss = self.rec_loss(mean_recon, batch)
        mean_loss = mean_loss.mean(dim=(2,3,4)).sum()/T
        frame_loss = self.rec_loss(frame_recon, batch)
        frame_loss = frame_loss.mean(dim=(2,3,4)).sum()
        loss = mean_loss + frame_loss

        self.log('valid_MSE', loss, prog_bar=False) # total loss 
        self.log('valid_mean', mean_loss, prog_bar=False) # loss for reconstruction from mean vector 
        self.log('valid_frame', frame_loss, prog_bar=False) # loss for reconstruction from frame vector

        if batch_idx == 0:
            with torch.no_grad():
                video_recs = torch.cat((batch, mean_recon, frame_recon), dim=-2)  
                video_recs = video_recs.repeat(1,1,3,1,1)
                self.logger.experiment.add_video('valid_recs', video_recs[:8], global_step=self.current_epoch,
                                                 fps=12)
        return loss 
    
    def on_validation_epoch_end(self):
        save_path = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(save_path, exist_ok=True)

        valid_mse = self.trainer.callback_metrics.get('valid_MSE')
        if not hasattr(self, 'best_mse'):
            self.best_mse = float('inf')
    
        if valid_mse < self.best_mse:
            self.best_mse = valid_mse
    
    def configure_optimizers(self):
        rec_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return rec_optimizer