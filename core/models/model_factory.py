import os
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from core.models import predrnn_v2, action_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.scaler = GradScaler()
        networks_map = {
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.network.half()  # Convert model to float16
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device).half()
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device).half()
        self.optimizer.zero_grad()

        with autocast():
            next_frames, loss = self.network(frames_tensor, mask_tensor)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device).half()
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device).half()

        with autocast():
            next_frames, _ = self.network(frames_tensor, mask_tensor)

        return next_frames.detach().cpu().numpy()
