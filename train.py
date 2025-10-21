import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from degradate import apply_degradation
from RealESRGAN.rrdbnet_arch import RRDBNet


class HRDataset(Dataset):
    def __init__(self, hr_folder, scale, patch_size=128):
        self.files = sorted(glob(os.path.join(hr_folder, '*')))
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        w, h = hr.size
        ph = self.patch_size * self.scale
        if w < ph or h < ph:
            hr = hr.resize((max(ph, w), max(ph, h)), Image.BICUBIC)
            w, h = hr.size
        x = np.random.randint(0, w - ph + 1)
        y = np.random.randint(0, h - ph + 1)
        hr = hr.crop((x, y, x + ph, y + ph))

        lr_pil = hr.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        try:
            degraded_lr = apply_degradation(lr_pil)
        except Exception:
            degraded_lr = lr_pil

        hr_arr = np.array(hr).astype(np.float32) / 255.0
        lr_arr = np.array(degraded_lr).astype(np.float32) / 255.0
        hr_arr = hr_arr * 2.0 - 1.0
        lr_arr = lr_arr * 2.0 - 1.0
        hr = torch.from_numpy(hr_arr).permute(2,0,1)
        lr = torch.from_numpy(lr_arr).permute(2,0,1)
        return lr, hr
    
def save_checkpoint(model, out_path):
    state = {'params': model.state_dict()}
    torch.save(state, out_path)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=args.scale
    ).to(device)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        # accept various key names
        if 'params' in ckpt:
            net.load_state_dict(ckpt['params'], strict=False)
        elif 'params_ema' in ckpt:
            net.load_state_dict(ckpt['params_ema'], strict=False)
        else:
            net.load_state_dict(ckpt, strict=False)

    dataset = HRDataset(args.data, scale=args.scale, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.L1Loss()
    optimizer = Adam(net.parameters(), lr=args.lr)

    net.train()
    global_step = 0
    for epoch in range(1, args.epochs+1):
        running_loss = 0.0
        for i, (lr, hr) in enumerate(loader):
            lr = lr.to(device)    # [B,3,H,W]
            hr = hr.to(device)    # [B,3,H*scale,W*scale]

            pred = net(lr)
            loss = criterion(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"Epoch {epoch} Step {global_step} Loss {avg:.6f}")
                running_loss = 0.0

        # save per epoch
        checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pth")
        os.makedirs(args.out_dir, exist_ok=True)
        save_checkpoint(net, checkpoint_path)
        print("Saved", checkpoint_path)

    # final save
    final_path = os.path.join(args.out_dir, args.out_name)
    save_checkpoint(net, final_path)
    print("Training finished. Final weights saved to", final_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to HR images folder')
    p.add_argument('--scale', type=int, default=4, choices=[2,4,8])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patch-size', type=int, default=128)
    p.add_argument('--out-dir', default='weights')
    p.add_argument('--out-name', default='RealESRGAN_x4.pth')
    p.add_argument('--pretrained', default=None, help='path to pretrained weights (optional)')
    p.add_argument('--log-interval', type=int, default=50)
    args = p.parse_args()
    train(args)