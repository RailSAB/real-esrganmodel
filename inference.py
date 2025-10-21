import os
import argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from degradate import apply_degradation
from RealESRGAN.rrdbnet_arch import RRDBNet

def make_divisible_crop(img, scale):
    w, h = img.size
    w2 = (w // scale) * scale
    h2 = (h // scale) * scale
    if w2 == 0 or h2 == 0:
        return img
    left = (w - w2) // 2
    top = (h - h2) // 2
    return img.crop((left, top, left + w2, top + h2))

def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'params' in ckpt:
            state = ckpt['params']
        elif 'params_ema' in ckpt:
            state = ckpt['params_ema']
        else:
            # may be whole state_dict
            state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)

def tensor_to_pil(img_t):
    # img_t: torch.Tensor CxHxW in [-1,1]
    arr = img_t.clamp(-1, 1).add(1).div(2).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(arr)

def run(args):
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)
    net = net.to(device)
    load_checkpoint_to_model(net, args.weights, device)
    net.eval()

    files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for fp in tqdm(files, desc='images'):
        name = os.path.splitext(os.path.basename(fp))[0]
        hr = Image.open(fp).convert('RGB')
        hr = make_divisible_crop(hr, args.scale)

        # create LR by downsampling HR then degrading
        w, h = hr.size
        lr_size = (w // args.scale, h // args.scale)
        lr = hr.resize(lr_size, Image.BICUBIC)
        try:
            degraded_lr = apply_degradation(lr)
        except Exception:
            degraded_lr = lr

        # prepare tensor normalized to [-1,1]
        lr_arr = np.array(degraded_lr).astype(np.float32) / 255.0
        lr_arr = lr_arr * 2.0 - 1.0
        lr_t = torch.from_numpy(lr_arr).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(lr_t)
        pred = pred.squeeze(0)
        sr_pil = tensor_to_pil(pred)
        # save SR and LR (and optionally original HR)
        sr_pil.save(os.path.join(args.out_dir, f"{name}_sr.png"))
        degraded_lr.save(os.path.join(args.out_dir, f"{name}_lr.png"))
        hr.save(os.path.join(args.out_dir, f"{name}_hr.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='path to weights .pth (e.g. /kaggle/working/weights/RealESRGAN_x4.pth)')
    p.add_argument('--input', required=True, help='input HR folder (e.g. /kaggle/input/realimages)')
    p.add_argument('--out-dir', default='/kaggle/working/results')
    p.add_argument('--scale', type=int, default=4, choices=[2,4,8])
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    run(args)