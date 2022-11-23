#https://github.com/tcapelle/metnet_pytorch/blob/master/01_model.ipynb

import torch
# seq_len=5
# i=3
# times = (torch.eye(seq_len)[i-1]).float().unsqueeze(-1).unsqueeze(-1)
# print(f"times : {times.shape}")

def condition_time( i=0, size=(6, 4), seq_len=5):
    "create one hot encoded time image-layers, i in [1, seq_len]"
    assert i < seq_len
    times = (
        (torch.eye(seq_len, dtype=torch.long)[i])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    print(f"times : {times.shape}")
    
    ones = torch.ones(1, *size)
    print(f"ones : {ones.shape}")
    
    out = times * ones
    print(f"out : {out.shape} ")
    # print(f"out : {out.shape},{out}")
    
    
    bs =2
    out_re=out.repeat(bs, seq_len, 1, 1, 1)
    print(f"out_re : {out_re.shape} ")
    
    
    return out

# condition_time()
from metnet import MetNet
from metnet import MetNet2

import torch
import torch.nn.functional as F

model = MetNet(
        hidden_dim=32,
        forecast_steps=24,
        input_channels=16,
        output_channels=12,
        sat_channels=12,
        input_size=32,
        )
model2 = MetNet2(
        forecast_steps=8,
        input_size=64,
        num_input_timesteps=6,
        upsampler_channels=128,
        lstm_channels=32,
        encoder_channels=64,
        center_crop_size=16,
        )
# # MetNet expects original HxW to be 4x the input size
# x = torch.randn((2, 12, 16, 128, 128))
# out = []
# for lead_time in range(24):
#         out.append(model(x, lead_time))
# out = torch.stack(out, dim=1)
# # MetNet creates predictions for the center 1/4th
# y = torch.randn((2, 24, 12, 8, 8))
# print(F.mse_loss(out, y))



x = torch.randn((2, 6, 12, 256, 256))
out = []
for lead_time in range(1):
        out.append(model2(x, lead_time))
out = torch.stack(out, dim=1)
y = torch.rand((2,8,12,64,64))
print(F.mse_loss(out, y))