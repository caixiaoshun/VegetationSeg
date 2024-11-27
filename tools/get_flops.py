import argparse
from mmseg.apis import init_model
from thop import profile
import torch
from thop import clever_format
from vegseg import models

def get_args():
    parser = argparse.ArgumentParser(description="Get the FLOPs of a segmentor")
    parser.add_argument("config", help="train config file path")

    args = parser.parse_args()
    return args.config

def main():
    config = get_args()
    model = init_model(config=config,device="cpu")
    model.eval()
    fake_data = torch.randn((1,3,256,256))
    macs, params = profile(model, inputs=(fake_data, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs:{macs},params:{params}")

if __name__ == "__main__":
    main()