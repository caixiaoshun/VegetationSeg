import torch
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str)
    parse.add_argument("--head_path", type=str)
    parse.add_argument("--auxiliary_head_path")
    args = parse.parse_args()
    return args.weight_path, args.head_path, args.auxiliary_head_path


def main():
    weight_path, head_path,auxiliary_head_path = get_args()
    weight = torch.load(weight_path, map_location="cpu")
    state_dict = weight["state_dict"]
    head_state_dict = {}
    auxiliary_head_dict = {}
    for k, v in state_dict.items():
        if "decode_head" in k:
            head_state_dict[k] = v
        elif "auxiliary_head" in k:
            auxiliary_head_dict[k] = v
    torch.save(head_state_dict, head_path)
    torch.save(auxiliary_head_dict,auxiliary_head_path)


if __name__ == "__main__":
    # example usage:python tools/extract_head.py --weight_path work_dirs/dinov2_upernet/best_mIoU_iter_10810.pth --head_path work_dirs/dinov2_upernet/decode_head.pth --auxiliary_head_path work_dirs/dinov2_upernet/auxiliary_head.pth
    main()
