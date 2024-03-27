# -*- coding: utf-8 -*-

import torch
import argparse
from collections import OrderedDict

def change_model(args):
    fgd_model = torch.load(args.fgd_path)
    all_name = []
    for name, v in fgd_model["state_dict"].items():
        if name.startswith("student."):
            all_name.append((name[8:], v))
        else:
            continue
    state_dict = OrderedDict(all_name)
    fgd_model['state_dict'] = state_dict
    fgd_model.pop('optimizer')
    torch.save(fgd_model, args.output_path) 

           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--lgd_path', type=str, default='--',
                        metavar='N',help='lgd_model path')
    parser.add_argument('--output_path', type=str, default='--',metavar='N',
                        help = 'pair path')
    args = parser.parse_args()
    change_model(args)
