import torch


def show(pt_dir):
    p = torch.load(pt_dir)
    for parameter,weight in p.items():
        print(parameter, weight)


show('/data0/private/caohanwen/OpenSoCo/checkpoint/roberta-small-null/transformed_null.pt')
