import torch

CPU = torch.device('cpu')
chk = torch.load('./coco_train/coco_prefix-000.pt', map_location=CPU)
print(chk)