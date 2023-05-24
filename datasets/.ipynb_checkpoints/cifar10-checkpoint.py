import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.autograd import Variable


class CIFAR10():
    def __init__(self, data_path, samps=1000, bt_size=1000):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        per_class_num = samps//10 or samps
        label_include = 10 if samps > 10 else samps
        
        train_set = datasets.CIFAR10(root=data_path, train=True,
                                                download=True, transform=transform_train)

        updated_train_data, updated_train_targets = [], []
        updated_val_data, updated_val_targets = [], []  # val set are those samples that are not included in the train
        count = torch.zeros(label_include)
        for i in range(50000):
            target_i = train_set.__dict__['targets'][i]
            sample_i = train_set.__dict__['data'][i]
            if target_i in torch.arange(label_include).tolist() and count[target_i] < per_class_num and torch.sum(count).item() < samps:
                updated_train_data.append(sample_i)
                updated_train_targets.append(target_i)
                count[target_i] += 1
            else:
                updated_val_data.append(sample_i)
                updated_val_targets.append(target_i)

        train_set.__dict__['targets'] = updated_train_targets
        train_set.__dict__['data'] = updated_train_data
        # load training and test set here:

        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=bt_size,
                                                  num_workers=4,
                                                  pin_memory=True)
        testset = datasets.CIFAR10(root=data_path, train=False,
                                               download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=bt_size,
                                                 num_workers=4,
                                                 pin_memory=True)