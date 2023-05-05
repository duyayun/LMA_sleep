import yaml
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import pickle as pk
from tqdm import tqdm
def custom_collate_fn(batch):
    # Separate the input data (X) and labels (Y) from the batch
    X = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Process the input data and labels as needed
    # For example, you can convert them to Tensors or perform other transformations
    X_tensor = torch.stack(X)  # assuming X is a list of tensors
    labels_tensor = torch.tensor(labels)  # assuming labels is a list of integers or floats
    
    # Return the processed data and labels
    return X_tensor, labels_tensor


class SleepStudySubjectDataset(Dataset):

    def __init__(self, path):
        try:
            with open(path.split('.')[0]+'pk', 'rb') as f:
                self.npval = pk.load(f)

        except Exception as e:


            df = pd.read_parquet(path).dropna()
            data = df[['x','y','z','ma_hr', 'ma_rr','chest_orientation','Epoch', 'QuantizedStage']]
            data = data.groupby('Epoch').apply(lambda x: x.values)
            self.npval = np.stack([x for x in data if x.shape==(30000,8)])
            with open(path.split('.')[0]+'pk', 'wb') as f:
                pk.dump(self.npval, f)
        self.X = self.npval[:, :,:-2].astype(np.float32)
        self.label = np.median(self.npval[:, :,-1], axis=1).astype(np.float32)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if int(self.label[idx]) > 0.0:
            label = 1.0
        else:
            label = self.label[idx]
        return (torch.unsqueeze(torch.tensor(self.X[idx,:]), dim=0), torch.tensor(label))
    
    

def prepare_dataloaders(config):
    train_file_paths = config["data"]["train"]
    val_file_paths = config["data"]["val"]
    test_file_paths = config["data"]["test"]

    train_datasets = [SleepStudySubjectDataset(file_path) for file_path in tqdm(train_file_paths, desc="Loading Train Data")]
    val_datasets = [SleepStudySubjectDataset(file_path) for file_path in tqdm(val_file_paths, desc="Loading Validation Data")] if val_file_paths else []
    test_datasets = [SleepStudySubjectDataset(file_path) for file_path in tqdm(test_file_paths, desc="Loading Test Data")] if test_file_paths else []

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn) if test_dataset else None
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    test_file = r'D:\ShiyuanDuan\SleepStudy\outputs\sub007.parquet'
    sl = SleepStudySubjectDataset(test_file)
    print(len(sl))
    # data_loader = DataLoader(sl, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
    # for i, (data, target) in enumerate(data_loader):
    #     # Your testing logic here
    #     print(f"Batch {i + 1}:")
    #     print("Data:", data)
    #     print("Target:", target)
    #     print(data.size())


    # with open("./conf/config_train.yaml", "r") as file:
    #     config = yaml.safe_load(file)

    # train, val, test = prepare_dataloaders(config=config)
    # print(len(train))
    # print(len(val))
    # print(len(test))
    