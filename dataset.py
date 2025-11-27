import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GeoLifeDataset(Dataset):
    def __init__(self, data_path, max_len=50):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare sequences
        loc_seq = sample['X']
        user_seq = sample['user_X']
        weekday_seq = sample['weekday_X']
        start_min_seq = sample['start_min_X']
        dur_seq = sample['dur_X']
        diff_seq = sample['diff']
        target = sample['Y']
        
        # Pad or truncate
        seq_len = len(loc_seq)
        if seq_len > self.max_len:
            loc_seq = loc_seq[-self.max_len:]
            user_seq = user_seq[-self.max_len:]
            weekday_seq = weekday_seq[-self.max_len:]
            start_min_seq = start_min_seq[-self.max_len:]
            dur_seq = dur_seq[-self.max_len:]
            diff_seq = diff_seq[-self.max_len:]
            seq_len = self.max_len
        
        return {
            'loc': torch.LongTensor(loc_seq),
            'user': torch.LongTensor(user_seq),
            'weekday': torch.LongTensor(weekday_seq),
            'start_min': torch.LongTensor(start_min_seq),
            'duration': torch.FloatTensor(dur_seq),
            'diff': torch.LongTensor(diff_seq),
            'seq_len': seq_len,
            'target': torch.LongTensor([target])
        }


def collate_fn(batch):
    max_len = max([item['seq_len'] for item in batch])
    
    batch_loc = []
    batch_user = []
    batch_weekday = []
    batch_start_min = []
    batch_duration = []
    batch_diff = []
    batch_mask = []
    batch_target = []
    
    for item in batch:
        seq_len = item['seq_len']
        pad_len = max_len - seq_len
        
        # Pad sequences
        batch_loc.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item['loc']]))
        batch_user.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item['user']]))
        batch_weekday.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item['weekday']]))
        batch_start_min.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item['start_min']]))
        batch_duration.append(torch.cat([torch.zeros(pad_len, dtype=torch.float), item['duration']]))
        batch_diff.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item['diff']]))
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), torch.ones(seq_len, dtype=torch.bool)])
        batch_mask.append(mask)
        
        batch_target.append(item['target'])
    
    return {
        'loc': torch.stack(batch_loc),
        'user': torch.stack(batch_user),
        'weekday': torch.stack(batch_weekday),
        'start_min': torch.stack(batch_start_min),
        'duration': torch.stack(batch_duration),
        'diff': torch.stack(batch_diff),
        'mask': torch.stack(batch_mask),
        'target': torch.cat(batch_target)
    }


def get_dataloaders(batch_size=128, max_len=50):
    train_dataset = GeoLifeDataset('data/geolife/geolife_transformer_7_train.pk', max_len=max_len)
    val_dataset = GeoLifeDataset('data/geolife/geolife_transformer_7_validation.pk', max_len=max_len)
    test_dataset = GeoLifeDataset('data/geolife/geolife_transformer_7_test.pk', max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader
