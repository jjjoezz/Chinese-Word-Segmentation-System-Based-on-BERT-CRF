import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda item: len(item[0]), reverse=True)
        x, y = zip(*batch)

        # 强制限制最大长度为 512
        MAX_LEN = 512
        lengths = [min(len(i), MAX_LEN) for i in x]
        max_len = max(lengths)

        batch_x = torch.zeros(len(x), max_len, dtype=torch.long)
        batch_y = torch.zeros(len(x), max_len, dtype=torch.long)
        mask = torch.zeros(len(x), max_len, dtype=torch.bool)

        for i in range(len(x)):
            seq_len = lengths[i]
            batch_x[i, :seq_len] = torch.tensor(x[i][:seq_len])
            batch_y[i, :seq_len] = torch.tensor(y[i][:seq_len])
            mask[i, :seq_len] = True

        return batch_x, batch_y, mask, lengths