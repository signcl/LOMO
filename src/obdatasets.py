import utils
import logging
from torch.utils.data import Dataset

class OBDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(OBDataset, self).__init__()
        logging.warning("Loading data...")
        self.data = utils.jload(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }
