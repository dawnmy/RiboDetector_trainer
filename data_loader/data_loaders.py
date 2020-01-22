from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import SeqFeature
from data_loader.seq_encoder import all_seqs_x
from pathlib import Path
import torch


class SeqDataLoader(BaseDataLoader):
    """
    sequence data loader using BaseDataLoader
    """

    def __init__(self, seq_data, min_seq_len, batch_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):
        neg_seq_file, pos_seq_file = seq_data["0"], seq_data["1"]

        neg_seq = all_seqs_x(neg_seq_file, min_seq_len)
        pos_seq = all_seqs_x(pos_seq_file, min_seq_len)

        data = torch.FloatTensor(neg_seq + pos_seq)

        data_target = torch.LongTensor([0] * len(neg_seq) + [1] * len(pos_seq))

        dataset = SeqFeature(data, data_target)
        self.dataset = dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
