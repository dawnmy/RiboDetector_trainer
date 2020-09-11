from torchvision import datasets, transforms
from base import BaseDataLoader

# from data_loader.seq_encoder import all_seqs_x
from data_loader.seq_encoder import load_encoded_seqs, load_encoded_seq_reads
from data_loader.dataset import SeqFeature
from torch import FloatTensor, LongTensor


def create_dataset_from_longseqs(seq_file_dict, read_len=100, step_sizes=None):
    seq_data = []
    target_data = []
    step_sizes = [10] * len(seq_file_dict) if step_sizes is None else step_sizes
    idx = 0
    for target, seq_file in seq_file_dict.items():
        encoded_seq = load_encoded_seq_reads(seq_file, read_len, step_sizes[idx])
        seq_data.extend(encoded_seq)
        target_data.extend([int(target)] * len(encoded_seq))
        idx += 1
    dataset = SeqFeature(FloatTensor(seq_data), LongTensor(target_data))
    return dataset


def create_dataset_from_reads(seq_file_dict, min_seq_len=100):
    seq_data = []
    target_data = []
    #step_sizes = [10] * len(seq_file_dict) if step_sizes is None else step_sizes
    idx = 0
    for target, seq_file in seq_file_dict.items():
        encoded_seq = load_encoded_seqs(seq_file, min_seq_len)
        seq_data.extend(encoded_seq)
        target_data.extend([int(target)] * len(encoded_seq))
        idx += 1
    dataset = SeqFeature(FloatTensor(seq_data), LongTensor(target_data))
    return dataset


class SeqDataLoader(BaseDataLoader):
    """
    sequence data loader using BaseDataLoader
    """

    def __init__(self, seq_data, min_seq_len, batch_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):
        step_sizes = [10, 10]
        self.dataset = create_dataset_from_longseqs(seq_data, min_seq_len, step_sizes)

        print('Training set samples: {}'.format(len(self.dataset)))

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
