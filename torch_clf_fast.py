import torch
import torch.nn as nn
import os
from tqdm import tqdm
# import torchvision
from torch.utils.data import Dataset, DataLoader
from data_loader.dataset import SeqFeature
# from SeqEncoder import all_seqs_x
from data_loader.data_loaders import create_dataset_from_longseqs, create_dataset_from_reads
import math
from model import metric


# Global setting
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 30
BATCH_SIZE = 2048
HIDDEN_UNITS = 128
N_LAYERS = 1
N_CLASSES = 2
# DROPOUT = 0.2
LR = 0.005          # learning rate
THREADS = 4         # number of threads to load the data; 2 will be the best
INPUT_SIZE = 4     # input width


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,      # width of input
            hidden_size=HIDDEN_UNITS,     # number of rnn hidden unit
            num_layers=N_LAYERS,       # number of RNN layers
            batch_first=True,
            bidirectional=True,
            # dropout=DROPOUT
        )

        self.out = nn.Linear(HIDDEN_UNITS * 2, N_CLASSES)    # out put layer

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


def train(model,
          loss_func,
          optimizer,
          metrics,
          train_loader,
          epoch):
    '''
    training an epoch
    '''
    # Set model to train mode
    # model.train(True)
    # Iterate training set

    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # gives batch data
        # b_data = data.to(device)
        output = model(data)             # rnn output
        loss = loss_func(output, target)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        torch.cuda.empty_cache()

        # print("The loss of epoch {}: {}".format(epoch, loss))
        # print(output)
    print('###### Training on epoch {} ######'.format(epoch))
    print('Loss: {} '.format(loss))
    (recall, precision, accuracy, F1, mcc) = metrics(output, target)
    print('Recall: {} \nPrecision: {} \nAccuracy: {} \nF1: {} \nMCC: {}\n\n'.format(recall, precision, accuracy, F1, mcc))


if __name__ == "__main__":

    print("Batch size: {} \nHidden units: {} \nNubmer layers: {} \nLearning rate: {}".format(BATCH_SIZE,
                                                                                             HIDDEN_UNITS,
                                                                                             N_LAYERS,
                                                                                             LR))

    # train_seq_files = {0: '../datasets/illumina-non-rrna-reads-exclude-tail1000.fasta',
    #                    1: '../datasets/set1-illumina-rrna-reads-exclude-tail1000.fasta'}

    # train_seq_files = {0: '../datasets/new_mRNA_lenlt1900.fa',
    #                    1: '../datasets/new_rRNA_2016.fa'}

    train_seq_files = {0: '../datasets/rrna_mrna_dbs/prokaryotes.eukaryotes.cdna.mm.id0.7cov0.7_rep_seq_le300.fasta',
                       1: '../datasets/rrna_mrna_dbs/SILVA_138.1_SSU_LSU.fasta'}

    seq_sliding_step_sizes = [50, 10]
    read_len = 100

    train_seq_data = create_dataset_from_longseqs(train_seq_files, read_len, seq_sliding_step_sizes)

    test_seq_files = {0: '../datasets/illumina-non-rrna-reads.fasta',
                      1: '../datasets/set1-illumina-rrna-reads.fasta'}

    test_seq_data = create_dataset_from_reads(test_seq_files, read_len)

    print('Training data size: {}\n\n'.format(len(train_seq_data)))

    train_loader = DataLoader(
        train_seq_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=THREADS,
        pin_memory=torch.cuda.is_available()
    )

    rnn = RNN().to(device)
    rnn.train()

    optimizer = torch.optim.Adam(
        rnn.parameters(),
        # amsgrad=True,
        lr=LR)   # optimize all parameters
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    # training and testing
    for epoch in range(1, EPOCH + 1):
        train(rnn,
              loss_func,
              optimizer,
              metric.all,
              train_loader,
              epoch)

        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)  # gives batch data
        #     # b_data = data.to(device)
        #     output = rnn(data)             # rnn output
        #     loss = loss_func(output, target)   # cross entropy loss
        #     optimizer.zero_grad()           # clear gradients for this training step
        #     loss.backward()                 # backpropagation, compute gradients
        #     optimizer.step()                # apply gradients
        #     torch.cuda.empty_cache()

        # # print("The loss of epoch {}: {}".format(epoch, loss))
        # # print(output)
        # print('###### Training on epoch {} ######'.format(epoch))
        # print('Loss: {} '.format(loss))
        # (recall, precision, accuracy, F1, mcc) = metric.all(output, target)
        # print('Recall: {} \nPrecision: {} \nAccuracy: {} \nF1: {} \nMCC: {}\n\n'.format(recall, precision, accuracy, F1, mcc))
        # lr_scheduler.step()
    arch = type(rnn).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': rnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_data': train_seq_files
    }
    torch.save(state, "Ribodetector_{}epochs_readlen{}_step{}-{}_test.plt".format(
        EPOCH, read_len, *seq_sliding_step_sizes))

    torch.save(rnn, "Ribodetector_{}epochs_readlen{}_step{}-{}_test.model".format(
        EPOCH, read_len, *seq_sliding_step_sizes))

    test_output_list = []

    rnn.eval()
    for idx, batch_data in enumerate(tqdm(torch.split(test_seq_data.data, 10000))):
        with torch.no_grad():
            batch_data = batch_data.to(device)
            batch_output = rnn(batch_data)
            test_output_list.append(batch_output)
            del batch_data, batch_output

    test_output = torch.cat(test_output_list, 0)

    print('######## Testing dataset performance #########')
    (recall, precision, accuracy, F1, mcc) = metric.all(test_output, test_seq_data.target)
    print('Recall: {} \nPrecision: {} \nAccuracy: {} \nF1: {} \nMCC: {}\n\n'.format(recall, precision, accuracy, F1, mcc))
