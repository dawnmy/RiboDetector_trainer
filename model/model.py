import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# # Hyper Parameters
# INPUT_SIZE = 4     # input width
# HIDDEN_UNITS = 128
# N_LAYERS = 1
# N_CLASSES = 2
# hidden_fh = open('../datasets/intermediate_output/Tuf.head2k.hidden.txt', 'w')


class SeqModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 batch_first=True,
                 bidirectional=True):
        super(SeqModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,      # width of input
            hidden_size=hidden_size,     # number of rnn hidden unit
            num_layers=num_layers,       # number of RNN layers
            batch_first=batch_first,
            bidirectional=bidirectional,
            # dropout=DROPOUT
        )

        self.out = nn.Linear(hidden_size * 2, num_classes)    # output layer

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        # for i in r_out[:, -1, :]:
        #     hidden_fh.write('\t'.join(map(str, i.tolist())) + '\n')
        out = self.out(r_out[:, -1, :])
        return out
