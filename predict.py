import os
import gzip
import torch
import argparse
# import logging
# from Bio import SeqIO
from tqdm import tqdm
from functools import partial
import model.model as module_arch
from parse_predict_config import ConfigParser
from data_loader.seq_encoder import all_seqs_x, get_seq_format
from data_loader.fastx_parser import seq_parser
# from data_loader import readseq

cd = os.path.dirname(os.path.abspath(__file__))

class Predictor:
    def __init__(self, config, args, chunk_size):
        self.config = config
        self.args = args
        self.logger = config.get_logger('predict', 1)
        self.chunk_size = chunk_size

    def get_state_dict(self):
        if self.args.len < 50:
            self.logger.error('Sequence length is too short to classify!')
            raise RuntimeError("Sequence length must be set to not shorter than 50.")
        elif 50 <= self.args.len < 100:
            self.state_file = os.path.join(cd, config['state_file']['read_len50'])
            self.len = 50
        elif 100 <= self.args.len < 150:
            self.state_file = os.path.join(cd, config['state_file']['read_len100'])
            self.len = 100
        else:
            self.state_file = os.path.join(cd, config['state_file']['read_len150'])
            self.len = 150

    def load_model(self):
        if self.args.deviceid is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.deviceid
        self.get_state_dict()
        model = self.config.init_obj('arch', module_arch)
        self.logger.info(model)
        self.logger.info('Loading model file: {} ...'.format(self.state_file))
        state = torch.load(self.state_file)
        state_dict = state['state_dict']
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        if torch.cuda.is_available() and self.args.cpu == False:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def predict(self):
        self.input = self.args.input
        self.output = self.args.output
        self.rrna = self.args.rrna
        if len(self.input) != len(self.output) or len(self.input) > 2:
            self.logger.error('The number of input and output sequence files is invalid!')
            raise RuntimeError(
                "Input or output should have no more than two files and they should the same number of files.")
        if self.rrna != None and len(self.rrna) != len(self.input):
            self.logger.error('The number of output rRNA sequence files is invalid!')
            raise RuntimeError(
                "Ouput rRNA should have no more than two files and they should the same number with input files.")

        # self.logger.info('Predicting ...')

        self.pred_labels = []
        for i, in_seq in enumerate(self.input):
            pred_labels_i = []
            pred_data = torch.FloatTensor(all_seqs_x(in_seq, self.len))
            self.logger.info('Predicting sequence file {}...'.format(i + 1))

            for idx, data in enumerate(tqdm(torch.split(pred_data, self.chunk_size))):
                with torch.no_grad():
                    data = data.to(self.device)
                    output = self.model(data)
                    batch_labels = torch.argmax(output, dim=1)

                    pred_labels_i.extend(batch_labels)
                    del data, output, batch_labels

            n_samples = len(pred_data)
            self.pred_labels.append(pred_labels_i)
        self.logger.info('Predicting {} sequences finished!'.format(n_samples))

    def output_seq(self):
        norrna_count = 0
        seq_format = get_seq_format(self.input[0])
        seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
        _open = partial(gzip.open, mode='rt') if seq_format.endswith("gz") else open

        self.logger.info('Writing output non-rRNA sequences into file: {}...'.format(", ".join(self.output)))

        if len(self.pred_labels) == 2:
            if self.rrna is not None:
                self.logger.info('Writing output rRNA sequences into file: {}...'.format(", ".join(self.rrna)))
                rrna1_fh = open(self.rrna[0], "w")
                rrna2_fh = open(self.rrna[1], "w")
            with open(self.output[0], "w") as out1_fh, open(self.output[1], "w") as out2_fh:
                with _open(self.input[0]) as r1_fh, _open(self.input[1]) as r2_fh:
                    for idx, (r1, r2) in enumerate(tqdm(zip(seq_parser(r1_fh, seq_type), seq_parser(r2_fh, seq_type)))):
                        if self.pred_labels[0][idx] == self.pred_labels[1][idx] == 0:
                            # seq_r2_header = record.id.replace("/1", "/2")
                            out1_fh.write('\n'.join(r1) + '\n')
                            out2_fh.write('\n'.join(r2) + '\n')
                            # SeqIO.write(r1, out1_fh, seq_type)
                            # SeqIO.write(r2, out2_fh, seq_type)
                            norrna_count += 1
                        elif self.pred_labels[0][idx] == self.pred_labels[1][idx] == 1:
                            if self.rrna is not None:
                                # seq_r2_header = record.id.replace("/1", "/2")
                                rrna1_fh.write('\n'.join(r1) + '\n')
                                rrna2_fh.write('\n'.join(r2) + '\n')
                                # SeqIO.write(r1, rrna1_fh, seq_type)
                                # SeqIO.write(r2, rrna2_fh, seq_type)
                        else:
                            if self.args.ensure == "rrna":
                                # seq_r2_header = record.id.replace("/1", "/2")
                                out1_fh.write('\n'.join(r1) + '\n')
                                out2_fh.write('\n'.join(r2) + '\n')
                                # SeqIO.write(r1, out1_fh, seq_type)
                                # SeqIO.write(r2, out2_fh, seq_type)
                                norrna_count += 1
                            elif self.args.ensure == "norrna":
                                if self.rrna is not None:
                                    # seq_r2_header = record.id.replace("/1", "/2")
                                    rrna1_fh.write('\n'.join(r1) + '\n')
                                    rrna2_fh.write('\n'.join(r2) + '\n')
                                    # SeqIO.write(r1, rrna1_fh, seq_type)
                                    # SeqIO.write(r2, rrna2_fh, seq_type)
                            else:
                                continue
            if self.rrna is not None:
                rrna1_fh.close()
                rrna2_fh.close()

        else:
            if self.rrna is not None:
                self.logger.info('Writing output rENA sequences into file: {}...'.format(", ".join(self.rrna)))
                rrna_fh = open(self.rrna[0], "w")
            with open(self.output[0], "w") as out_fh:
                with _open(self.input[0]) as read_fh:
                    for idx, record in enumerate(tqdm(seq_parser(read_fh, seq_type))):
                        if self.pred_labels[0][idx] == 0:
                            out_fh.write('\n'.join(record) + '\n')
                            # SeqIO.write(record, out_fh, seq_type)
                            norrna_count += 1
                        else:
                            if self.rrna is not None:
                                rrna_fh.write('\n'.join(record) + '\n')
                                # SeqIO.write(record, rrna_fh, seq_type)
            if self.rrna is not None:
                rrna_fh.close()
        self.logger.info('Finished writing {} non-rRNA sequences!'.format(norrna_count))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='rRNA sequence detector')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path')
    args.add_argument('-d', '--deviceid', default=None, type=str,
                      help='indices of GPUs to enable. Quotated comma-separated device ID numbers. (default: all)')
    args.add_argument('-l', '--len', type=int, required=True,
                      help='Sequencing read length, should be not smaller than 50.')
    args.add_argument('-i', '--input', default=None, type=str, nargs='*', required=True,
                      help='path to input sequence files, the second file will be considered as second end if two files given.')
    args.add_argument('-o', '--output', default=None, type=str, nargs='*', required=True,
                      help='path to the output sequence files after rRNAs removal (same number of files as input)')
    args.add_argument('-r', '--rrna', default=None, type=str, nargs='*',
                      help='path to the output sequence file of detected rRNAs (same number of files as input)')
    args.add_argument('-e', '--ensure', default="rrna", type=str, choices=['rrna', 'norrna', 'both'],
                      help='''Applicable when given paired end reads; 
norrna: remove as many as possible rRNAs, output non-rRNAs with high confidence; 
rrna: vice versa, rRNAs with high confidence;
both: both non-rRNA and rRNA prediction with high confidence''')
    args.add_argument('--cpu', default=False, action='store_true',
                      help='Use CPU even GPU is available. Useful when GPUs and GPU RAM are occupied.')

    # args = parser.parse_args()
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.config is None:
        config_file = os.path.join(cd, 'predict_config.json')
    else:
        config_file = args.config
    config = ConfigParser.from_json(config_file)
    seq_pred = Predictor(config, args, 1000)
    seq_pred.load_model()
    seq_pred.predict()
    seq_pred.output_seq()
