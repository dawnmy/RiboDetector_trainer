# from Bio import SeqIO
from pathlib import Path
import gzip
from mimetypes import guess_type
from functools import partial
from data_loader.fastx_parser import seq_parser

BASE_DICT = {"A": (1, 0, 0, 0),
             "C": (0, 1, 0, 0),
             "G": (0, 0, 1, 0),
             "T": (0, 0, 0, 1),
             "U": (0, 0, 0, 1)
             }

ZERO_LIST = (0, 0, 0, 0)


def get_seq_format(seq_file):
    fa_exts = [".fasta", ".fa", ".fna", ".fas"]
    fq_exts = [".fq", ".fastq"]
    encoding = guess_type(seq_file)[1]  # uses file extension
    if encoding is None:
        encoding = ""
    elif encoding == "gzip":
        encoding = "gz"
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))
    seq_filename = Path(seq_file).stem if encoding == 'gz' else Path(seq_file).name
    seq_file_ext = Path(seq_filename).suffix
    if seq_file_ext not in (fa_exts + fq_exts):
        raise ValueError("""Unknown extension {}. Only fastq and fasta sequence formats are supported. 
And the file must end with one of ".fasta", ".fa", ".fna", ".fas", ".fq", ".fastq" 
and followed by ".gz" or ".gzip" if they are gzipped.""".format(seq_file_ext))
    seq_format = "fa" + encoding if seq_file_ext in fa_exts else "fq" + encoding
    return seq_format


# def parse_seq_file(seq_file):
#     seq_format = get_seq_format(seq_file)

#     _open = open if seq_format.endswith("gz") else partial(gzip.open, mode='rt')
#     seq_type = "fasta" if seq_format.startswith("fa") else "fastq"

#     with _open(seq_file) as fh:
#         return SeqIO.parse(fh, seq_type)


def all_seqs_x(seq_file, min_seq_length):
    dataset = []
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith("gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    with _open(seq_file) as fh:
        # for record in SeqIO.parse(fh, seq_type):  # parse_seq_file(seq_file):
        #     seq = str(record.seq).upper()
        for record in seq_parser(fh, seq_type):
            features = seq_to_feature(record[1], min_seq_length)
            try:
                dataset.append(features)
            except NameError as e:
                print(NameError("Can not concatenate the np array", e))
        return dataset


def seq_to_feature(seq, min_seq_length):
    read_length = len(seq)
    if read_length > min_seq_length:
        start = (read_length - min_seq_length) // 2
        end = min_seq_length + start
        seq = seq[start:end]
    seq_feature = [BASE_DICT.get(base, ZERO_LIST) for base in seq]
    if read_length < min_seq_length:
        seq_feature.extend([ZERO_LIST] * (min_seq_length - read_length))

    return seq_feature


# if __name__ == "__main__":
#     mRNA_seq_file = "datasets/illumina-non-rrna-reads.fasta"
#     #rRNA_seq_file = "datasets/set1-illumina-rrna-reads-head1000.fasta"

#     #print("Extracting features from sequences")
#     mRNA_data = all_seqs_x(mRNA_seq_file, "fasta", 100)
#     #rRNA_data = all_seqs_x(rRNA_seq_file, "fasta", 100)

#     #train_data = np.expand_dims(np.concatenate((mRNA_data, rRNA_data)), axis=1)
#     #train_target = np.array([0] * 1000 + [1] * 1000)
#     print(len(mRNA_data))
