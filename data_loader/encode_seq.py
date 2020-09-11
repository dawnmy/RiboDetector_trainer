from multiprocessing import Pool
# from fastx_parser import seq_parser
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
import gzip
from itertools import chain
from pathlib import Path
from functools import partial
from mimetypes import guess_type
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


def load_encoded_seq(seq_file, read_len=100, step=20, cores=4):
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith(
        "gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    pool = Pool(processes=cores)
    partial_encode_seq_reads = partial(
        encode_seq_reads, read_len=read_len, step=step)

    # encoded_seq_read_list = []
    # count = 0
    with _open(seq_file) as fh:
        encoded_seq = pool.map_async(partial_encode_seq_reads, (seq for name,
                                                                seq in seq_parser(fh, seq_type))).get()
        # for _name, seq in seq_parser(fh, seq_type):
        #     count += 1
        #     seq_read = partial_seq_to_feature(seq)
        #     encoded_seq_read_list.extend(seq_read)

    # return result
    return list(chain.from_iterable(encoded_seq))
    # return encoded_seq_read_list


def encode_seq_reads(seq, read_len=100, step=10):
    seq_len = len(seq)
    seq_rc = Seq(seq, generic_dna).reverse_complement()
    seq_feature = [BASE_DICT.get(base, ZERO_LIST) for base in seq]
    seq_rc_feature = [BASE_DICT.get(base, ZERO_LIST) for base in seq_rc]
    encoded_read_list = []
    for feature in [seq_feature, seq_rc_feature]:
        for i in range(0, seq_len, step):
            if i + read_len < seq_len:

                read_feature = feature[i:i + read_len]
                encoded_read_list.append(read_feature)
            else:
                missing_len = read_len + i - seq_len
                if missing_len < read_len / 2:
                    read_feature = feature[i:] + [ZERO_LIST] * missing_len
                    encoded_read_list.append(read_feature)
                break
    return encoded_read_list
    # return seq_feature

# if __name__ == "__main__":
#     mRNA_seq_file = "datasets/illumina-non-rrna-reads.fasta"
#     # rRNA_seq_file = "datasets/set1-illumina-rrna-reads-head1000.fasta"

#     #print("Extracting features from sequences")
#     mRNA_data = all_seqs_x_multip(mRNA_seq_file)
#     # rRNA_data = all_seqs_x_multip(rRNA_seq_file)

#     #train_data = np.expand_dims(np.concatenate((mRNA_data, rRNA_data)), axis=1)
#     #train_target = np.array([0] * 1000 + [1] * 1000)
#     print(mRNA_data)
