'''
File: fastx_parser.py
Created Date: November 30th 2019
Author: ZL Deng <dawnmsg(at)gmail.com>
---------------------------------------
Last Modified: 30th November 2019 11:04:48 pm

Description:

This is a simple fastq and fasta parser which is about 5 times faster than
SeqIO in Biopython.
'''


def seq_parser(seq_fh, seq_type):
    if seq_type == 'fastq':
        record_line = 0  # before parse seq record
        for line in seq_fh:
            line = line.rstrip()
            # if the current line is header line
            if line[0] == '@' and record_line == 0:
                record_line = 1
                header = line
            # if it is the next line of the header which is the seq reads
            elif record_line == 1:
                seq = line
                record_line = 2
            # if it is the next line of seq, which is + line
            elif record_line == 2:
                endseq = line
                record_line = 3
            # quality line
            else:
                qual = line
                yield header, seq, endseq, qual
                header = seq = endseq = qual = ''
                record_line = 0

    else:
        seq = header = ''
        for line in seq_fh:
            line = line.rstrip()
            if line[0] == '>' and seq != '':
                yield header, seq
                header = line
                seq = ''
            else:
                seq += line.upper()
        if seq != '':
            yield header, seq


# if __name__ == '__main__':
#     with open("../../datasets/non-rrna-reads.fasta") as fh:
#         # for name, seq, qual in readseq(fh):
#         #     print(seq)
#         # for header, seq, endseq, qual in parseseq(fh, "fastq"):
#         #     print(qual)
#         # for header, seq in parseseq(fh, "fasta"):
#         #     print(seq)
#         for header, seq in biopython_parser(fh, "fasta"):
#             print(seq)
