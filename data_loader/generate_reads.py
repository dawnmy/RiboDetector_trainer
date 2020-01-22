import click
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna


@click.command()
@click.argument('seqfile', type=click.Path(exists=True))
@click.option('-l', '--readlen', help='The generated read length.', default=100)
@click.option('-s', '--step', help='The step length for generating the output reads.', default=5)
@click.argument('output', type=str)
def seq_to_reads(seqfile, readlen, step, output):
    out_fh = open(output, 'w')
    for record in SeqIO.parse(seqfile, 'fasta'):
        seq = str(record.seq)
        seq_len = len(seq)
        # header = record.id

        for i in range(0, seq_len, step):
            if i + readlen <= seq_len:
                header = '>{} {}:{}'.format(record.id, i, i + readlen)
                read = seq[i:i + readlen]
                read_rc = Seq(read, generic_dna).reverse_complement()
                out_fh.write('{}\n{}\n{}\n{}\n'.format(header, read, header + "-strand", read_rc))
            else:
                header = '>{} {}:{}'.format(record.id, i, i + seq_len)
                read = seq[i:]
                read_rc = Seq(read, generic_dna).reverse_complement()
                out_fh.write('{}\n{}\n{}\n{}\n'.format(header, read, header + "-strand", read_rc))
                break

    out_fh.close()


if __name__ == '__main__':
    seq_to_reads()
