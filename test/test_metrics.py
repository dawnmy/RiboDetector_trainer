import argparse


class ConfigParser:
    def __init__(self):
        self.level = "A"

    @classmethod
    def from_args(cls, args):
        if not isinstance(args, tuple):
            args = args.parse_args()
        cls.input = args.input
        return cls()


def main(config):
    print(config.input)
    print(config.level)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='rRNA sequence detector')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--input', default=None, type=str, nargs='*', required=True,
                      help='path to input sequence file, the second file will be considered as second end if two files given.')

    config = ConfigParser.from_args(args)
    main(config)
