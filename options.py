import argparse
import os
from utils import mkdirs


class Option:
    """This class defines options used during both training and CNN_PET_ADCN time. It also implements several helper
    functions such as parsing, printing, and saving the options. It also gathers additional options defined in
    <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and CNN_PET_ADCN."""
        # basic settings
        parser.add_argument('--name', type=str, default='VQCNN_PET_vis')
        parser.add_argument('--dataroot', type=str, default='D:\\Datasets\\ADNI_ALL\\ADNI.hdf5')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--modality', type=str, default='PET')
        parser.add_argument('--model', type=str, default='VQCNN_Single')

        # dataset parameters
        parser.add_argument('--aug', type=bool, default=True)
        parser.add_argument('--include_missing', type=bool, default=False)
        parser.add_argument('--task', type=str, default='ADCN')
        parser.add_argument('--remove_CN', type=bool, default=True)
        parser.add_argument('--add_ADCN', type=bool, default=False)

        # training parameters
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=60)
        parser.add_argument('--CLS_epochs', type=int, default=20)
        parser.add_argument('--warmup_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.01)

        # model parameters
        parser.add_argument('--dim', type=int, default=64)
        parser.add_argument('--code_num', type=int, default=128)
        parser.add_argument('--beta', type=float, default=0.25)
        parser.add_argument('--trans_enc_depth', type=int, default=3)
        parser.add_argument('--heads_num', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)

        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        print(f'Create opt file opt.txt')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        self.parser = self.initialize(self.parser)
        self.opt= self.parser.parse_args()
        self.print_options(self.opt)
        return self.opt
