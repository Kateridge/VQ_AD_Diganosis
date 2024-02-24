import os
import torch
from torch import nn


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        # create a logging file to store training losses
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self):
        fname = self.fname_template
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
                outdict[name] = module.state_dict()

        torch.save(outdict, fname)

    def load(self):
        fname = self.fname_template
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))

        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])

