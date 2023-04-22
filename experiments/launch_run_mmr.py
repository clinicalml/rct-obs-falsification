import argparse
import json
import numpy as np
import subprocess
import time
import codecs
import numpy as np

from distutils.util import strtobool
from collections import OrderedDict
from itertools import product

SLEEP = 10 
CTR_LIM = 12

def num_procs_open(procs):
    k = 0
    for p in procs:
        k += (p.poll() is None)
    return k

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch experiments')
    parser.add_argument('--script', type=str, default='whi_experiment.py',
                        help='Path to the experiment run script.')
    args = parser.parse_args()
    print(args)
    
    # Dictionary of lists in JSON config
    wparams = np.arange(-5,7,0.5)
    for wparam in wparams: 
        proc_args_ate_power = [
            'python', args.script, '-f', 'ATE', '-n', '10', '-s', 'debug', '-e', f'ATE-power-{wparam}-v2', \
            '-C', '2', '-E', 'True', '-r', 'linear', '-t', 'non_linear', '-w', '0.2', '-p', f'{wparam}'
        ]
        proc_args_mmr_power = [
            'python', args.script, '-f', 'MMR-Contrast-KEOPS', '-n', '10', '-s', 'debug', '-e', f'MMR-power-{wparam}-v2', \
            '-C', '2', '-E', 'True', '-r', 'linear', '-t', 'non_linear', '-w', '0.2', '-p', f'{wparam}'
        ]
        proc_args_mmr_level = [
            'python', args.script, '-f', 'MMR-Contrast-KEOPS', '-n', '10', '-s', 'debug', '-e', f'MMR-level-{wparam}-v2', \
            '-C', '0', '-E', 'False', '-r', 'linear', '-t', 'non_linear', '-w', '0.2', '-p', f'{wparam}'
        ]
        print('[Launching ' + ' '.join(proc_args_ate_power) + ']')
        subprocess.call(proc_args_ate_power)
        print('[Launching ' + ' '.join(proc_args_mmr_power) + ']')
        subprocess.call(proc_args_mmr_power)
        print('[Launching ' + ' '.join(proc_args_mmr_level) + ']')
        subprocess.call(proc_args_mmr_level)
        
