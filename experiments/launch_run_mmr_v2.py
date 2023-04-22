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
    parser.add_argument('-f', '--falsification_type', type=str, default='MMR-Absolute',
                        help='MMR-Absolute, MMR-Contrast, GATE, or ATE')
    parser.add_argument('-C', '--confounder_conc', type=int, default=0)
    parser.add_argument('-E', '--effect_mod_concealment', type=strtobool)
    parser.add_argument('-r', '--ctr_response_surface', type=str, default='linear')
    parser.add_argument('-t', '--reweighting_type', type=str, default='non_linear')
    parser.add_argument('-w', '--reweighting_factor', type=float, default=0.2)
    parser.add_argument('-p', '--wparam', type=float, default=2.)
    parser.add_argument('-s', '--save_folder_name', type=str, default='type1-error')                    
    parser.add_argument('-n', '--num_iters', type=int, default=1)
    parser.add_argument('-e', '--exp_name', type=str, default='mmr-contrast')
    parser.add_argument('-i', '--rct_size', type=float, default=3.)
    

    args = parser.parse_args()
    print(args)
    
    # Dictionary of lists in JSON config
    proc_args_rct1 = [
        'python', args.script, '-f', args.falsification_type, '-n', str(args.num_iters), \
        '-s', args.save_folder_name, '-e', f'{args.falsification_type}-rct3-C{args.confounder_conc}-v2-strong-confounding', \
        '-C', str(args.confounder_conc), '-E', str(args.effect_mod_concealment), '-r', 'linear', '-t', 'non_linear', \
        '-w', '0.2', '-i', '3.'
    ]
    proc_args_rct2 = [
        'python', args.script, '-f', args.falsification_type, '-n', str(args.num_iters), \
        '-s', args.save_folder_name, '-e', f'{args.falsification_type}-rct5-C{args.confounder_conc}-v2-strong-confounding', \
        '-C', str(args.confounder_conc), '-E', str(args.effect_mod_concealment), '-r', 'linear', '-t', 'non_linear', \
        '-w', '0.2', '-i', '5.'
    ]
    # proc_args_rct3 = [
    #     'python', args.script, '-f', args.falsification_type, '-n', str(args.num_iters), \
    #     '-s', args.save_folder_name, '-e', f'{args.falsification_type}-rct6-C{args.confounder_conc}-strong-confounding', \
    #     '-C', str(args.confounder_conc), '-E', str(args.effect_mod_concealment), '-r', 'linear', '-t', 'non_linear', \
    #     '-w', '0.2', '-i', '6.'
    # ]
    print('[Launching ' + ' '.join(proc_args_rct1) + ']')
    subprocess.call(proc_args_rct1)
    print('[Launching ' + ' '.join(proc_args_rct2) + ']')
    subprocess.call(proc_args_rct2)
    # print('[Launching ' + ' '.join(proc_args_rct3) + ']')
    # subprocess.call(proc_args_rct3)
    
