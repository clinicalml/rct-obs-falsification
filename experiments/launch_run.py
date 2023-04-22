import argparse
import json
import numpy as np
import subprocess
import time
import codecs

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
    # parser.add_argument('--config', type=str,
    #                     help='Path to the experiment config file.')
    parser.add_argument('--B', type=int, default=100,
                        help='number of bootstraps')
    parser.add_argument('--n_models', type=int, default=0,
                        help='Expected number of models.')
    parser.add_argument('-d', '--downsize', type=strtobool, default=True)
    parser.add_argument('-p', '--downsize_proportion', type=float, default=0.5)
    parser.add_argument('-s', '--save_folder_name', type=str, default='test')
    parser.add_argument('-t', '--split', type=str, default='train')
    parser.add_argument('-f', '--falsification_type', type=str, default='ATE')
    parser.add_argument('-o', '--obs_type', type=str, default='None')
    parser.add_argument('-l', '--selection_bias', type=float, default=0.05)
    parser.add_argument('--procs_lim', type=int, default=12,
                        help='Max number of processes to run in parallel.')
    args = parser.parse_args()
    print(args)
    
    # Dictionary of lists in JSON config
    param_names = ['falsification_type', 'bootstrap_seed', 'save_folder_name', \
        'exp_name', 'downsize', 'downsize_proportion', 'split', 'obs_type', 'selection_bias']
    param_vals  = [[args.falsification_type],[x for x in range(args.B)]]

    # Number of expected models & subsampling prob. range
    n_total_configs   = len(list(product(*param_vals)))
    n_expected_models = args.n_models if args.n_models > 0 else n_total_configs
    param_keep_prob   = min(float(n_expected_models) / n_total_configs, 1.0)

    # Iterate over the param configs and launch subprocesses
    run_idxs = []
    procs    = []
    j        = -1
    x = list(product(*param_vals))
    for param_set in product(*param_vals):
        # if np.random.rand() > param_keep_prob[]:
        #     continue
        j += 1
        run_idxs.append(j)

        # Assemble command line argument
        proc_args = [
            'python', args.script
        ]
        param_set = list(param_set) + \
            [args.save_folder_name, \
             param_set[0] + '_seed' + str(param_set[1]), \
             args.downsize, 
             args.downsize_proportion,
             args.split,
             args.obs_type, 
             args.selection_bias]
        for k, v in zip(param_names, param_set):
            proc_args += ['--{0}'.format(k), str(v)]


        # Launch as subprocess
        print("Launching model {0}".format(j))
        print("\t".join(
                    ["%s=%s" % (k, v) for k, v in zip(param_names, param_set)]
        ))
        p = subprocess.Popen(proc_args)
        procs.append(p)
        ctr = 0
        while True:
            k = num_procs_open(procs)
            ctr += 1
            if ctr >= CTR_LIM:
                ctr = 0
                print('{0} processes still running'.format(k))
            if num_procs_open(procs) >= args.procs_lim:
                time.sleep(SLEEP)
            else:
                break

    n = len(procs)
    ctr = 0
    while True:
        k = num_procs_open(procs)
        if k == 0:
            break
        ctr += 1
        if ctr >= CTR_LIM:
            ctr = 0
            print('{0} processes still running'.format(k))
        time.sleep(SLEEP)