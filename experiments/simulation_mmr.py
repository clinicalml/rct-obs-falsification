## TODO: merge this file with whi_experiment.py; lots of overlapping code 

from distutils.util import strtobool
from turtle import pos
import pandas as pd 
import numpy as np
import sys 
import os
import argparse
import pprint

from itertools import repeat
from numpy.random import default_rng
from scipy.stats import norm
sys.path.append('../data/')
from DataModule import DataModule, test_params
sys.path.append('../models/')
from falsifier import Falsifier
from baselines import MetaAnalyzer, SimpleBaseline, EvolvedMetaAnalyzer
from estimator import CATE, ATE
sys.path.append('../models_mmr/')
from falsifier_mmr import FalsifierMMR
from estimator_mmr import OutcomeEstimator
from multiprocessing import Pool 
from util import *
from model_util import *

def get_strata_Ns(strata, table, data_module): 
    '''
        A function that takes in the desired stratum, table, and 
        outputs the indices for that stratum
    '''
    normalized_strata = []
    for stratum in strata: 
        new_stratum = []
        for elem in stratum: 
            col, op, cutoff, norm_truth = elem
            if norm_truth: 
                new_elem = (col, op, data_module.get_normalized_cutoff(col,cutoff))
                new_stratum.append(new_elem)
            else: 
                new_stratum.append((col,op,cutoff))
        normalized_strata.append(tuple(new_stratum))

    all_indices = []
    for stratum in normalized_strata: 
        idxs_truth = pd.Series([True for _ in range(table.shape[0])])
        for elem in stratum: 
            cutoff = float(elem[2])
            if elem[1] == '==': 
                idxs_truth = idxs_truth&((table[elem[0]]-cutoff).abs()<1e-6)
            elif elem[1] == '<=': 
                idxs_truth = idxs_truth&(table[elem[0]] <= cutoff)
            elif elem[1] == '>=': 
                idxs_truth = idxs_truth&(table[elem[0]] >= cutoff)
            elif elem[1] == '<': 
                idxs_truth = idxs_truth&(table[elem[0]] < cutoff)
            elif elem[1] == '>': 
                idxs_truth = idxs_truth&(table[elem[0]] > cutoff)
            else: 
                raise ValueError('invalid operator (choose one of <,>,<=,>=,==)')
        all_indices.append(idxs_truth)
    
    return [np.where(g.values == 1.)[0].shape[0] for g in all_indices]

def run_simulation(num_iters=2,
                    alpha=0.05,
                    root='',
                    strata_mod='', 
                    strata_metadata_mod='',
                    params_mod='',
                    falsification_type='MMR-Absolute'): 
    # TODO: update params
    params = {
        'ihdp': True,
        'num_continuous': 4,
        'num_binary': 3,
        'confounding_type': 'random',
        'omega': -23, # [0.2,0.5,.75,1.,1.25]
        # 'gamma_coefs': [0.1,0.2,.5,.75,1.], # fig 1, fig 2
        # 'gamma_coefs': [0.2,0.5,.75,1.,1.25],
        'gamma_coefs': [1.,1.75,2.,2.25,2.75], # fig 1, fig3?
        # 'gamma_coefs': [5.,6.,7.,8.,9.],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'grand_seed': 10, # was 10 originally
        'confounder_seed': 0,
        'beta_seed': 4,
        'noise_seed': 0,
        'selection_seed': 42,
        'cross_fitting_seed': 42,
        'propensity_model': {
            'model_name': 'GradientBoostingClassifier',
            'hp': {
                'learning_rate': [0.01], 'n_estimators': [50],\
                'max_depth': [2], 'min_samples_leaf': [50], \
                'min_samples_split': [50], 'max_features': ['sqrt'],
                'random_state': [42]
            },
            'model_type': 'binary'
        },
        'selection_model': {
            'model_name': 'SelectionModelOracle',
            'hp': {},
            'model_type': 'binary'
        },
        'response_surface_1': {
            'model_name': 'ResponseSurfaceOracle-1',
            'hp': {},
            'model_type': 'continuous'
        },
        'response_surface_0': {
            'model_name': 'ResponseSurfaceOracle-0',
            'hp': {},
            'model_type': 'continuous'
        },        
        'obs_dict': {
            'num_obs': 1,
            'sizes': [1.], # TODO: play with this a little bit
            'confounder_concealment': [0], # will be concealed according to ordering of coefficients
            'effect_mod_concealment': [True],
            'missing_bias': [False]
        }, 
        'rct_dict': { 
            'size': 1.
        },
        'reweighting': True,
        'reweighting_type': 'non_linear',
        'reweighting_factor': 0.2,
        'kernel': 'laplace',
        'response_surface': {
            'ctr': 'non_linear', 
            'trt': 'linear'
        },
        'wparam': 2.
    }
    pprint.pprint(params_mod, sort_dicts=False)
    if params_mod != '':
        for i in params_mod:
            print(i)
            params[i[0]] = i[1]
    rng = default_rng(params['grand_seed'])
    confounder_seeds = rng.choice(range(1000), size=(num_iters,))
    noise_seeds = rng.choice(range(1000), size=(num_iters,))
    
    results = []
    for iter_ in range(num_iters): 
        print(f'Simulation Number {iter_+1}')
        params['confounder_seed'] = confounder_seeds[iter_]
        params['noise_seed'] = noise_seeds[iter_]

        ''' 
            Part 1: data simulation piece 
        ''' 
        # TODO: implement new setup for semi-synthetic data to use in the new story (i.e. the confounding effect is not linear?)
        print(f'data generation parameters:')
        pprint.pprint({k:params[k] for k in params.keys() if k != 'oracle_params'}, sort_dicts=False)
        if root != '':
            ihdp_data = DataModule(params = params, root = root)
        else: 
            ihdp_data = DataModule(params = params) # change root path to data and add it as argument
        ihdp_data.generate_dataset()
        data_dicts = ihdp_data.get_datasets()
        # add oracle params to params
        params['oracle_params'] = ihdp_data.oracle_params
        if falsification_type == 'GATE': 
            results_to_append = run_gate(ihdp_data, 
                                        data_dicts, 
                                        strata_mod, 
                                        strata_metadata_mod, 
                                        params, 
                                        alpha, 
                                        iter_=iter_) 
        elif falsification_type == 'ATE': 
            results_to_append = run_ate(ihdp_data, 
                                        data_dicts, 
                                        params, 
                                        alpha, 
                                        iter_=iter_) 
        elif 'MMR-Absolute' in falsification_type or 'MMR-Contrast' in falsification_type:
            results_to_append = run_mmr(ihdp_data,
                                        data_dicts, 
                                        params, 
                                        alpha, 
                                        falsification_type, 
                                        iter_=iter_)
        results = results + results_to_append

    return results       

def run_gate(data_table, 
            data_dicts, 
            strata_mod='', 
            strata_metadata_mod='', 
            params='',
            alpha=0.05, 
            iter_=0, 
            ate=False,
            return_by_strata=False): 
    if ate: 
        strata = [('ATE')]
    elif strata_mod == '':
        strata = [
            (('b.marr','==',1,False),('bw','<',2000,True)), 
            (('b.marr','==',1,False),('bw','>=',2000,True)),
            (('b.marr','==',0,False),('bw','<',2000,True)),
            (('b.marr','==',0,False),('bw','>=',2000,True))            
        ]
    else: 
        strata = strata_mod
    
    if ate: 
        strata_metadata = [('ATE',True)]
    elif strata_metadata_mod == '':
        strata_metadata = [
            ('lbw, married',True), # (group name, whether or not strata is supported on RCT)
            ('hbw, married',True),
            ('lbw, single',True),
            ('hbw, single',True)
        ]
    else: 
        strata_metadata = strata_metadata_mod

    if ate: 
        print(f'[Running ATE]')
        cate_estimator = ATE(data_table, params=params, ate=ate)
    else: 
        print(f'[Running GATE for strata -- {strata} -- and metadata -- {strata_metadata}]')
        # define CATE estimator 
        cate_estimator = CATE(data_table, strata=strata, strata_metadata=strata_metadata, params = params)
    theta_hats, sd_hats = cate_estimator.rct_estimate(rct_table=data_dicts['rct-full'], \
                        y_name='y_rct', trt_name='treat', full = True)
    strata_names_rct = cate_estimator.get_strata_names(ate=ate)

    full_theta_obs = []; full_sd_obs = []
    for _ , obs_table in enumerate(data_dicts['obs']): 
        if params['reweighting']: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate_reweight(
                                            obs_table=obs_table, \
                                            rct_table=data_dicts['rct-full'], \
                                            feat_importance=False)
        else: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)

        full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
    strata_names_obs = cate_estimator.get_strata_names(ate=ate)

    falsifier = Falsifier(alpha=alpha)
    (lci_out_aos, uci_out_aos), (lci_selected, uci_selected), acc, (lci_oracle, uci_oracle) = falsifier.run_validation(
                                                                            theta_hats, 
                                                                            sd_hats, 
                                                                            full_theta_obs, 
                                                                            full_sd_obs, 
                                                                            strata_names=strata_names_obs, 
                                                                            return_acc = True, 
                                                                            return_oracle=True)

    ## baselines
    meta_baseline   = MetaAnalyzer(alpha=alpha)
    simple_baseline = SimpleBaseline(alpha=alpha)
    evo_baseline    = EvolvedMetaAnalyzer(alpha=alpha)
    lci_out_meta, uci_out_meta = meta_baseline.compute_intervals(full_theta_obs, \
                                        full_sd_obs, strata_names=strata_names_obs)
    lci_out_simple, uci_out_simple = simple_baseline.compute_intervals(full_theta_obs, \
                                        full_sd_obs, strata_names=strata_names_obs)
    lci_out_evo, uci_out_evo = evo_baseline.compute_intervals(full_theta_obs, \
                                        full_sd_obs, strata_names_obs, theta_hats, sd_hats)
    lci_out_rct = []; uci_out_rct = []
    for i in range(len(theta_hats)): 
        uci_out_rct.append(theta_hats[i] + norm.ppf(1-alpha/2) * sd_hats[i])
        lci_out_rct.append(theta_hats[i] - norm.ppf(1-alpha/2) * sd_hats[i])
    print('')
    if ate: 
        print('RCT ATE')
        print(f'mean (std) ATE: {theta_hats[0]} ({sd_hats[0]})')                           
        print('OBS ATEs')
        for i in range(len(full_theta_obs)): 
            print(f'mean (std) ATE: {full_theta_obs[i]} ({full_sd_obs[i]})')
    else: 
        print('RCT CATEs')
        for j,strata_name in enumerate(strata_names_rct): 
            print(f'mean (std) CATE of {strata_name} group: {theta_hats[j]} ({sd_hats[j]})')                           
        for i in range(len(full_theta_obs)): 
            print(f'OBS {i+1} CATES:')
            for j,strata_name in enumerate(strata_names_obs):
                print(f'mean (std) CATE of {strata_name} group: {full_theta_obs[i][j]} ({full_sd_obs[i][j]})')
            print('')
    print('')
    results_to_append = []
    if not return_by_strata: 
        if len(acc) != 0: 
                acc_idxs = [int(x) for x in acc.split(',')]
        else: 
            acc_idxs = []
        print(acc)
        for k in range(len(full_theta_obs)): 
            results_add = {'iter': iter_}
            results_add['obs_study_num'] = k+1
            results_add['obs_study_size'] = params['obs_dict']['sizes'][k]
            results_add['reject'] = int(k not in acc_idxs)
            results_to_append.append(results_add)
        return results_to_append

    for d,stratum in enumerate(strata_metadata): 
        name, in_rct = stratum
        if len(lci_out_aos) == 0:
            lci_aos = np.nan; uci_aos = np.nan
        else:
            lci_aos = lci_out_aos[d]; uci_aos = uci_out_aos[d]

        results_add = {
            'iter': iter_, 
            'strata_name': name,
            'accept': acc,
            'lci_out_rct': lci_out_rct[d], 
            'uci_out_rct': uci_out_rct[d], 
            'lci_out_aos': lci_aos, 
            'uci_out_aos': uci_aos, 
            'lci_out_meta': lci_out_meta[d], 
            'uci_out_meta': uci_out_meta[d],
            'lci_out_evo': lci_out_evo[d], 
            'uci_out_evo': uci_out_evo[d],
            'lci_out_simple': lci_out_simple[d],
            'uci_out_simple': uci_out_simple[d],
            'lci_out_oracle': lci_oracle[d],
            'uci_out_oracle': uci_oracle[d]
        }
        for k in range(len(full_theta_obs)): 
            results_add[f'obs_{k}_estimate'] = full_theta_obs[k][d]
            results_add[f'obs_{k}_sd'] = full_sd_obs[k][d]
        
        results_to_append.append(results_add)

    return results_to_append

def run_ate(data_table, 
            data_dicts, 
            params,
            alpha=0.05, 
            iter_=0): 
    return run_gate(data_table, data_dicts, strata_mod='', strata_metadata_mod='', \
        params=params, alpha=alpha, iter_=iter_, ate=True)

def run_mmr(data_table,
            data_dicts, 
            params, 
            alpha, 
            falsification_type, 
            visualize=False,
            iter_=0): 
    
    '''
        Part 2: estimation of signals for OBS studies
    '''
    stacked_tables   = []
    mmr_test_signals = []
    results_to_append = []
    for _, obs_table in enumerate(data_dicts['obs']): 
        n1 = obs_table.shape[0]; n0 = data_dicts['rct-full'].shape[0]
        params['oracle_params']['selection_model']['P_S0'] = n0 / (n0+n1)
        oe = OutcomeEstimator(rct_table=data_dicts['rct-full'], # TODO: make this rct-full? 
                                obs_table=obs_table,
                                params=params, 
                                rct_partial=False)
        stacked_tables.append(oe.get_stacked_table())
        U_obs_a1, U_obs_a0 = oe.estimate_signals(S=1)
        
        '''
            Part 3: estimation of signals for RCT studies
        '''            
        U_rct_a1, U_rct_a0 = oe.estimate_signals(S=0)
        psi1 = U_obs_a1 - U_rct_a1 
        psi0 = U_obs_a0 - U_rct_a0  
        mmr_test_signals.append((psi0,psi1))
        

    '''
        Part 4: write new falsifier that incorporates MMR test 
        
    '''
    falsifier = FalsifierMMR(params=params, alpha=alpha,\
        kernel=params['kernel'], falsification_type=falsification_type)
    for k, _ in enumerate(data_dicts['obs']):  
        p_val = falsifier.run_test(stacked_tables[k], mmr_test_signals[k], B=100)
        results_add = {'iter': iter_}
        results_add['obs_study_num'] = k+1 
        results_add['obs_study_size'] = params['obs_dict']['sizes'][k]
        results_add['p_val'] = p_val
        results_add['reject'] = int(p_val < alpha)

        if visualize: 
            print('[Visualizing witness function!]')
            covariate_names = ['nnhealth', 'booze']; covariate_types = ['continuous', 'binary']
            f, Xmean_rep, covariate_idxs = falsifier.visualize_witness_func(stacked_tables[k], mmr_test_signals[k], \
                covariate_names=covariate_names, covariate_types=covariate_types)
            if len(covariate_names) == 2 and 'binary' in covariate_types:
                # precondition: always put binary covariate second
                pos_idxs = np.where(Xmean_rep[:,covariate_idxs[1]] == 1.)
                neg_idxs = np.where(Xmean_rep[:,covariate_idxs[1]] == 0.)
                x_coord  = Xmean_rep[:,covariate_idxs[0]]
                cov_mean, cov_std = data_table.get_normalizing_factors(covariate_names[0])
                x_coord_orig = (x_coord*cov_std)+cov_mean
                x_coord1 = x_coord_orig[pos_idxs]; x_coord0 = x_coord_orig[neg_idxs]
                f_coord1 = f[pos_idxs]; f_coord0 = f[neg_idxs]
                results_add['f_coord_pos'] = f_coord1; results_add['f_coord_neg'] = f_coord0 
                results_add['x_coord_pos'] = x_coord1; results_add['x_coord_neg'] = x_coord0
            elif len(covariate_names) == 2: 
                x1_coord = Xmean_rep[:,covariate_idxs[0]]
                x2_coord = Xmean_rep[:,covariate_idxs[1]]
                cov_mean1, cov_std1 = data_table.get_normalizing_factors(covariate_names[0])
                cov_mean2, cov_std2 = data_table.get_normalizing_factors(covariate_names[1])
                x1_coord_orig = (x1_coord*cov_std1)+cov_mean1
                x2_coord_orig = (x2_coord*cov_std2)+cov_mean2
                results_add['f'] = f
                results_add['x1_coord'] = x1_coord_orig
                results_add['x2_coord'] = x2_coord_orig
            elif len(covariate_names) == 1: 
                x1_coord = Xmean_rep[:,covariate_idxs[0]]
                cov_mean1, cov_std1 = data_table.get_normalizing_factors(covariate_names[0])
                x1_coord_orig = (x1_coord*cov_std1)+cov_mean1
                results_add['f'] = f
                results_add['x1_coord'] = x1_coord_orig
            results_add['covariate_names'] = covariate_names
        results_to_append.append(results_add) 
    
    return results_to_append

def get_nuisance_params(use_oracle=False): 
    if use_oracle: 
        return [('selection_model', {
            'model_name': 'SelectionModelOracle', 
            'hp': {},
            'model_type': 'binary'
        }),
        ('response_surface_1', {
            'model_name': 'ResponseSurfaceOracle-1',
            'hp': {},
            'model_type': 'continuous'
        }),
        ('response_surface_0', {
            'model_name': 'ResponseSurfaceOracle-0',
            'hp': {},
            'model_type': 'continuous'
        })]
    else: 
        return [('selection_model', {
            'model_name': 'GradientBoostingClassifier',
            'hp': {
                'learning_rate': [0.01], 'n_estimators': [50],\
                'max_depth': [2], 'min_samples_leaf': [50], \
                'min_samples_split': [50], 'max_features': ['sqrt'],
                'random_state': [42]
            },
            'model_type': 'binary'
        }),
        ('response_surface_1', {
            'model_name': 'LinearRegression',
            'hp': {},
            'model_type': 'continuous'
        }),
        ('response_surface_0', {
            'model_name': 'LinearRegression',
            'hp': {},
            'model_type': 'continuous'
            # 'model_name': 'RandomForestRegressor', 
            # 'hp': {'n_estimators': [100,250,350], 'max_depth': [10], \
            #         'min_samples_split': [10], 'max_features': ['sqrt']
            #     }, 
            # 'model_type': 'continuous'
        })]

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--falsification_type', type=str, default='MMR-Absolute',
                        help='MMR-Absolute, MMR-Contrast, GATE, or ATE')
    parser.add_argument('-c', '--config', type=str, default='configs/demo.yaml',
                        help='give path to simulation config.')
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
    parser.add_argument('-o', '--oracle_estimator', type=strtobool, default=False)
    
    args   = parser.parse_args()    
    config_info = read_yaml(path=args.config)
    cc = args.confounder_conc
    params_mod = [
        ('obs_dict', {
            'num_obs': 1,
            'sizes': [3.,4.,5.,7.,10], # TODO: play with this a little bit
            'confounder_concealment': [cc,cc,cc,cc,cc], # will be concealed according to ordering of coefficients
            'missing_bias': [False,False,False,False,False], 
            # 'sizes': [1.], 
            # 'confounder_concealment': [cc],
            # 'missing_bias': [False],
            'effect_mod_concealment': args.effect_mod_concealment
        }),
        ('rct_dict', {
            'size': args.rct_size
        }),
        ('response_surface', {
            'ctr': args.ctr_response_surface, 
            'trt': 'linear'
        }),
        ('reweighting_type', args.reweighting_type),
        ('reweighting_factor', args.reweighting_factor), 
        ('confounding_type', 'linear-XZ-v2'), 
        ('num_continuous', 7),
        ('num_binary', 0), 
        ('wparam', args.wparam)
    ]
    nuisance_params = get_nuisance_params(args.oracle_estimator)
    params_mod = params_mod + nuisance_params
    results = run_simulation(num_iters=args.num_iters, params_mod=params_mod, falsification_type=args.falsification_type) # config_info['params_mod'][0]
    pprint.pprint(results, sort_dicts=False)
    R_inter = pd.DataFrame(results)
    R_inter.to_csv(os.path.join(f"./mmr_results/{args.save_folder_name}/{args.exp_name}.csv"))
    
    # parallel_simulation(save_folder_name = config_info['save_folder_name'],
    #                       num_iters = config_info['num_iters'],
    #                       alpha = config_info['alpha'],
    #                       root = config_info['root'],
    #                       strata_mod = config_info['strata_mod'],
    #                       strata_metadata_mod = config_info['strata_metadata_mod'],
    #                       params_mod = config_info['params_mod'])
