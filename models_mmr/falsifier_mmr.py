import imp
import pandas as pd 
import numpy as np
import model_util
import itertools
import multiprocessing
import time 
from numpy.random import default_rng
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from pykeops.numpy import generic_sum 
from pykeops.numpy import Genred

def bootstrap(rng, 
              b=100, 
              num_rct=100,
              num_obs=100,
              N=100,
              h=None,
              X=None,
              contrast=None):

    # computing bootstrap weights
    t = time.time()
    p_rct = np.ones(num_rct) / num_rct 
    multi_rct = rng.multinomial(num_rct, p_rct)
    p_obs = np.ones(num_obs) / num_obs
    multi_obs = rng.multinomial(num_obs, p_obs) 
    ret = np.concatenate((multi_obs,multi_rct))
    rho = (ret - 1) / N 
    if h is None: 
        # do keops
        M1_term = Genred(' ((Sum(X1 * X2)* gamma + 1)*(Sum(X1 * X2)* gamma + 1)*(Sum(X1 * X2)* gamma + 1)) * psi_prime', 
                                [   f'X1 = Vj({X.shape[1]})',
                                    f'X2 = Vi({X.shape[1]})', 
                                    'gamma = Pm(1)',
                                    'psi_prime = Vi(1)'], 
                                reduction_op='Sum', 
                                axis=0)
        Mboot_term = Genred(' ((Sum(X * X)* gamma + 1)*(Sum(X * X)* gamma + 1)*(Sum(X * X)* gamma + 1)) * Square(psi_prime)', 
                                [f'X = Vi({X.shape[1]})', 
                                'gamma = Pm(1)',
                                'psi_prime = Vi(1)'], 
                                reduction_op='Sum', 
                                axis=0)
        

        gamma = 1.0 / X.shape[1]
        Xcont = np.ascontiguousarray(X, dtype=np.float64)
        gamma_cont = np.ascontiguousarray(np.array([gamma]), dtype=np.float64)
        psi_prime  = contrast[:,None]*rho[:,None]
        psi_prime_cont = np.ascontiguousarray(psi_prime, dtype=np.float64)
        s3_2 = Mboot_term(Xcont, gamma_cont, psi_prime_cont)
        s3_1 = M1_term(Xcont, Xcont, gamma_cont, psi_prime_cont)
        return (N*(np.dot(s3_1.squeeze(), psi_prime.squeeze()) - s3_2.squeeze()))
    
    M_full   = np.outer(rho, rho) * h
    M   = np.sum(M_full)
    return N*M

class FalsifierMMR(): 
    
    def __init__(self, 
                 params={}, 
                 alpha=0.05, 
                 kernel='polynomial', 
                 falsification_type='MMR-Absolute' 
                ):

        self.alpha  = alpha
        self.params = params
        self.rng = default_rng(params['grand_seed'])
        self.kernel = kernel
        self.falsification_type = falsification_type
        self.bw_sq  = None
        valid_kernels = ['rbf','polynomial', 'laplace']
        assert kernel in valid_kernels, 'invalid kernel specified.'
    
    def run_test(self, 
                 stacked_table, 
                 signals, 
                 B=1000, 
                 parallel=False
                ): 
        ''' 
            1 set of signals 
        '''
        
        # define RBF kernel with bw based on median heuristic 
        X, _, _, _ = model_util._get_numpy_arrays(self.params, stacked_table)
        N = X.shape[0]

        # compute M-squared statistic 
        if 'KEOPS' not in self.falsification_type: 
            if self.kernel == 'rbf': 
                t = time.time()
                distX = pairwise_distances(X,X)
                print(f'[1a] time elapsed for pairwise distances: {time.time()-t}')
                t = time.time() 
                self.bw_sq = np.median(distX[np.tril_indices(distX.shape[0],k=-1)]) / 2
                print(f'[1b] time elapsed for median heuristic computation: {time.time()-t}')
                rbf_kernel = RBF(length_scale=self.bw_sq)
                k = rbf_kernel(X,X)
            elif self.kernel == 'polynomial': 
                t = time.time()
                k = polynomial_kernel(X,X)
                print(f'[2] time elapsed for kernel computation: {time.time()-t}')
            elif self.kernel == 'laplace': 
                t = time.time() 
                k = laplacian_kernel(X,X)
                print(f'[2] time elapsed for kernel computation: {time.time() - t}')

            if self.falsification_type == 'MMR-Absolute':       
                h = k*(np.outer(signals[0],signals[0]) + np.outer(signals[1],signals[1]))
            elif self.falsification_type == 'MMR-Contrast':
                contrast = signals[1]-signals[0]
                t = time.time()
                h = k*np.outer(contrast, contrast)
                print(f'[3] time elapsed for h computation: {time.time()-t}')

            t = time.time()
            np.fill_diagonal(h,0)
            M2 = np.sum(h) * (1 / (N * (N-1)))
            print(f'[4] time elapsed for M^2 computation: {time.time()-t}')
        else: 
            # TODO: implement keops version of kernel computatioon  
            if self.falsification_type == 'MMR-Contrast-KEOPS': 
                contrast = signals[1]-signals[0]
                # contrast = np.random.normal(0., 0.01, size=len(signals[0]))
                t = time.time()
                M2_term = Genred(' ((Sum(X * X)* gamma + 1)*(Sum(X * X)* gamma + 1)*(Sum(X * X)* gamma + 1)) * Square(C)', 
                                [f'X = Vi({X.shape[1]})', 
                                    'gamma = Pm(1)',
                                    'C = Vi(1)'], 
                                reduction_op='Sum', 
                                axis=0)
                M1_term = Genred(' ((Sum(X1 * X2)* gamma + 1)*(Sum(X1 * X2)* gamma + 1)*(Sum(X1 * X2)* gamma + 1)) * C', 
                                [   f'X1 = Vj({X.shape[1]})',
                                    f'X2 = Vi({X.shape[1]})', 
                                    'gamma = Pm(1)',
                                    'C = Vi(1)'], 
                                reduction_op='Sum', 
                                axis=0)
                gamma = 1.0 / X.shape[1]
                Xcont = np.ascontiguousarray(X, dtype=np.float64)
                gamma_cont = np.ascontiguousarray(np.array([gamma]), dtype=np.float64)
                contrast_cont = np.ascontiguousarray(contrast[:,None], dtype=np.float64)
                s3_2 = M2_term(Xcont, gamma_cont, contrast_cont)
                s3_1 = M1_term(Xcont, Xcont, gamma_cont, contrast_cont)
                M2 = (1 / (N * (N-1)))*(np.dot(s3_1.squeeze(), contrast) - s3_2.squeeze())
                print(f'[total time for kernel and M2 computation: {time.time()-t}]')
        
        # stratified bootstrap 
        if self.params['ihdp']: 
            num_rct = len(stacked_table[stacked_table['S'] == 0])
            num_obs = len(stacked_table[stacked_table['S'] == 1])
        else: 
            num_rct = len(stacked_table[stacked_table['OS'] == 0])
            num_obs = len(stacked_table[stacked_table['OS'] == 1])
        M2_boot = []
        if parallel: 
            args = [([self.rng, b, num_rct, num_obs, N, h],) for b in range(B)]
            args = tuple(args)
            pool = multiprocessing.Pool(8) 
            print('starting parallel bootstrap...')
            M2_boot = pool.starmap(bootstrap, args)
        else: 
            tb = time.time()
            for b in range(B): 
                if b % 20 == 0: 
                    print(f'MMR bootstrap iter {b}')
                if 'KEOPS' in self.falsification_type: 
                    t = time.time()
                    M2_boot.append(bootstrap(self.rng, b, num_rct, num_obs, N, h=None, X=X, contrast=contrast))
                    print(f'time elapsed for 1 bootstrap computation: {time.time()-t}')
                else: 
                    M2_boot.append(bootstrap(self.rng, b, num_rct, num_obs, N, h=h, X=None, contrast=None))
            print(f'[5] time elapsed for all bootstrap computations: {time.time()-tb}')
        M2_boot = np.array(M2_boot)
        # gamma = np.quantile(np.array(M2_boot),1-self.alpha)
        t = time.time()
        p_val = (np.sum(M2_boot >= N*M2) + 1) / (B + 1)
        print(f'[6] time elapsed for p-value computation: {time.time()-t}')
        return p_val

    def get_witness_func(self, Xi, stacked_table, signals):
        X, _, _, _ = model_util._get_numpy_arrays(self.params, stacked_table)
        psi0 = signals[0]; psi1 = signals[1]
        
        # define kernel
        if self.kernel == 'rbf':    
            rbf_kernel = RBF(length_scale=self.bw_sq)
            k = rbf_kernel(X,Xi)
        elif self.kernel == 'polynomial': 
            k = polynomial_kernel(X,Xi) 

        # f0, f1
        f1 = np.mean((psi1[:,None]*k),axis=0)
        f0 = np.mean((psi0[:,None]*k),axis=0)
        
        if self.falsification_type == 'MMR-Absolute': 
            return (f1,f0)
        elif 'MMR-Contrast' in self.falsification_type: 
            return f1-f0
    
    def visualize_witness_func(self, stacked_table, signals, \
            covariate_names=['b.head'], covariate_types=['continuous']):
        assert len(covariate_names) <= 2, 'can give up to two covariates to alter.' 
        assert len(covariate_types) <= 2, 'can give up to two covariates to alter.'
        assert len(covariate_names) == len(covariate_types) 
        if self.params['ihdp']: 
            stacked_table_subset = stacked_table.drop(columns=['y_hat','S','treat'], inplace=False)
        else: 
            stacked_table_subset = stacked_table.drop(columns=['ID', 'OS', 'HRTARM', 'EVENT']\
                 + [f for f in stacked_table.columns.values if f.endswith('_E') or f.endswith('_DY')],\
                    inplace=False)
        X = stacked_table_subset.values
        col_names = stacked_table_subset.columns.values

        Xi_seqs = []; covariate_idxs = []
        for i in range(len(covariate_names)): 
            name = covariate_names[i]
            cov_type = covariate_types[i]
            covariate_number = np.where(col_names == name)[0][0]
            covariate_idxs.append(covariate_number)
            
            if cov_type == 'continuous':   
                minXi = np.min(X[:,covariate_number])
                maxXi = np.max(X[:,covariate_number])
                Xi_seq = np.linspace(minXi, maxXi, num=100)
            else: 
                Xi_seq = np.unique(X[:,covariate_number])
            
            Xi_seqs.append(Xi_seq)
        
        Xmean = np.mean(X,axis=0)

        if len(Xi_seqs) == 2: 
            res = list(itertools.product(Xi_seqs[0], Xi_seqs[1]))
            Xi1 = [x[0] for x in res]; Xi2 = [x[1] for x in res]
            Xi1_idx = covariate_idxs[0]; Xi2_idx = covariate_idxs[1]
            Xmean_rep = np.tile(Xmean, (len(res),1))
            Xmean_rep[:,Xi1_idx] = Xi1 
            Xmean_rep[:,Xi2_idx] = Xi2
        else: 
            Xi_seq = Xi_seqs[0]
            Xmean_rep = np.tile(Xmean, (len(Xi_seq),1))
            Xmean_rep[:,covariate_idxs[0]] = Xi_seq

        f = self.get_witness_func(Xmean_rep, stacked_table, signals)
        return f, Xmean_rep, covariate_idxs