import pandas as pd
import numpy as np

def _stack(params, rct_table, obs_table): 
    ''' 
        Stack the tables (and adjust number of covariates)
    '''
    if params['ihdp']: 
        rct_table = rct_table.copy()
        num_covariates = params['num_continuous'] + params['num_binary']
        rct_table.insert(loc=0, column=f'S', value=np.zeros((rct_table['treat'].shape[0],)))
        obs_table.insert(loc=0, column=f'S', value=np.ones((obs_table['treat'].shape[0],)))
        rct_table.drop(columns=['y1_rct','y0_rct'], inplace=True)
        obs_table.drop(columns=['y1_obs','y0_obs'], inplace=True)
        rct_rename = {'y_rct': 'y_hat'}
        rct_rename.update({f'xprime_rct{i+1}':f'xprime{i+1}' \
                                            for i in range(num_covariates)})
        obs_rename = {'y_obs': 'y_hat'}
        obs_rename.update({f'xprime_obs{i+1}':f'xprime{i+1}' \
                                            for i in range(num_covariates)})
        rct_table.rename(columns=rct_rename, inplace=True)
        obs_table.rename(columns=obs_rename, inplace=True)
    
    pooled_table = pd.concat((obs_table, rct_table),axis=0,sort=False).reset_index(drop=True)
    pooled_table = pooled_table.dropna(axis=1)
    return pooled_table

def _get_numpy_arrays(params, table): 
        if params['ihdp']: 
            X = table.drop(columns=['y_hat','S','treat'], inplace=False).values
            Y = table['y_hat'].values
            T = table['treat'].values
            S = table['S'].values
        else: 
            X = table.drop(columns=['ID', 'OS', 'HRTARM', 'EVENT']\
                 + [f for f in table.columns.values if f.endswith('_E') or f.endswith('_DY')],\
                    inplace=False).values
            Y = table['EVENT'].values
            T = table['HRTARM'].values
            S = table['OS'].values

        return X, Y, T, S