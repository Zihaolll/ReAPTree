from ReAPtree_Experiments import Experiments

import os
import joblib
import numpy as np

dataset_name = 'banknote'  # or 'breast_cancer', 'banknote', 'compas', 'iris'
EnsembleMechanism = 'GBDT'  # or 'GBDT'

path = 'Explanation_Task_ReAPtree'
load_path_information = os.path.join(path, dataset_name, 'dataset_information.pkl')
dataset_information = joblib.load(load_path_information)

load_path_bbx = os.path.join(path, dataset_name, EnsembleMechanism)
train_idx = np.load(os.path.join(load_path_bbx, 'train.npy'), allow_pickle=True)
val_idx = np.load(os.path.join(load_path_bbx, 'val.npy'), allow_pickle=True)
test_idx = np.load(os.path.join(load_path_bbx, 'test.npy'), allow_pickle=True)
TE = joblib.load(os.path.join(load_path_bbx, 'TreeEnsemble.pkl'))

subspace_param = {'peaks or valleys': 'valleys', 'param': 10}
SubspaceAPtree_param = {
    'max_step': 5,  # Integer indicating the maximum step of the ReAPtree
    'test_idx': test_idx,

    # vector_distance can be one of: 'L1', 'L2', 'KL', 'CL', 'NL'.
    # Note:
    # - When 'vector_distance' is 'CL' or 'NL', 'value_space' MUST be 'probability'.
    # - If the original model is GBDT, 'NL' is NOT supported.
    'vector_distance': 'KL',

    # value_space specifies how the prediction values are represented:
    # - For 'CL' or 'NL', value_space must be 'probability'.
    # - For 'L1', 'L2', or 'KL', value_space can also be 'probability'.
    # - For Random Forest (RF), 'onehot' is allowed.
    # - For GBDT, ONLY 'probability' is allowed.
    'value_space': 'probability',

    'subspace_param': subspace_param,
    'subaptree_max_step': 5,  # Integer indicating the maximum step of the SubspaceAPtree

    # n_jobs specifies the number of parallel threads to use.
    # -1 indicates using all available CPU cores.
    'n_jobs': -1
}

exp = Experiments(TE, dataset_information, SubspaceAPtree_param)
result = exp.run([])
print(result)
