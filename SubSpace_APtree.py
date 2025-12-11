from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from Explainer_SubSpace import Explain_RF_via_SimpleFunction
import numpy as np
from sklearn import metrics
import time
from GBF_to_interval import _parse_GBF_interval
from Create_SubSpace import CreateSubspace
from cedf import set_ecdf
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import sys
import copy
import numpy as np


def dataset_interval_predict(dataset, interval_set, predictions):
    dataset = np.asarray(dataset)
    interval_set = np.asarray(interval_set)
    predictions = np.asarray(predictions)

    M, n = dataset.shape
    K = interval_set.shape[0]
    C = predictions.shape[1]

    lower = interval_set[:, :, 0]
    upper = interval_set[:, :, 1]

    ge_lower = dataset[:, None, :] >= lower[None, :, :]
    le_upper = dataset[:, None, :] <= upper[None, :, :]
    inside = (ge_lower & le_upper).all(axis=-1)

    default_pred = predictions.mean(axis=0)

    final_pred = np.zeros((M, C))
    for m in range(M):
        mask = inside[m]
        if mask.any():
            final_pred[m] = predictions[mask].mean(axis=0)
        else:
            final_pred[m] = default_pred

    return final_pred


class SubspaceAPtree:
    def __init__(self, TreeEnsemble, dataset_information, max_step, test_idx, val_idx=None, task_region=None,
                 measure='Distribution', vector_distance='NL', value_space_switch='onehot', ecdf=None,
                 subspace_param={None}, subaptree_max_step=3, n_jobs=-1):
        self.rf = TreeEnsemble
        self.TE = TreeEnsemble
        self.di = dataset_information
        self.max_step = max_step
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.task_region = task_region
        self.ms = measure
        self.vds = vector_distance
        self.ecdf = ecdf
        self.subaptree_max_step = subaptree_max_step
        self.n_jobs = n_jobs
        self.vss = value_space_switch

        if self.vds == 'CL' or self.vds == 'NL':
            if self.vss != 'onehot':
                print('The chosen vector distance is not compatible with the prediction value space.', self.vds, self.vss)
                sys.exit()

        self.subspace_param = subspace_param

        self.test_X = self.di['X'][self.test_idx]
        self.test_Y = self.di['Y'][self.test_idx]
        self.TE_Y = self.TE.predict(self.test_X)
        self.TE_accuracy = metrics.accuracy_score(self.test_Y, self.TE_Y)
        self.TE_f1_score = metrics.f1_score(self.test_Y, self.TE_Y, average='macro')

        if self.ecdf is None:
            self.ecdf = set_ecdf(self.di['X'])

        if isinstance(self.TE, RandomForestClassifier):
            self.forest_name = 'RF'
        elif isinstance(self.TE, GradientBoostingClassifier):
            self.forest_name = 'GBDT'

        self.subspace_list = CreateSubspace(self.TE, self.di['space_lower_bound'], self.di['space_upper_bound'],
                                            self.di['space_lower_bound'], self.di['space_upper_bound'],
                                            filter_paramters=self.subspace_param, ecdf=self.ecdf).generate_subspaces()

        print('n_subspace', len(self.subspace_list))

    def generate_subspaceAPtree(self):
        SSAPtree_I = []
        SSAPtree_V = []
        SSAPtree_D = []

        def process_one(subspace):
            Explainer = Explain_RF_via_SimpleFunction(
                rf=self.TE,
                dataset_information=self.di,
                val_idx=self.val_idx,
                measure_switch=self.ms,
                vector_distance_switch=self.vds,
                value_space_switch=self.vss,
                ecdf=self.ecdf,
                task_region=subspace)


            SSAPtree = Explainer.generate_explanation(self.subaptree_max_step)
            SSAPtree.pruning()
            SF_Iset, SF_Vset, SF_Dset = SSAPtree.get_interval_and_value()
            return SF_Iset, SF_Vset, SF_Dset


        with tqdm_joblib(tqdm(desc="Processing subspaces", total=len(self.subspace_list))):
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(process_one)(subspace) for subspace in self.subspace_list
            )

        for SF_Iset, SF_Vset, SF_Dset in results:
            SSAPtree_I.append(SF_Iset)
            SSAPtree_V.append(SF_Vset)
            SSAPtree_D.append(SF_Dset)

        self.IS = np.vstack(SSAPtree_I)
        self.VS = np.vstack(SSAPtree_V)
        self.DS = np.vstack(SSAPtree_D)

        self.SSAPtree_result = self.analysis_interval_prediction(self.IS, self.VS)

        return SSAPtree_I, SSAPtree_V, SSAPtree_D

    def generate_wholespaceAPtree(self, wholespace_vds='CL'):

        SSAPtree_I, SSAPtree_V, SSAPtree_D = self.generate_subspaceAPtree()

        IS = np.vstack(SSAPtree_I)
        VS = np.vstack(SSAPtree_V)
        DS = np.vstack(SSAPtree_D)
        self.IandP = {}
        self.IandP['interval set'] = IS
        self.IandP['prediction set'] = VS

        Explainer = Explain_RF_via_SimpleFunction(rf=self.IandP,
                                                  dataset_information=self.di,
                                                  val_idx=self.val_idx,
                                                  measure_switch=self.ms,
                                                  vector_distance_switch=wholespace_vds,
                                                  ecdf=self.ecdf)
        SF = Explainer.generate_explanation(self.max_step)
        self.SF_before_pruning = copy.deepcopy(SF)
        SF.pruning()

        result, p_v, p_c = self.analysis_SF(SF)
        result['TE_accuracy'] = self.TE_accuracy
        result['TE_f1'] = self.TE_f1_score
        result['SubSpace_n'] = len(self.subspace_list)
        return result, p_v, p_c

    def analysis_SF(self, SimpleFunction):
        SF_result = {}
        SF = SimpleFunction
        SF_Iset, SF_Vset, SF_Dset = SF.get_interval_and_value()
        SF_Y_class = []
        SF_Y_vector = []
        SF_Y_step = []
        for i in range(self.test_X.shape[0]):
            x = self.test_X[i]
            v, _, step = SF.value_and_step_count(x)
            SF_Y_vector.append(v)
            SF_Y_class.append(self.TE.classes_[np.argmax(v)])
            SF_Y_step.append(step)

        SF_result['partitions_n'] = len(SF_Iset)

        SF_result['fidelity_accuracy'] = metrics.accuracy_score(self.TE_Y, SF_Y_class)
        SF_result['fidelity_f1'] = metrics.f1_score(self.TE_Y, SF_Y_class, average='macro')

        SF_result['predict_accuracy'] = metrics.accuracy_score(self.test_Y, SF_Y_class)
        SF_result['predict_f1'] = metrics.f1_score(self.test_Y, SF_Y_class, average='macro')

        return SF_result, SF_Y_vector, SF_Y_class

    def analysis_interval_prediction(self, intervals, predictions):
        IP_result = {}
        Iset = intervals
        Vset = predictions

        IP_pred_prob = dataset_interval_predict(self.test_X, Iset, Vset)
        IP_pred = self.TE.classes_[np.argmax(IP_pred_prob, axis=1)]

        IP_result['partitions_n'] = len(Iset)

        IP_result['fidelity_accuracy'] = metrics.accuracy_score(self.TE_Y, IP_pred)
        IP_result['fidelity_f1'] = metrics.f1_score(self.TE_Y, IP_pred, average='macro')

        IP_result['predict_accuracy'] = metrics.accuracy_score(self.test_Y, IP_pred)
        IP_result['predict_f1'] = metrics.f1_score(self.test_Y, IP_pred, average='macro')

        return IP_result
