from SubSpace_APtree import SubspaceAPtree
import pandas as pd
import numpy as np
import copy


class Experiments:
    def __init__(self, TreeEnsemble, dataset_information, SubspaceAPtree_param):
        self.TE = TreeEnsemble
        self.dataset_information = dataset_information
        self.param = SubspaceAPtree_param
        self.SAPtree = SubspaceAPtree(self.TE, self.dataset_information, max_step=self.param['max_step'],
                                      test_idx=self.param['test_idx'],
                                      vector_distance=self.param['vector_distance'],
                                      value_space_switch=self.param['value_space'],
                                      subspace_param=self.param['subspace_param'],
                                      subaptree_max_step=self.param['subaptree_max_step'],
                                      n_jobs=self.param['n_jobs'])

    def run(self, cut_param_list):

        result, _, _ = self.SAPtree.generate_wholespaceAPtree()
        for cut_param in cut_param_list:
            if cut_param < self.SAPtree.max_step:
                CSF = copy.deepcopy(self.SAPtree.SF_before_pruning)
                CSF.cutting(cut_param)
                CSF.pruning()
                result_cut, _, _ = self.SAPtree.analysis_SF(CSF)
                result_c = {k + "_%d" % cut_param: v for k, v in result_cut.items()}

                result.update(result_c)

        result_pd = pd.DataFrame.from_dict(result, orient='index')
        return result_pd


