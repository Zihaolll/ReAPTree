from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from GBF_to_interval import _parse_GBF_interval
from Generate_SimpleFunction import Generate_SimpleFunction
from Gain_Function import GainFunc
from Get_SplitPoint_Set import interval_set_to_split_point_set
import numpy as np
from RF_to_interval import _parse_forest_interval
import copy
import xgboost as xgb


class Explain_RF_via_SimpleFunction:
    def __init__(self, rf, dataset_information, val_idx=None, measure_switch='Lebesgue',
                 vector_distance_switch='NL',
                 value_space_switch='onehot', task_region=None, ecdf=None, m_topn=-1):
        self.rf = rf
        if isinstance(self.rf, RandomForestClassifier):
            self.forest_name = 'RF'
        elif isinstance(self.rf, GradientBoostingClassifier):
            self.forest_name = 'GBDT'
        elif isinstance(self.rf, dict):
            self.forest_name = 'IntervalAndPrediction'

        self.di = dataset_information
        self.val_idx = val_idx
        self.ms = measure_switch
        self.vds = vector_distance_switch
        self.vss = value_space_switch
        self.ecdf = ecdf
        self.m_topn = m_topn
        if task_region is not None:
            self.task_region = task_region
        else:
            self.task_region = np.array([self.di['space_lower_bound'], self.di['space_upper_bound']]).T



        self.gainfunc = GainFunc(rf=self.rf, dataset_information=self.di, val_idx=self.val_idx,
                                 measure_switch=self.ms, vector_distance_switch=self.vds,
                                 value_space_switch=self.vss, ecdf=self.ecdf, m_topn=self.m_topn,
                                 task_region=self.task_region)

        if self.forest_name == 'RF':
            interval_set, _ = _parse_forest_interval(rf, left_bounds=self.di['space_lower_bound'],
                                                     right_bounds=self.di['space_upper_bound'],
                                                     subspace_left_bounds=self.task_region[:, 0],
                                                     subspace_right_bounds=self.task_region[:, 1])
        elif self.forest_name == 'GBDT':
            interval_set, _ = _parse_GBF_interval(rf, left_bounds=self.di['space_lower_bound'],
                                                  right_bounds=self.di['space_upper_bound'],
                                                  subspace_left_bounds=self.task_region[:, 0],
                                                  subspace_right_bounds=self.task_region[:, 1])
        elif self.forest_name == 'IntervalAndPrediction':
            interval_set = self.rf['interval set']

        for interval in interval_set:
            lower_ok = np.all(interval[:, 0] < self.task_region[:, 0])
            upper_ok = np.all(interval[:, 1] > self.task_region[:, 1])
            if not lower_ok and upper_ok:
                print("interval", interval)
                print("task region", self.task_region)

        self.sp_set = interval_set_to_split_point_set(interval_set)

    def generate_explanation(self, max_step):
        SF = Generate_SimpleFunction(init_interval=self.task_region, split_point_set=self.sp_set,
                                     max_step=max_step, gainfunction=self.gainfunc)

        SF.split(0)
        return SF
