import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from Interval_IOU import IoU, IoU_set
from RF_to_interval import _parse_forest_interval
from Exhaustivity_intersect import ExhaustivityIntersect
import time
from statsmodels.distributions.empirical_distribution import ECDF
from Distribution_measure import distribution_measure_set, distribution_measure
from GBF_to_interval import _parse_GBF_interval
# from scipy.special import rel_entr
import sys
from Distance_Optimize import barycenter


class SimpleFuncDis:
    def __init__(self, integration_domain, integration_data=None, measure_switch='Lebesgue',
                 vector_distance_switch='NL', ecdf=None):
        self.domain = integration_domain
        self.ms = measure_switch
        self.vds = vector_distance_switch
        self.data = integration_data
        self.ecdf = ecdf

        if self.ecdf is None:
            self.ecdf = {i: ECDF(self.data.T[i]) for i in range(self.data.shape[1])}

    def calculate(self, sf1_intervals, sf1_probas, sf2_intervals, sf2_probas):

        IS1 = sf1_intervals.copy()
        PS1 = sf1_probas.copy()
        IS2 = sf2_intervals.copy()
        PS2 = sf2_probas.copy()
        dis = 0
        MU = 0
        for i in range(len(IS2)):
            I, P = IS2[i], PS2[i]

            I_area = IoU(self.domain, I).intersect_rate_for_interval1()
            intersect_set = IoU_set(I, IS1)

            if self.ms == 'Lebesgue':
                mu = intersect_set.intersect_rate_for_interval1() * I_area  # Lebesgue measure: area(intersect/space)

            elif self.ms == 'Distribution':
                mu = distribution_measure_set(intersect_set.intersect(), self.ecdf)
                mu[intersect_set.none_mask] = 0
                if sum(intersect_set.none_mask - 1) == 0:
                    avg_p = np.ones(P.shape[0]) / P.shape[0]

                    PS1 = np.tile(avg_p, (PS1.shape[0], 1))
                    mu[intersect_set.none_mask] = distribution_measure(I, self.ecdf) / PS1.shape[0]

            if self.vds == 'NL':
                P_class = np.argmax(P)
                vd = (1 - PS1[:, P_class])
            elif self.vds == 'CL':
                P_class = np.argmax(P)
                vd = (np.argmax(PS1, axis=1) != P_class) + 0
            elif self.vds == 'L2':
                vd = np.linalg.norm((PS1 - P), axis=1)
            elif self.vds == 'KL':
                eps = 1e-10
                P_class = np.argmax(P)
                vd = -np.log(np.clip(PS1[:, P_class], eps, 1.0))

            muvd_set = mu * vd

            dis += sum(muvd_set)
            MU += mu

        return dis

    def search_min_dis(self, sf1_intervals, sf1_probas, interval):

        IS1 = sf1_intervals.copy()
        PS1 = sf1_probas.copy()
        I = interval
        # dis = 0
        # MU = 0

        I_area = IoU(self.domain, I).intersect_rate_for_interval1()
        intersect_set = IoU_set(I, IS1)

        if self.ms == 'Lebesgue':
            mu = intersect_set.intersect_rate_for_interval1() * I_area

        elif self.ms == 'Distribution':
            muuuu = distribution_measure_set(intersect_set.intersect(), self.ecdf)
            none_mask = intersect_set.none_mask
            muuuu[none_mask] = 0
            mu = muuuu
            if sum(intersect_set.none_mask - 1) == 0:
                avg_p = np.ones(PS1.shape[1]) / PS1.shape[1]
                # print("avg_p", avg_p)
                PS1 = np.tile(avg_p, (PS1.shape[0], 1))
                mu[intersect_set.none_mask] = distribution_measure(I, self.ecdf) / PS1.shape[0]

        best_p, _, dis_min, _ = barycenter(PS1, mu, metric=self.vds)

        return dis_min, best_p


class RFSimpleFuncDis:
    def __init__(self, rf, dataset_information, val_idx=None, measure_switch='Lebesgue', vector_distance_switch='NL',
                 task_region=None, ecdf=None, m_topn=-1):
        self.rf = rf
        if isinstance(self.rf, RandomForestClassifier):
            self.forest_name = 'RF'
        elif isinstance(self.rf, GradientBoostingClassifier):
            self.forest_name = 'GBDT'
        elif isinstance(self.rf, dict):
            self.forest_name = 'IntervalAndPrediction'

        self.di = dataset_information
        if task_region is not None:
            self.task_region = task_region
        else:
            self.task_region = np.array([self.di['space_lower_bound'], self.di['space_upper_bound']]).T
        self.ms = measure_switch
        self.vds = vector_distance_switch
        self.ecdf = ecdf
        self.val_X = None

        if self.vds == 'NL':
            ti = time.time()
            if self.forest_name == 'RF':
                self.intervals, self.probas = _parse_forest_interval(self.rf,
                                                                     left_bounds=self.di['space_lower_bound'],
                                                                     right_bounds=self.di['space_upper_bound'],
                                                                     subspace_left_bounds=self.task_region[:, 0],
                                                                     subspace_right_bounds=self.task_region[:, 1])
            elif self.forest_name == 'GBDT':
                self.intervals, self.probas = _parse_GBF_interval(self.rf,
                                                                  left_bounds=self.di['space_lower_bound'],
                                                                  right_bounds=self.di['space_upper_bound'],
                                                                  subspace_left_bounds=self.task_region[:, 0],
                                                                  subspace_right_bounds=self.task_region[:, 1])

        else:
            if self.forest_name == 'IntervalAndPrediction':
                self.intervals, self.probas = self.rf['interval set'], self.rf['prediction set']
            else:

                self.intervals, self.probas, _ = ExhaustivityIntersect(self.rf,
                                                                       left_bounds=self.di['space_lower_bound'],
                                                                       right_bounds=self.di['space_upper_bound'],
                                                                       subspace_left_bounds=self.task_region[:, 0],
                                                                       subspace_right_bounds=self.task_region[:,
                                                                                             1]).Exh_intersect(
                    measure=self.ms, ecdf=self.ecdf, topn=m_topn)

        self.SFD = SimpleFuncDis(integration_data=self.val_X, integration_domain=self.task_region,
                                 measure_switch=self.ms, vector_distance_switch=self.vds, ecdf=self.ecdf)

    def calculate_dis(self, SFintervals, SFprobas):
        self.dis = self.SFD.calculate(self.intervals, self.probas, SFintervals, SFprobas)
        if self.vds == 'NL':
            self.dis /= self.rf.n_estimators

        return self.dis

    def calculate_min_dis_one_interval(self, interval):

        self.min_dis, self.best_p = self.SFD.search_min_dis(self.intervals, self.probas, interval)

        return self.min_dis, self.best_p
