import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from RF_to_interval import _parse_forest_interval
from GBF_to_interval import _parse_GBF_interval
from Get_SplitPoint_Set import interval_set_to_split_point_set_keep_repetition
from scipy.stats import gaussian_kde
import itertools
from Distribution_measure import distribution_measure
from Interval_IOU import IoU_set


def KDE(arr, qurary_points=None):
    kde = gaussian_kde(arr)
    if qurary_points is None:
        query_points = np.unique(arr)[1:-1]
    density = kde(query_points)
    return query_points, density


def get_internal_low_density_segments(density, threshold):
    is_low_density = density < threshold
    segments = []
    start_idx = None

    for i in range(len(is_low_density)):
        if is_low_density[i] and start_idx is None:
            start_idx = i
        elif not is_low_density[i] and start_idx is not None:
            end_idx = i - 1
            if start_idx > 0 and i < len(is_low_density) - 1:
                segments.append((start_idx, end_idx))
            start_idx = None

    return segments


class CreateSubspace:
    def __init__(self, forest, forest_space_lower_bound, forest_space_upper_bound,
                 task_space_lower_bound, task_space_upper_bound, ecdf,
                 filter_paramters=None, stop_C=50, curr_depth=0, max_depth=5):
        self.forest = forest
        self.f_space_lower_bound = forest_space_lower_bound
        self.f_space_upper_bound = forest_space_upper_bound
        self.max_depth = max_depth
        self.curr_depth = curr_depth

        self.whole_interval = np.array([task_space_lower_bound, task_space_upper_bound]).T
        if isinstance(forest, RandomForestClassifier):
            self.forest_name = 'RF'
            self.interval_set, self.proba_set = _parse_forest_interval(forest,
                                                                       left_bounds=self.f_space_lower_bound,
                                                                       right_bounds=self.f_space_upper_bound,
                                                                       subspace_left_bounds=task_space_lower_bound,
                                                                       subspace_right_bounds=task_space_upper_bound,
                                                                       )
        elif isinstance(forest, GradientBoostingClassifier):
            self.forest_name = 'GBDT'
            self.interval_set, self.proba_set = _parse_GBF_interval(forest,
                                                                    left_bounds=self.f_space_lower_bound,
                                                                    right_bounds=self.f_space_upper_bound,
                                                                    subspace_left_bounds=task_space_lower_bound,
                                                                    subspace_right_bounds=task_space_upper_bound,
                                                                    )
        self.split_set_repetition = interval_set_to_split_point_set_keep_repetition(self.interval_set,
                                                                                    self.whole_interval)
        self.filter_paramters = filter_paramters
        self.ecdf = ecdf
        self.subspaces_set = []
        self.stop_C = stop_C

    def search_split_value(self):
        split_value = []

        if self.filter_paramters == None:
            self.threshold_percent = 10
            self.search_switch = 'valleys'
        else:
            self.search_switch = self.filter_paramters['peaks or valleys']
            self.threshold_percent = self.filter_paramters['param']

        feature_idx_list = [i for i in range(self.whole_interval.shape[0])]

        for feature_idx in range(self.whole_interval.shape[0]):
            if np.unique(self.split_set_repetition[feature_idx]).size > 10 \
                    and self.split_set_repetition[feature_idx].size > 20:
                query_points, density = KDE(self.split_set_repetition[feature_idx])

                threshold = np.percentile(density, self.threshold_percent)

                if self.search_switch == 'valleys':

                    internal_segments = get_internal_low_density_segments(density, threshold)

                    representative_points = []
                    for start, end in internal_segments:
                        segment_points = query_points[start:end + 1]
                        representative_point = segment_points[len(segment_points) // 2]
                        representative_points.append(representative_point)
                elif self.search_switch == 'peaks':
                    unpeaks_mask = density < threshold

                    unpeaks_value = np.unique(query_points[unpeaks_mask])
                    representative_size = min(np.sum(unpeaks_value.shape[0]), 1)
                    representative_points = np.random.choice(unpeaks_value, size=representative_size,
                                                             replace=False).tolist()

            else:
                representative_points = []

            split_value.append(representative_points)

        n_features = self.whole_interval.shape[0]

        unique_counts = []
        for i in range(n_features):
            vals = np.asarray(self.split_set_repetition[i])
            unique_counts.append(np.unique(vals).size if vals.size > 0 else 0)

        if len(unique_counts) == 0 or np.max(unique_counts) == 0:
            return split_value
        max_unique_count = np.max(unique_counts)

        if all(len(v) == 0 for v in split_value) and max_unique_count > 20:
            best_feature = int(np.argmax(unique_counts))

            feature_values = np.sort(np.unique(self.split_set_repetition[best_feature]))
            mid_index = len(feature_values) // 2
            mid_point = feature_values[mid_index]

            split_value[best_feature] = [mid_point]

        return split_value

    def generate_subspaces(self):
        if self.curr_depth >= self.max_depth:
            return []

        split_point_list = self.search_split_value()

        n_dimensions = len(self.whole_interval)
        assert len(split_point_list) == n_dimensions, "split_point_list must match the number of dimensions."

        dimension_intervals = []
        for i in range(n_dimensions):

            bounds = self.whole_interval[i]
            splits = sorted(split_point_list[i])

            if not splits:
                dimension_intervals.append([bounds.copy()])
            else:

                intervals = []
                splits = sorted(splits)
                intervals.append([bounds[0], splits[0]])
                for j in range(len(splits) - 1):
                    intervals.append([splits[j], splits[j + 1]])
                intervals.append([splits[-1], bounds[1]])
                dimension_intervals.append(intervals)

        subspace_set = list(itertools.product(*dimension_intervals))
        num_subspaces = len(subspace_set)

        if num_subspaces >= 2:
            for subspace in subspace_set:
                subspace = np.array(subspace)
                # print(subspace)

                if distribution_measure(subspace, self.ecdf) >= 0.0001:
                    if self.stop_generate_subspace(subspace):

                        self.subspaces_set.extend([subspace])
                    else:
                        CS = CreateSubspace(self.forest, self.f_space_lower_bound, self.f_space_upper_bound,
                                            subspace[:, 0], subspace[:, 1],
                                            filter_paramters=self.filter_paramters, ecdf=self.ecdf,
                                            curr_depth=self.curr_depth + 1, )
                        self.subspaces_set.extend(CS.generate_subspaces())
        else:

            self.subspaces_set.extend([np.array(subspace_set[0])])

        return self.subspaces_set

    def stop_generate_subspace(self, subspace):
        stop = True

        IoU = IoU_set(subspace, self.interval_set)
        area_rate = IoU.intersect_rate_for_interval1()[~IoU.none_mask]
        IF = 1 - 4 * ((area_rate - 0.5) ** 2)

        Complexity = np.sum(IF)

        sub_interval_num = area_rate.shape[0]

        if Complexity > self.stop_C:
            stop = False

        return stop


