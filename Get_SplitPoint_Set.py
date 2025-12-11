import numpy as np
def interval_set_to_split_point_set(interval_set):
    I = interval_set
    feature_n = I.shape[1]
    sp_set = []
    for i in range(feature_n):
        point_set = np.unique(I[:, i, :])
        point_set = np.sort(point_set)[1:-1]
        for value in point_set:
            sp_set.append([i, value])
    return sp_set


def interval_set_to_split_point_set_keep_repetition(interval_set, whole_interval=None):
    I = interval_set
    feature_n = I.shape[1]
    sp_set = {}
    for i in range(feature_n):
        point_set = I[:, i, :].flatten()
        if whole_interval is not None:
            point_set_mask = (whole_interval[i, 1] >= point_set) & (point_set > whole_interval[i, 0])
            point_set = point_set[point_set_mask]

        if sum(point_set_mask+0) > 4:
            sorted_point_set = np.sort(point_set)

            min_value = sorted_point_set[0]
            max_value = sorted_point_set[-1]
            mask = (point_set != min_value) & (point_set != max_value)
            point_set = point_set[mask]
            sp_set[i] = point_set
        else:
            sp_set[i] = np.array([])

    return sp_set