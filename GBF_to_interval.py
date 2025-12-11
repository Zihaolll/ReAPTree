import numpy as np
from Interval_IOU import IoU_set, IoU


def interval_updata_left(current_interval, threshold):
    if current_interval[0] < threshold:
        current_interval[1] = threshold
    return current_interval


def interval_updata_right(current_interval, threshold):
    if current_interval[1] > threshold:
        current_interval[0] = threshold
    return current_interval


def _parse_GBF_interval(model, left_bounds, right_bounds, subspace_left_bounds, subspace_right_bounds):
    n_features = model.n_features_in_
    leaf_interval = []
    leaf_pridict = []

    for step_indx in range(model.estimators_.shape[0]):
        for class_indx in range(model.estimators_.shape[1]):
            tree = model.estimators_[step_indx, class_indx]
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            nodes = np.ones((n_nodes, n_features, 2))
            nodes[:, :, 0] = nodes[:, :, 0] * left_bounds
            nodes[:, :, 1] = nodes[:, :, 1] * right_bounds

            for i in range(n_nodes):

                if (children_left[i] != children_right[i]):
                    nodes[children_left[i]] = nodes[i].copy()
                    nodes[children_right[i]] = nodes[i].copy()

                    current_interval = nodes[i][feature[i]]

                    nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                    nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(), threshold[i])

                else:

                    if model.n_classes_ <= 2:
                        pridict_values = model.learning_rate * values[i][0]
                    else:
                        pridict_values = np.zeros(shape=[model.n_classes_])
                        pridict_values[class_indx] = model.learning_rate * values[i][0]

                    subspace = np.array([subspace_left_bounds, subspace_right_bounds]).T
                    intersect = IoU(nodes[i], subspace).intersect()

                    if intersect is not None:
                        leaf_pridict.append(pridict_values)
                        leaf_interval.append(intersect)

    leaf_interval_set, leaf_proba_set = np.array(leaf_interval), np.array(leaf_pridict)
    return leaf_interval_set, leaf_proba_set


def _parse_GBFeachTree_interval_with(model, left_bounds, right_bounds,
                                     subspace_left_bounds, subspace_right_bounds):
    n_features = model.n_features_in_
    GBF_interval = []
    GBF_proba = []
    GBF_flux = []

    subspace = np.array([subspace_left_bounds, subspace_right_bounds]).T

    for step_indx in range(model.estimators_.shape[0]):
        for class_indx in range(model.estimators_.shape[1]):
            leaf_interval = []
            leaf_pridict = []
            leaf_flux = []

            tree = model.estimators_[step_indx, class_indx]
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            nodes = np.ones((n_nodes, n_features, 2))
            nodes[:, :, 0] = nodes[:, :, 0] * left_bounds
            nodes[:, :, 1] = nodes[:, :, 1] * right_bounds

            for i in range(n_nodes):
                if children_left[i] != children_right[i]:

                    nodes[children_left[i]] = nodes[i].copy()
                    nodes[children_right[i]] = nodes[i].copy()
                    current_interval = nodes[i][feature[i]]
                    nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                    nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(), threshold[i])
                else:

                    if model.n_classes_ <= 2:

                        pridict_values = model.learning_rate * values[i][0]
                    else:
                        pridict_values = np.zeros(shape=[model.n_classes_])
                        pridict_values[class_indx] = model.learning_rate * values[i][0]

                    leaf_pridict.append(pridict_values)
                    leaf_interval.append(nodes[i])
                    leaf_flux.append(values[i].sum())

            if len(leaf_interval) == 0:

                leaf_interval_set = np.zeros((0, n_features, 2))
                leaf_proba_set = np.zeros((0, model.n_classes_ if hasattr(model, 'n_classes_') else 1))
                leaf_flux_set = np.zeros((0,))
            else:
                leaf_interval_set = np.array(leaf_interval)
                leaf_proba_set = np.array(leaf_pridict)
                leaf_flux_set = np.array(leaf_flux)

                if leaf_proba_set.ndim == 1:
                    leaf_proba_set = leaf_proba_set.reshape((-1, 1))

            if leaf_interval_set.shape[0] == 0:

                GBF_interval.append(leaf_interval_set)
                GBF_proba.append(leaf_proba_set)
                GBF_flux.append(leaf_flux_set)
            else:
                subspace_leaf_interval_set = IoU_set(subspace, leaf_interval_set).intersect()

                disempty_mask = np.where(np.isnan(subspace_leaf_interval_set).any(axis=(1, 2)), 0, 1).astype(bool)

                filtered_intervals = subspace_leaf_interval_set[disempty_mask]
                filtered_probas = leaf_proba_set[disempty_mask]
                filtered_flux = leaf_flux_set[disempty_mask]

                GBF_interval.append(filtered_intervals)
                GBF_proba.append(filtered_probas)
                GBF_flux.append(filtered_flux)

    return GBF_interval, GBF_proba, GBF_flux
