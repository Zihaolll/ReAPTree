
import pandas as pd
from sklearn.datasets import load_iris


def read_data_iris():
    iris = load_iris()
    data = pd.DataFrame(iris.data[:], columns=iris.feature_names)
    data['class'] = iris.target
    y_column = 'class'
    feature_types = ['float'] * 4
    x_columns = iris.feature_names

    return data, x_columns, y_column, feature_types


def read_data_banknote():
    x_columns = ['x' + str(i) for i in range(4)]
    y_column = 'class'
    data = pd.read_csv('DataSet/banknote.txt', names=x_columns + [y_column])
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types


def read_data_breast_cancer_data():
    x_columns = ['Clump thickness',
                 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    y_column = 'class'
    feature_types = ['float'] * 9
    data = pd.read_csv('DataSet/breast-cancer-wisconsin.data', names=x_columns + [y_column])
    data = data[data['Bare Nuclei'] != '?']
    data['Bare Nuclei'] = [int(i) for i in data['Bare Nuclei']]
    return data, x_columns, y_column, feature_types


def read_data_compas():
    data = pd.read_csv('DataSet/compas')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types


def get_dataset(dataset_name):
    if dataset_name == 'iris':
        return read_data_iris()
    elif dataset_name == 'banknote':
        return read_data_banknote()
    elif dataset_name == 'breast_cancer':
        return read_data_breast_cancer_data()
    elif dataset_name == 'compas':
        return read_data_compas()


