# -*- coding: UTF-8 -*-

""" Preprocessing techniques
"""

from sklearn.preprocessing \
    import LabelEncoder, \
           OneHotEncoder, \
           RobustScaler, \
           StandardScaler, \
           MinMaxScaler


def labelize(data, col):
    le = LabelEncoder()
    le.fit(data[col])
    data[f'{col}_enc'] = le.transform(data[col])
    return data.drop(columns=[col])


def ohe(data, col):
    ohe = OneHotEncoder(sparse = False)
    ohe.fit(data[[col]])
    transf_ohe = ohe.transform(data[[col]])
    categories = [f'{col}_{c}' for c in ohe.categories_[0]]

    for index, category in enumerate(categories):
        data[category] = transf_ohe.T[index]

    return data.drop(columns=[col])


def robust(data, cols):
    r_scaler = RobustScaler()
    r_scaler.fit(data[cols])
    return r_scaler.transform(data[cols])


def standard(data, cols):
    s_scaler = StandardScaler()
    s_scaler.fit(data[cols])
    return s_scaler.transform(data[cols])


def min_max(data, cols):
    normalizer = MinMaxScaler()
    normalizer.fit(data[cols])
    return normalizer.transform(data[cols])
