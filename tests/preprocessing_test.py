# -*- coding: UTF-8 -*-

# Import from standard library
import os
import random
import nml_toolbox
import pandas as pd
import seaborn as sns
# Import from our lib
from nml_toolbox.preprocessing import robust
import pytest


def test_robust():
    data = sns.load_dataset('iris')
    col = 'sepal_length'

    initial = data[col].copy()
    processed_data = robust(data, [col])
    data[col] = processed_data
    assert len(initial) == len(data[col]), f"Column length has changed ({len(initial)} -> {len(data[col])})"

    random_index = random.randint(0, len(data[col]))
    assert round(initial[random_index], 3) != round(data[col][random_index], 3), f"Data has not changed after preprocessing"
