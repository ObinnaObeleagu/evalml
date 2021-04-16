import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal

from evalml.pipelines.components.transformers.preprocessing import (
    DropNullRowsTransformer
)


def test_drop_null_rows_transformer_init():
    X = pd.DataFrame({"a": [np.nan, 1, 2]})
    drop_null_transformer = DropNullRowsTransformer(columns=['a'])
    X_t = drop_null_transformer.fit_transform(X)
