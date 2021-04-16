from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class DropNullRowsTransformer(Transformer):
    """Transformer to drop rows with null values"""
    name = "Drop Null Rows Transformer"
    hyperparameter_ranges = {}

    def __init__(self, columns, random_seed=0, **kwargs):
        """

        Arguments:
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        if columns and not isinstance(columns, list):
            raise ValueError(f"Parameter columns must be a list. Received {type(columns)}.")

        parameters = {"columns": columns}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X = infer_feature_types(X)
        self._check_input_for_columns(X)

        return self

    def _check_input_for_columns(self, X):
        cols = self.parameters.get("columns") or []

        column_names = X.columns

        missing_cols = set(cols) - set(column_names)
        if missing_cols:
            raise ValueError(
                "Columns {} not found in input data".format(', '.join(f"'{col_name}'" for col_name in missing_cols))
            )

    def transform(self, X, y=None):
        """Transforms data X
        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X
        """
        X_ww = infer_feature_types(X)
        X_t = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        cols = self.parameters.get("columns")
        if len(cols) > 0:
            s = X_t[cols]
            indices_to_drop = s[s.isnull().any(axis=1)].index.tolist()
        X_dropped = X_t.drop(indices_to_drop, axis=0)
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_dropped)
