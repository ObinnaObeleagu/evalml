import copy

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.utils import make_balancing_dictionary
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class BaseSampler(Transformer):
    """Base Sampler component. Used as the base class of all sampler components"""

    def fit(self, X, y):
        """Resample the data using the sampler. Since our sampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be none")
        return self

    def _prepare_data(self, X, y):
        """Transforms the input data to pandas data structure that our sampler can ingest.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn, pd.DataFrame, pd.Series: Prepared X and y data, both woodwork and pandas
        """
        X = infer_feature_types(X)
        if y is None:
            raise ValueError("y cannot be none")
        y = infer_feature_types(y)
        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        return X, y, X_pd, y_pd

    def transform(self, X, y=None):
        """No transformation needs to be done here.

        Arguments:
            X (ww.DataFrame): Training features. Ignored.
            y (ww.DataColumn): Target features. Ignored.

        Returns:
            ww.DataTable, ww.DataColumn: X and y data that was passed in.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y


class BaseOverSampler(BaseSampler):
    """Base Oversampler component. Used as the base class of all imbalance-learn oversampler components"""

    def __init__(self, sampler, parameters, component_obj, random_seed):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        self.sampler = {"SMOTE": im.SMOTE,
                        "SMOTENC": im.SMOTENC,
                        "SMOTEN": im.SMOTEN}[sampler]
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fits the Oversampler to the data.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        super().fit(X, y)
        self._initialize_oversampler(X, y, self.sampler)

    def _initialize_oversampler(self, X, y, sampler_class):
        """Initializes the oversampler with the given sampler_ratio or sampler_ratio_dict. If a sampler_ratio_dict is provided, we will opt to use that.
        Otherwise, we use will create the sampler_ratio_dict dictionary.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features
            sampler_class (imblearn.BaseSampler): The sampler we want to initialize
        """
        _, _, _, y_pd = self._prepare_data(X, y)
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None and len(self.parameters['sampling_ratio_dict']):
            # dictionary provided and takes priority
            sampler_params['sampling_strategy'] = self.parameters['sampling_ratio_dict']
        else:
            # no dictionary provided. We pass the float if we have a binary situation
            sampling_ratio = self.parameters['sampling_ratio']
            dic = make_balancing_dictionary(y_pd, sampling_ratio)
            sampler_params['sampling_strategy'] = dic
        sampler = sampler_class(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler

    def fit_transform(self, X, y):
        """Fit and transform the data using the data sampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Sampled X and y data
        """
        self.fit(X, y)
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)
