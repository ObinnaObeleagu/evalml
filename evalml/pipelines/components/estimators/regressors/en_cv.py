from sklearn.linear_model import ElasticNetCV as SKElasticNetCV
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetCVRegressor(Estimator):
    """Elastic Net CV Regressor."""
    name = "Elastic Net CV Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)
        en_regressor = SKElasticNetCV(random_state=random_seed,
                                    **parameters)
        super().__init__(parameters=parameters,
                         component_obj=en_regressor,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        return self._component_obj.coef_
