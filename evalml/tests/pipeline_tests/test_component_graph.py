from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from woodwork.logical_types import Double, Integer

from evalml.exceptions import MissingComponentError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    ElasticNetClassifier,
    Estimator,
    Imputer,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    SelectColumns,
    StandardScaler,
    TargetImputer,
    TextFeaturizer,
    Transformer,
    Undersampler,
)
from evalml.pipelines.components.transformers.transformer import (
    TargetTransformer,
)
from evalml.utils import infer_feature_types


class DummyTransformer(Transformer):
    name = "Dummy Transformer"

    def __init__(self, parameters=None, random_seed=0):
        parameters = parameters or {}
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y):
        return self


class TransformerA(DummyTransformer):
    """copy class"""


class TransformerB(DummyTransformer):
    """copy class"""


class TransformerC(DummyTransformer):
    """copy class"""


class DummyEstimator(Estimator):
    name = "Dummy Estimator"
    model_family = None
    supported_problem_types = None

    def __init__(self, parameters=None, random_seed=0):
        parameters = parameters or {}
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y):
        return self


class EstimatorA(DummyEstimator):
    """copy class"""


class EstimatorB(DummyEstimator):
    """copy class"""


class EstimatorC(DummyEstimator):
    """copy class"""


@pytest.fixture
def dummy_components():
    return TransformerA, TransformerB, TransformerC, EstimatorA, EstimatorB, EstimatorC


@pytest.fixture
def example_graph():
    graph = {
        "Imputer": [Imputer],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x"],
        "OneHot_ElasticNet": [OneHotEncoder, "Imputer.x"],
        "Random Forest": [RandomForestClassifier, "OneHot_RandomForest.x"],
        "Elastic Net": [ElasticNetClassifier, "OneHot_ElasticNet.x"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }
    return graph


def test_init(example_graph):
    comp_graph = ComponentGraph()
    assert len(comp_graph.component_dict) == 0

    graph = example_graph
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression",
    ]
    assert comp_graph.compute_order == expected_order


def test_init_str_components():
    graph = {
        "Imputer": ["Imputer"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x"],
        "Logistic Regression": [
            "Logistic Regression Classifier",
            "Random Forest",
            "Elastic Net",
        ],
    }
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression",
    ]
    assert comp_graph.compute_order == expected_order


def test_invalid_init():
    invalid_graph = {"Imputer": [Imputer], "OHE": OneHotEncoder}
    with pytest.raises(
        ValueError, match="All component information should be passed in as a list"
    ):
        ComponentGraph(invalid_graph)

    with pytest.raises(
        ValueError, match="may only contain str or ComponentBase subclasses"
    ):
        ComponentGraph(
            {
                "Imputer": [Imputer(numeric_impute_strategy="most_frequent")],
                "OneHot": [OneHotEncoder],
            }
        )

    graph = {
        "Imputer": [Imputer(numeric_impute_strategy="constant", numeric_fill_value=0)]
    }
    with pytest.raises(
        ValueError, match="may only contain str or ComponentBase subclasses"
    ):
        ComponentGraph(graph)

    graph = {
        "Imputer": ["Imputer", "Fake"],
        "Fake": ["Fake Component", "Estimator"],
        "Estimator": [ElasticNetClassifier],
    }
    with pytest.raises(MissingComponentError):
        ComponentGraph(graph)


def test_init_bad_graphs():
    graph = {
        "Imputer": [Imputer],
        "OHE": [OneHotEncoder, "Imputer.x", "Estimator"],
        "Estimator": [RandomForestClassifier, "OHE.x"],
    }
    with pytest.raises(ValueError, match="given graph contains a cycle"):
        ComponentGraph(graph)

    graph = {
        "Imputer": [Imputer],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x"],
        "OneHot_ElasticNet": [OneHotEncoder, "Imputer.x"],
        "Random Forest": [RandomForestClassifier],
        "Elastic Net": [ElasticNetClassifier],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }
    with pytest.raises(ValueError, match="graph is not completely connected"):
        ComponentGraph(graph)

    graph = {
        "Imputer": ["Imputer"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x"],
        "Elastic Net": ["Elastic Net Classifier"],
        "Logistic Regression": [
            "Logistic Regression Classifier",
            "Random Forest",
            "Elastic Net",
        ],
    }
    with pytest.raises(ValueError, match="graph has more than one final"):
        ComponentGraph(graph)


def test_order_x_and_y():
    graph = {
        "Imputer": [Imputer],
        "OHE": [OneHotEncoder, "Imputer.x", "Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OHE.x"],
    }
    component_graph = ComponentGraph(graph).instantiate({})
    assert component_graph.compute_order == ["Imputer", "OHE", "Random Forest"]


def test_list_raises_error():
    component_list = ["Imputer", "One Hot Encoder", RandomForestClassifier]
    with pytest.raises(
        ValueError,
        match="component_dict must be a dictionary which specifies the components and edges between components",
    ):
        ComponentGraph(component_list)


def test_instantiate_with_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert not isinstance(component_graph.get_component("Imputer"), Imputer)
    assert not isinstance(
        component_graph.get_component("Elastic Net"), ElasticNetClassifier
    )

    parameters = {
        "OneHot_RandomForest": {"top_n": 3},
        "OneHot_ElasticNet": {"top_n": 5},
        "Elastic Net": {"max_iter": 100},
    }
    component_graph.instantiate(parameters)

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression",
    ]
    assert component_graph.compute_order == expected_order

    assert isinstance(component_graph.get_component("Imputer"), Imputer)
    assert isinstance(
        component_graph.get_component("Random Forest"), RandomForestClassifier
    )
    assert isinstance(
        component_graph.get_component("Logistic Regression"),
        LogisticRegressionClassifier,
    )
    assert component_graph.get_component("OneHot_RandomForest").parameters["top_n"] == 3
    assert component_graph.get_component("OneHot_ElasticNet").parameters["top_n"] == 5
    assert component_graph.get_component("Elastic Net").parameters["max_iter"] == 100


def test_instantiate_without_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert (
        component_graph.get_component("OneHot_RandomForest").parameters["top_n"] == 10
    )
    assert component_graph.get_component("OneHot_ElasticNet").parameters["top_n"] == 10
    assert component_graph.get_component(
        "OneHot_RandomForest"
    ) is not component_graph.get_component("OneHot_ElasticNet")

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression",
    ]
    assert component_graph.compute_order == expected_order


def test_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({})
    with pytest.raises(ValueError, match="Cannot reinstantiate a component graph"):
        component_graph.instantiate({"OneHot": {"top_n": 7}})


def test_bad_instantiate_can_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match="Error received when instantiating component"):
        component_graph.instantiate(
            parameters={"Elastic Net": {"max_iter": 100, "fake_param": None}}
        )

    component_graph.instantiate({"Elastic Net": {"max_iter": 22}})
    assert component_graph.get_component("Elastic Net").parameters["max_iter"] == 22


def test_get_component(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_component("OneHot_ElasticNet") == OneHotEncoder
    assert (
        component_graph.get_component("Logistic Regression")
        == LogisticRegressionClassifier
    )

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_component("Fake Component")

    component_graph.instantiate(
        {
            "OneHot_RandomForest": {"top_n": 3},
            "Random Forest": {"max_depth": 4, "n_estimators": 50},
        }
    )
    assert component_graph.get_component("OneHot_ElasticNet") == OneHotEncoder()
    assert component_graph.get_component("OneHot_RandomForest") == OneHotEncoder(
        top_n=3
    )
    assert component_graph.get_component("Random Forest") == RandomForestClassifier(
        n_estimators=50, max_depth=4
    )


def test_get_estimators(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match="Cannot get estimators until"):
        component_graph.get_estimators()

    component_graph.instantiate({})
    assert component_graph.get_estimators() == [
        RandomForestClassifier(),
        ElasticNetClassifier(),
        LogisticRegressionClassifier(),
    ]

    component_graph = ComponentGraph({"Imputer": ["Imputer", "X", "y"]})
    component_graph.instantiate({})
    assert component_graph.get_estimators() == []


def test_parents(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_inputs("Imputer") == []
    assert component_graph.get_inputs("OneHot_RandomForest") == ["Imputer.x"]
    assert component_graph.get_inputs("OneHot_ElasticNet") == ["Imputer.x"]
    assert component_graph.get_inputs("Random Forest") == ["OneHot_RandomForest.x"]
    assert component_graph.get_inputs("Elastic Net") == ["OneHot_ElasticNet.x"]
    assert component_graph.get_inputs("Logistic Regression") == [
        "Random Forest",
        "Elastic Net",
    ]

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_inputs("Fake component")

    component_graph.instantiate({})
    assert component_graph.get_inputs("Imputer") == []
    assert component_graph.get_inputs("OneHot_RandomForest") == ["Imputer.x"]
    assert component_graph.get_inputs("OneHot_ElasticNet") == ["Imputer.x"]
    assert component_graph.get_inputs("Random Forest") == ["OneHot_RandomForest.x"]
    assert component_graph.get_inputs("Elastic Net") == ["OneHot_ElasticNet.x"]
    assert component_graph.get_inputs("Logistic Regression") == [
        "Random Forest",
        "Elastic Net",
    ]

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_inputs("Fake component")


def test_get_last_component(example_graph):
    component_graph = ComponentGraph()
    with pytest.raises(
        ValueError, match="Cannot get last component from edgeless graph"
    ):
        component_graph.get_last_component()

    component_graph = ComponentGraph(example_graph)
    assert component_graph.get_last_component() == LogisticRegressionClassifier

    component_graph.instantiate({})
    assert component_graph.get_last_component() == LogisticRegressionClassifier()

    component_graph = ComponentGraph({"Imputer": [Imputer]})
    assert component_graph.get_last_component() == Imputer

    component_graph = ComponentGraph(
        {"Imputer": [Imputer], "OneHot": [OneHotEncoder, "Imputer"]}
    )
    assert component_graph.get_last_component() == OneHotEncoder

    component_graph = ComponentGraph({"Imputer": [Imputer], "OneHot": [OneHotEncoder]})
    with pytest.raises(
        ValueError, match="Cannot get last component from edgeless graph"
    ):
        component_graph.get_last_component()


@patch("evalml.pipelines.components.Transformer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_fit(mock_predict, mock_fit, mock_fit_transform, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_fit_transform.return_value = pd.DataFrame(X)
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 3
    assert mock_predict.call_count == 2


@patch("evalml.pipelines.components.Imputer.fit_transform")
@patch("evalml.pipelines.components.OneHotEncoder.fit_transform")
def test_fit_correct_inputs(
    mock_ohe_fit_transform, mock_imputer_fit_transform, X_y_binary
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series(y)
    graph = {"Imputer": [Imputer], "OHE": [OneHotEncoder, "Imputer.x", "Imputer.y"]}
    expected_x = pd.DataFrame(index=X.index, columns=X.columns).fillna(1.0)
    expected_x.ww.init()

    expected_y = pd.Series(index=y.index).fillna(0)
    mock_imputer_fit_transform.return_value = tuple((expected_x, expected_y))
    mock_ohe_fit_transform.return_value = expected_x
    component_graph = ComponentGraph(graph).instantiate({})
    component_graph.fit(X, y)
    assert_frame_equal(expected_x, mock_ohe_fit_transform.call_args[0][0])
    assert_series_equal(expected_y, mock_ohe_fit_transform.call_args[0][1])


@patch("evalml.pipelines.components.Transformer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_fit_features(
    mock_predict, mock_fit, mock_fit_transform, example_graph, X_y_binary
):
    X, y = X_y_binary
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({})

    mock_X_t = pd.DataFrame(np.ones(pd.DataFrame(X).shape))
    mock_fit_transform.return_value = mock_X_t
    mock_fit.return_value = Estimator
    mock_predict.return_value = pd.Series(y)

    component_graph.fit_features(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 2
    assert mock_predict.call_count == 2


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict(mock_predict, mock_fit, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    component_graph.predict(X)
    assert (
        mock_predict.call_count == 5
    )  # Called twice when fitting pipeline, thrice when predicting
    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict_repeat_estimator(mock_predict, mock_fit, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = pd.Series(y)

    graph = {
        "Imputer": [Imputer],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x"],
        "OneHot_Logistic": [OneHotEncoder, "Imputer.x"],
        "Random Forest": [RandomForestClassifier, "OneHot_RandomForest.x"],
        "Logistic Regression": [LogisticRegressionClassifier, "OneHot_Logistic.x"],
        "Final Estimator": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Logistic Regression",
        ],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    assert (
        not component_graph.get_component("Logistic Regression")._component_obj
        == component_graph.get_component("Final Estimator")._component_obj
    )

    component_graph.predict(X)
    assert mock_predict.call_count == 5
    assert mock_fit.call_count == 3


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict")
def test_compute_final_component_features(
    mock_en_predict, mock_rf_predict, mock_ohe, mock_imputer, example_graph, X_y_binary
):
    X, y = X_y_binary
    mock_imputer.return_value = pd.DataFrame(X)
    mock_ohe.return_value = pd.DataFrame(X)
    mock_en_predict.return_value = pd.Series(np.ones(X.shape[0]))
    mock_rf_predict.return_value = pd.Series(np.zeros(X.shape[0]))
    X_expected = pd.DataFrame(
        {"Random Forest": np.zeros(X.shape[0]), "Elastic Net": np.ones(X.shape[0])}
    )
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.compute_final_component_features(X)
    assert_frame_equal(X_expected, X_t)
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 4


@patch(f"{__name__}.DummyTransformer.transform")
def test_compute_final_component_features_single_component(mock_transform, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    mock_transform.return_value = X
    component_graph = ComponentGraph(
        {"Dummy Component": [DummyTransformer]}
    ).instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.compute_final_component_features(X)
    assert_frame_equal(X, X_t)


@patch("evalml.pipelines.components.Imputer.fit_transform")
def test_fit_y_parent(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {
        "Imputer": [Imputer],
        "OHE": [OneHotEncoder, "Imputer.x", "Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OHE.x"],
    }
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.fit(X, y)
    mock_fit_transform.assert_called_once()


def test_predict_empty_graph(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    component_graph = ComponentGraph()
    component_graph.instantiate({})

    component_graph.fit(X, y)
    X_t = component_graph.predict(X)
    assert_frame_equal(X, X_t)


@patch("evalml.pipelines.components.OneHotEncoder.fit_transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
def test_predict_transformer_end(mock_fit_transform, mock_transform, X_y_binary):
    X, y = X_y_binary
    graph = {"Imputer": [Imputer], "OHE": [OneHotEncoder, "Imputer.x"]}
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))
    mock_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.fit(X, y)
    output = component_graph.predict(X)
    assert_frame_equal(pd.DataFrame(X), output)


def test_no_instantiate_before_fit(X_y_binary):
    X, y = X_y_binary
    graph = {
        "Imputer": [Imputer],
        "OHE": [OneHotEncoder, "Imputer.x"],
        "Estimator": [RandomForestClassifier, "OHE.x"],
    }
    component_graph = ComponentGraph(graph)
    with pytest.raises(
        ValueError,
        match="All components must be instantiated before fitting or predicting",
    ):
        component_graph.fit(X, y)


@patch("evalml.pipelines.components.Imputer.fit_transform")
def test_multiple_y_parents(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {
        "Imputer": [Imputer],
        "OHE": [OneHotEncoder, "Imputer.x"],
        "Estimator": [RandomForestClassifier, "Imputer.y", "OHE.y"],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))
    with pytest.raises(
        ValueError, match="Cannot have multiple `y` parents for a single component"
    ):
        component_graph.fit(X, y)


def test_component_graph_order(example_graph):
    component_graph = ComponentGraph(example_graph)
    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression",
    ]
    assert expected_order == component_graph.compute_order

    component_graph = ComponentGraph({"Imputer": [Imputer]})
    expected_order = ["Imputer"]
    assert expected_order == component_graph.compute_order


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
def test_computation_input_custom_index(index):
    graph = {
        "OneHot": [OneHotEncoder],
        "Random Forest": [RandomForestClassifier, "OneHot.x"],
        "Elastic Net": [ElasticNetClassifier, "OneHot.x"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }

    X = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    y = pd.Series([1, 2, 1, 2, 1])
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.predict(X)
    assert_index_equal(X_t.index, pd.RangeIndex(start=0, stop=5, step=1))
    assert not X_t.isna().any(axis=None)


@patch(f"{__name__}.EstimatorC.predict")
@patch(f"{__name__}.EstimatorB.predict")
@patch(f"{__name__}.EstimatorA.predict")
@patch(f"{__name__}.TransformerC.transform")
@patch(f"{__name__}.TransformerB.transform")
@patch(f"{__name__}.TransformerA.transform")
def test_component_graph_evaluation_plumbing(
    mock_transa,
    mock_transb,
    mock_transc,
    mock_preda,
    mock_predb,
    mock_predc,
    dummy_components,
):
    (
        TransformerA,
        TransformerB,
        TransformerC,
        EstimatorA,
        EstimatorB,
        EstimatorC,
    ) = dummy_components
    mock_transa.return_value = pd.DataFrame(
        {"feature trans": [1, 0, 0, 0, 0, 0], "feature a": np.ones(6)}
    )
    mock_transb.return_value = pd.DataFrame({"feature b": np.ones(6) * 2})
    mock_transc.return_value = pd.DataFrame({"feature c": np.ones(6) * 3})
    mock_preda.return_value = pd.Series([0, 0, 0, 1, 0, 0])
    mock_predb.return_value = pd.Series([0, 0, 0, 0, 1, 0])
    mock_predc.return_value = pd.Series([0, 0, 0, 0, 0, 1])
    graph = {
        "transformer a": [TransformerA],
        "transformer b": [TransformerB, "transformer a"],
        "transformer c": [TransformerC, "transformer a", "transformer b"],
        "estimator a": [EstimatorA],
        "estimator b": [EstimatorB, "transformer a"],
        "estimator c": [
            EstimatorC,
            "transformer a",
            "estimator a",
            "transformer b",
            "estimator b",
            "transformer c",
        ],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    X = pd.DataFrame({"feature1": np.zeros(6), "feature2": np.zeros(6)})
    y = pd.Series(np.zeros(6))
    component_graph.fit(X, y)
    predict_out = component_graph.predict(X)

    assert_frame_equal(mock_transa.call_args[0][0], X)
    assert_frame_equal(
        mock_transb.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
            },
            columns=["feature trans", "feature a"],
        ),
    )
    assert_frame_equal(
        mock_transc.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
                "feature b": np.ones(6) * 2,
            },
            columns=["feature trans", "feature a", "feature b"],
        ),
    )
    assert_frame_equal(mock_preda.call_args[0][0], X)
    assert_frame_equal(
        mock_predb.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
            },
            columns=["feature trans", "feature a"],
        ),
    )
    assert_frame_equal(
        mock_predc.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
                "estimator a": pd.Series([0, 0, 0, 1, 0, 0], dtype="int64"),
                "feature b": np.ones(6) * 2,
                "estimator b": pd.Series([0, 0, 0, 0, 1, 0], dtype="int64"),
                "feature c": np.ones(6) * 3,
            },
            columns=[
                "feature trans",
                "feature a",
                "estimator a",
                "feature b",
                "estimator b",
                "feature c",
            ],
        ),
    )
    assert_series_equal(pd.Series([0, 0, 0, 0, 0, 1], dtype="int64"), predict_out)


def test_input_feature_names(example_graph):
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate(
        {"OneHot_RandomForest": {"top_n": 2}, "OneHot_ElasticNet": {"top_n": 3}}
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Imputer"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_RandomForest"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_ElasticNet"] == ["column_1", "column_2"]
    assert input_feature_names["Random Forest"] == [
        "column_2",
        "column_1_a",
        "column_1_b",
    ]
    assert input_feature_names["Elastic Net"] == [
        "column_2",
        "column_1_a",
        "column_1_b",
        "column_1_c",
    ]
    assert input_feature_names["Logistic Regression"] == [
        "Random Forest",
        "Elastic Net",
    ]


def test_iteration(example_graph):
    component_graph = ComponentGraph(example_graph)

    expected = [
        Imputer,
        OneHotEncoder,
        ElasticNetClassifier,
        OneHotEncoder,
        RandomForestClassifier,
        LogisticRegressionClassifier,
    ]
    iteration = [component for component in component_graph]
    assert iteration == expected

    component_graph.instantiate({"OneHot_RandomForest": {"top_n": 32}})
    expected = [
        Imputer(),
        OneHotEncoder(),
        ElasticNetClassifier(),
        OneHotEncoder(top_n=32),
        RandomForestClassifier(),
        LogisticRegressionClassifier(),
    ]
    iteration = [component for component in component_graph]
    assert iteration == expected


def test_custom_input_feature_types(example_graph):
    X = pd.DataFrame(
        {
            "column_1": ["a", "a", "a", "b", "b", "b", "c", "c", "d"],
            "column_2": [1, 2, 3, 3, 4, 4, 5, 5, 6],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(X, {"column_2": "categorical"})

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate(
        {"OneHot_RandomForest": {"top_n": 2}, "OneHot_ElasticNet": {"top_n": 3}}
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Imputer"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_RandomForest"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_ElasticNet"] == ["column_1", "column_2"]
    assert input_feature_names["Random Forest"] == [
        "column_1_a",
        "column_1_b",
        "column_2_4",
        "column_2_5",
    ]
    assert input_feature_names["Elastic Net"] == [
        "column_1_a",
        "column_1_b",
        "column_1_c",
        "column_2_3",
        "column_2_4",
        "column_2_5",
    ]
    assert input_feature_names["Logistic Regression"] == [
        "Random Forest",
        "Elastic Net",
    ]


def test_component_graph_dataset_with_different_types():
    # Checks that types are converted correctly by Woodwork. Specifically, the standard scaler
    # should convert column_3 to float, so our code to try to convert back to the original boolean type
    # will catch the TypeError thrown and not convert the column.
    # Also, column_4 will be treated as a datetime feature, but the identical column_5 set as natural language
    # should be treated as natural language, not as datetime.
    graph = {
        "Text": [TextFeaturizer],
        "Imputer": [Imputer, "Text.x"],
        "OneHot": [OneHotEncoder, "Imputer.x"],
        "DateTime": [DateTimeFeaturizer, "OneHot.x"],
        "Scaler": [StandardScaler, "DateTime.x"],
        "Random Forest": [RandomForestClassifier, "Scaler.x"],
        "Elastic Net": [ElasticNetClassifier, "Scaler.x"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        }
    )
    X["column_4"] = [
        str((datetime(2021, 5, 21, 12, 0, 0) + timedelta(minutes=5 * x)))
        for x in range(len(X))
    ]
    X["column_5"] = X["column_4"]

    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(
        X, {"column_2": "categorical", "column_5": "NaturalLanguage"}
    )

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    def check_feature_names(input_feature_names):
        assert input_feature_names["Text"] == [
            "column_1",
            "column_2",
            "column_3",
            "column_4",
            "column_5",
        ]
        text_columns = [
            "DIVERSITY_SCORE(column_5)",
            "MEAN_CHARACTERS_PER_WORD(column_5)",
            "POLARITY_SCORE(column_5)",
            "LSA(column_5)[0]",
            "LSA(column_5)[1]",
        ]

        assert (
            input_feature_names["Imputer"]
            == [
                "column_1",
                "column_2",
                "column_3",
                "column_4",
            ]
            + text_columns
        )
        assert (
            input_feature_names["OneHot"]
            == [
                "column_1",
                "column_2",
                "column_3",
                "column_4",
            ]
            + text_columns
        )
        assert sorted(input_feature_names["DateTime"]) == sorted(
            [
                "column_3",
                "column_4",
                "column_1_a",
                "column_1_b",
                "column_1_c",
                "column_1_d",
                "column_2_1",
                "column_2_2",
                "column_2_3",
                "column_2_4",
                "column_2_5",
                "column_2_6",
            ]
            + text_columns
        )
        assert sorted(input_feature_names["Scaler"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            )
        )
        assert sorted(input_feature_names["Random Forest"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            )
        )
        assert sorted(input_feature_names["Elastic Net"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            )
        )
        assert input_feature_names["Logistic Regression"] == [
            "Random Forest",
            "Elastic Net",
        ]

    check_feature_names(component_graph.input_feature_names)
    component_graph.input_feature_names = {}
    component_graph.predict(X)
    check_feature_names(component_graph.input_feature_names)


@patch("evalml.pipelines.components.RandomForestClassifier.fit")
def test_component_graph_types_merge_mock(mock_rf_fit):
    graph = {
        "Select numeric col_2": [SelectColumns],
        "Imputer numeric col_2": [Imputer, "Select numeric col_2.x"],
        "Scaler col_2": [StandardScaler, "Imputer numeric col_2.x"],
        "Select categorical col_1": [SelectColumns],
        "Imputer categorical col_1": [Imputer, "Select categorical col_1.x"],
        "OneHot col_1": [OneHotEncoder, "Imputer categorical col_1.x"],
        "Pass through col_3": [SelectColumns],
        "Random Forest": [
            RandomForestClassifier,
            "Scaler col_2.x",
            "OneHot col_1.x",
            "Pass through col_3.x",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    # woodwork would infer this as boolean by default -- convert to a numeric type
    X = infer_feature_types(X, {"column_3": "integer"})

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    # if the user-specified woodwork types are being respected, we should see the pass-through column_3 staying as numeric,
    # meaning it won't cause a modeling error when it reaches the estimator
    component_graph.instantiate(
        {
            "Select numeric col_2": {"columns": ["column_2"]},
            "Select categorical col_1": {"columns": ["column_1"]},
            "Pass through col_3": {"columns": ["column_3"]},
        }
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Random Forest"] == (
        ["column_2", "column_1_a", "column_1_b", "column_1_c", "column_1_d", "column_3"]
    )
    assert isinstance(mock_rf_fit.call_args[0][0].ww.logical_types["column_3"], Integer)
    assert isinstance(mock_rf_fit.call_args[0][0].ww.logical_types["column_2"], Double)


def test_component_graph_preserves_ltypes_created_during_pipeline_evaluation():

    # This test checks that the component graph preserves logical types created during pipeline evaluation
    # The other tests ensure that logical types set before pipeline evaluation are preserved

    class ZipCodeExtractor(Transformer):
        name = "Zip Code Extractor"

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X = pd.DataFrame({"zip_code": pd.Series(["02101", "02139", "02152"] * 3)})
            X.ww.init(logical_types={"zip_code": "PostalCode"})
            return X

    class ZipCodeToAveragePrice(Transformer):
        name = "Check Zip Code Preserved"

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X = infer_feature_types(X)
            original_columns = list(X.columns)
            X = X.ww.select(["PostalCode"])
            # This would make the test fail if the componant graph
            assert len(X.columns) > 0, "No Zip Code!"
            X.ww["average_apartment_price"] = pd.Series([1000, 2000, 3000] * 3)
            X = X.ww.drop(original_columns)
            return X

    graph = {
        "Select non address": [SelectColumns],
        "OneHot": [OneHotEncoder, "Select non address.x"],
        "Select address": [SelectColumns],
        "Extract ZipCode": [ZipCodeExtractor, "Select address.x"],
        "Average Price From ZipCode": [ZipCodeToAveragePrice, "Extract ZipCode.x"],
        "Random Forest": [
            RandomForestClassifier,
            "OneHot.x",
            "Average Price From ZipCode.x",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
            "address": [f"address-{i}" for i in range(9)],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])

    # woodwork would infer this as boolean by default -- convert to a numeric type
    X.ww.init(semantic_tags={"address": "address"})

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    # if the user-specified woodwork types are being respected, we should see the pass-through column_3 staying as numeric,
    # meaning it won't cause a modeling error when it reaches the estimator
    component_graph.instantiate(
        {
            "Select non address": {"columns": ["column_1", "column_2", "column_3"]},
            "Select address": {"columns": ["address"]},
        }
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert sorted(input_feature_names["Random Forest"]) == sorted(
        [
            "column_2",
            "column_1_a",
            "column_1_b",
            "column_1_c",
            "column_1_d",
            "column_3",
            "average_apartment_price",
        ]
    )


def test_component_graph_types_merge():
    graph = {
        "Select numeric": [SelectColumns],
        "Imputer numeric": [Imputer, "Select numeric.x"],
        "Select text": [SelectColumns],
        "Text": [TextFeaturizer, "Select text.x"],
        "Imputer text": [Imputer, "Text.x"],
        "Scaler": [StandardScaler, "Imputer numeric.x"],
        "Select categorical": [SelectColumns],
        "Imputer categorical": [Imputer, "Select categorical.x"],
        "OneHot": [OneHotEncoder, "Imputer categorical.x"],
        "Select datetime": [SelectColumns],
        "Imputer datetime": [Imputer, "Select datetime.x"],
        "DateTime": [DateTimeFeaturizer, "Imputer datetime.x"],
        "Select pass through": [SelectColumns],
        "Random Forest": [
            RandomForestClassifier,
            "Scaler.x",
            "OneHot.x",
            "DateTime.x",
            "Imputer text.x",
            "Select pass through.x",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        }
    )
    X["column_4"] = [
        str((datetime(2021, 5, 21, 12, 0, 0) + timedelta(minutes=5 * x)))
        for x in range(len(X))
    ]
    X["column_5"] = X["column_4"]
    X["column_6"] = [42.0] * len(X)
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(X, {"column_5": "NaturalLanguage"})

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    component_graph.instantiate(
        {
            "Select numeric": {"columns": ["column_2"]},
            "Select categorical": {"columns": ["column_1", "column_3"]},
            "Select datetime": {"columns": ["column_4"]},
            "Select text": {"columns": ["column_5"]},
            "Select pass through": {"columns": ["column_6"]},
        }
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Random Forest"] == (
        [
            "column_2",
            "column_3",
            "column_1_a",
            "column_1_b",
            "column_1_c",
            "column_1_d",
            "column_4_year",
            "column_4_month",
            "column_4_day_of_week",
            "column_4_hour",
            "DIVERSITY_SCORE(column_5)",
            "MEAN_CHARACTERS_PER_WORD(column_5)",
            "POLARITY_SCORE(column_5)",
            "LSA(column_5)[0]",
            "LSA(column_5)[1]",
            "column_6",
        ]
    )


def test_component_graph_sampler():
    graph = {
        "Imputer": [Imputer],
        "OneHot": [OneHotEncoder, "Imputer.x"],
        "Undersampler": [Undersampler, "OneHot.x"],
        "Random Forest": [RandomForestClassifier, "Undersampler.x", "Undersampler.y"],
        "Elastic Net": [ElasticNetClassifier, "Undersampler.x", "Undersampler.y"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_inputs("Imputer") == []
    assert component_graph.get_inputs("OneHot") == ["Imputer.x"]
    assert component_graph.get_inputs("Undersampler") == ["OneHot.x"]
    assert component_graph.get_inputs("Random Forest") == [
        "Undersampler.x",
        "Undersampler.y",
    ]
    assert component_graph.get_inputs("Elastic Net") == [
        "Undersampler.x",
        "Undersampler.y",
    ]
    assert component_graph.get_inputs("Logistic Regression") == [
        "Random Forest",
        "Elastic Net",
    ]


def test_component_graph_dataset_with_target_imputer():
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, np.nan])
    graph = {
        "Target Imputer": [TargetImputer],
        "OneHot": [OneHotEncoder, "Target Imputer.x", "Target Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OneHot.x", "Target Imputer.y"],
        "Elastic Net": [ElasticNetClassifier, "OneHot.x", "Target Imputer.y"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
            "Target Imputer.y",
        ],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_inputs("Target Imputer") == []
    assert component_graph.get_inputs("OneHot") == [
        "Target Imputer.x",
        "Target Imputer.y",
    ]
    assert component_graph.get_inputs("Random Forest") == [
        "OneHot.x",
        "Target Imputer.y",
    ]
    assert component_graph.get_inputs("Elastic Net") == [
        "OneHot.x",
        "Target Imputer.y",
    ]

    component_graph.fit(X, y)
    predictions = component_graph.predict(X)
    assert not pd.isnull(predictions).any()


@patch("evalml.pipelines.components.estimators.LogisticRegressionClassifier.fit")
def test_component_graph_sampler_y_passes(mock_estimator_fit):
    pytest.importorskip(
        "imblearn.over_sampling", reason="Cannot import imblearn, skipping tests"
    )
    # makes sure the y value from oversampler gets passed to the estimator
    X = pd.DataFrame({"a": [i for i in range(100)], "b": [i % 3 for i in range(100)]})
    y = pd.Series([0] * 90 + [1] * 10)
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "SMOTE Oversampler": ["SMOTE Oversampler", "Imputer.x", "y"],
        "Standard Scaler": [
            "Standard Scaler",
            "SMOTE Oversampler.x",
            "SMOTE Oversampler.y",
        ],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Standard Scaler.x",
            "SMOTE Oversampler.y",
        ],
    }

    component_graph = ComponentGraph(component_graph)
    component_graph.instantiate({})
    component_graph.fit(X, y)
    assert len(mock_estimator_fit.call_args[0][0]) == len(
        mock_estimator_fit.call_args[0][1]
    )
    assert len(mock_estimator_fit.call_args[0][0]) == int(1.25 * 90)


def test_component_graph_equality(example_graph):
    different_graph = {
        "Target Imputer": [TargetImputer],
        "OneHot": [OneHotEncoder, "Target Imputer.x", "Target Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OneHot.x", "Target Imputer.y"],
        "Elastic Net": [ElasticNetClassifier, "OneHot.x", "Target Imputer.y"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
            "Target Imputer.y",
        ],
    }

    same_graph_different_order = {
        "Imputer": [Imputer],
        "OneHot_ElasticNet": [OneHotEncoder, "Imputer.x"],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x"],
        "Random Forest": [RandomForestClassifier, "OneHot_RandomForest.x"],
        "Elastic Net": [ElasticNetClassifier, "OneHot_ElasticNet.x"],
        "Logistic Regression": [
            LogisticRegressionClassifier,
            "Random Forest",
            "Elastic Net",
        ],
    }

    component_graph = ComponentGraph(example_graph, random_seed=0)
    component_graph_eq = ComponentGraph(example_graph, random_seed=0)
    component_graph_different_seed = ComponentGraph(example_graph, random_seed=5)
    component_graph_not_eq = ComponentGraph(different_graph, random_seed=0)
    component_graph_different_order = ComponentGraph(
        same_graph_different_order, random_seed=0
    )

    component_graph.instantiate({})
    component_graph_eq.instantiate({})
    component_graph_different_seed.instantiate({})
    component_graph_not_eq.instantiate({})
    component_graph_different_order.instantiate({})

    assert component_graph == component_graph
    assert component_graph == component_graph_eq

    assert component_graph != "not a component graph"
    assert component_graph != component_graph_different_seed
    assert component_graph != component_graph_not_eq
    assert component_graph != component_graph_different_order


def test_component_graph_equality_same_graph():
    # Same component nodes and edges, just specified in a different order in the input dictionary
    cg = ComponentGraph(
        {
            "Component A": [DateTimeFeaturizer],
            "Component B": [OneHotEncoder],
            "Random Forest": [RandomForestClassifier, "Component A.x", "Component B.x"],
        }
    )

    cg2 = ComponentGraph(
        {
            "Component B": [OneHotEncoder],
            "Component A": [DateTimeFeaturizer],
            "Random Forest": [RandomForestClassifier, "Component A.x", "Component B.x"],
        }
    )
    cg2 == cg


@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_component_graph(return_dict, example_graph, caplog):
    component_graph = ComponentGraph(example_graph, random_seed=0)
    component_graph.instantiate({})
    expected_component_graph_dict = {
        "Imputer": {
            "name": "Imputer",
            "parameters": {
                "categorical_impute_strategy": "most_frequent",
                "numeric_impute_strategy": "mean",
                "categorical_fill_value": None,
                "numeric_fill_value": None,
            },
        },
        "One Hot Encoder": {
            "name": "One Hot Encoder",
            "parameters": {
                "top_n": 10,
                "features_to_encode": None,
                "categories": None,
                "drop": "if_binary",
                "handle_unknown": "ignore",
                "handle_missing": "error",
            },
        },
        "Random Forest Classifier": {
            "name": "Random Forest Classifier",
            "parameters": {"n_estimators": 100, "max_depth": 6, "n_jobs": -1},
        },
        "Elastic Net Classifier": {
            "name": "Elastic Net Classifier",
            "parameters": {
                "C": 1,
                "l1_ratio": 0.15,
                "n_jobs": -1,
                "solver": "saga",
                "penalty": "elasticnet",
                "multi_class": "auto",
            },
        },
        "Logistic Regression Classifier": {
            "name": "Logistic Regression Classifier",
            "parameters": {
                "penalty": "l2",
                "C": 1.0,
                "n_jobs": -1,
                "multi_class": "auto",
                "solver": "lbfgs",
            },
        },
    }
    component_graph_dict = component_graph.describe(return_dict=return_dict)
    if return_dict:
        assert component_graph_dict == expected_component_graph_dict
    else:
        assert component_graph_dict is None

    out = caplog.text
    for component in component_graph.component_instances.values():
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


class LogTransform(TargetTransformer):
    name = "Log Transform"

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X, y
        y = infer_feature_types(y)
        return X, infer_feature_types(np.log(y))

    def inverse_transform(self, y):
        y = infer_feature_types(y)
        return infer_feature_types(np.exp(y))


class DoubleTransform(TargetTransformer):
    name = "Double Transform"

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X, y
        y = infer_feature_types(y)
        return X, infer_feature_types(y * 2)

    def inverse_transform(self, y):
        y = infer_feature_types(y)
        return infer_feature_types(y / 2)


class SubsetData(Transformer):
    """To simulate a transformer that modifies the target but is not a target transformer, e.g. a sampler."""

    name = "Subset Data"

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.iloc[:50]
        y_new = None
        if y is not None:
            y_new = y.iloc[:50]
        return X_new, y_new


component_graphs = [
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "Log": [LogTransform],
                "Random Forest": ["Random Forest Regressor", "Imputer.x", "Log.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "Log": [LogTransform],
                "Double": [DoubleTransform, "Log.y"],
                "Random Forest": ["Random Forest Regressor", "Imputer.x", "Double.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y / 2)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "Log": [LogTransform, "Imputer.x"],
                "Double": [DoubleTransform, "Log.x", "Log.y"],
                "Random Forest": ["Random Forest Regressor", "Double.x", "Double.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y / 2)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "OneHot": [OneHotEncoder, "Imputer.x"],
                "DateTime": [DateTimeFeaturizer, "OneHot.x"],
                "Log": [LogTransform],
                "Double": [DoubleTransform, "Log.y"],
                "Random Forest": ["Random Forest Regressor", "DateTime.x", "Double.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y / 2)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "OneHot": [OneHotEncoder, "Imputer.x"],
                "DateTime": [DateTimeFeaturizer, "OneHot.x"],
                "Log": [LogTransform],
                "Double": [DoubleTransform, "Log.y"],
                "Double2": [DoubleTransform, "Double.y"],
                "Random Forest": ["Random Forest Regressor", "DateTime.x", "Double2.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y / 4)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": ["Imputer"],
                "Double": [DoubleTransform],
                "DateTime 1": ["DateTime Featurization Component", "Imputer"],
                "ET": ["Extra Trees Regressor", "DateTime 1.x", "Double.y"],
                "Double 2": [DoubleTransform],
                "DateTime 2": ["DateTime Featurization Component", "Imputer"],
                "Double 3": [DoubleTransform, "Double 2.y"],
                "RandomForest": [
                    "Random Forest Regressor",
                    "DateTime 2.x",
                    "Double 3.y",
                ],
                "DateTime 3": ["DateTime Featurization Component", "Imputer"],
                "Double 4": [DoubleTransform],
                "Catboost": ["Random Forest Regressor", "DateTime 3.x", "Double 4.y"],
                "Logistic Regression": [
                    "Linear Regressor",
                    "Catboost",
                    "RandomForest",
                    "ET",
                    "Double 3.y",
                ],
            }
        ),
        lambda y: infer_feature_types(y / 4),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "OneHot": [OneHotEncoder, "Imputer.x"],
                "DateTime": [DateTimeFeaturizer, "OneHot.x"],
                "Log": [LogTransform],
                "Double": [DoubleTransform, "Log.y"],
                "Double2": [DoubleTransform, "Double.y"],
                "Subset": [SubsetData, "DateTime.x", "Double2.y"],
                "Random Forest": ["Random Forest Regressor", "Subset.x", "Subset.y"],
            }
        ),
        lambda y: infer_feature_types(np.exp(y / 4)),
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "Random Forest": ["Random Forest Regressor", "Imputer.x"],
            }
        ),
        lambda y: y,
    ),
    (
        ComponentGraph(
            {
                "Imputer": [Imputer],
                "DateTime": [DateTimeFeaturizer, "Imputer.x"],
                "OneHot": [OneHotEncoder, "DateTime.x"],
                "Random Forest": ["Random Forest Regressor", "OneHot.x"],
            }
        ),
        lambda y: y,
    ),
    (ComponentGraph({"Random Forest": ["Random Forest Regressor"]}), lambda y: y),
    (
        ComponentGraph(
            {
                "Imputer": ["Imputer"],
                "Double": [DoubleTransform],
                "DateTime 1": ["DateTime Featurization Component", "Imputer"],
                "ET": ["Extra Trees Regressor", "DateTime 1.x", "Double.y"],
                "Double 2": [DoubleTransform],
                "DateTime 2": ["DateTime Featurization Component", "Imputer"],
                "Double 3": [DoubleTransform, "Double 2.y"],
                "RandomForest": [
                    "Random Forest Regressor",
                    "DateTime 2.x",
                    "Double 3.y",
                ],
                "DateTime 3": ["DateTime Featurization Component", "Imputer"],
                "Double 4": [DoubleTransform],
                "Linear": ["Linear Regressor", "DateTime 3.x", "Double 4.y"],
                "Logistic Regression": [
                    "Linear Regressor",
                    "Linear",
                    "RandomForest",
                    "ET",
                ],
            }
        ),
        lambda y: y,
    ),
]


@pytest.mark.parametrize("component_graph,answer_func", component_graphs)
def test_component_graph_inverse_transform(
    component_graph, answer_func, X_y_regression
):
    X, y = X_y_regression
    y = pd.Series(np.abs(y))
    X = pd.DataFrame(X)
    component_graph.instantiate({})
    component_graph.fit(X, y)
    predictions = component_graph.predict(X)
    answer = component_graph.inverse_transform(predictions)
    expected = answer_func(predictions)
    pd.testing.assert_series_equal(answer, expected)


def test_final_component_features_does_not_have_target():
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])

    cg = ComponentGraph(
        {
            "Imputer": ["Imputer"],
            "OneHot": ["One Hot Encoder", "Imputer.x"],
            "TargetImputer": ["Target Imputer", "OneHot.x", "OneHot.y"],
            "Logistic Regression": [
                "Logistic Regression Classifier",
                "TargetImputer.x",
                "TargetImputer.y",
            ],
        }
    )
    cg.instantiate({})
    cg.fit(X, y)

    final_features = cg.compute_final_component_features(X, y)
    assert "TargetImputer.y" not in final_features.columns


@patch("evalml.pipelines.components.Imputer.fit_transform")
def test_component_graph_with_X_y_inputs_X(mock_fit):
    class DummyColumnNameTransformer(Transformer):
        name = "Dummy Column Name Transform"

        def __init__(self, parameters=None, random_seed=0):
            super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

        def fit(self, X, y):
            return self

        def transform(self, X, y=None):
            return X.rename(columns=lambda x: x + "_new", inplace=False)

    X = pd.DataFrame(
        {
            "column_1": [0, 2, 3, 1, 5, 6, 5, 4, 3],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        }
    )

    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    graph = {
        "DummyColumnNameTransformer": [DummyColumnNameTransformer, "X", "y"],
        "Imputer": ["Imputer", "DummyColumnNameTransformer.x", "X", "y"],
        "Random Forest": ["Random Forest Classifier", "Imputer.x", "y"],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    mock_fit.return_value = X
    assert component_graph.get_inputs("DummyColumnNameTransformer") == ["X", "y"]
    assert component_graph.get_inputs("Imputer") == [
        "DummyColumnNameTransformer.x",
        "X",
        "y",
    ]

    component_graph.fit(X, y)

    # Check that we have columns from both the output of DummyColumnNameTransformer as well as the original columns since "X" was specified
    assert list(mock_fit.call_args[0][0].columns) == [
        "column_1_new",
        "column_2_new",
        "column_1",
        "column_2",
    ]


@patch("evalml.pipelines.components.Imputer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
def test_component_graph_with_X_y_inputs_y(mock_fit, mock_fit_transform):
    X = pd.DataFrame(
        {
            "column_1": [0, 2, 3, 1, 5, 6, 5, 4, 3],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    graph = {
        "Log": [LogTransform, "X", "y"],
        "Imputer": ["Imputer", "Log.x", "y"],
        "Random Forest": ["Random Forest Classifier", "Imputer.x", "Log.y"],
    }
    mock_fit_transform.return_value = X
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_inputs("Log") == ["X", "y"]
    assert component_graph.get_inputs("Imputer") == ["Log.x", "y"]
    assert component_graph.get_inputs("Random Forest") == ["Imputer.x", "Log.y"]

    component_graph.fit(X, y)
    # Check that we use "y" for Imputer, not "Log.y"
    assert_series_equal(mock_fit_transform.call_args[0][1], y)
    # Check that we use "Log.y" for RF
    assert_series_equal(mock_fit.call_args[0][1], infer_feature_types(np.log(y)))
