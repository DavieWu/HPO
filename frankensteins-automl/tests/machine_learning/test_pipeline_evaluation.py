from time import perf_counter
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceRestProblem,
)
from frankensteins_automl.machine_learning.arff_reader import read_arff
from frankensteins_automl.machine_learning.pipeline.pipeline_evaluator import (
    PipelineEvaluator,
)
from frankensteins_automl.search_space.search_space_reader import (
    create_search_space,
)

search_space = create_search_space(
    "res/search_space/ml-plan-ul.json",
    "res/search_space/scikit-learn-classifiers-tpot.json",
    "res/search_space/scikit-learn-preprocessors-tpot.json",
)
c1 = search_space.get_component_by_name("sklearn.pipeline.make_pipeline")
c2 = search_space.get_component_by_name("sklearn.preprocessing.Binarizer")
c3 = search_space.get_component_by_name("sklearn.tree.DecisionTreeClassifier")
required_interfaces = [
    {
        "interface": {
            "name": "AbstractPreprocessor",
            "construction_key": 0,
            "id": c1.get_required_interfaces()[0]["id"],
        },
        "satisfied": True,
        "component_id": "a3f1fa38-0979-11ea-ba87-309c23b50ce0",
        "satisfied_with": "a3f20a5a-0979-11ea-ba87-309c23b50ce0",
    },
    {
        "interface": {
            "name": "BasicClassifier",
            "construction_key": 1,
            "id": c1.get_required_interfaces()[1]["id"],
        },
        "satisfied": True,
        "component_id": "a3f1fa38-0979-11ea-ba87-309c23b50ce0",
        "satisfied_with": "a3f307ac-0979-11ea-ba87-309c23b50ce0",
    },
]
component_mapping = {
    "a3f1fa38-0979-11ea-ba87-309c23b50ce0": c1,
    "a3f20a5a-0979-11ea-ba87-309c23b50ce0": c2,
    "a3f307ac-0979-11ea-ba87-309c23b50ce0": c3,
}
parameter_config = {
    "a3f1fa38-0979-11ea-ba87-309c23b50ce0": {},
    "a3f20a5a-0979-11ea-ba87-309c23b50ce0": {"threshold": 0.505},
    "a3f307ac-0979-11ea-ba87-309c23b50ce0": {
        "criterion": "gini",
        "max_depth": 6,
        "min_samples_split": 11,
        "min_samples_leaf": 11,
    },
}
rest_problem = SearchSpaceRestProblem(required_interfaces, component_mapping)
data_x, data_y, _, _ = read_arff("res/datasets/blood_transfusion.arff", 4)
evaluator = PipelineEvaluator(
    data_x, data_y, "sklearn.pipeline.make_pipeline", rest_problem, 123
)


class TestPipelineEvaluation:
    def test_pipeline_evaluation_value(self):
        assert c1 is not None
        assert c1.get_name() == "sklearn.pipeline.make_pipeline"
        assert c2 is not None
        assert c3 is not None
        score = evaluator.evaluate_pipeline(parameter_config, 10)
        assert score > 0.5

    def test_pipeline_evaluation_timeout(self):
        start_time = perf_counter()
        evaluator.evaluate_pipeline(parameter_config, 0.01)
        stop_time = perf_counter()
        assert (stop_time - start_time) < 0.05
