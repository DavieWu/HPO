import numpy
from threading import Event
from time import perf_counter
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.optimizers.baysian.smac_optimizer import SMAC
from frankensteins_automl.optimizers.evolution.genetic_algorithm import (
    GeneticAlgorithm,
)
from frankensteins_automl.optimizers.hyperband.hyperband_optimizer import (
    Hyperband,
)
from frankensteins_automl.optimizers.search.discretization import (
    discretization_search,
)
from frankensteins_automl.optimizers.search.random_search import RandomSearch
from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)

component_mapping = {
    "vars": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": [],
            "requiredInterface": [],
            "parameter": [
                {
                    "name": "x",
                    "type": "double",
                    "default": 2.0,
                    "min": -5.0,
                    "max": 5.0,
                    "construction_key": 1,
                },
                {
                    "name": "y",
                    "type": "double",
                    "default": -2.0,
                    "min": -5.0,
                    "max": 5.0,
                    "construction_key": 2,
                },
            ],
        }
    )
}

optimizer_classes = [
    SMAC,
    GeneticAlgorithm,
    Hyperband,
    discretization_search.DiscretizationSearch,
    RandomSearch,
]


class TestEvaluator:
    def evaluate_pipeline(self, params, *args):
        return -(params["vars"]["x"]) ** 2 - (params["vars"]["y"]) ** 2


class TestOptimizers:
    def test_optimization_result_improvements(self):
        for optimizer_class in optimizer_classes:
            domain = OptimizationParameterDomain(component_mapping)
            optimizer = optimizer_class(
                domain,
                TestEvaluator(),
                2.0,
                1,
                numpy.random.RandomState(seed=1),
            )
            _, score = optimizer.perform_optimization(10, Event())
            assert score > -8.0

    def test_optimizer_uses_warmstart(self):
        existing_score = -2.4200000000000004
        for optimizer_class in optimizer_classes:
            domain = OptimizationParameterDomain(component_mapping)
            domain.add_result(numpy.array([1.1, -1.1]), existing_score)
            optimizer = optimizer_class(
                domain,
                TestEvaluator(),
                2.0,
                1,
                numpy.random.RandomState(seed=1),
            )
            _, score = optimizer.perform_optimization(0.1, Event())
            assert existing_score <= score

    def test_optimization_timeout(self):
        timeout_in_seconds = 5
        for optimizer_class in optimizer_classes:
            domain = OptimizationParameterDomain(component_mapping)
            optimizer = optimizer_class(
                domain,
                TestEvaluator(),
                2.0,
                1,
                numpy.random.RandomState(seed=1),
            )
            start_time = perf_counter()
            optimizer.perform_optimization(timeout_in_seconds, Event())
            stop_time = perf_counter()
            assert (stop_time - start_time) < (timeout_in_seconds + 1)
            assert (stop_time - start_time) > (timeout_in_seconds - 1)
