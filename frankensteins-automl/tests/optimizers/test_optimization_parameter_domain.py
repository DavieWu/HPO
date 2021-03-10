import numpy
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)

component_mapping = {
    "abc": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [
                {
                    "id": "required1",
                    "name": "requiredA",
                    "construction_key": "key_a",
                },
                {
                    "id": "required2",
                    "name": "requiredB",
                    "construction_key": 2,
                },
                {
                    "id": "required3",
                    "name": "requiredC",
                    "construction_key": "key_b",
                },
            ],
            "parameter": [
                {
                    "name": "testDouble",
                    "type": "double",
                    "default": 0.53,
                    "min": 0.05,
                    "max": 1.01,
                    "construction_key": 1,
                },
                {
                    "name": "testInt",
                    "type": "int",
                    "default": 6,
                    "min": 1,
                    "max": 11,
                    "construction_key": 0,
                },
                {
                    "name": "testCat",
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                    "construction_key": "key_c",
                },
            ],
        }
    ),
    "def": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [{"id": "required1", "name": "requiredA"}],
            "parameter": [
                {
                    "name": "testCat",
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                }
            ],
        }
    ),
    "ghi": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [{"id": "required1", "name": "requiredA"}],
            "parameter": [],
        }
    ),
}


class TestOptimizationParameterDomain:
    def test_default_configuration(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert numpy.array_equal(
            domain.get_default_config(), [0.53, 6.0, 0, 0]
        )

    def test_random_configuration_completeness(self):
        domain = OptimizationParameterDomain(component_mapping)
        random_config = domain.draw_random_config()
        assert random_config is not None
        assert len(random_config) == 4
        assert numpy.all(random_config >= domain.get_min_vector())
        assert numpy.all(random_config <= domain.get_max_vector())

    def test_parameter_descriptions(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert domain.get_parameter_descriptions() == {
            "abc": {
                "testDouble": {
                    "type": "double",
                    "default": 0.53,
                    "min": 0.05,
                    "max": 1.01,
                    "construction_key": 1,
                },
                "testInt": {
                    "type": "int",
                    "default": 6,
                    "min": 1,
                    "max": 11,
                    "construction_key": 0,
                },
                "testCat": {
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                    "construction_key": "key_c",
                },
            },
            "def": {
                "testCat": {
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                }
            },
            "ghi": {},
        }

    def test_queuing_order(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert not domain.has_results()
        r1 = numpy.array([123.0])
        r2 = numpy.array([456.0])
        r3 = numpy.array([789.0])
        domain.add_result(r1, 12)
        domain.add_result(r2, 3)
        domain.add_result(r3, 42)
        assert domain.has_results()
        top_score, top_candidate = domain.get_top_results(1)[0]
        assert top_score == 42
        assert numpy.array_equal(r3, top_candidate)
        top_results = domain.get_top_results(3)
        expected = [(42, r3), (12, r1), (3, r2)]
        for i in range(3):
            assert top_results[i][0] == expected[i][0]
            assert numpy.array_equal(top_results[i][1], expected[i][1])

    def test_duplicates_not_inserted(self):
        domain = OptimizationParameterDomain(component_mapping)
        r1 = numpy.array([123.0])
        r2 = numpy.array([456.0])
        domain.add_result(r1, 12)
        domain.add_result(r2, 3)
        domain.add_result(r2, 12)
        top_results = domain.get_top_results(3)
        expected = [(12, r1), (3, r2)]
        for i in range(1):
            assert top_results[i][0] == expected[i][0]
            assert numpy.array_equal(top_results[i][1], expected[i][1])

    def test_min_vector(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert numpy.array_equal(
            domain.get_min_vector(), [0.05, 1.0, 0.0, 0.0]
        )

    def test_max_vector(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert numpy.array_equal(
            domain.get_max_vector(), [1.01, 11.0, 2.0, 2.0]
        )

    def test_config_vector_transformation(self):
        domain = OptimizationParameterDomain(component_mapping)
        config = {
            "abc": {"testDouble": 0.53, "testInt": 6, "testCat": "a"},
            "def": {"testCat": "a"},
            "ghi": {},
        }
        vector = [0.53, 6.0, 0.0, 0.0]
        assert numpy.array_equal(domain.config_to_vector(config), vector)
        assert domain.config_from_vector(vector) == config
