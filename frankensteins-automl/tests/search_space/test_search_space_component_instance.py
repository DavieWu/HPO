from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)
from frankensteins_automl.search_space.search_space_component_instance import (
    SearchSpaceComponentInstance,
)
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

description = {
    "name": "sklearn.feature_selection.RFE",
    "providedInterface": [
        "sklearn.feature_selection.RFE",
        "FeatureSelection",
        "AbstractPreprocessor",
        "BasicPreprocessor",
    ],
    "requiredInterface": [
        {
            "id": "estimator",
            "name": "sklearn.ensemble.forest.ExtraTreesClassifier",
            "construction_key": "estimator",
        }
    ],
    "parameter": [
        {
            "name": "step",
            "type": "double",
            "default": 0.53,
            "min": 0.05,
            "max": 1.01,
        }
    ],
}

description_with_wrong_path = {
    "name": "abc.def.GHI",
    "providedInterface": [
        "sklearn.feature_selection.RFE",
        "FeatureSelection",
        "AbstractPreprocessor",
        "BasicPreprocessor",
    ],
}


class TestSearchSpaceComponentInstance:
    def test_instance_construction(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert instance.get_component() == component

    def test_invalid_parameter_configuration(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert not instance.set_parameter_config({"step": 0.01})

    def test_wrong_number_of_required_interfaces(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert not instance.set_required_interfaces({})
        assert not instance.set_required_interfaces(
            {
                "estimator": ExtraTreesClassifier(),
                "additional": ExtraTreesClassifier(),
            }
        )

    def test_missing_required_interfaces(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert not instance.set_required_interfaces(
            {"wrong": ExtraTreesClassifier()}
        )

    def test_wrong_required_interfaces(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert not instance.set_required_interfaces({})

    def test_pipeline_element_construction(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert instance.set_parameter_config({"step": 0.3})
        estimator = ExtraTreesClassifier()
        assert instance.set_required_interfaces({"estimator": estimator})
        pipeline_element = instance.construct_pipeline_element()
        assert isinstance(pipeline_element, RFE)
        assert pipeline_element.get_params(deep=False) == {
            "step": 0.3,
            "estimator": estimator,
            "n_features_to_select": None,
            "verbose": 0,
        }

    def test_pipeline_element_construction_without_params(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        estimator = ExtraTreesClassifier()
        assert instance.set_required_interfaces({"estimator": estimator})
        assert instance.construct_pipeline_element() is None

    def test_pipeline_element_construction_withput_interfaces(self):
        component = SearchSpaceComponent(description)
        instance = SearchSpaceComponentInstance(component)
        assert instance.set_parameter_config({"step": 0.3})
        assert instance.construct_pipeline_element() is None

    def test_pipeline_element_construction_with_wrong_path(self):
        component = SearchSpaceComponent(description_with_wrong_path)
        instance = SearchSpaceComponentInstance(component)
        assert instance.construct_pipeline_element() is None
