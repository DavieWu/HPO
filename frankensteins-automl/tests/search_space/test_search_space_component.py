from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)


component_description = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "requiredInterface": [
        {"id": "required1", "name": "requiredA", "construction_key": "key_a"},
        {"id": "required2", "name": "requiredB", "construction_key": 2},
        {"id": "required3", "name": "requiredC", "construction_key": "key_b"},
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


component_description_with_unknown_type = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "requiredInterface": [{"id": "required1", "name": "requiredA"}],
    "parameter": [
        {"name": "testAbc", "type": "abc"},
        {
            "name": "testCat",
            "default": "a",
            "type": "cat",
            "values": ["a", "b", "c"],
        },
    ],
}

component_description_without_params = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "requiredInterface": [{"id": "required1", "name": "requiredA"}],
    "parameter": [],
}

component_description_without_required_interfaces = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "parameter": [],
}


class TestSearchSpaceComponent:
    def test_search_space_component_creation(self):
        component = SearchSpaceComponent(component_description)
        assert component.get_name() == "testComponent"
        assert component.get_provided_interfaces() == [
            "providedA",
            "providedB",
            "providedC",
        ]
        assert component.get_required_interfaces() == [
            {
                "id": "required1",
                "name": "requiredA",
                "construction_key": "key_a",
            },
            {"id": "required2", "name": "requiredB", "construction_key": 2},
            {
                "id": "required3",
                "name": "requiredC",
                "construction_key": "key_b",
            },
        ]

    def test_correct_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "testCat": "b"}
        )

    def test_unknown_category_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "testCat": "d"}
        )

    def test_too_small_value_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 0.04, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 0, "testCat": "d"}
        )

    def test_too_big_value_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 1.02, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 12, "testCat": "d"}
        )

    def test_wrong_type_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 1, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5.5, "testCat": "d"}
        )

    def test_parameter_amount_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {
                "testDouble": 0.5,
                "testInt": 5,
                "testCat": "b",
                "testAdditional": 1,
            }
        )
        assert not component.validate_parameter_config(
            {"testDouble": 1, "testInt": 5}
        )

    def test_missing_param_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "anotherCat": "d"}
        )

    def test_unkwon_type_param_configuration_validation(self):
        component = SearchSpaceComponent(
            component_description_with_unknown_type
        )
        assert component.validate_parameter_config(
            {"testAbc": 1, "testCat": "a"}
        )

    def test_has_parameter(self):
        component_with_params = SearchSpaceComponent(component_description)
        component_without_params = SearchSpaceComponent(
            component_description_without_params
        )
        assert component_with_params.has_parameter()
        assert not component_without_params.has_parameter()

    def test_has_required_interafaces(self):
        component_with_required_interfaces = SearchSpaceComponent(
            component_description
        )
        compomemt_without_required_interfaces = SearchSpaceComponent(
            component_description_without_required_interfaces
        )
        assert component_with_required_interfaces.has_required_interfaces()
        assert (
            not compomemt_without_required_interfaces.has_required_interfaces()
        )

    def test_contrsuction_args_creation(self):
        component = SearchSpaceComponent(component_description)
        postional, keyword = component.create_construction_args_from_config(
            {"testDouble": 1.2, "testInt": 2, "testCat": "a"},
            {"required1": "abc", "required2": "def", "required3": "ghi"},
        )
        assert postional == [2, 1.2, "def"]
        assert keyword == {"key_a": "abc", "key_b": "ghi", "key_c": "a"}

    def test_default_param_config(self):
        component = SearchSpaceComponent(component_description)
        default_config = component.create_default_parameter_config()
        assert default_config == {
            "testDouble": 0.53,
            "testInt": 6,
            "testCat": "a",
        }

    def test_random_param_config(self):
        component = SearchSpaceComponent(component_description)
        random_config = component.draw_random_parameter_config()
        assert "testDouble" in random_config
        assert "testInt" in random_config
        assert "testCat" in random_config
