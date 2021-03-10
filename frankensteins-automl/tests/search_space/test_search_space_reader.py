from frankensteins_automl.search_space import search_space_reader
from frankensteins_automl.search_space.search_space import SearchSpace


class TestSearchSpaceReader:
    def test_non_string_path(self):
        assert search_space_reader.create_search_space(123) is None

    def test_non_json_path(self):
        assert search_space_reader.create_search_space("README.md") is None

    def test_json_parsing(self):
        search_space = search_space_reader.create_search_space(
            "res/search_space/ml-plan-ul.json"
        )
        assert isinstance(search_space, SearchSpace)

    def test_assigned_ids(self):
        search_space = search_space_reader.create_search_space(
            "res/search_space/ml-plan-ul.json"
        )
        c1 = search_space.get_component_by_name(
            "sklearn.pipeline.make_pipeline"
        )
        c2 = search_space.get_component_by_name("sklearn.pipeline.make_union")
        components = [c1, c2]
        used_ids = []
        for component in components:
            for ri in component.get_required_interfaces():
                assert "id" in ri
                assert ri["id"] not in used_ids
                used_ids.append(ri["id"])
