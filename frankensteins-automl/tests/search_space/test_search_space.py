from frankensteins_automl.search_space import search_space_reader


search_space = search_space_reader.create_search_space(
    "res/search_space/scikit-learn-classifiers-tpot.json"
)


class TestSearchSpaceReader:
    def test_component_retrievement(self):
        component = search_space.get_component_by_name(
            "sklearn.naive_bayes.GaussianNB"
        )
        assert component.get_name() == "sklearn.naive_bayes.GaussianNB"

    def test_non_existing_component_retrievement(self):
        assert search_space.get_component_by_name("abc.def.GHI") is None

    def test_interface_retrievement(self):
        providing_components = search_space.get_components_providing_interface(
            "BaseLearner"
        )
        providing_components_names = []
        for component in providing_components:
            providing_components_names.append(component.get_name())
        components = [
            "sklearn.naive_bayes.GaussianNB",
            "sklearn.naive_bayes.BernoulliNB",
            "sklearn.naive_bayes.MultinomialNB",
            "sklearn.tree.DecisionTreeClassifier",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.GradientBoostingClassifier",
            "sklearn.neighbors.KNeighborsClassifier",
            "sklearn.svm.LinearSVC",
        ]
        providing_components_names.sort()
        components.sort()
        assert providing_components_names == components

    def test_not_existing_interface_retrievement(self):
        providing_components = search_space.get_components_providing_interface(
            "abc"
        )
        assert providing_components is None
