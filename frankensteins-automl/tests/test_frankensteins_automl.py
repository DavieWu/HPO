from time import perf_counter
from frankensteins_automl.frankensteins_automl import (
    FrankensteinsAutoMLConfig,
    FrankensteinsAutoML,
)


class TestFrankensteinsAutoMl:
    def test_stop_after_short_timeout(self):
        timeout_in_seconds = 90.0
        tolerance = timeout_in_seconds * 0.1
        config = FrankensteinsAutoMLConfig(
            "res/datasets/blood_transfusion.arff", 4
        )
        config.timeout_in_seconds = timeout_in_seconds
        config.timout_for_optimizers_in_seconds = 30.0
        config.timeout_for_pipeline_evaluation = 10.0

        automl = FrankensteinsAutoML(config)
        start_time = perf_counter()
        automl_results = automl.run()
        stop_time = perf_counter()

        candidate = automl_results["pipeline_object"]
        score = automl_results["search_score"]

        assert (stop_time - start_time) < (timeout_in_seconds + tolerance)
        assert (stop_time - start_time) > (timeout_in_seconds - tolerance)
        assert candidate is not None
        assert score is not None
        assert score > 0

    def test_reproducible_results(self):
        # TODO
        pass
