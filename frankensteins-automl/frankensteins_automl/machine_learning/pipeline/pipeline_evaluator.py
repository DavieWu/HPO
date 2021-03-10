import logging
import pickle
import time
import warnings
import multiprocessing
from pubsub import pub
from frankensteins_automl.event_listener import event_topics
from frankensteins_automl.machine_learning.pipeline import pipeline_constructor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

topic = event_topics.FRANKENSTEINS_AUTOML_TOPIC


class PipelineEvaluator:
    def __init__(
        self,
        data_x,
        data_y,
        start_component_name,
        satisfied_rest_problem,
        random_seed,
    ):
        self.data_x = data_x
        self.data_y = data_y
        self.start_component_name = start_component_name
        self.rest_problem = satisfied_rest_problem
        self.random_seed = random_seed
        self.mp_context = multiprocessing.get_context()

    def evaluate_pipeline(self, pipeline_parameter_config, timeout, ratio=1.0):
        def _calc_cross_val(pipeline, data_x, data_y, shared_value):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                splitter = StratifiedShuffleSplit(
                    n_splits=2, random_state=self.random_seed, train_size=0.7
                )
                try:
                    score = cross_val_score(
                        pipeline,
                        data_x,
                        data_y,
                        cv=splitter,
                        error_score="raise",
                    ).mean()
                    logger.debug(f"{pipeline} achieved : {score}")
                    with shared_value.get_lock():
                        shared_value.value = score
                except Exception as e:
                    logger.exception(f"Error while scoring pipeline: {e}")
                for warning in w:
                    logger.debug(f"Pipeline evaluation warning: {warning}")

        score = 0.0
        pipeline = pipeline_constructor.construct_pipeline(
            self.start_component_name,
            self.rest_problem,
            pipeline_parameter_config,
        )
        if pipeline is not None:
            pipeline_data = pickle.dumps(pipeline).hex()
            data_x = self.data_x
            data_y = self.data_y
            # Try to create a subset of the training data
            # with the given ratio
            try:
                if ratio < 1.0:
                    data_x, _, data_y, _ = train_test_split(
                        data_x,
                        data_y,
                        train_size=ratio,
                        random_state=self.random_seed,
                    )
            finally:
                # Score the pipeline with the cross-validation
                # mean on the data or the sample split

                managed_score = self.mp_context.Value("d", 0.0)
                process = self.mp_context.Process(
                    target=_calc_cross_val,
                    daemon=True,
                    args=(pipeline, data_x, data_y, managed_score),
                )
                time.sleep(2.0)
                try:
                    process.start()
                except Exception as e:
                    process.close()
                    logger.error(
                        f"Wasn't able to start evaluation process: {e}"
                    )
                    return 0.0
                process.join(timeout - 2.0)
                score = managed_score.value
                if process.is_alive():
                    process.kill()
                else:
                    process.close()
                pub.sendMessage(
                    topic,
                    payload={
                        "event_type": "SOLUTION_SCORED",
                        "pipeline_data": pipeline_data,
                        "score": score,
                    },
                )
        else:
            logger.warning("Constructed pipeline is None")
        return score
