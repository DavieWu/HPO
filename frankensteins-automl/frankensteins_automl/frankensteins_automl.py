import logging
import random
from pubsub import pub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from frankensteins_automl.event_listener import (
    event_logger,
    event_rest_sender,
    event_topics,
    optimizer_event_counter,
    solution_scored_listener,
)
from frankensteins_automl.machine_learning.arff_reader import read_arff
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
from frankensteins_automl.mcts.mcts_search import MctsSearchConfig, MctsSearch

logger = logging.getLogger(__name__)

topic = event_topics.FRANKENSTEINS_AUTOML_TOPIC


class FrankensteinsAutoMLConfig:
    def __init__(self):
        self.data_path = None
        self.data_target_column_index = None
        self.data_x = None
        self.data_y = None
        self.perform_pipeline_validation = False
        self.validation_ratio = 0.3
        self.timeout_in_seconds = 600.0
        self.timeout_for_optimizers_in_seconds = 30.0
        self.timeout_for_pipeline_evaluation = 10.0
        self.simulation_runs_amount = 3
        self.random_node_selection = False
        self.random_seed = random.randint(0, 30)
        self.optimizers = [
            SMAC,
            GeneticAlgorithm,
            Hyperband,
            discretization_search.DiscretizationSearch,
            RandomSearch,
        ]
        self.count_optimizer_calls = False
        self.event_logging = False
        self.event_send_url = None

    def data_input_from_arff_file(self, data_path, data_target_column_index):
        self.data_path = data_path
        self.data_target_column_index = data_target_column_index

    def direct_data_input(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y


class FrankensteinsAutoML:
    def __init__(self, config):
        self.config = config

    def run(self):
        if self.config is None:
            logger.error(
                "No config for Frankensteins AutoML given. Cannot run."
            )
        elif not isinstance(self.config, FrankensteinsAutoMLConfig):
            logger.error(
                "Config is not a FrankensteinsAutoMLConfig. Cannot run."
            )
        else:
            # Activate event listeners
            solution_scored_listener.activate()
            if self.config.event_logging:
                event_logger.activate()
            if self.config.event_send_url is not None:
                event_rest_sender.activate(self.config.event_send_url)
            optimizer_call_counter = None
            if self.config.count_optimizer_calls:
                optimizer_call_counter = optimizer_event_counter
                optimizer_call_counter.activate()
            # Set random seed for Python random
            seed = int(self.config.random_seed)
            logger.debug(f"Applying random seed {seed}")
            random.seed(seed)

            logger.debug("Load and split data")
            if self._load_data():
                logger.debug("Construct and init search")
                search = self._construct_search(self.search_data)
                logger.debug("Start search")
                pub.sendMessage(topic, payload={"event_type": "SEARCH_START"})
                search.run_search()

                pipeline, search_score = (
                    solution_scored_listener.get_best_solution()
                )
                pub.sendMessage(
                    topic,
                    payload={
                        "event_type": "SEARCH_FINISHED",
                        "score": search_score,
                    },
                )
                results = {
                    "pipeline_object": pipeline,
                    "search_score": search_score,
                }
                if pipeline is None:
                    logger.warning(
                        "Was not able to find a valid pipeline in time"
                    )
                    logger.warning("Please increase the given timeout")
                else:
                    logger.debug(f"Best pipeline: {pipeline}")
                    logger.debug(f"Score of best pipeline: {search_score}")
                    results["pipeline_object"] = pipeline
                    if self.config.perform_pipeline_validation:
                        logger.debug("Validate best pipeline")
                        pipeline.fit(self.search_data[0], self.search_data[1])
                        predictions = pipeline.predict(self.validation_data[0])
                        score = accuracy_score(
                            predictions, self.validation_data[1]
                        )
                        logger.debug(
                            f"Validation score of best pipeline: {score}"
                        )
                        results["validation_score"] = score
                # Add optimizer call counts to result if requested
                if optimizer_call_counter is not None:
                    results[
                        "optimizer_calls"
                    ] = optimizer_call_counter.get_optimizer_call_counts()
                return results

    def _load_data(self):
        data_x = None
        data_y = None
        if self.config.data_x is not None and self.config.data_y is not None:
            data_x = self.config.data_x
            data_y = self.config.data_y
        elif self.config.data_path is not None:
            data_x, data_y, _, _ = read_arff(
                self.config.data_path, self.config.data_target_column_index
            )
        else:
            logger.error(
                "No input data is given! "
                "Either provide a path to an ARFF file or a data array."
            )
            return False
        if data_x is None or data_y is None:
            return False
        if self.config.perform_pipeline_validation:
            search_x, validate_x, search_y, validate_y = train_test_split(
                data_x,
                data_y,
                test_size=self.config.validation_ratio,
                random_state=self.config.random_seed,
                stratify=data_y,
            )
            self.search_data = (search_x, search_y)
            self.validation_data = (validate_x, validate_y)
            return True
        else:
            self.search_data = (data_x, data_y)
            return True

    def _construct_search(self, search_data):
        config = MctsSearchConfig(
            search_data[0], search_data[1], self.config.random_seed
        )
        config.search_timeout = self.config.timeout_in_seconds
        config.optimization_time_budget = (
            self.config.timeout_for_optimizers_in_seconds
        )
        config.timeout_for_pipeline_evaluation = (
            self.config.timeout_for_pipeline_evaluation
        )
        config.simulation_runs_amount = self.config.simulation_runs_amount
        config.random_node_selection = self.config.random_node_selection
        return MctsSearch(config, self.config.optimizers)
