import logging
import random
import time
from numpy.random import RandomState
from threading import Thread, Event
from frankensteins_automl.event_listener import event_topics
from frankensteins_automl.mcts.monte_carlo_simulation_runner import (
    MonteCarloSimulationRunner,
)
from frankensteins_automl.mcts.mcts_search_graph import (
    MctsGraphGenerator,
    MctsGraphNode,
)
from frankensteins_automl.machine_learning.pipeline import pipeline_evaluator
from frankensteins_automl.search_space.search_space_reader import (
    create_search_space,
)

logger = logging.getLogger(__name__)
topic = event_topics.MCTS_TOPIC


class MctsSearchConfig:
    def __init__(self, data_x, data_y, seed):
        self.search_timeout = 600.0
        self.optimization_time_budget = 30.0
        self.timeout_for_pipeline_evaluation = 10.0
        self.pipeline_evaluator_class = pipeline_evaluator.PipelineEvaluator
        basepath = "res/search_space/"
        self.search_space_files = [
            f"{basepath}frankensteins_automl_topologies.json",
            f"{basepath}frankensteins_automl_data_preprocessors.json",
            f"{basepath}frankensteins_automl_feature_preprocessors.json",
            f"{basepath}frankensteins_automl_classifiers.json",
        ]
        start_component_module = (
            "frankensteins_automl.machine_learning.pipeline"
        )
        start_component_class = ".pipeline_constructor"
        self.start_component_name = (
            f"{start_component_module}{start_component_class}.build_topology"
        )
        self.random_node_selection = False
        self.simulation_runs_amount = 3
        self.data_x = data_x
        self.data_y = data_y
        self.seed = seed
        self.numpy_random_state = RandomState(seed=seed)

    def __str__(self):
        config = {
            "search_timeout": self.search_timeout,
            "optimization_time_budget": self.optimization_time_budget,
            "timeout_for_pipeline_evaluation": (
                self.timeout_for_pipeline_evaluation
            ),
            "pipeline_evaluator_class": self.pipeline_evaluator_class,
            "search_space_files": self.search_space_files,
            "start_component_name": self.start_component_name,
            "random_node_selection": self.random_node_selection,
            "simulation_runs_amount": self.simulation_runs_amount,
            "seed": self.seed,
        }
        return str(config)


class MctsSearch:
    def __init__(self, search_config, optimizers):
        self.config = search_config
        self.search_space = create_search_space(
            *self.config.search_space_files
        )
        self.stop_event = Event()
        self.graph_generator = MctsGraphGenerator(
            self.search_space,
            self.config.start_component_name,
            optimizers,
            self.config.pipeline_evaluator_class,
            self.stop_event,
            self.config.timeout_for_pipeline_evaluation,
            self.config.data_x,
            self.config.data_y,
            self.config.seed,
            self.config.numpy_random_state,
        )
        self.root_node = self.graph_generator.get_root_node()
        self.random_selection = self.config.random_node_selection

    def run_search(self):
        try:
            logger.info(f"Start MCTS with config: {self.config}")
            search_thread = Thread(target=self._search_loop, daemon=True)
            search_thread.start()
            time.sleep(self.config.search_timeout - 5)
            logger.info("Stopping MCTS ...")
            self.stop_event.set()
            search_thread.join(timeout=5)
        except Exception as e:
            logger.exception(f"Error during search: {e}")

    def _search_loop(self):
        while True:
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
            logger.debug("Next MCTS search loop iteration")
            logger.debug("Select candidate node")
            candidate_node = self._select_candidate_node()
            logger.debug("Expand candidate node")
            expanded_nodes = self._candidate_node_expansion(candidate_node)
            logger.debug("Simulate expanded node")
            reached_leaf_nodes = self._simulation_of_expanded_nodes(
                expanded_nodes, self.stop_event
            )
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
            if not self.random_selection:
                logger.debug("Perform back propagation")
                self._back_propagation(reached_leaf_nodes)
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
        logger.debug("End of MCTS search loop reached")

    def _select_candidate_node(self):
        current_node = self.root_node
        while (current_node.get_successors() is not None) and (
            len(current_node.get_successors()) > 0
        ):
            successors = current_node.get_successors()
            current_node = successors[0]
            best_score = current_node.get_node_value()
            all_identical = True
            if not self.random_selection:
                for successor in successors:
                    node_value = successor.get_node_value()
                    if node_value != best_score:
                        all_identical = False
                        if node_value > best_score:
                            current_node = successor
                            best_score = node_value
            # Select a random successor, if this behavior is requested
            # or if all node values are identical to prevent
            # always choosing the first successor
            if all_identical:
                logger.debug("Select next MCTS candidate node randomly")
                current_node = random.choice(successors)

        logger.debug(f"Next candidate for expansion is {current_node}")
        return current_node

    def _candidate_node_expansion(self, candidate_node):
        if candidate_node.is_leaf_node():
            return [candidate_node]
        successors = self.graph_generator.generate_successors(candidate_node)
        return successors

    def _simulation_of_expanded_nodes(self, expanded_nodes, stop_event):
        def optimize_leaf_node(leaf_node, time_budget):
            return leaf_node.perform_optimization(time_budget, stop_event)

        runner = MonteCarloSimulationRunner(
            expanded_nodes,
            self.config.simulation_runs_amount,
            self.graph_generator,
            optimize_leaf_node,
            stop_event,
        )
        return runner.run(self.config.optimization_time_budget)

    def _back_propagation(self, leaf_nodes):
        while len(leaf_nodes) > 0:
            node, score = leaf_nodes.pop(0)
            logger.debug(f"Recalculate {node.get_node_id()} with {score}")
            if isinstance(node, MctsGraphNode):
                node.recalculate_node_value(score)
            predecessor = node.get_predecessor()
            if predecessor is not None:
                leaf_nodes.append((predecessor, score))
