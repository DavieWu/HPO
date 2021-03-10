import logging
import math
from pubsub import pub
from frankensteins_automl.event_listener import event_topics
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceGraphNode,
    SearchSpaceGraphGenerator,
)

logger = logging.getLogger(__name__)
search_topic = event_topics.SEARCH_GRAPH_TOPIC
optimization_topic = event_topics.OPTIMIZATION_TOPIC


class MctsGraphNode(SearchSpaceGraphNode):
    def __init__(
        self, predecessor, rest_problem, specified_interface, optimizer
    ):
        super().__init__(predecessor, rest_problem, specified_interface)
        self.optimizer = optimizer
        self.parameter_domain = None
        if (
            self.rest_problem.is_satisfied() and super().is_leaf_node()
        ) and self.optimizer is None:
            self.parameter_domain = OptimizationParameterDomain(
                self.rest_problem.get_component_mapping()
            )
        self.node_value = 1.0
        self.simulation_visits = 0
        self.score_avg = 0.0
        self.best_optimization_value = 0.0

    def is_search_space_leaf_node(self):
        return super().is_leaf_node()

    def is_leaf_node(self):
        if super().is_leaf_node():
            return self.optimizer is not None
        else:
            False

    def get_node_value(self):
        return self.node_value

    def get_score_avg(self):
        return self.score_avg

    def get_simulation_visits(self):
        return self.simulation_visits

    def get_optimizer(self):
        return self.optimizer

    def get_parameter_domain(self):
        return self.parameter_domain

    def get_stop_signal_of_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer.get_optimization_run_stop_event()

    def recalculate_node_value(self, new_result):
        logger.debug(
            f"Update {self.node_id} from {self.node_value} with {new_result}"
        )
        self.simulation_visits = self.simulation_visits + 1
        logger.debug(f"Number of visits: {self.simulation_visits}")
        self.score_avg = self.score_avg + (
            (new_result - self.score_avg) / self.simulation_visits
        )
        logger.debug(f"Score Avg: {self.score_avg}")
        exploration_factor = 0.0
        predecessor = self.get_predecessor()
        if predecessor is not None and isinstance(predecessor, MctsGraphNode):
            if self.simulation_visits < (
                predecessor.get_simulation_visits() + 1
            ):
                exploration_factor = math.log(
                    float(predecessor.get_simulation_visits() + 1)
                )
                exploration_factor = exploration_factor / float(
                    self.simulation_visits
                )
                exploration_factor = math.sqrt(exploration_factor)
                exploration_factor = exploration_factor * 1.41421
                logger.debug(f"Exploration factor: {exploration_factor}")
        else:
            logger.debug(
                f"Can't calc expl factor since predecessor is {predecessor}"
            )
        self.node_value = self.score_avg + exploration_factor
        logger.debug(f"New node value: {self.node_value}")
        if predecessor is not None:
            pub.sendMessage(
                search_topic,
                payload={
                    "event_type": "WEIGHT_UPDATE",
                    "from": self.predecessor.get_node_id(),
                    "to": self.get_node_id(),
                    "weight": self.node_value,
                },
            )

    def perform_optimization(self, time_budget, stop_event):
        logger.debug(f"Start optimization in {self.node_id}")
        if self.optimizer is not None:
            pub.sendMessage(
                optimization_topic,
                payload={"optimizer_class": type(self.optimizer).__name__},
            )
            if stop_event.is_set():
                logger.debug(
                    f"Stop optimization in {self.node_id} because stop event"
                )
                return

            # Check if there are any parameters to optimizer
            optimization_required = True
            param_domain = self.predecessor.get_parameter_domain()
            if param_domain is not None:
                optimization_required = param_domain.get_has_parameter()

            config = param_domain.get_default_config()
            score = 0.0

            # If there are parameters, perform an optimization
            # Skip optimization and score empty config otherwise
            if optimization_required:
                config, score = self.optimizer.perform_optimization(
                    time_budget
                )
            else:
                vector = param_domain.config_to_vector(config)
                score = self.optimizer._score_candidate(vector)

            # Check if a new best score was found
            if score > self.best_optimization_value:
                self.best_optimization_value = score
                pub.sendMessage(
                    search_topic,
                    payload={
                        "event_type": "OPTIMIZER_UPDATE",
                        "id": self.node_id,
                        "score": score,
                    },
                )
            return config, score
        else:
            logger.warning("Node has no optimizer!")


class MctsGraphGenerator(SearchSpaceGraphGenerator):
    def __init__(
        self,
        search_space,
        initial_component_name,
        optimizer_classes,
        pipeline_evaluator_class,
        mcts_stop_event,
        timeout_for_pipeline_evaluation,
        data_x,
        data_y,
        seed,
        numpy_random_state,
    ):
        super().__init__(search_space, initial_component_name)
        self.root_node = MctsGraphNode(
            None, self.root_node.get_rest_problem(), None, None
        )
        self.pipeline_evaluator_class = pipeline_evaluator_class
        self.mcts_stop_event = mcts_stop_event
        self.timeout_for_pipeline_evaluation = timeout_for_pipeline_evaluation
        self.data_x = data_x
        self.data_y = data_y
        self.optimizer_classes = optimizer_classes
        self.seed = seed
        self.numpy_random_state = numpy_random_state
        # Send event for new root node
        root_node_payload = self.root_node.get_event_payload()
        root_node_payload["node_class"] = type(self.root_node).__name__
        root_node_payload["event_type"] = "NEW_NODE"
        root_node_payload["node_type"] = "root"
        pub.sendMessage(search_topic, payload=root_node_payload)

    def get_node_successors(self, node):
        if node.is_leaf_node():
            return []
        else:
            successors = []
            node_type = "inner"
            is_indirect_leaf = False
            if isinstance(node, MctsGraphNode):
                is_indirect_leaf = node.is_search_space_leaf_node()
            else:
                is_indirect_leaf = node.is_leaf_node()
            if is_indirect_leaf:
                for optimizer_class in self.optimizer_classes:
                    node_type = "optimizer"
                    pipeline_evaluator = self.pipeline_evaluator_class(
                        self.data_x,
                        self.data_y,
                        self.initial_component_name,
                        node.get_rest_problem(),
                        self.seed,
                    )
                    optimizer = optimizer_class(
                        node.get_parameter_domain(),
                        pipeline_evaluator,
                        self.timeout_for_pipeline_evaluation,
                        self.mcts_stop_event,
                        self.seed,
                        self.numpy_random_state,
                    )
                    successors.append(
                        MctsGraphNode(
                            node,
                            node.get_rest_problem(),
                            type(optimizer).__name__,
                            optimizer,
                        )
                    )
                logger.debug(f"{len(successors)} successors for optimizers")
            else:
                successors = list(
                    map(
                        lambda n: MctsGraphNode(
                            node,
                            n.get_rest_problem(),
                            n.get_specified_interface(),
                            None,
                        ),
                        super().get_node_successors(node),
                    )
                )
            # Send event for new successor node
            for successor in successors:
                new_node_payload = successor.get_event_payload()
                new_node_payload["event_type"] = "NEW_NODE"
                new_node_payload["node_class"] = type(successor).__name__
                new_node_payload["node_type"] = node_type
                pub.sendMessage(search_topic, payload=new_node_payload)
            node.set_successors(successors)
            return successors
