import logging
from time import time
from frankensteins_automl.mcts.monte_carlo_simulation_runner import (
    MonteCarloSimulationRunner,
)
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from frankensteins_automl.optimizers.search.discretization import (
    discretization_helper,
)
from frankensteins_automl.search_space.graphs import GraphGenerator, GraphNode


logger = logging.getLogger(__name__)

SIMULATION_RUNS_AMOUNT = 3


class DiscretizationSearch(AbstractOptimizer):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        timeout_for_pipeline_evaluation,
        mcts_stop_event,
        seed,
        numpy_random_state,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
            mcts_stop_event,
            seed,
            numpy_random_state,
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)
        self.graph_generator = DiscretizationGraphGenerator(
            self.parameter_domain
        )

    def perform_optimization(self, optimization_time_budget):
        def score_atomic_discretization(node, _):
            config_vector = self.parameter_domain.config_to_vector(
                node.get_discretization().get_config()
            )
            return config_vector, self._score_candidate(config_vector)

        root = self.graph_generator.get_root_node()
        run_stop = time() + optimization_time_budget
        while (not self._is_stop_event_set()) and (time() < run_stop):
            # Stop optimization if the root node is covered
            # since then the whole graph was evaluated
            if root.is_covered():
                break
            current_node = root

            # Find unexpanded, not covered node
            # where the highest rated sub-graph is rooted
            while current_node.get_successors() != [] and (
                not self._is_stop_event_set()
            ):
                successors = current_node.get_successors()
                successors = list(
                    filter(lambda n: not n.is_covered(), successors)
                )
                # A successor node was found, which is covered
                # Perform backpropagation and restart search from the root
                # Except when the root node is covered, than abort
                if len(successors) == 0:
                    if not current_node.is_covered():
                        current_node.backpropagate()
                    if root.get_node_id() == current_node.get_node_id():
                        break
                    else:
                        current_node = root
                        continue
                # A uncovered node was found
                # The best successor can be selected for next iteration
                best_successor = successors[0]
                best_successor_score = successors[0].get_best_successor_score()
                for successor in successors:
                    successor_score = successor.get_best_successor_score()
                    if successor_score > best_successor_score:
                        best_successor = successor
                        best_successor_score = successor_score
                current_node = best_successor

            if current_node == root or current_node is None:
                break

            # Expand this node
            expanded_nodes = self.graph_generator.generate_successors(
                current_node
            )
            current_node.set_successors(expanded_nodes)

            # Score expanded nodes with Monte Carlo simulations
            runner = MonteCarloSimulationRunner(
                expanded_nodes,
                SIMULATION_RUNS_AMOUNT,
                self.graph_generator,
                score_atomic_discretization,
                self.mcts_stop_event,
            )

            # Check if the simulations found a new best candidate
            # and backpropagate the results from the leafs
            results = runner.run(self.pipeline_evaluation_timeout)
            for result in results:
                leaf, candidate, score = result
                leaf.best_successor_score = score
                leaf.backpropagate()
                if score > self.best_score:
                    self.best_score = score
                    self.best_candidate = candidate
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )


class DiscretizationGraphNode(GraphNode):
    def __init__(self, predecessor, discretization):
        super().__init__(predecessor)
        self.discretization = discretization
        self.covered = self.discretization.is_atomic()
        self.best_successor_score = float("-inf")

    def backpropagate(self):
        if not self.covered:
            all_successores_covered = True
            for successor in self.successors:
                if not successor.is_covered():
                    all_successores_covered = False
                successor_score = successor.get_best_successor_score()
                if successor_score > self.best_successor_score:
                    self.best_successor_score = successor_score
            self.covered = all_successores_covered
            if self.predecessor is not None:
                self.predecessor.backpropagate()

    def get_discretization(self):
        return self.discretization

    def is_covered(self):
        return self.covered

    def get_best_successor_score(self):
        return self.best_successor_score

    def is_leaf_node(self):
        return self.discretization.is_atomic()


class DiscretizationGraphGenerator(GraphGenerator):
    def __init__(self, parameter_domain):
        self.parameter_domain = parameter_domain
        self.root_node = DiscretizationGraphNode(
            None,
            discretization_helper.Discretization(
                parameter_domain.get_parameter_descriptions()
            ),
        )

    def get_root_node(self):
        return self.root_node

    def get_node_successors(self, node):
        refinements = discretization_helper.refine_discretization(
            node.get_discretization()
        )
        successors = []
        for refinement in refinements:
            successors.append(DiscretizationGraphNode(node, refinement))
        return successors
