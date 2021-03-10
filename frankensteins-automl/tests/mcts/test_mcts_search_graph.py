from numpy.random import RandomState
from frankensteins_automl.machine_learning.pipeline import pipeline_evaluator
from frankensteins_automl.mcts.mcts_search_graph import (
    MctsGraphGenerator,
    MctsGraphNode,
)
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.optimizers.search.random_search import RandomSearch
from frankensteins_automl.search_space import search_space_reader

# Generate a big enoug search space graph
search_space = search_space_reader.create_search_space(
    "res/search_space/ml-plan-ul.json",
    "res/search_space/scikit-learn-classifiers-tpot.json",
    "res/search_space/scikit-learn-preprocessors-tpot.json",
)

# Generate a MCTS Graph by Breadth First Search
generator = MctsGraphGenerator(
    search_space,
    "sklearn.pipeline.make_pipeline",
    [RandomSearch],
    pipeline_evaluator.PipelineEvaluator,
    10,
    [
        [-1.0],
        [1.0],
        [-2.0],
        [2.0],
        [-3.0],
        [3.0],
        [-4.0],
        [4.0],
        [-5.0],
        [5.0],
        [-6.0],
        [6.0],
        [-7.0],
        [7.0],
        [-8.0],
        [8.0],
        [-9.0],
        [9.0],
        [-10.0],
        [10.0],
    ],
    [
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ],
    1,
    RandomState(seed=1),
)
root = generator.get_root_node()
open_list = [root]
optimizer_leaf_nodes = []
while len(open_list) > 0 and len(optimizer_leaf_nodes) < 25:
    node = open_list[0]
    open_list = open_list[1:]
    successors = generator.generate_successors(node)
    if len(successors) > 0:
        open_list.extend(successors)
    else:
        optimizer_leaf_nodes.append(node)


class TestSearchSpaceGraph:
    def test_leaf_optimizer_nodes(self):
        for optimizer_leaf in optimizer_leaf_nodes:
            assert isinstance(optimizer_leaf, MctsGraphNode)
            assert isinstance(optimizer_leaf.get_optimizer(), RandomSearch)
            assert isinstance(
                optimizer_leaf.get_predecessor().get_parameter_domain(),
                OptimizationParameterDomain,
            )
            assert optimizer_leaf.get_predecessor().get_optimizer() is None
            assert optimizer_leaf.get_parameter_domain() is None
            assert optimizer_leaf.is_leaf_node()
            assert optimizer_leaf.get_predecessor().is_search_space_leaf_node()
            assert not optimizer_leaf.get_predecessor().is_leaf_node()
            assert optimizer_leaf.get_node_value() == 0.0
            assert optimizer_leaf.get_simulation_visits() == 0
            assert optimizer_leaf.get_score_avg() == 0.0
            optimizer_leaf.recalculate_node_value(1.0)
            optimizer_leaf.recalculate_node_value(2.0)
            optimizer_leaf.recalculate_node_value(3.0)
            assert optimizer_leaf.get_node_value() == 2.0
            assert optimizer_leaf.get_simulation_visits() == 3
            assert optimizer_leaf.get_score_avg() == 2.0
