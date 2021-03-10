import logging
import random
import time
import uuid
from threading import Thread
from queue import Queue
from frankensteins_automl.mcts.mcts_search_graph import MctsGraphNode

logger = logging.getLogger(__name__)


class MonteCarloSimulationRunner:
    def __init__(
        self,
        start_nodes,
        runs_amount,
        graph_generator,
        leaf_score_method,
        mcts_stop_event,
    ):
        self.start_nodes = start_nodes * runs_amount
        self.runs_amount = runs_amount
        self.graph_generator = graph_generator
        self.leaf_score_method = leaf_score_method
        self.mcts_stop_event = mcts_stop_event
        self.optimizer_stop_event_queue = Queue()

    def run(self, timeout):
        logger.debug(
            f"Starting Monte Carlo simulations at: {self.start_nodes}"
        )
        results = []
        try:
            runs = list(
                map(
                    lambda n: RandomSearchRun(
                        n,
                        self.graph_generator,
                        self.leaf_score_method,
                        timeout,
                        self.mcts_stop_event,
                        self.optimizer_stop_event_queue,
                    ),
                    self.start_nodes,
                )
            )
            run_status = {}
            for r in runs:
                if not self.mcts_stop_event.is_set():
                    run_status[r.get_id()] = False
                    r.start()
            logger.debug("Simulation threads started")

            time.sleep(timeout)

            logger.debug("Set optimization stop events")
            while self.optimizer_stop_event_queue.empty():
                event = self.optimizer_stop_event_queue.get()
                if not event.is_set():
                    event.set()

            logger.debug("Simulation threads are joining ...")
            for r in runs:
                r.join(timeout=2)
                logger.debug(f"[Random Search {r.get_id()}] stopped")
            logger.debug("Simulation threads stopped")

            for r in runs:
                if r.is_alive():
                    run_status[r.get_id()] = False
                result = r.get_result()
                if result is not None:
                    run_status[r.get_id()] = result
                    if result is not None:
                        results.append(result)

            logger.debug(f"Simulation threads finished with: {run_status}")
        except Exception as e:
            logger.exception(f"Error during Simulations: {e}")
        return results


class RandomSearchRun(Thread):
    def __init__(
        self,
        start_node,
        graph_generator,
        leaf_score_method,
        leaf_score_timeout,
        mcts_stop_event,
        optimizer_stop_event_queue,
    ):
        super().__init__()
        self.start_node = start_node
        self.leaf_node = None
        self.graph_generator = graph_generator
        self.leaf_score_method = leaf_score_method
        self.leaf_score_timeout = leaf_score_timeout
        self.mcts_stop_event = mcts_stop_event
        self.id = str(uuid.uuid1())
        self.result = None
        self.daemon = True
        self.optimizer_stop_event_queue = optimizer_stop_event_queue

    def get_id(self):
        return self.id

    def get_result(self):
        return self.result

    def run(self):
        prompt = f"[Random Search {self.id}]"
        logger.debug(f"{prompt} Start simulation at {self.start_node}")
        current_node = self.start_node
        result = None
        score = 0.0
        while self.leaf_node is None:
            if self.mcts_stop_event.is_set():
                logger.debug(f" Early stop event")
                break
            if current_node.is_leaf_node():
                logger.debug(f"{prompt} Found leaf {current_node}")
                self.leaf_node = current_node
            else:
                successors = current_node.get_successors()
                if successors is None or successors == []:
                    logger.debug(f"{prompt} Generating successors")
                    successors = self.graph_generator.generate_successors(
                        current_node
                    )
                current_node = random.choice(successors)
                logger.debug(f"{prompt} Next successor: {current_node}")
        if self.leaf_node is not None:
            if isinstance(self.leaf_node, MctsGraphNode):
                self.optimizer_stop_event_queue.put(
                    self.leaf_node.get_stop_signal_of_optimizer()
                )
                logger.debug(
                    f"{prompt} Start scoring of leaf {self.leaf_node}"
                )
                result, score = self.leaf_score_method(
                    self.leaf_node, self.leaf_score_timeout
                )
                logger.debug(f"{prompt} Optimization finished: {score}")
            else:
                t = type(self.leaf_node)
                logger.error(f"{prompt} ended in an inner node of type: {t}")
        else:
            logger.warning(f"{prompt} Found no leaf node!")
        self.result = (self.leaf_node, score)
