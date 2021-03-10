import logging
import numpy
import random
from abc import ABC, abstractmethod
from threading import Event

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        pipeline_evaluation_timeout,
        mcts_stop_event,
        seed,
        numpy_random_state,
    ):
        super().__init__()
        self.parameter_domain = parameter_domain
        self.pipeline_evaluator = pipeline_evaluator
        self.pipeline_evaluation_timeout = pipeline_evaluation_timeout
        self.mcts_stop_event = mcts_stop_event
        self.min_vector = self.parameter_domain.get_min_vector()
        self.max_vector = self.parameter_domain.get_max_vector()
        self.seed = seed
        self.numpy_random_state = numpy_random_state
        self.optimization_run_stop_event = Event()

    def get_optimization_run_stop_event(self):
        return self.optimization_run_stop_event

    def _is_stop_event_set(self):
        return self.mcts_stop_event.is_set() or (
            self.optimization_run_stop_event.is_set()
        )

    def _score_candidate(self, candidate):
        candidate = numpy.clip(candidate, self.min_vector, self.max_vector)
        score = self.parameter_domain.get_score_of_result(candidate)
        if score is None and not self.optimization_run_stop_event.is_set():
            configuration = self.parameter_domain.config_from_vector(candidate)
            score = self.pipeline_evaluator.evaluate_pipeline(
                configuration, self.pipeline_evaluation_timeout
            )
            self.parameter_domain.add_result(candidate, score)
        return score

    def _random_transform_candidate(self, candidate, number_of_changes):
        number_of_changes = min(len(candidate), number_of_changes)
        indices = random.sample(range((len(candidate))), number_of_changes)
        for index in indices:
            lower_bound = max(self.min_vector[index], candidate[index] - 0.5)
            upper_bound = min(self.max_vector[index], candidate[index] + 0.5)
            candidate[index] = random.uniform(lower_bound, upper_bound)
        return candidate

    @abstractmethod
    def perform_optimization(self, optimization_time_budget):
        pass
