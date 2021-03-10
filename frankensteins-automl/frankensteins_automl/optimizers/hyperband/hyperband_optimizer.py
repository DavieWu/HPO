import logging
import random
from time import time
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from frankensteins_automl.optimizers.hyperband.hyperband_runner import (
    HyperbandRunner,
)

logger = logging.getLogger(__name__)


class Hyperband(AbstractOptimizer):
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
        self.hyperband_runner = HyperbandRunner(
            self._select_hyperband_candidates,
            super()._score_candidate,
            super()._random_transform_candidate,
            self._check_candidate_for_early_stop,
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)

    def perform_optimization(self, optimization_time_budget):
        spend_time_budget = 0.0
        last_hyperband_run = 0.0
        est_time = 0.0
        while (est_time <= (optimization_time_budget)) and (
            not self._is_stop_event_set()
        ):
            run_start = time()
            run_candidate, run_score = self.hyperband_runner.run(
                self.optimization_run_stop_event, self.mcts_stop_event
            )
            if run_score >= self.best_score:
                self.best_score = run_score
                self.best_candidate = run_candidate
            last_hyperband_run = time() - run_start
            spend_time_budget = spend_time_budget + last_hyperband_run
            est_time = spend_time_budget + last_hyperband_run
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )

    def _select_hyperband_candidates(self):
        # 50:50 chance between using one of the top 50 candidates
        # (if there are already 50 candidates)
        # and a random candidate
        top_candidates = self.parameter_domain.get_top_results(50)
        i = random.randint(0, 99)
        if i < len(top_candidates):
            return top_candidates[i][1]
        else:
            return self.parameter_domain.draw_random_config()

    def _check_candidate_for_early_stop(self, score, best_score, n_iterations):
        # Currently not used, but could be applied in later experiments
        return False
