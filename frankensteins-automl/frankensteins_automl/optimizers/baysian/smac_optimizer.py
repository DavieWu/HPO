import logging
import numpy
from smac.configspace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType


logger = logging.getLogger(__name__)


class SMAC(AbstractOptimizer):
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
        self.configuration_space = self._create_configuration_space()
        self.candidates_for_warmstart_history = 100

    def _create_configuration_space(self):
        config_space = ConfigurationSpace()
        min = self.parameter_domain.get_min_vector()
        max = self.parameter_domain.get_max_vector()
        for i in range(len(min)):
            param = None
            if min[i] == max[i]:
                param = UniformFloatHyperparameter(
                    name=str(i), lower=min[i], upper=max[i] + 0.0001
                )
            else:
                param = UniformFloatHyperparameter(
                    name=str(i), lower=min[i], upper=max[i]
                )
            config_space.add_hyperparameter(param)
        return config_space

    def _create_scenario(self, optimization_time_budget):
        return Scenario(
            {
                "cs": self.configuration_space,
                "run_obj": "quality",
                "wallclock_limit": optimization_time_budget,
                "deterministic": "true",
                "output_dir": None,
            }
        )

    def _create_run_history(self):
        runhistory = RunHistory()
        candidates = []
        candidates.extend(
            self.parameter_domain.get_top_results(
                int(self.candidates_for_warmstart_history * 0.5)
            )
        )
        if len(candidates) == int(self.candidates_for_warmstart_history * 0.5):
            for _ in range(int(self.candidates_for_warmstart_history * 0.5)):
                candidates.append(self.parameter_domain.get_random_result())
        for score, candidate in candidates:
            runhistory.add(
                config=Configuration(
                    self.configuration_space,
                    values=self._vector_to_smac_dict(candidate),
                ),
                cost=(1 - score),
                time=self.pipeline_evaluation_timeout,
                status=StatusType.SUCCESS,
                seed=self.seed,
            )
        return runhistory

    def _smac_dict_to_vector(self, config):
        length = len(config)
        vector = numpy.zeros(length)
        for i in range(length):
            vector[i] = config[str(i)]
        return vector

    def _vector_to_smac_dict(self, vector):
        length = len(vector)
        config = {}
        for i in range(length):
            config[str(i)] = vector[i]
        return config

    def perform_optimization(self, optimization_time_budget):
        def _evaluate_config(config):
            vector = self._smac_dict_to_vector(config.get_dictionary())
            score = self._score_candidate(vector)
            return 1 - score

        if not self._is_stop_event_set():
            smac = SMAC4HPO(
                scenario=self._create_scenario(optimization_time_budget),
                tae_runner=_evaluate_config,
                rng=self.numpy_random_state,
                runhistory=self._create_run_history(),
            )
            try:
                candidate = smac.optimize()
            finally:
                candidate = smac.solver.incumbent
            vector = self._smac_dict_to_vector(candidate.get_dictionary())
            score = self._score_candidate(vector)
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = self._smac_dict_to_vector(
                    candidate.get_dictionary()
                )
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )
