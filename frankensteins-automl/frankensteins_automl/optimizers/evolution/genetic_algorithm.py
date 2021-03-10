import logging
import math
import numpy
import random
from time import time
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)


logger = logging.getLogger(__name__)


class GeneticAlgorithm(AbstractOptimizer):
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
        self.individuals_per_generation = 100

    def perform_optimization(self, optimization_time_budget):
        best_from_domain = self.parameter_domain.get_top_results(1)
        if best_from_domain is not None and len(best_from_domain) > 0:
            self.best_score, self.best_candidate = best_from_domain[0]
        logger.debug(f"Genetic algorithm starts")
        self._evolution_loop(optimization_time_budget)
        logger.debug(f"Genetic algorithm ends with score: {self.best_score}")
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )

    def _evolution_loop(self, optimization_time_budget):
        run_stop = time() + optimization_time_budget
        generation = self._init_first_generation()
        while (not self._is_stop_event_set()) and (time() < run_stop):
            next_generation = self._generate_next_generation(generation)
            generation = []
            for individual in next_generation:
                score = self._score_candidate(individual)
                generation.append((score, individual))
                if score > self.best_score:
                    self.best_candidate = individual
                    self.best_score = score
                    logger.debug(
                        f"New best score in genetic algorithm: {score}"
                    )
                if self._is_stop_event_set() or (time() > run_stop):
                    break
            if self._is_stop_event_set() or (time() > run_stop):
                logger.info("Stop event in genetic algorithm")
                break

    def _init_first_generation(self):
        generation = []
        # Fill 20% of the first generation with the top existing results
        generation.extend(
            self.parameter_domain.get_top_results(
                int(self.individuals_per_generation * 0.2)
            )
        )
        # Fill the other 80% with random candidates
        for _ in range(self.individuals_per_generation - len(generation)):
            if self._is_stop_event_set():
                break
            individual = self.parameter_domain.draw_random_config()
            score = self._score_candidate(individual)
            generation.append((score, individual))
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = individual
        return generation

    def _generate_next_generation(self, generation):
        # Select the best 20% of the previous generation
        # and produce five copies as offsprings of each
        generation.sort(key=lambda i: i[0])
        selection_amount = int(self.individuals_per_generation * 0.2)
        last_generations_top = generation[-selection_amount:]
        next_generation_offsprings = []
        for candidate in last_generations_top:
            for _ in range(5):
                next_generation_offsprings.append(numpy.copy(candidate[1]))
        next_generation = []
        # Select random 5% for keeping
        selection_amount = int(self.individuals_per_generation * 0.05)
        indices = random.sample(
            range((len(next_generation_offsprings))), selection_amount
        )
        indices.sort(reverse=True)
        for index in indices:
            next_generation.append(next_generation_offsprings[index])
            del next_generation_offsprings[index]
        # Select random 5% as pairs for cross overs
        # (transform to the next even number if necessary)
        selection_amount = int(self.individuals_per_generation * 0.05)
        selection_amount = int(math.ceil(selection_amount / 2.0) * 2)
        indices = random.sample(
            range((len(next_generation_offsprings))), selection_amount
        )
        indices.sort(reverse=True)
        for i in range(int(len(indices) / 2)):
            new_1, new_2 = self._one_point_crossover(
                next_generation_offsprings[indices[i * 2]],
                next_generation_offsprings[indices[i * 2 + 1]],
            )
            del next_generation_offsprings[indices[i * 2]]
            del next_generation_offsprings[indices[i * 2 + 1]]
            next_generation.append(new_1)
            next_generation.append(new_2)
        # Use the remaining 90% for point mutations
        for offspring in next_generation_offsprings:
            next_generation.append(
                self._random_transform_candidate(offspring, 1)
            )
        return next_generation

    def _one_point_crossover(self, individual_1, individual_2):
        new_individual_1 = numpy.copy(individual_1)
        new_individual_2 = numpy.copy(individual_1)
        # Just perform a crossover if the vectors long enough
        # Return the unchanged individuals otherwise
        if (len(individual_1) - 1) > 1:
            crossover_point = random.randint(0, len(individual_1) - 1)
            for i in range(crossover_point, len(individual_1) - 1):
                new_individual_1[i] = individual_2[i]
                new_individual_2[i] = individual_1[i]
        return new_individual_1, new_individual_2
