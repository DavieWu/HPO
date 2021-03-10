import numpy

from math import log, ceil


# Adapted from https://github.com/zygmuntz/hyperband


class HyperbandRunner:
    def __init__(
        self,
        get_params_function,
        try_params_function,
        change_params_function,
        early_stop_function,
    ):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.change_params = change_params_function
        self.early_stop = early_stop_function

        self.max_iter = 81  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

    # can be called multiple times
    def run(self, optimization_run_stop_event, mcts_stop_event, skip_last=0):
        best_candidate = None
        best_score = float("-inf")

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                scores = []
                early_stops = []

                # Change configuration at one position for the actual searching
                for i, t in enumerate(T):
                    T[i] = self.change_params(t, 1)

                for t in T:

                    score = self.try_params(t)

                    scores.append(score)
                    early_stops.append(
                        self.early_stop(score, best_score, n_iterations)
                    )
                    # keeping track of the best result so far
                    if score >= best_score:
                        best_score = score
                        best_candidate = t

                    if mcts_stop_event.is_set() or (
                        optimization_run_stop_event.is_set()
                    ):
                        return best_candidate, best_score

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = numpy.argsort(scores)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[-int(n_configs / self.eta) :]

        return best_candidate, best_score
