import os
import sys
from functools import reduce

import numpy as np
from leap_ec.problem import ScalarProblem
from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip


def binary_array_to_decimal(binary_array):
    return reduce(lambda a,b: 2*a+b, binary_array)

class SquareRoot(ScalarProblem):

    def __init__(self, target_number: int, maximize=False):
        super().__init__(maximize)
        self.target_number = target_number

    def evaluate(self, phenome):
        return np.square(np.square(binary_array_to_decimal(phenome)) - self.target_number)


##############################
# main
##############################
if __name__ == '__main__':
    l = 10
    pop_size = 10
    generations = 100

    problem = SquareRoot(625)


    ##############################
    # Visualizations
    ##############################
    # Setting up some visualization probes in advance
    # Doing it here allow us to use subplots to arrange them nicely
    plt.figure(figsize=(18, 5))

    plt.subplot(131)
    p1 = probe.SumPhenotypePlotProbe(
        xlim=(0, l),
        ylim=(0, l),
        problem=problem,
        ax=plt.gca())

    plt.subplot(132)
    p2 = probe.FitnessPlotProbe(ax=plt.gca(), xlim=(0, generations))

    plt.subplot(133)
    p3 = probe.PopulationMetricsPlotProbe(
        metrics=[ probe.pairwise_squared_distance_metric ],
        xlim=(0, generations),
        title='Population Diversity',
        ax=plt.gca())

    plt.tight_layout()
    viz_probes = [ p1, p2, p3 ]


    ##############################
    # Run!
    ##############################
    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                                problem=problem,  # Fitness function

                                # Representation
                                representation=Representation(
                                    # Initialize a population of integer-vector genomes
                                    initialize=create_binary_sequence(length=l)
                                ),

                                # Operator pipeline
                                pipeline=[
                                    ops.tournament_selection(k=2),
                                    ops.clone,
                                    # Apply binomial mutation: this is a lot like
                                    # additive Gaussian mutation, but adds an integer
                                    # value to each gene
                                    mutate_bitflip(expected_num_mutations=1),
                                    ops.UniformCrossover(p_swap=0.4),
                                    ops.evaluate,
                                    ops.pool(size=pop_size),
                                    # Collect fitness statistics to stdout
                                    probe.FitnessStatsCSVProbe(stream=sys.stdout),
                                    *viz_probes  # Inserting the additional probes we defined above
                                ]
                                )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()

    plt.close('all')
    # print best genome
    best = min(final_pop, key=lambda ind: ind.fitness)
    print(f"Best genome: {np.array(best.genome).astype(int)}, fitness: {best.fitness}")
    # print decimal value of best genome
    print(f"Decimal value: {binary_array_to_decimal(np.array(best.genome).astype(int))}")

