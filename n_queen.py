import random


class InitialPopulation:

    def __init__(self, population_size, board_dimension):
        """
        :param population_size: an int representing size of population
        :param board_dimension: an int representing chess board dimension

        generate random population based on the given values and store it in self.population as a list of individuals
        """
        self.k = population_size
        self.n = board_dimension
        self.population = list()

        def generate_population():
            for i in range(self.k):
                individual = list()
                for j in range(self.n):
                    individual.append(random.randint(1, self.n))
                self.population.append(individual)

        generate_population()


def genetic_algorithm(population, advance):
    """
    :param population: a list of individuals
    :param advance: True/False  True: search until it finds the solution
                                False: search until it finds the solution or 1000,000 iteration has been accomplished
    :return: the solution, if it has been found ow. returns False
    """

    def fitness_function(individual):
        """
        :param individual: each state
        :return: number of non attacking pairs of queens
        """
        fitness = 0
        n = len(individual)
        for index in range(n - 1):
            for j in range(index + 1, n):
                if abs(index - j) != abs(individual[index] - individual[j]):  # no diagonal collisions
                    if index != j and individual[index] != individual[j]:  # no horizontal collisions
                        fitness += 1
        return fitness

    def crossover(individual_1, individual_2):
        """
        :param individual_1: one parent
        :param individual_2: another parent
        :return: child of given parents
        """
        n = len(individual_1)
        crossover_point = random.randint(0, n - 1)

        new_individual = list()
        new_individual[0:crossover_point] = individual_1[0:crossover_point]
        new_individual[crossover_point:n] = individual_2[crossover_point:n]

        return new_individual

    def mutate(individual):
        n = len(individual)
        point = random.randint(0, n - 1)
        value = random.randint(1, n)
        mutated_child = individual
        mutated_child[point] = value

        return mutated_child

    def random_selection(population_arr, probabilities_arr):
        """
        :param population_arr: list of individuals of given population
        :param probabilities_arr: list of probabilities for individuals in the same order of given population list
        :return: a individual which was randomly selected based on the given probabilities
        """
        # map the probabilities on a continues line
        prob_sum = sum(probabilities_arr)
        # set a random threshold to select first individual which has lower or equal local_sum
        threshold = random.uniform(0, prob_sum)
        local_sum = 0
        for index in range(len(probabilities_arr)):
            if probabilities_arr[index] + local_sum >= threshold:
                return population_arr[index]
            local_sum += probabilities_arr[index]

    def probability(individual, fitness):
        return float(fitness_function(individual) / sum(fitness))

    iteration = 0
    # compute max fitness
    max_fitness = len(population[0]) * (len(population[0]) - 1) / 2
    population = population
    while True:
        if not advance and iteration > 1000000:
            print('no solution has be found in 1000000 iterations')
            return False

        mutation_prob = 0.03
        fitness_arr = [fitness_function(individual) for individual in population]
        # check if it reached the solution in given population or not
        if max_fitness in fitness_arr:
            goal_population = list()
            for i in range(len(fitness_arr)):
                if fitness_arr[i] == max_fitness:
                    goal_population = population[i]
                    break
            print('goal population:     {}'.format(goal_population))
            return goal_population

        probabilities = [probability(individual, fitness_arr) for individual in population]

        new_population = list()
        # generate new individuals based on the last population
        for i in range(len(population)):
            x = random_selection(population, probabilities)
            y = random_selection(population, probabilities)
            child = crossover(x, y)
            if random.random() <= mutation_prob:
                child = mutate(child)
            new_population.append(child)

        population = new_population
        iteration += 1


if __name__ == '__main__':
    in_pop = InitialPopulation(population_size=20, board_dimension=8)
    genetic_algorithm(in_pop.population, False)
