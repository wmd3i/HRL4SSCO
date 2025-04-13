import numpy as np
import random


class GA:
    def __init__(self, env, population_size=100, num_generations=100, mutation_rate=0.01):
        self.env = env
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def encode(self, nodes):
        encoded = [0] * self.env.n
        for node in nodes:
            encoded[node] = 1
        return encoded

    def decode(self, encoded):
        return [i for i in range(len(encoded)) if encoded[i] == 1]

    def compute_num_nodes(self, individual):
        return sum([sum(day) for day in individual])

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            total_chosen_nodes = set()
            remaining_budget = self.env.budget
            for _ in range(self.env.T):
                available_nodes = list(
                    set(self.env.get_not_visited()) - total_chosen_nodes)
                num_to_choose = random.randint(
                    0, min(len(available_nodes), remaining_budget))
                chosen_nodes = random.sample(available_nodes, num_to_choose)
                total_chosen_nodes.update(chosen_nodes)
                remaining_budget -= len(chosen_nodes)
                individual.append(self.encode(chosen_nodes))
            population.append(individual)
        return population

    def eval_fitness(self, individual):
        self.env.reset()
        if self.compute_num_nodes(individual) > self.env.budget:
            return -float('inf')
        total_reward = 0
        for day_plan in individual:
            decoded_plan = self.decode(day_plan)
            _, reward, done = self.env.step(decoded_plan)
            total_reward += reward
            if done:
                break
        return total_reward

    def select_parents(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, score in zip(population, fitness_scores):
            current += score
            if current > pick:
                return individual
        return population[-1]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        def enforce_node_limit(child):
            total_nodes = self.compute_num_nodes(child)
            if total_nodes > self.env.budget:
                for day in child:
                    for i in range(len(day)):
                        if total_nodes <= self.env.budget:
                            break
                        if day[i] == 1:
                            day[i] = 0
                            total_nodes -= 1
            return child

        child1 = enforce_node_limit(child1)
        child2 = enforce_node_limit(child2)
        return child1, child2

    def mutate(self, individual):
        total_nodes = self.compute_num_nodes(individual)
        for day in range(len(individual)):
            if random.random() < self.mutation_rate:
                if total_nodes < self.env.budget:
                    all_nodes = set(range(self.env.n))
                    visited_nodes = set(
                        np.where(np.array(individual).sum(axis=0))[0])
                    available_nodes = list(all_nodes - visited_nodes)
                    if available_nodes:
                        node_to_add = random.choice(available_nodes)
                        individual[day][node_to_add] = 1
                        total_nodes += 1
                elif total_nodes > 0:
                    chosen_nodes = [i for i, x in enumerate(
                        individual[day]) if x == 1]
                    if chosen_nodes:
                        node_to_remove = random.choice(chosen_nodes)
                        individual[day][node_to_remove] = 0
                        total_nodes -= 1
        return individual

    def run(self):
        population = self.init_population()
        best_solution = None
        best_fitness = -float('inf')
        for generation in range(self.num_generations):
            fitness_scores = [self.eval_fitness(
                individual) for individual in population]
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.select_parents(population, fitness_scores)
                parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend(
                    [self.mutate(child1), self.mutate(child2)])
            population = new_population
            current_best_fitness = max(fitness_scores)
            current_best_solution = population[fitness_scores.index(
                current_best_fitness)]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution
        best_policy = [len(self.decode(day)) for day in best_solution]
        return best_policy, best_fitness


class Greedy:
    def __init__(self, env):
        self.env = env

    def run(self):
        self.env.reset()
        total_reward = 0
        seeding_policy = []
        remaining_budget = self.env.budget
        max_dist = np.max(self.env.dist)

        for t in range(self.env.T):
            day_plan = []
            current_node = self.env.current_node
            available_nodes = [node for node in range(
                self.env.n) if self.env.profit[node] > 0]
            total_distance = 0

            while remaining_budget > 0 and available_nodes:
                best_node = None
                best_profit = -float('inf')

                for node in available_nodes:
                    distance = self.env.dist[current_node, node]
                    if (total_distance + distance) / max_dist <= self.env.max_daily_distance:
                        profit = self.env.profit[node]
                        if profit > best_profit:
                            best_profit = profit
                            best_node = node

                if best_node is None:
                    break

                day_plan.append(best_node)
                total_distance += self.env.dist[current_node, best_node]
                remaining_budget -= 1
                available_nodes.remove(best_node)
                current_node = best_node

            seeding_policy.append(len(day_plan))
            _, reward, _ = self.env.step(day_plan)
            total_reward += reward

        return seeding_policy, total_reward
