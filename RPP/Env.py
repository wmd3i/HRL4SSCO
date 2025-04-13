import numpy as np


class Env:
    def __init__(self, loc, initial_profit=None, budget=20, T=10, max_daily_distance=1.5, distance_penalty=5, start_node=0):
        # parameters
        self.n = loc.shape[0]
        self.loc = np.array(loc)
        self.T = T
        self.budget = budget
        self.max_daily_distance = max_daily_distance
        self.distance_penalty = distance_penalty
        self.start_node = start_node

        # Node profits
        if initial_profit is None:
            self.initial_profit = np.random.uniform(0.5, 1.0, self.n)
        else:
            self.initial_profit = np.array(initial_profit)
        self.profit_factor = 0.05  # random profit factor
        self.profit_min = 0.3
        self.eta = 0.1  # coefficient for submodular function

        # Distance matrix
        self.dist = np.sqrt(
            ((loc[:, np.newaxis, :] - loc[np.newaxis, :, :]) ** 2).sum(axis=2))

        self.reset()

    def reset(self):
        self.t = 1
        self.profit = self.initial_profit.copy()
        self.profit[self.start_node] = 0
        self.remaining_budget = self.budget
        self.current_node = self.start_node

    def get_state(self):
        loc = self.loc / np.max(self.loc)
        profit = self.profit.reshape(-1, 1)
        state = np.concatenate([loc, profit], axis=1)
        return state

    def get_not_visited(self):
        return np.where(self.profit > 0)[0]

    def get_context(self):
        context = np.array([self.current_node, self.start_node])
        return context

    def cal_total_distance(self, action, context=None):
        if context is None:
            context = self.get_context()
        if len(action) == 0 and self.t < self.T:
            return 0
        elif len(action) == 0 and self.t == self.T:
            return self.dist[context[0], context[1]]

        total_distance = self.dist[context[0], action[0]]
        total_distance += sum(self.dist[action[i], action[i + 1]]
                              for i in range(len(action) - 1))
        if self.t == self.T:
            total_distance += self.dist[action[-1], context[1]]
        return total_distance / np.max(self.dist)

    def step(self, action):
        assert len(action) <= self.remaining_budget, "Budget exceeded"

        if len(action) == 0:
            return self.get_state(), 0, self.t == self.T

        total_distance = self.cal_total_distance(action)

        # compute reward
        reward = self.cal_reward(action)
        self.remaining_budget -= len(action)
        self.profit[action] = 0
        not_visited_nodes = [node for node in range(
            self.n) if self.profit[node] > 0]
        self.profit[not_visited_nodes] = np.maximum(
            self.profit[not_visited_nodes] + self.profit_factor * np.random.uniform(-1, 1, len(not_visited_nodes)), self.profit_min)

        if total_distance > self.max_daily_distance:
            reward -= self.distance_penalty * \
                (total_distance - self.max_daily_distance)

        # print(
        #     f"total_distance/max_dist: {total_distance / np.max(self.dist + 1e-5)}")

        if len(action) > 0:
            self.current_node = action[-1]
        self.t += 1
        done = self.t > self.T
        return self.get_state(), reward, done

    def cal_reward(self, nodes):
        # submodular function
        if len(nodes) == 0:
            return 0
        reward = sum(self.profit[node] for node in nodes) - \
            self.eta * (max(len(nodes) - 1, 0)) ** 2
        return reward

    def reward_by_simulation(self, state, context, action):

        total_distance = self.cal_total_distance(action, context=context)

        # compute reward
        reward = sum(state[:, 2][action]) - \
            self.eta * (max(len(action) - 1, 0)) ** 2

        if total_distance > self.max_daily_distance:
            reward -= self.distance_penalty * \
                (total_distance - self.max_daily_distance)

        # print(
        #     f"total_distance/max_dist: {total_distance / np.max(self.dist + 1e-5)}")

        if len(action) > 0:
            self.current_node = action[-1]
        self.t += 1
        done = self.t > self.T
        return reward
