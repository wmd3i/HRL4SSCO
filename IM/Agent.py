# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from Env import *
from Network import *
from ReplayBuffer import *
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

CAPACITY = 20000
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env

        self.weighted_edge = self.env.weighted
        self.A = torch.Tensor(self.env.A).to(device)  # n * n

        self.feature_size = 3
        self.embedding_size = 64

        # discount
        self.discount = 0.99

        # network
        # self.network = NetworkI(
        #     node_feature_size=self.feature_size, embedding_layers_size=self.embedding_size).to(device)
        # self.network_target = NetworkI(
        #     node_feature_size=self.feature_size, embedding_layers_size=self.embedding_size).to(device)

        self.network = SimpleModelI(
            node_feature_size=self.feature_size).to(device)
        self.network_target = SimpleModelI(
            node_feature_size=self.feature_size).to(device)

        self.network_sub = SimpleModel(
            self.feature_size).to(device)
        self.network_sub_target = SimpleModel(
            self.feature_size).to(device)

        # self.network_sub = SimpleGNN(
        #     self.feature_size, embed_dim=128).to(device)
        # self.network_sub_target = SimpleGNN(
        #     self.feature_size, embed_dim=128).to(device)

        # loss function
        self.loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.optimizer_sub = optim.Adam(self.network_sub.parameters(), lr=1e-4)

        # experience replay
        self.memory = ReplayBuffer(CAPACITY)
        self.memory_sub = ReplayBuffer_sub(CAPACITY)

        # other parameters
        self.learning_step = [0, 0]
        self.target_update = 10

        self.epsilon_start = 0.9
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1
        self.epsilon = self.epsilon_start

        self.temperature = [10, 10]

    def set_graph(self, graph):
        _, budget, T = self.env.get_info()
        self.env = Env(graph=graph, budget=budget, T=T)
        self.A = torch.Tensor(self.env.A).to(device)

    def update_target(self):
        if self.learning_step[0] % self.target_update == 0:
            self.network_target.load_state_dict(self.network.state_dict())
        if self.learning_step[1] % self.target_update == 0:
            self.network_sub_target.load_state_dict(
                self.network_sub.state_dict())

    def state_action(self, state, action):
        output_state = state.copy()
        if len(action) > 0:
            seeds = action
            self.env.set_state_index(output_state, seeds, 1)
        return output_state

    def Q(self, state, option, target=False):
        net = self.network if target == False else self.network_target
        state = torch.Tensor(state).to(device)
        option = torch.Tensor(option).to(device)
        q_value = net(state, option, self.A).to("cpu")
        return q_value

    def Q_sub(self, state, option, action, target=False):
        q_value = self.Q_sub_each(state, option, target=target)
        action = torch.Tensor(action).long()
        action[action == -1] = self.env.n
        action = action.reshape(-1, 1, 1)
        q_value = q_value.gather(1, action).squeeze()
        # q_value is a tensor of shape (batch_size, )
        return q_value

    # Q-value of each node
    def Q_sub_each(self, state, option, mask=None, target=False):
        net = self.network_sub if target == False else self.network_sub_target
        state = torch.Tensor(np.array(state)).to(device)
        option = torch.Tensor(np.array(option)).to(device)
        # q_value = net(state, option, self.A).to("cpu")
        q_value = net(state, self.A, mask).to("cpu")
        return q_value

    def update_q_function(self, budget_policy='agent', seeding_policy='agent'):
        self.train()

        S, O, A, R, done, train_cumulative_reward = self.run_episode(
            budget_policy=budget_policy, seeding_policy=seeding_policy, epsilon=self.epsilon)

        # compute the gain
        G = [0] * len(R)
        G[-1] = R[-1]
        for i in range(len(R) - 2, -1, -1):
            G[i] = R[i] + self.discount * G[i + 1]

        # experience replay
        horizon = len(S) - 1
        for i in range(horizon):
            self.memory.push(S[i], O[i], G[i], S[i+1], done[i])

            # use -1 to represent null action []
            if len(A[i]) == 0:
                self.memory_sub.push(
                    S[i], np.array([0]), np.array(-1), R[i], S[i+1], np.array([1]), done[i])
                continue

            sta, act, rew_list = [], [], []  # for simulation
            k = len(A[i])
            for j in range(k + 1):
                iter_num = 20
                if j == 0:
                    sta.append(S[i])
                    rew_list.append(self.run_simulation(
                        S[i], [], iter_num=iter_num))

                act_cur = int(A[i][j]) if j < k else -1
                act.append(act_cur)

                s_next = self.state_action(S[i], act) if j < k else S[i+1]
                sta.append(s_next)

                if j < k:
                    rew_list.append(self.run_simulation(
                        S[i], act, iter_num=iter_num))

            k_eff = S[i][A[i], 0].sum()
            rew_diff = np.array(
                [max(rew_list[j + 1] - rew_list[j], 0) for j in range(k)])
            rew_diff_scaled = ((rew_diff + 1e-5) / (sum(rew_diff) +
                               1e-5 * k_eff) * k_eff) if k_eff > 0 else np.zeros(k)
            for j in range(k + 1):
                if j < k:
                    rew_cur_scaled = rew_diff_scaled[j]
                    opt = 1 if O[i][2] - j > 0 else 0
                    self.memory_sub.push(sta[j], np.array([1]), np.array(
                        act[j]), rew_cur_scaled, sta[j+1], np.array([opt]), False)
                else:
                    opt = 1 if O[i+1][2] > 0 else 0
                    self.memory_sub.push(
                        sta[j], np.array([0]), np.array(-1), R[i] - k_eff, sta[j+1], np.array([opt]), done[i])

            # loss first layer
            batch = self.memory.sample(batch_size=min(
                BATCH_SIZE, self.memory.__len__()))
            self.optimizer.zero_grad()
            loss = self.cal_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.learning_step[0] += 1

            # loss of second layer
            batch = self.memory_sub.sample(batch_size=min(
                BATCH_SIZE, self.memory_sub.__len__()))
            self.optimizer_sub.zero_grad()
            loss_sub = self.cal_loss_sub(batch)
            loss_sub.backward()
            self.optimizer_sub.step()
            self.learning_step[1] += 1

            # update target_net
            self.update_target()
            self.epsilon = max(
                self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss.item(), loss_sub.item(), train_cumulative_reward

    def train_model(self, num_epochs=10, num_iterations=10, test=True, save_interval=(None, None)):
        comment = "n = {}, T = {}, budget = {}, discount = {}, num_epoch = {}, num_iterations = {}".format(
            self.env.n, self.env.T, self.env.budget, self.discount, num_epochs, num_iterations)
        writer = SummaryWriter(comment=comment)

        for epoch in range(num_epochs):
            loss_list, loss_sub_list, train_cumulative_reward_list, test_cumulative_reward_list = [], [], [], []

            if epoch < np.floor(num_epochs / 2):
                budget_policy, seeding_policy = 'average', 'agent'
            else:
                budget_policy, seeding_policy = 'agent', 'agent'

            # budget_policy, seeding_policy = 'agent', 'score'

            # budget_policy, seeding_policy = 'average', 'agent'

            graph = nx.erdos_renyi_graph(n=self.env.n, p=0.01, directed=True)
            self.set_graph(graph)

            for episode in range(num_iterations):
                loss, loss_sub, train_cumulative_reward = self.update_q_function(
                    budget_policy=budget_policy, seeding_policy=seeding_policy)
                loss_list.append(loss)
                loss_sub_list.append(loss_sub)
                train_cumulative_reward_list.append(train_cumulative_reward)

                # test
                if test:
                    self.eval()
                    _, _, _, _, _, test_cumulative_reward = self.run_episode(
                        budget_policy=budget_policy, seeding_policy=seeding_policy, epsilon=0)
                    test_cumulative_reward_list.append(test_cumulative_reward)

                # save model
                if save_interval[0] is not None and (epoch * num_iterations + episode + 1) % save_interval[0] == 0:
                    path = save_interval[1] if save_interval[1] is not None else 'model'
                    model_name = f'/model_{epoch}_epoch_{episode}_episode_{loss}_{loss_sub}.pth'
                    torch.save(self.state_dict(), path + model_name)

            writer.add_scalar('Train_loss', np.mean(loss_list), epoch)
            writer.add_scalar('Train_loss_sub', np.mean(loss_sub_list), epoch)
            writer.add_scalar('Train_cumulative Reward', np.mean(
                train_cumulative_reward_list), epoch)
            writer.add_scalar('Test cumulative Reward', np.mean(
                test_cumulative_reward_list) if test else 0, epoch)

            print(f'Epoch {epoch}, MSE loss: {np.mean(loss_list)}, MSE loss_sub: {np.mean(loss_sub_list)}, train average reward: {np.mean(train_cumulative_reward_list)}, test average cumulative reward: {np.mean(test_cumulative_reward_list) if test else 0}')

        writer.close()

        return (train_cumulative_reward_list, test_cumulative_reward_list) if test else (train_cumulative_reward_list, [])

    def get_option(self, state, t_r, budget_r, is_null, epsilon=0):
        if budget_r == 0:
            return 0
        if t_r == 1:
            return budget_r

        lower = 1 if is_null else 0
        budget_options = range(lower, budget_r + 1)

        if np.random.rand() < epsilon:
            return np.random.choice(budget_options)

        Q_values = np.array([self.Q(state, [t_r, budget_r, i]).detach().numpy()[
                            0] for i in budget_options])

        optimal_budget = budget_options[np.argmax(Q_values)]
        return optimal_budget

    def static_seeding_policy(self, t_r, budget_r, p=5):
        # p: period
        d = int(np.floor(self.env.T / p))   # number of periods
        k = int(np.floor(self.env.budget / d))
        if budget_r == 0:
            budget = 0
        elif t_r == 1:
            budget = budget_r
        elif t_r % p == self.env.T % p:
            budget = min(budget_r, k)
        else:
            budget = 0
        return budget

    def normal_policy(self, t_r, budget_r):
        if t_r == self.env.T:
            budget = budget_r
        else:
            budget = 0
        return budget

    def average_policy(self, t_r, budget_r):
        mean_budget = max(int(self.env.budget / self.env.T), 1)
        if budget_r == 0:
            return 0
        if t_r == 1:
            return budget_r
        else:
            return mean_budget

    def get_action(self, state, budget, epsilon):
        if budget == 0:
            return []

        if np.random.rand() < epsilon:
            action = np.random.choice(self.env.nodes, budget, replace=False)
            return list(action)  # return a list of seeds

        action = []
        for i in range(budget):
            seed = self.get_one_action(state, 1)
            action.append(seed)
            self.env.set_state_index(state, seed, 1)
        return action

    def get_one_action(self, state, option):
        possible_seeds = self.env.get_nodes_type(state, 'inactive')
        if len(possible_seeds) == 0:
            possible_seeds = self.env.nodes

        if option == 0:
            return []

        state = np.expand_dims(state, axis=0)
        option = np.expand_dims(option, axis=0)
        each_node_reward = self.Q_sub_each(state, option).detach().numpy()
        each_node_reward = each_node_reward[0, :, 0]
        max_indices = each_node_reward[possible_seeds].argsort()[-1]
        seed = np.array(possible_seeds)[max_indices]
        return seed

    def get_action_score(self, state, budget, mode='score'):
        action = []
        # set of inactive nodes
        inactive_nodes = self.env.get_nodes_type(state, 'inactive')
        if len(inactive_nodes) == 0:
            return action
        elif len(inactive_nodes) <= budget:
            return inactive_nodes
        else:
            inactive_nodes = self.env.sort_by_value(inactive_nodes, mode)
            action = inactive_nodes[:budget]
        return action

    def run_episode(self, budget_policy='agent', seeding_policy='agent', beam_search=False, epsilon=0.1):
        # budget policies
        budget_policies = {
            'agent': lambda state, t_r: self.get_option(state, t_r, self.env.budget_r, True, epsilon=epsilon),
            'static': lambda _, t_r: self.static_seeding_policy(t_r, self.env.budget_r),
            'normal': lambda _, t_r: self.normal_policy(t_r, budget_r=self.env.budget_r),
            'average': lambda _, t_r: self.average_policy(t_r, budget_r=self.env.budget_r),
            'greedy': lambda _, t_r: self.greedy_policy(t_r, self.env.budget_r),
            'sof': lambda state, t_r: self.sof_policy(state, self.env.budget_r, t_r)
        }

        # seeding policies
        seeding_actions = {
            'agent': lambda state, budget_cur: self.beam_search(state, budget_cur) if beam_search else self.get_action(state=state, budget=budget_cur, epsilon=epsilon),
            'score': lambda state, budget_cur: self.get_action_score(state=state, budget=budget_cur, mode='score'),
            'degree': lambda state, budget_cur: self.get_action_score(state=state, budget=budget_cur, mode='degree')
        }

        S, O, A, R, done = [], [], [], [], []
        cumulative_reward = 0
        self.env.reset()
        is_null = True

        for t in range(self.env.T):

            state = self.env.state.copy()
            S.append(state)
            t_r = self.env.T - t
            budget_cur = budget_policies[budget_policy](state, t_r)

            option = [t_r, self.env.budget_r, budget_cur]
            O.append(option)
            self.env.budget_r -= budget_cur

            action = seeding_actions[seeding_policy](state, budget_cur)
            A.append(action)
            next_state, reward, is_done, is_null = self.env.step(
                self.env.state, action)
            self.env.state = next_state.copy()

            R.append(reward)
            done.append(is_done)

            if t == self.env.T - 1:
                S.append(next_state)
                O.append([0, 0, 0])

            cumulative_reward += reward * (self.discount ** t)

        seeding_policy = (np.array(O))[:, 2]

        return S, O, A, R, done, cumulative_reward

    def cal_loss(self, batch):
        loss_list = []
        for memory in batch:
            state, option, gain = memory.state, memory.option, memory.gain
            gain = torch.tensor(gain, dtype=torch.float32)

            # calculate prediction and target
            prediction = self.Q(state, option)
            loss = (prediction - gain) ** 2
            loss_list.append(loss)

        total_loss = torch.stack(loss_list).mean()
        return total_loss

    def cal_loss_sub(self, batch):
        batch_data = Transition_sub(*zip(*batch))
        state_batch = batch_data.state
        option_batch = np.array(batch_data.option)
        action_batch = np.array(batch_data.action)
        reward_batch = torch.Tensor(batch_data.reward)
        next_state_batch = batch_data.next_state
        next_option_batch = np.array(batch_data.next_option)
        done_batch = torch.BoolTensor(batch_data.done)

        prediction = self.Q_sub(state_batch, option_batch, action_batch)
        q_values = self.Q_sub_each(
            next_state_batch, next_option_batch).detach()
        q_values_ = q_values[:, :-1, :].squeeze(-1)
        next_action_batch = torch.full(
            (q_values_.size(0), 1), self.env.n, dtype=torch.long)
        max_action_indices = q_values_.max(1)[1].view(-1, 1)

        option_batch = option_batch.reshape(-1)
        index_of_selected = option_batch == 1
        next_action_batch[index_of_selected] = max_action_indices[index_of_selected]
        next_action_batch = next_action_batch.reshape(-1, 1, 1)

        next_q_value = q_values.gather(1, next_action_batch).squeeze()

        next_q_value[done_batch] = 0

        discount_batch = torch.ones_like(reward_batch)
        discount_batch[option_batch == 0] = self.discount
        target = reward_batch + discount_batch * next_q_value
        loss_mean = self.loss_fn(prediction, target)

        return loss_mean

    # def cal_loss_sub(self, batch):
    #     loss_v = []
    #     for memory in batch:
    #         state, option, action, reward, next_state, next_option, done = (
    #             memory.state, memory.option, memory.action, memory.reward, memory.next_state, memory.next_option, memory.done
    #         )

    #         prediction = self.Q_sub(state, option, action)

    #         if done:
    #             target = torch.tensor(reward, dtype=torch.float32)
    #         else:
    #             next_action = self.get_one_action(next_state, next_option)
    #             next_action = np.array(next_action)
    #             target = reward + self.discount * \
    #                 self.Q_sub(next_state, next_option,
    #                            next_action, target=True).detach()

    #         loss = (prediction - target) ** 2
    #         loss_v.append(loss.unsqueeze(0))

    #     loss_v = torch.cat(loss_v, dim=0)
    #     loss_mean = loss_v.mean()
    #     return loss_mean

    # def if_add(self, state, action, v, t_r, theta=0.2):
    #     L = 20
    #     print("if_add: ", action, v, t_r)
    #     M_a = (self.g_U(state, list(set(action) | set([v])), t_r, L) - self.g_U(state, action, t_r, L)) / (
    #         self.g_U(state, [v], t_r, L) - self.g_U(state, [], t_r, L) + 0.0001)
    #     M_t = (self.h_function(state, action, v, t_r, t_r, L) - self.h_function(state, action,
    #            v, t_r, t_r - 1, L)) / (self.h_function(state, action, v, t_r, t_r, L) + 0.0001)
    #     alpha_t = 1 - 1/t_r
    #     I = alpha_t * M_a + (1 - alpha_t) * M_t
    #     if I >= theta:
    #         return True
    #     return False

    # def g_U(self, state, action, t_r, L=500):
    #     state = state
    #     count = 0
    #     for i in range(L):
    #         R = self.one_sample(state, t_r)
    #         if (set(R) & set(action)):
    #             count = count + 1
    #     return self.env.n * count / L

    # def h_function(self, state, action, v, t, t1, L=200):
    #     g = 0
    #     for i in range(L):
    #         next_state, _, _, _ = self.env.step(state, action)
    #         for j in range(t - 1):
    #             next_state, _, _, _ = self.env.step(next_state, [])
    #         g = g + self.g_U(next_state, [v], t1, L) - \
    #             len(self.env.get_active(next_state))
    #     return g/L

    # def one_sample(self, state, t_r):
    #     t = t_r
    #     active_nodes = self.env.get_active(state)
    #     # randomly choose a node
    #     v = random.randint(0, self.env.n - 1)
    #     R = []
    #     queue = []
    #     queue.append(v)
    #     visited = []
    #     while queue and (t > 0):
    #         size = len(queue)
    #         for j in range(size):
    #             s = queue.pop(0)
    #             R.append(s)
    #             if s in active_nodes:
    #                 return self.env.nodes
    #             for i in self.env.graph.neighbors(s):
    #                 prob = self.env.get_edge_prob(i)
    #                 if i not in visited and i not in queue and random.uniform(0, 1) < prob:
    #                     queue.append(i)
    #             visited.append(s)
    #         t = t - 1
    #     return R

    def beam_search(self, state, budget, beta=10, gamma=10):
        state = state.copy()
        action = [([], 0)]
        state_cache = {}  # Cache the state-action pairs to avoid repeated calculations

        if budget == 0:
            return []

        for i in range(budget):
            new_action = []

            for par_action, _ in action:
                action_key = tuple(par_action)
                if action_key not in state_cache:
                    state_cache[action_key] = self.state_action(
                        state, par_action)
                state_ = state_cache[action_key]
                option = np.array(1)

                mask = np.ones(self.env.n + 1)
                possible_nodes = self.env.get_nodes_type(state_, 'inactive')
                mask[possible_nodes] = 0
                state_batch = state_.reshape(1, -1, self.feature_size)
                node_rewards = self.Q_sub_each(
                    state_batch, option, mask).detach().numpy().flatten()
                node_rewards = node_rewards[:-1]
                expansion_seed = np.argpartition(-node_rewards, gamma)[:gamma]

                for seed in expansion_seed:
                    # Modify the list to contain tuples of (action, reward)
                    new_par_action = par_action.copy()
                    new_par_action.append(seed)
                    new_action.append((new_par_action, 0))

            action = new_action
            for idx, (new_par_action, _) in enumerate(action):
                action_key = tuple(new_par_action)
                if action_key not in state_cache:
                    state_cache[action_key] = self.state_action(
                        state, new_par_action)
                state_ = state_cache[action_key]

                inactive_nodes = self.env.get_nodes_type(state_, 'inactive')
                inactive_nodes = self.env.sort_by_value(
                    inactive_nodes, 'score')
                length = len(new_par_action)
                complete_action = inactive_nodes[:(
                    budget-length)] + new_par_action
                # rew = self.run_simulation(state, complete_action, iter_num=10)
                rew = self.reward_by_score(state, complete_action)
                action[idx] = (new_par_action, rew)

            # select top beta action
            action = sorted(action, key=lambda x: x[1], reverse=True)[:beta]

        # select the best action
        action = max(action, key=lambda x: x[1])[0]
        return action

    def run_simulation(self, state, action, iter_num=10):
        rew = 0
        for _ in range(iter_num):
            _, rew_, _, _ = self.env.step(state, action)
            rew += rew_
        rew = rew / iter_num
        return rew

    def reward_by_score(self, state, action):
        # Get the inactive neighbors for the given state and action
        inactive_nodes = self.env.get_inactive_neighbors(state, action)

        # Use a dictionary to count occurrences of each node
        node_counts = Counter(inactive_nodes)

        # Calculate the reward, taking into account multiple occurrences of the same node
        reward = sum((1 - (1 - self.env.get_edge_prob(node)) ** count)
                     for node, count in node_counts.items())
        return reward
