# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from Env import *
from Network import *
from ReplayBuffer import *
from torch.utils.tensorboard import SummaryWriter

CAPACITY = 20000
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env

        # parameters
        self.feature_size = 3
        self.discount = 0.99

        # network
        self.OptNet = OptionNetwork(node_dim=self.feature_size).to(device)
        self.OptNet_target = OptionNetwork(
            node_dim=self.feature_size).to(device)

        self.ActNet = ActionNetwork(node_dim=self.feature_size).to(device)
        self.ActNet_target = ActionNetwork(
            node_dim=self.feature_size).to(device)

        # loss function
        self.loss_fn = nn.MSELoss()

        # optimizer
        self.optimizer = optim.Adam(self.OptNet.parameters(), lr=1e-2)
        self.optimizer2_sub = optim.Adam(self.ActNet.parameters(), lr=1e-4)

        # experience buffer
        self.buffer = ReplayBuffer(CAPACITY)
        self.buffer_sub = ReplayBuffer_sub(CAPACITY)

        # eploration rate
        self.epsilon_start = 0.9
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1
        self.epsilon = [self.epsilon_start, self.epsilon_start]

        # target update rate
        self.learning_step = [0, 0]
        self.target_update = 10

    def set_loc(self, loc):
        self.env.loc = loc
        self.env.dist = np.sqrt(
            ((loc[:, np.newaxis, :] - loc[np.newaxis, :, :]) ** 2).sum(axis=2))
        self.env.n = loc.shape[0]
        self.env.initial_profit = np.random.uniform(0.5, 1.0, loc.shape[0])
        self.env.reset()

    def update_target(self):
        if self.learning_step[0] % self.target_update == 0:
            self.OptNet_target.load_state_dict(self.OptNet.state_dict())
        if self.learning_step[1] % self.target_update == 0:
            self.ActNet_target.load_state_dict(self.ActNet.state_dict())

    def Q(self, state, context, option, target=False):
        # state: (batch_size, n, feature_size), option: (batch_size, 3), context: (batch_size, 2)
        net = self.OptNet_target if target else self.OptNet
        state = torch.tensor(state, dtype=torch.float32).to(device)
        context = torch.tensor(context, dtype=torch.float32).to(device)
        option = torch.tensor(option, dtype=torch.float32).to(device)
        return net(state, option, context).to('cpu')

    def Q_sub(self, state, context, mask=None, target=False):
        net = self.ActNet_target if target else self.ActNet
        state = torch.tensor(state, dtype=torch.float32).to(device)
        context = torch.tensor(context, dtype=torch.float32).to(device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool).to(device)
        q_value = net(state, context, mask).to('cpu')
        return q_value

    def get_option(self, state, context, t_r, budget_r, epsilon=0):
        if budget_r == 0:
            return 0
        if t_r == 1:
            return budget_r

        # budget in [0, budget_r]
        budget_options = range(budget_r + 1)

        # epsilon-greedy
        if np.random.rand() < epsilon:
            return np.random.choice(budget_options)

        Q_values = [self.Q(np.expand_dims(state, axis=0), np.expand_dims(context, axis=0), np.expand_dims([t_r, budget_r, i], axis=0)).detach().numpy()[
            0] for i in budget_options]

        return budget_options[np.argmax(Q_values)]

    def get_action(self, state, context, budget, epsilon=0):
        if budget == 0:
            return []

        if np.random.rand() < epsilon:
            return list(np.random.choice(self.env.n, budget, replace=False))

        cur_state = state.copy()
        action = []
        context_expanded = np.expand_dims(context, axis=0)

        for _ in range(budget):
            possible_nodes = [i for i in range(
                self.env.n) if cur_state[i, 2] > 0]
            cur_state_expanded = np.expand_dims(cur_state, axis=0)
            each_node_reward = self.Q_sub(
                cur_state_expanded, context_expanded).detach().numpy()[0, :, 0]
            max_index = np.argmax(each_node_reward[possible_nodes])
            node = possible_nodes[max_index]

            action.append(node)
            cur_state[node, 2] = 0
            context_expanded = np.expand_dims(
                np.array([node, context[1]]), axis=0)
        return action

    def run_episode(self, budegt_policy='agent', beam_search=False, epsilon=[0, 0]):

        S, C, O, A, R, done = [], [], [], [], [], []
        cumulative_reward = 0
        self.env.reset()

        for t in range(self.env.T):

            state, context = self.env.get_state(), self.env.get_context()
            S.append(state)
            C.append(context)
            # get option
            t_r = self.env.T - t
            if budegt_policy == 'average':
                budget_cur = self.average_policy(
                    t_r, self.env.remaining_budget)
            else:
                budget_cur = self.get_option(
                    state, context, t_r, self.env.remaining_budget, epsilon[0])
            option = [t_r, self.env.remaining_budget, budget_cur]
            O.append(option)
            # get action
            if beam_search:
                action = self.beam_search(
                    state, context, budget_cur, beta=10, gamma=10)
            else:
                action = self.get_action(
                    state, context, budget_cur, epsilon[1])
            A.append(action)

            # step
            next_state, reward, is_done = self.env.step(action)

            R.append(reward)
            done.append(False) if t < self.env.T - 1 else done.append(True)
            cumulative_reward += reward * (self.discount ** t)

            if t == self.env.T - 1:
                S.append(next_state)
                C.append(self.env.get_context())
                O.append([0, 0, 0])
                break

        return S, C, O, A, R, done, cumulative_reward

    def update_q_function(self, budget_policy='agent'):
        self.train()

        S, C, O, A, R, done, train_cumulative_reward = self.run_episode(
            budegt_policy=budget_policy, epsilon=self.epsilon)

        # compute the gain
        G = [0] * len(R)
        G[-1] = R[-1]
        for i in range(len(R) - 2, -1, -1):
            G[i] = R[i] + self.discount * G[i + 1]

        # experience replay
        horizon = len(S) - 1
        for i in range(horizon):
            self.buffer.push(S[i], C[i], O[i], G[i])

            # use -1 to represent null action []
            if len(A[i]) == 0:
                self.buffer_sub.push(
                    S[i], C[i], np.array([0]), np.array([-1]), R[i], S[i+1], C[i+1], np.array([1]), done[i])
                continue

            # sublayer
            k = len(A[i])
            sta, cont, opt, act, rew = [], [], [], [], []
            for j in range(k + 1):
                sta_t = self.state_action(S[i], act)
                sta.append(sta_t)
                cont_t = np.array([act[-1][0], C[i][1]]
                                  ) if len(act) > 0 else np.array(C[i])
                cont.append(cont_t)
                opt_t = np.array([1]) if j < k else np.array([0])
                opt.append(opt_t)
                act_t = np.array([A[i][j]]) if j < k else np.array([-1])
                act.append(act_t)
                rew_t = S[i][act_t, 2][0] if j < k else 0  # profit
                rew.append(rew_t)

                if j == k:
                    sta.append(S[i+1])
                    cont.append(C[i+1])
                    opt_ = np.array([1]) if O[i+1][2] > 0 else np.array([0])
                    opt.append(opt_)

            rew = np.array(rew)
            print(f"rew:{rew}")
            rew = (rew + 1e-5) / (sum(rew) + k * 1e-5) * (R[i] - 0)

            for j in range(k + 1):
                is_done = done[i] if j == k else False
                self.buffer_sub.push(
                    sta[j], cont[j], opt[j], act[j], rew[j], sta[j+1], cont[j+1], opt[j+1], is_done)

        # update q function
        batch = self.buffer.sample(min(BATCH_SIZE, len(self.buffer)))
        self.optimizer.zero_grad()
        loss = self.cal_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.learning_step[0] += 1

        batch_sub = self.buffer_sub.sample(
            min(BATCH_SIZE, len(self.buffer_sub)))
        self.optimizer2_sub.zero_grad()
        loss_sub = self.cal_loss_sub(batch_sub)
        loss_sub.backward()
        self.optimizer2_sub.step()
        self.learning_step[1] += 1

        self.update_target()
        self.epsilon[1] = max(
            self.epsilon[1] * self.epsilon_decay, self.epsilon_min)
        if budget_policy == 'agent':
            self.epsilon[0] = max(
                self.epsilon[0] * self.epsilon_decay, self.epsilon_min)
        return loss.item(), loss_sub.item(), train_cumulative_reward

    def train_model(self, num_epochs=10, num_iterations=10, test=True):
        comment = "n = {}, T = {}, budget = {}, discount = {}, num_epoch = {}, num_iterations = {}".format(
            self.env.n, self.env.T, self.env.budget, self.discount, num_epochs, num_iterations)
        writer = SummaryWriter(comment=comment)

        for epoch in range(num_epochs):
            loss_list, loss_sub_list, train_cumulative_reward_list, test_cumulative_reward_list = [], [], [], []

            if epoch < np.floor(num_epochs / 2):
                budget_policy = 'average'
            else:
                budget_policy = 'agent'

            loc = np.random.rand(self.env.n, 2) * 100
            self.set_loc(loc)

            for episode in range(num_iterations):
                print(f"epoch:{epoch}, episode:{episode} start")

                loss, loss_sub, train_cumulative_reward = self.update_q_function(
                    budget_policy=budget_policy)
                loss_list.append(loss)
                loss_sub_list.append(loss_sub)
                train_cumulative_reward_list.append(train_cumulative_reward)

                if test:
                    self.eval()
                    _, _, _, _, _, _, test_cumulative_reward = self.run_episode()
                    test_cumulative_reward_list.append(test_cumulative_reward)

            writer.add_scalar('Train_loss', np.mean(loss_list), epoch)
            writer.add_scalar('Train_loss_sub', np.mean(loss_sub_list), epoch)
            writer.add_scalar('Train_cumulative Reward', np.mean(
                train_cumulative_reward_list), epoch)
            writer.add_scalar('Test cumulative Reward', np.mean(
                test_cumulative_reward_list) if test else 0, epoch)

            print(f'Epoch {epoch}, MSE loss: {np.mean(loss_list)}, MSE loss_sub: {np.mean(loss_sub_list)}, train average reward: {np.mean(train_cumulative_reward_list)}, test average cumulative reward: {np.mean(test_cumulative_reward_list) if test else 0}')

        writer.close()

        return (train_cumulative_reward_list, test_cumulative_reward_list) if test else (train_cumulative_reward_list, [])

    def cal_loss(self, batch):
        batch_data = Transition(*zip(*batch))
        state = torch.tensor(batch_data.state, dtype=torch.float32)
        context = torch.tensor(batch_data.context, dtype=torch.float32)
        option = torch.tensor(batch_data.option, dtype=torch.float32)
        gain = torch.tensor(batch_data.gain, dtype=torch.float32)
        prediction = self.Q(state, context, option).squeeze(-1)
        loss = self.loss_fn(prediction, gain)
        return loss

    def cal_loss_sub(self, batch):

        batch_data = Transition_sub(*zip(*batch))
        state = batch_data.state
        context = torch.tensor(batch_data.context, dtype=torch.float32)
        option = torch.tensor(batch_data.option, dtype=torch.float32)
        action = torch.tensor(batch_data.action, dtype=torch.int64)
        reward = torch.tensor(batch_data.reward, dtype=torch.float32)
        next_state = batch_data.next_state
        next_context = torch.tensor(
            batch_data.next_context, dtype=torch.float32)
        next_option = torch.tensor(batch_data.next_option, dtype=torch.float32)
        done = torch.BoolTensor(batch_data.done)

        q_values = self.Q_sub(state, context).squeeze(-1)
        action[action == -1] = self.env.n
        q_value = q_values.gather(1, action)
        q_values_next = self.Q_sub(
            next_state, next_context, target=True).detach().squeeze(-1)
        next_action = q_values_next.argmax(
            dim=1, keepdim=True)  # size: [batch_size, 1]
        next_action[next_option == 0] = self.env.n
        next_q_value = q_values_next.gather(1, next_action).squeeze()
        next_q_value[done] = 0
        discount = torch.ones_like(reward)
        option = option.view(-1)
        discount[option == 0] = self.discount
        target = reward + discount * next_q_value
        q_value_target = reward + self.discount * next_q_value
        q_value = q_value.squeeze()
        loss = self.loss_fn(q_value, q_value_target)
        return loss

    def state_action(self, state, action):
        output_state = state.copy()
        if len(action) > 0:
            output_state[action, 2] = 0
        return output_state

    def average_policy(self, t_r, budget_r):
        if budget_r == 0:
            return 0
        if t_r == 1:
            return budget_r
        mean_budget = max(int(self.env.budget / self.env.T), 1)
        return mean_budget

    # def beam_search(self, state, context, budget, beta=20, gamma=20):
    #     state = state.copy()
    #     action = [([], 0)]
    #     state_cache = {}  # Cache the state-action pairs to avoid repeated calculations

    #     if budget == 0:
    #         return []

    #     for i in range(budget):
    #         new_action = []

    #         for par_action, _ in action:
    #             action_key = tuple(par_action)
    #             if action_key not in state_cache:
    #                 state_cache[action_key] = self.state_action(
    #                     state, par_action)

    #             state_ = state_cache[action_key]
    #             context_ = par_action[-1] if len(par_action) > 0 else context

    #             mask = np.ones(self.env.n + 1)
    #             possible_nodes = [i for i in range(
    #                 self.env.n) if state_[i, 2] > 0]
    #             mask[possible_nodes] = 0
    #             state_batch = state_.reshape(1, -1, self.feature_size)
    #             context_batch = context_.reshape(1, -1)
    #             node_rewards = self.Q_sub(
    #                 state=state_batch, context=context_batch, mask=mask).detach().numpy().squeeze()
    #             node_rewards = node_rewards[:-1]
    #             expansion_seed = np.argpartition(-node_rewards, gamma)[:gamma]

    #             for seed in expansion_seed:
    #                 # Modify the list to contain tuples of (action, reward)
    #                 new_par_action = par_action.copy()
    #                 new_par_action.append(seed)
    #                 new_action.append((new_par_action, 0))

    #         action = new_action
    #         for idx, (new_par_action, _) in enumerate(action):
    #             action_key = tuple(new_par_action)
    #             if action_key not in state_cache:
    #                 state_cache[action_key] = self.state_action(
    #                     state, new_par_action)
    #             state_ = state_cache[action_key]
    #             context_ = new_par_action[-1] if len(
    #                 new_par_action) > 0 else context

    #             unvisited_nodes = np.where(state_[:, 2] > 0)[0]
    #             unvisited_nodes_sorted = unvisited_nodes[np.argsort(
    #                 -state_[unvisited_nodes, 2])]

    #             length = len(new_par_action)
    #             complete_action = new_par_action + \
    #                 list(unvisited_nodes_sorted[:(budget - length)])

    #             # rew = sum(state_[complete_action, 2]) - self.env.eta * (max(budget - 1, 0)) ** 2
    #             rew = sum(state_[complete_action, 2])
    #             action[idx] = (new_par_action, rew)

    #         # select top beta action
    #         action = sorted(action, key=lambda x: x[1], reverse=True)[:beta]

    #     # select the best action
    #     action = max(action, key=lambda x: x[1])[0]
    #     return action

    def beam_search(self, state, context, budget, beta=20, gamma=20):
        state = state.copy()
        actions = [([], 0)]

        if budget == 0:
            return []

        for _ in range(budget):
            new_actions = []

            for par_action, _ in actions:
                state_ = self.state_action(state, par_action)
                context_ = par_action[-1] if par_action else context

                mask = np.ones(self.env.n + 1)
                possible_nodes = [i for i in range(
                    self.env.n) if state_[i, 2] > 0]
                mask[possible_nodes] = 0

                state_batch = state_.reshape(1, -1, self.feature_size)
                context_batch = context_.reshape(1, -1)
                node_rewards = self.Q_sub(
                    state=state_batch, context=context_batch, mask=mask).detach().numpy().squeeze()[:-1]

                expansion_seeds = np.argpartition(-node_rewards, gamma)[:gamma]

                for seed in expansion_seeds:
                    new_par_action = par_action + [seed]
                    new_actions.append((new_par_action, 0))

            actions = new_actions

            for idx, (new_par_action, _) in enumerate(actions):
                state_ = self.state_action(state, new_par_action)
                context_ = new_par_action[-1] if new_par_action else context

                unvisited_nodes = np.where(state_[:, 2] > 0)[0]
                unvisited_nodes_sorted = unvisited_nodes[np.argsort(
                    -state_[unvisited_nodes, 2])]
                length = len(new_par_action)
                complete_action = new_par_action + \
                    list(unvisited_nodes_sorted[:(budget - length)])

                reward = sum(state_[complete_action, 2]) - \
                    self.env.eta * (max(budget - 1, 0)) ** 2
                total_distance = self.env.cal_total_distance(
                    complete_action, context)
                if total_distance > self.env.max_daily_distance:
                    reward -= self.env.distance_penalty * \
                        (total_distance - self.env.max_daily_distance)
                actions[idx] = (new_par_action, reward)

            actions = sorted(actions, key=lambda x: x[1], reverse=True)[:beta]

        best_action = max(actions, key=lambda x: x[1])[0]
        return best_action
