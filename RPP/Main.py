import networkx as nx
import numpy as np
import sys
from Agent import Agent
from Env import Env
from Baselines import *

params = {
    'index': 1,
    'n': 100,     # number of nodes
    'test_n': 100,
    'budget': 20,  # number of budget
    'T': 10,     # number of time steps
    'num_epoch': 20,
    'num_iteration': 10,
    'num_test': 100,
}


def generate_job_name(params):
    return f"job{params['index']}-n={params['n']}-text_n={params['test_n']}-budget={params['budget']}-T={params['T']}-epoch={params['num_epoch']}-iteration={params['num_iteration']}"


# run the testing rounds and write results
def run_test_rounds(model, params, job_name):

    txt_name = f"results/{job_name}-test.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        for i in range(params['num_test']):
            print(f"test round: {i}")
            _, _, _, _, _, _, cumulative_reward = model.run_episode(
                beam_search=True)
            print(f"cumulative reward of round {i}: {cumulative_reward}")
            cumulative_reward_test.append(cumulative_reward)
        avg_reward = np.average(cumulative_reward_test)
        std_dev = np.std(cumulative_reward_test)
        print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")


if __name__ == "__main__":
    job_name = generate_job_name(params)

    # Training phase
    txt_name = f"results/{job_name}-training.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        loc = np.random.rand(params['n'], 2) * 100
        env = Env(loc=loc, budget=params['budget'], T=params['T'])
        model = Agent(env=env)

        # train_cumulative_reward_list, test_cumulative_reward_list = model.train_model(
        #     num_epochs=params['num_epoch'], num_iterations=params['num_iteration']
        # )

        # print("------------------------------ Total Results ------------------------------")
        # print("train_cumulative_reward_list: ", train_cumulative_reward_list)
        # print("test_cumulative_reward_list: ", test_cumulative_reward_list)

    # Testing phase
    loc = np.random.rand(params['test_n'], 2) * 100
    model.set_loc(loc)
    model.eval()
    model.discount = 1.0

    # Run test rounds for all policy pairs
    run_test_rounds(model, params, job_name)

    # baseline
    txt_name = f"results/{job_name}-greedy.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        greedy = Greedy(env=model.env)
        for i in range(params['num_test']):
            print(f"test round: {i}")
            seeding_policy, cumulative_reward = greedy.run()
            print(f"cumulative reward of round {i}: {cumulative_reward}")
            print(f"seeding_policy: {seeding_policy}")
            cumulative_reward_test.append(cumulative_reward)
        avg_reward = np.average(cumulative_reward_test)
        std_dev = np.std(cumulative_reward_test)
        print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")

    txt_name = f"results/{job_name}-ga.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        ga = GA(env=model.env)
        for i in range(params['num_test']):
            print(f"test round: {i}")
            seeding_policy, cumulative_reward = greedy.run()
            print(f"cumulative reward of round {i}: {cumulative_reward}")
            print(f"seeding_policy: {seeding_policy}")
            cumulative_reward_test.append(cumulative_reward)
        avg_reward = np.average(cumulative_reward_test)
        std_dev = np.std(cumulative_reward_test)
        print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")
