import networkx as nx
import numpy as np
import sys
from Agent import Agent
from Env import Env

params = {
    'index': 1,
    'n': 200,     # number of nodes
    'test_n': 200,
    'budget': 20,  # number of budget
    'test_budget': 100,
    'T': 10,     # number of time steps
    'num_epoch': 20,
    'num_iteration': 15,
    'num_test': 100,
}


def generate_job_name(params):
    return f"job{params['index']}-n={params['n']}-test_n={params['test_n']}-budget={params['budget']}-test_budget={params['test_budget']}-T={params['T']}-epoch={params['num_epoch']}-iteration={params['num_iteration']}"


# run the testing rounds and write results
def run_test_rounds(model, params, job_name, policy_pairs):
    for policy in policy_pairs:
        budget_policy, seeding_policy = policy
        txt_name = f"results/{job_name}-{budget_policy}-{seeding_policy}.txt"
        with open(txt_name, 'w', encoding='utf-8') as f:
            sys.stdout = f
            cumulative_reward_test = []
            for i in range(params['num_test']):
                print(f"test round: {i}")
                _, _, _, _, _, cumulative_reward = model.run_episode(
                    print_info=True,
                    budget_policy=budget_policy,
                    seeding_policy=seeding_policy,
                    beam_search=('agent' in seeding_policy)
                )
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
        g = nx.erdos_renyi_graph(params['n'], 0.01, directed=True)
        env = Env(graph=g, budget=params['budget'], T=params['T'])
        model = Agent(env=env)

        train_cumulative_reward_list, test_cumulative_reward_list = model.train_model(
            num_epochs=params['num_epoch'], num_iterations=params['num_iteration']
        )

        print("------------------------------ Total Results ------------------------------")
        print("train_cumulative_reward_list: ", train_cumulative_reward_list)
        print("test_cumulative_reward_list: ", test_cumulative_reward_list)

    # Testing phase
    g = nx.erdos_renyi_graph(params['test_n'], 0.01, directed=True)
    model.set_graph(g)
    model.env.budget = params['test_budget']
    model.env.reset()
    model.eval()

    # Define different budget and seeding policy pairs
    policy_pairs = [
        ('agent', 'agent'), ('average', 'score'), ('average', 'degree'),
        ('average', 'agent'), ('static', 'score'), ('static', 'degree'),
        ('static', 'agent'), ('normal', 'score'), ('normal', 'degree'),
        ('normal', 'agent'), ('agent', 'score'), ('agent', 'degree')
    ]

    # Run test rounds for all policy pairs
    run_test_rounds(model, params, job_name, policy_pairs)
