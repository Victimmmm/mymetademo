import numpy as np
import Env
from Env import ModelEnv
import matplotlib.pyplot as plt
import Q_learning as QL

# Model parameters  
t_sens = Env.t_sens
task_granularity = Env.task_granularity
Number_of_UAVs= Env.Number_of_UAVs
Number_of_VSPs= Env.Number_of_VSPs
Wmn= Env.Wmn
t_req= Env.t_req

# Get the environment and extract the number of actions
Env = ModelEnv(t_sens=t_sens, Wmn=Wmn, Number_of_UAVs= Number_of_UAVs, L= task_granularity, t_req= t_req)
n_episodes = 2000
n_shift_deep = 25

#Q learning
Env = ModelEnv(t_sens=t_sens, Wmn=Wmn, Number_of_UAVs= Number_of_UAVs, L= task_granularity, t_req= t_req)

costs_QL = []
avg_cost_QL = []
QL_epsilon = 1
QL_epsilon_min = 0.001
QL_anneal_epoch = 100
Q_table = QL.make_Q_table()

episode_shift = 200
for i in range(n_episodes + episode_shift):
    done = False
    cost = 0
    Env.reset()
    observation = Env.get_state()
    while not done:
        action = QL.choose_action(observation, Q_table, QL_epsilon)
        observation_, penalty , done, info = Env.step(action)
        cost += penalty
        QL.update_Q_table(Q_table, observation, action, observation_, penalty, done)
        observation = observation_

    # Epsilon-greedy
    QL_epsilon = (1 - QL_epsilon_min) * (max((QL_anneal_epoch - i) / float(QL_anneal_epoch), 0)) \
                    + QL_epsilon_min

    costs_QL.append(cost)

    avg_cost = np.mean(costs_QL[max(0, i - 400):(i + 1)])
    avg_cost_QL.append(avg_cost)
    print('episode', i + 1, 'cost %.3f' % cost, 'average cost %.3f' % avg_cost)
    
# Evaluation
x = [i+1 for i in range(n_episodes)]

# Total cost
plt.plot(x, avg_cost_QL[episode_shift: ], 'g-', label='QL', linewidth=1.75)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total cost', fontsize=12)
plt.xlim([0, 2000])
plt.grid()
plt.legend(fontsize=12)
plt.show()