import Env
import numpy as np

alpha = 0.3
gamma = 1

Q_steps=40
gamma_EC_steps=10
t_tot_steps=5
t_req_steps=5
f_v_steps=4
f_e_steps=4

Q_max= Env.task_size
gamma_EC_max=10*pow(10,9/10)
gamma_EC_min=10*pow(10,-3/10)
t_tot_max=7
t_req_min=Env.t_req
t_req_max=10
f_v_max=15
f_e_max=15

Q_steps_length=Q_max/Q_steps
gamma_EC_steps_length=(gamma_EC_max-gamma_EC_min)/gamma_EC_steps
t_tot_steps_length=t_tot_max/t_tot_steps
t_req_steps_length=(t_req_max-t_req_min)/t_req_steps
f_v_steps_length=f_v_max/f_v_steps
f_e_steps_length=f_e_max/f_e_steps

def find_state_indices(state):
    Q_index=int(state[0]/Q_steps_length)
    gamma_EC_index=int((state[1]-gamma_EC_min)/gamma_EC_steps_length)
    t_tot_index=int(state[2]/t_tot_steps_length)
    t_req_index=int((state[3]-t_req_min)/t_req_steps_length)
    f_v_index=int(state[4]/f_v_steps_length)
    f_e_index=int(state[5]/f_e_steps_length)
    return(Q_index,gamma_EC_index,t_tot_index,t_req_index,f_v_index,f_e_index)
    
def make_Q_table():
    return np.ones((Q_steps+1 , gamma_EC_steps +1, t_tot_steps +1, 
                    t_req_steps +1, f_v_steps+1 , f_e_steps+1, Env.task_granularity + 1))*10000

def choose_action(state, Q_table, epsilon):
    s0, s1, s2, s3, s4, s5 = find_state_indices(state)
    temp = np.random.rand()
    if temp > epsilon:
        action = np.argmin(Q_table[s0, s1, s2, s3, s4, s5])
    else:
        action = np.random.choice(Env.task_granularity + 1)
    return int(action)

def update_Q_table(Q_table, state, action, next_state, reward, done):
    s0, s1, s2, s3, s4, s5 = find_state_indices(state)
    s_0, s_1, s_2, s_3, s_4, s_5 = find_state_indices(next_state)
    Q_table[s0, s1, s2, s3, s4, s5, int(action)] = (1 - alpha)*Q_table[s0, s1, s2, s3, s4, s5, int(action)] + \
                                                      alpha*(reward + gamma*np.min(Q_table[s_0, s_1, s_2, s_3, s_4, s_5]))\
                                                           *(1 - int(done))