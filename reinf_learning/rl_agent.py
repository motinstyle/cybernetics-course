import kuimaze
from kuimaze import HardMaze
import numpy as np
import time

# function that explores the enviroment and sets q_values of states for different actions
# returns policy
def learn_policy(env):
    all_states = env.get_all_states()
    q_table = np.zeros([all_states[-1][0] + 1, all_states[-1][1] + 1, 4], dtype=float)
    alpha = 0.2
    start = time.time()

    # main loop
    while True:

        # setting start conditions
        state = env.reset()[0:2]
        finished = False
        
        # one episode
        while not finished:
            
            # choose action from state
            action = env.action_space.sample()

            # take action, observe // change finished if needed
            observation, reward, finished, _ = env.step(action)

            # evaluating the max value from all actions in the state
            max_value = float("-inf") 
            for new_action in range(4):
                if q_table[observation[0]][observation[1]][new_action] > max_value:
                    max_value = q_table[observation[0]][observation[1]][new_action]

            # updating the q_value and moving to the new state
            q_table[state[0]][state[1]][action] += alpha * (reward + max_value - q_table[state[0]][state[1]][action]) 
            state = observation[0:2]

        # time breakpoint 
        if time.time() - start > 19:
            break

    # getting policy from q_values
    policy = {state: -1 for state in all_states}
    for state in all_states:
        max_value = float("-inf")
        for action in range(4):
            if q_table[state[0]][state[1]][action] > max_value:
                max_action = action
                max_value = q_table[state[0]][state[1]][action]
        policy[state] = max_action

    
    #return q_table
    return policy


