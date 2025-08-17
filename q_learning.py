# Nicholas Moreland
# 05/02/2024

import numpy as np
from random import random, shuffle

# Action class
class Action:
    arrows = ['^', '<', 'v', '>']
    names = ['up', 'left', 'down', 'right']
    translate = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    # Initialize the action
    def __init__(self, direction):
        direction = direction.lower()
        self.direction = Action.names.index(direction)

    # Apply the action to the given state
    def apply_to(self, state):
        Ty, Tx = Action.translate[self.direction]
        return state[0] + Ty, state[1] + Tx

    # Return the action that results from rotating the current action left
    def rotate_left(self):
        return Action(Action.names[(self.direction + 1) % 4])

    # Return the action that results from rotating the current action right
    def rotate_right(self):
        return Action(Action.names[(self.direction - 1) % 4])

    # Return a list of all actions
    @staticmethod
    def actions():
        return [Action(name) for name in Action.names]

    # Return whether the action is equal to the other action
    def __eq__(self, other):
        return False if other is None else self.direction == other.direction

    # Return a hash of the action
    def __hash__(self):
        return hash(self.direction)

    # Return a string representation of the action
    def __str__(self):
        return Action.arrows[self.direction]

    # Return a string representation of the action
    def __repr__(self):
        return Action.arrows[self.direction]

class Agent:
    total_actions_executed = 0

    # Initialize the actor
    def __init__(self, env):
        self.env = env
        self.cur_location = env.start_state

    # Return the current state and reward
    def sense_state_and_reward(self):
        return self.cur_location, self.env.reward(self.cur_location)

    # Execute the given action
    def execute_action(self, action):
        self.cur_location = self.env.execute_action(self.cur_location, action)
        Agent.total_actions_executed += 1

    # Return a string representation of the actor
    def __str__(self):
        return f'{self.cur_location}'

class Environment:
    # Initialize the environment
    def __init__(self, data_file, ntr):
        obstacles = []
        terminals = {}
        start_state = None

        # Read the data file
        with open(data_file, 'r') as filestream:
            rows = filestream.readlines()
            num_rows = len(rows)
            num_cols = len(rows[0].split(','))

            # Parse the data file
            for y, row in enumerate(rows):
                for x, val in enumerate(row.split(',')):
                    pos = (num_rows - y, x + 1)
                    val = val.strip()

                    if val == 'X':
                        obstacles.append(pos)
                    elif val == 'I':
                        start_state = pos
                    elif val != '.':
                        terminals[pos] = float(val)

        # Set the environment variables
        self.obstacles = obstacles
        self.terminals = terminals
        self.start_state = start_state
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.ntr = ntr

    # Return the reward for the given state
    def reward(self, state):
        return self.ntr if state not in self.terminals else self.terminals[state]

    # Execute the given action in the given state
    def execute_action(self, state, action):
        rand = random()
        if rand < 0.1:
            action = action.rotate_left()
        elif rand > 0.9:
            action = action.rotate_right()

        y, x = action.apply_to(state)
        if (y, x) in self.obstacles:
            return state
        elif (y <= 0 or y > self.num_rows) or (x <= 0 or x > self.num_cols):
            return state
        return (y, x)

# Update the Q values
def update_q(s_p, r_p, s, r, a, Q, N, gamma, env):
    if s_p in env.terminals:
        Q[(s_p, None)] = r_p

    if s is not None:
        N[(s, a)] = N.get((s, a), 0) + 1
        c = 20 / (19 + N[(s, a)])

        prev_Q = (1 - c) * Q.get((s, a), 0)
        new_Q = c * (r + gamma * utility(Q, s_p))
        Q[(s, a)] = prev_Q + new_Q

# Return the utility of the given state
def utility(Q, state):
    return max([Q.get((state, action), 0) for action in Action.actions() + [None]])

# Return the next action to take
def next_action(Q, N, Ne, state):
    f_vals = {}
    for action in Action.actions():
        f_vals[action] = f(Q.get((state, action), 0), N.get((state, action), 0), Ne)

    max_v = max(f_vals.values())
    ties = [key for key, value in f_vals.items() if value == max_v]
    shuffle(ties)
    return ties[0]

# Return the best action for the given state
def best_action(Q, state):
    return Action(Action.names[np.argmax([Q.get((state, action), 0) for action in Action.actions()])])

# Return the f value
def f(u, n, ne):
    return 1 if n < ne else u
    
def AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne):
    # Set up the environment
    env = Environment(environment_file, ntr)

    # Q values for (s,a)
    Q = {}
    # N = # of occurrences of (s,a)
    N = {}

    # Run the Q-learning algorithm
    while Agent.total_actions_executed < number_of_moves:
        s = None  # previous state
        r = None  # previous reward
        a = None  # next action
        actor = Agent(env)  # actor starts at the start state of env
        
        # Run the actor
        while Agent.total_actions_executed < number_of_moves:
            s_p, r_p = actor.sense_state_and_reward()
            update_q(s_p, r_p, s, r, a, Q, N, gamma, env)

            if s_p in env.terminals:
                break

            a = next_action(Q, N, Ne, s_p)
            actor.execute_action(a)
            s, r = s_p, r_p

    # Compute the utilities and the policy
    utilities = np.zeros((env.num_rows, env.num_cols))
    policy = np.empty((env.num_rows, env.num_cols), dtype=str)

    # Fill in the utilities and the policy
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            state = (i + 1, j + 1)

            utility_val = utility(Q, state)
            best_action_val = best_action(Q, state)

            if state in env.terminals:
                policy[i, j] = 'o'
                utilities[i, j] = env.terminals[state]
            elif state in env.obstacles:
                policy[i, j] = 'x'
                utilities[i, j] = 0
            else:
                policy[i, j] = best_action_val.__str__()
                utilities[i, j] = utility_val

    print('utilities:')
    for row in np.flip(utilities, axis=0):
        for val in row:
            print('%6.3f ' % val, end='')
        print()
    print()

    print('policy:')
    for row in np.flip(policy, axis=0):
        for val in row:
            print('%6s ' % val, end='')
        print()