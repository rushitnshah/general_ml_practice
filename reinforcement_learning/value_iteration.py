# # Value iteration in python
import numpy as np



class LineEnv:
    def __init__(self, nstates, noise=0.0):
        self.nstates = nstates
        self.nactions = 2
        self.noise = noise
        self.states = np.array([state for state in range(self.nstates)])
        self.actions = np.array([act for act in range(self.nactions)])
        self.terminal_state = self.states[-1]
        self._init_transition_dynamics()
        self._init_reward_function()
    
    def _init_reward_function(self):
        self.rewards = [float('-Inf') for i in range(self.nstates)]
        for state in self.states:
            if state == self.terminal_state:
                self.rewards[state] = 10
            else:
                self.rewards[state] = -1
        self.rewards = np.array(self.rewards)
    
    def _init_transition_dynamics(self):
        self.transition_dynamics = [
            [
                [
                    0.0 for k in range(self.nstates)
                ] for j in range(self.nactions)
            ] for i in range(self.nstates)
        ]

        for state in self.states:
            if state == self.terminal_state:
                for action in self.actions:
                    for next_state in self.states:
                        self.transition_dynamics[state][action][next_state] = 0.0
                    self.transition_dynamics[state][action][state] = 1.0
            else:
                for action in self.actions:
                    if action == 0: # left
                        if state == 0:
                            self.transition_dynamics[state][action][state] = 1.0
                        else:
                            self.transition_dynamics[state][action][state - 1] = 1-self.noise
                            self.transition_dynamics[state][action][state + 1] = self.noise / (self.nactions - 1)
                    if action == 1: # right
                        if state == 0:
                            self.transition_dynamics[state][action][state + 1] = 1-self.noise
                            self.transition_dynamics[state][action][state] = self.noise
                        else:
                            self.transition_dynamics[state][action][state + 1] = 1-self.noise
                            self.transition_dynamics[state][action][state - 1] = self.noise / (self.nactions - 1)
        self.transition_dynamics = np.array(self.transition_dynamics)

if __name__=='__main__':
    linenv = LineEnv(nstates=10, noise=0.5)
    states, actions, rewards, dynamics = linenv.states, \
                                            linenv.actions, \
                                            linenv.rewards, \
                                            linenv.transition_dynamics

    # Value Iteration
    niters = 250
    gamma = 0.9
    V_old = np.array([0.0 for _ in range(len(states))])
    V = V_old
    for iter in range(niters):
        vlist = [round(x,2) for x in V_old]
        if iter % 10 == 0:
            print(f"iter {iter}: {vlist}")
        
        for state in states:
            q_values = []
            for action in actions:
                q = sum(dynamics[state][action]*(rewards + gamma*V_old))
                q_values.append(q)
            V[state] = max(q_values)
        V_old = V