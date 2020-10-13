import numpy as np
from ddpg import Agent
from memory import ReplayBuffer
from noise import OUNoise
import torch
import torch.nn.functional as F
import torch.optim as optim

NUM_UPDATES = 1         # number of updates
UPDATE_EVERY = 2        # training interval
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 1         # learn every # iterations
NOISE_LEVEL = 100000    # number of iterations before removing noise 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    
    def __init__(self, num_agents, state_size, action_size, random_seed):
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.agents = [
            Agent(state_size, action_size, random_seed, i) 
            for i in range(num_agents)
        ]
        self.memory = ReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
       
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise_counter):
        actions = []
        
        for agent, state in zip(self.agents, states):
            action = agent.act(state, noise_counter)
            actions.append(action)
        
        out = np.array(actions).reshape(1, -1)
        
        return out
            
    def step(self, states, actions, rewards, next_states, dones, t):
        
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)

        # add to shared replay memory 
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if t % LEARN_EVERY == 0:
            if len(self.memory) >= BATCH_SIZE:
                # use the same for all agents
                e = self.memory.sample()
                experiences = [e for _ in range(self.num_agents)]

                # each agent learns (loops over each agent in self.learn())
                self.learn(experiences, GAMMA)
                
    def learn(self, sample, gamma):
        
        next_actions = []
        actions = []
        
        # loop over each agent
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = sample[i]
            
            # get agent_id
            agent_id = torch.tensor([i]).to(device)
            
            # extract agent i state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state) # predict action
            actions.append(action)
            
            # extract agent i next state and get action via target actor network
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
        
        # let each agent learn from his experiences
        for i, agent in enumerate(self.agents):
            agent.learn(sample[i], GAMMA, actions, next_actions, i)
