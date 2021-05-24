#it is a modified version of the original ddpg_agent.py to adapt multiple agents
import numpy as np
import random
import copy
from collections import namedtuple, deque


from model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
LR_CRITIC = 1e-3  # critic learning rate
LR_ACTOR = 1e-3  # actor learning rate
WEIGHT_DECAY = 0.0  # L2 weight decay
TAU = 1e-2  # soft target update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#for compactness I did not create a seperate file
class Agent():
    """Main DDPG agent that collects the experience through the number of episodes, we need two instance from it
    mostly copy paste from earlier projects. It also means that I wilol use 2 separte networks for the two Agents"""

    def __init__(self, state_size, action_size, random_seed):
        """
        state_size = in our example is 24, action size is 2
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor network
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network
        self.critic_local = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Perform hard copy
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise proccess
        # define Ornstein-Uhlenbeck process
        self.noise = OUNoise(action_size, random_seed)

    def reset(self):
        """Resets the noise process to mean"""
        self.noise.reset()

    def act(self, state, add_noise=True):
    
        """Returns actions for given state as per current policy.
        add_noise if true: add bias to the agent
        """
        #without bacprop, we are evaluating here
        state = torch.from_numpy(state).float().to(device)  
        self.actor_local.eval() 
        with torch.no_grad(): 
        # deterministic action based on Actor's forward pass.
            action = self.actor_local(state).cpu().data.numpy()
        # set back training mode
        self.actor_local.train()  

        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Extrapolate experience into (state, action, reward, next_state, done) tuples
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # r + γ * Q-values(a,s)
        Q_targets = rewards + (gamma * Q_targets_next *(1 - dones))  

        # Compute critic loss using MSE
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #based on the attemp 3 suggestion from the 2. assignement - clip gradients
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # gets mu(s)
        actions_pred = self.actor_local(states)  
        # gets V(s,a)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_copy_weights(self, target, source):
        """
        Copy weights from source to target network
        @Params:
        1. target: copy weights into (destination).
        2. source: copy weights from (source).
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class MADDPG():
    def __init__(self, num_agents=2, state_size=24, action_size=2, random_seed=2):
        """
        Handles the multiple agents, passes the S, A
        
        """
        self.num_agents = num_agents

        # Creating agents
        self.agents = [Agent(state_size, action_size, random_seed)
                       for _ in range(self.num_agents)]

        # As it was suggested I have also created a shared ExperienceReplay buffer
        self.memory = ReplayBuffer(
            action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=random_seed)

    def act(self, states, add_noise=True):
        """Handles multiple agents"""
        actions = []
        for state, agent in zip(states, self.agents):
            # get action from a single agent
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions

    def reset(self):
        """Reset the noise level of multiple agents"""
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """
        creating the experience buffer
        from: S, A, R, next_state
        done means if the episode ended.
        """
        # Save trajectories to Replay buffer
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i],
                            rewards[i], next_states[i], dones[i])

        # check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if(len(self.memory) > BATCH_SIZE):
            for _ in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience)

    def learn(self, experiences, gamma=GAMMA):
        """Learn from an agents experiences. performs batch learning for multiple agents simultaneously"""
        for agent in self.agents:
            agent.learn(experiences, gamma)

    def saveModel(self):
        """saves the finished model"""
      
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),
                        f"actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(),
                        f"critic_agent_{i}.pth")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        # suggested improvement from last assignement
        dx = self.theta * (self.mu - x) + self.sigma * \
             np.random.standard_normal(size=x.shape)

        #dx = self.theta * (self.mu - x) + self.sigma * \
        #    np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
