# Project details and results

The project uses 4 Neural Nets. 2 for each agent (actor, critic)
All neural net is a 24x128x128x2 size, using RELU and batch normalization (Input 24, 2 hidden layer and the output is 2)
It uses a shared Replay Buffer. 
  
### Hyperparameter tuning

```

BUFFER_SIZE = int(1e5)   # replay buffer size - we can define how many experiences we would like to collect
BATCH_SIZE = 128         # minibatch size - That depends of the memory mainly. The 128 was sufficient for this project. 
GAMMA = 0.99            # discount factor - I left it as a standard
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor - We are using Adam optimizer, and with the combination of a buffer size it had a nice speed to converge
LR_CRITIC = 1e-3        # learning rate of the critic - - We are using Adam optimizer, and with the combination of a buffer size it had a nice speed to converge
WEIGHT_DECAY = 0.0        # L2 weight decay

### Ornstein-Uhlenbeck process Noise parameters:
MU=0.
THETA=0.15
SIGMA=0.1
  
```

### Changes on the algorithm:

After a standard implementation of the DDPG coming originally from DeepMind https://arxiv.org/abs/1509.02971

I have used gradient clipping when training the critic network as it was suggested in the last project

```
 nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
  
```

I adopted the ddpg_agent to handle multiple Agents, they have their own Actor, Critic Networks.
  
After countless days of Hyperparameter tuning and changes in the code, the performance was not enough.
  
Sometimes after a few tousand of episode the score have reached around 0.36
I have tried everything...
 
Then I started to realize adding Noise is a very, very critical part of this project so I paid more attention of the Ornstein-Uhlenbeck process.
  
- First I had to admit that I had a significant bias towards positive values and that indeed resulted poor training
After replacing the generator that draws samples from a Standard Normal distribution and thus generate a different nois sample for each agent as opposed to applying same noise to all agents.

```
dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
  
  VS

dx = self.theta * (self.mu - x) + self.sigma * \
             np.random.standard_normal(size=x.shape)
```

Hyperparameter of the Ornstein-Uhlenbeck process:

It resulted also a significant different playing with the parameters of the noise and of course same setting but different random seed could cause big dirrence.
  
In retrospectively the key was to add adequate noise to the training process

### MADDPG ALgorithm - Multi Agent Deep Deterministic Policy gradient

It is basically an adoption of the DDPG algorythm for multiple Agents.
The DDPG is a policy based method and not a value-based as the DQN was in the last project. The policy-based methods directly learn the optimal policy, without having to maintain a separate value function estimate as we did at last time. So we had an in between step to obtain the optimal policy from the optimal action-value function.

#### OK. And? Why are we not using here also the DQN based model?

There are several reason for that:
* The main one is that the DQN produces a discrete action selection. We could try to solve the environment by discretizing the continous output, but it would be too difficult to learn. The policy based methods are well suited for continous action spaces. (Continous vs discrete: we do not only have a choice of move left or right, but in more like "how much left or right" where the how much is in a continous space like between -100 and +100)
* It is simpler because we are not need to store additional data, like action values
* It can learn true stochastic policies

#### Learning the Optimal policy directly OK. But what is "Gradient"?

Policy gradient methods are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.
Yes, ascent and not descent as we have used in all deep learning projects. The ascent means that we are not minimizing a loss function, but maximaizing the expected return.

#### D for Deep, P - Policy, G - gradient.

What is the extra D at the begining - Deterministic. It means that for a certain state we will get always the same action. If it would be Stochastic Policy - we would get a probability for a state.

#### Hmm. But we have used 2 networks one of them was Actor another one was Critic..

Well, actually we have used 2 actor and 2 Critic networks, but first what is Actor - Critic in a nutshell:

Actor - Critic is marrying the the two world of Policy and Value methods. 
The policy based methods reinforce the good actions an penalize the bad ones. After a lots of experience we have increased the probabilty of actions that led to a win. And decreased the ones that led to losses. (problem: High varience, slow learner)
The value based are always guessing - like the last project DQN compared a guess with a guess... (problem: Introduces bias and over or under estimation)

If you would dig more into the Actor-Critic methods, you might say that DDPG is not really one.

At DDPG what is the role of the two types of networks?

Actor: Optimal policy, deterministicly - outputs the best believed action for any given state

Critic: Learns evaluates the optimal action - value function  

In the DQN we had a local and the target network. And after a bunch of time steps we just copied the parameters from the local to the target network. 
In DDPG we also have to maintain the local and the target Critic networks, and from the local to target copy the parameters. But we not copying in big chunks, instead of we are using a soft update: We "blending" a very small portions from the local network to the target at each step. (Like 0.01%)
Same for Actor and Critic networks.

We get faster convergence using this startegy. The soft update can actually used at DQN as well.

More details in the code, with comments.

### The training result
  
 The scores and the rolling scores. The environment was solved in 233
 (This was the best result)

![result](https://github.com/plitkei/RL-Unity-MLAgents-Python-Pytorch-tennis.app/blob/main/result.jpg)

### Future improvements

Better hyperparameter tuning, using the same network for both agents might help for faster training, extra care of adding noise
