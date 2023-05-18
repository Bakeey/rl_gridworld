import numpy as np
import gymnasium as gym

from datetime import datetime
from collections import deque

import rl_gridworld.envs.rl_gridworld

class QNAgent:
    def __init__(self, no_episodes: int = 1000, alpha = 1.0, size: int = 4, battery_life = 1000, render_mode: str = "human"):
        self.env = rl_gridworld.envs.DysonEnv(size = size, render_mode=render_mode) # gym.make('rl_gridworld/Dyson-v0', render_mode="human")
        self.env.action_space.seed(42)

        self.observation, self.info = self.env.reset(seed=42)

        self.state_size = 2
        self.action_size = self.env.action_space.n
        self.EPISODES = no_episodes
        self.memory = deque(maxlen=2000)

        # Optimism in the face of uncertainty
        self.q_values = np.ones((size,size,4), dtype=float) * max(self.env.reward_range)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.0 # 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999

        self.learning_rate = alpha


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return # TODO
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
 
    # implement the epsilon-greedy policy
    def act(self, state):
        # implement the epsilon-greedy policy
        if  np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            action_values = self.q_values[state]
            return np.argmax(action_values)

    # DONE
    # implement the Q-learning
    def replay(self):
        # if len(self.memory) < self.train_start:
        #     return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros_like(state)
        action, reward, done = [None]*self.batch_size, [None]*self.batch_size, [None]*self.batch_size

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):
            state[i]      = minibatch[i][0].reshape(self.state_size,)
            action[i]     = minibatch[i][1]
            reward[i]     = minibatch[i][2]
            next_state[i] = minibatch[i][3].reshape(self.state_size,)
            done[i]       = minibatch[i][4]

        # compute value function of current(call it target) and value function of next state(call it target_next)
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        
        for i in range(self.batch_size):
            # correction on the Q value for the action used,
            # if done[i] is true, then the target should be just the final reward
            if done[i]:
                target[i,action[i]] = -100 # reward[i]
                
            else:
                # else, use Bellman Equation
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # target = max_a' (r + gamma*Q_target_next(s', a'))
                target[i,action[i]] = (1-self.learning_rate) * target[i,action[i]] +\
                    self.learning_rate * (reward[i] + self.gamma * np.max(target_next[i]))
            
    def training(self):
        rewards = np.zeros(self.EPISODES)
        for e in range(self.EPISODES):
            observation, info = self.env.reset() # returns observations, maybe reshape?
            state = tuple(observation['agent'])
            done = False
            truncated = False
            i = 0
            while (not done) and (not truncated):
                # if you have graphic support, you can render() to see the animation. 
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                next_state = tuple(next_state['agent'])
                                    
                self.remember(state, action, reward, next_state, done)

                self.q_values[state+(action,)] = (1-self.learning_rate) * self.q_values[state+(action,)] +\
                    self.learning_rate * (reward + self.gamma * np.max(self.q_values[next_state]))
                
                
                i += reward # reward
                if done or truncated:  
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("learning rate: {}, episode: {}/{}, score: {}, e: {:.2}, time: {}".format(self.learning_rate, e+1, self.EPISODES, i, self.epsilon, timestampStr))
                    rewards[e] = i
                    # save model option
                    # if i >= 500:
                    #     print("Saving trained model as cartpole-dqn-training.h5")
                    #     self.save("./save/cartpole-dqn-training.h5")
                    #     return rewards # remark this line if you want to train the model longer
                # self.replay()

                # update Q values
                state = next_state


        return rewards


def main():

    agent = QNAgent(render_mode=None) # "human")
    agent.training()


    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.env.step(env.env.action_space.sample())
        print(reward)
        if terminated or truncated:
            observation, info = env.env.reset()

    env.env.close()

if __name__=="__main__":
    exit(main())