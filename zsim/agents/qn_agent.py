import numpy as np
import matplotlib.pyplot as plt
import random


class QLearningAgent(object):
    def __init__(self, log_dir, env, agent_opts):
        self.log_dir = log_dir
        self.env = env
        self.agent_opts = agent_opts
        num_steps = agent_opts["num_steps"]
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        # For plotting metrics
        all_episodes = []
        all_penalties = []
        all_rewards = []
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

        for i in range(1, num_steps):
            state = env.reset()

            epochs, penalties, reward, episodic_reward = 0, 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(q_table[state]) # Exploit learned values

                next_state, reward, done, info = env.step(action)

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                episodic_reward += reward
                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            all_rewards.append(episodic_reward)
            all_episodes.append(i)
            if i % 100 == 0:
                #clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")

        plt.plot(all_episodes, all_rewards, 'r--')
        plt.show()
