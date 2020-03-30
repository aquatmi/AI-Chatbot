from cartpole import CartPoleAgent
import numpy as np

# reward function has been edited to reward not having the pole perfectly upright
# this is to make the gameplay more interesting.
# however, this led to the ai learning to just let the pole lean to a side and run off the edge
# as this would give it more points than playing normally for the same amount of time
# so I added a loss in points for getting close to the edge of the screen.
# this means the bot tries to have the pole at an angle as much as it can without going to the edges


# child class of CartPoleAgent, only the train function is overwritten
class RiskyAgent(CartPoleAgent):
    FILE_NAME = "cartpole-deepq-risky.h5"
    def train(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            score = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    # next_state[0][2] is the angle the pole is at, bigger angle, more risky = good!
                    # next_state[0][0] is distance of the cart from center, further from center = bad!
                    # should encourage riskier play
                    reward = reward + abs(next_state[0][2]) - abs(next_state[0][0])
                    score += reward
                else:
                    reward = -100
                i += 1
                if done:
                    print(
                        "episode: {}/{}, frames lasted: {}, score: {} e: {:.2}".format(
                            e, self.EPISODES, i, (score + 1), self.epsilon))
                    if i == 500:
                        reward = reward + 200
                        print("Saving trained model as cartpole-dqn-risky.h5")
                        self.save("cartpole-deepq-risky.h5")
                        # return
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()
        self.save("cartpole-deepq-risky.h5")

if __name__ == "__main__":
    agent = RiskyAgent()
    # agent.train()
    agent.test(agent.FILE_NAME, show=False)
