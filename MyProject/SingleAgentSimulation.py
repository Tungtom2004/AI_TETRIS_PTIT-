from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from CustomAgent import Agent

env = TetrisSingleEnv()
done = False
state = env.reset()
agent = Agent()

while not done:
    img = env.render(mode='rgb_array')
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)