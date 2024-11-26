from TetrisBattle.envs.tetris_env import TetrisDoubleEnv, TetrisSingleEnv
from notebook_video_writer import VideoWriter
from CustomAgent import Agent

env = TetrisSingleEnv()
done = False
state = env.reset()
agent = Agent(0, env)

agent.train(1000)