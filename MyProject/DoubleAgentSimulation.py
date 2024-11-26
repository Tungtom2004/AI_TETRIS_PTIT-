from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from CustomAgent import Agent
from notebook_video_writer import VideoWriter
env = TetrisDoubleEnv()

done = False
state = env.reset()
agent_list = [Agent(0), Agent(1)]

with VideoWriter(fps=50) as vw:
  while not done:
      vw.add(env.render(mode='rgb_array'))
      action = agent_list[env.game_interface.getCurrentPlayerID()].choose_action(state)
      state, reward, done, _ = env.step(action)
      env.take_turns()

  vw.add(env.render(mode='rgb_array'))