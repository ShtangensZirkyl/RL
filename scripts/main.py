import os
from environment import Environment
from network import DQN, Net, Net4
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

rewards = []
EPISODES = 50000
MEMORY_CAPACITY = 10000


class DeepRL(object):
    def __init__(self, network=Net, mem=MEMORY_CAPACITY):
        self.net = DQN(network)
        self.env = Environment()
        self.episodes = len(self.env.data)
        self.memory_capacity = mem
        self.rewards = []

    def train(self):
        print("The DQN is collecting experience...")
        step_counter_list = []
        for episode in tqdm.tqdm(range(self.episodes)):
            state = self.env.reset()
            step_counter = 0
            while True:
                if episode % 1000 == 0:
                    self.save_fig(episode, step_counter)
                step_counter += 1
                action = self.net.choose_action(state)
                next_state, reward, done = self.env.step(action)

                self.net.store_trans(state, action, reward, next_state)
                if self.net.memory_counter >= self.memory_capacity:
                    self.net.learn()
                if done:
                    step_counter_list.append(step_counter)
                    self.rewards.append(reward)
                    break

                state = next_state

    def save_fig(self, episode, step_counter):
        if not os.path.isdir('frames'):
            os.mkdir('frames')
        os.chdir('frames')
        if not os.path.isdir('episode' + str(episode)):
            os.mkdir('episode' + str(episode))
        os.chdir('episode' + str(episode))
        fig = self.env.draw_map()
        fig.savefig(str(step_counter) + '.png')
        os.chdir('../..')


def make_gif_animation(_path):
    os.chdir(_path)
    for dirs in os.listdir():
        os.chdir(dirs)
        frames = []
        frames_len = len([f for f in os.listdir()
                          if f.endswith('.png')])
        for frame_number in range(frames_len):
            frame = Image.open(str(frame_number) + '.png')
            frames.append(frame)
        frames[0].save(
            'episode.gif',
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=400,
            loop=0
        )
        os.chdir('..')
    os.chdir('..')


if __name__ == '__main__':
    drl = DeepRL(network=Net4)
    drl.train()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(drl.rewards)
    fig.savefig('rewards.png')  # save the figure to file
    m = 0
    p = 0
    for i in drl.rewards:
        if i < 0:
            m += 1
        else:
            p += 1
    print(m / (m + p))
    make_gif_animation('frames')
    drl.save('../models/net4.json')
