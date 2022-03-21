import torch
import tqdm
from environment import Environment
import os
from network import Net4
from network import Net
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


class test_DeepRL(object):
    def __init__(self):
        self.net = Net4()
        self.net.load_state_dict(torch.load('../models/Net4.json'))
        self.net.eval()
        self.env = Environment()
        self.episodes = len(self.env.data)
        self.rewards = []

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_value = self.net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0]
        return action

    def save_fig(self, episode, step_counter):
        if not os.path.isdir('test_frames'):
            os.mkdir('test_frames')
        os.chdir('test_frames')
        if not os.path.isdir('episode' + str(episode)):
            os.mkdir('episode' + str(episode))
        os.chdir('episode' + str(episode))
        fig = self.env.draw_map()
        fig.savefig(str(step_counter) + '.png')
        os.chdir('../..')

    def test(self):
        for episode in tqdm.tqdm(range(self.episodes)):
            state = self.env.reset()
            step_counter = 0
            while True:
                if episode % 100 == 0:
                    self.save_fig(episode, step_counter)
                step_counter += 1
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                if done:
                    self.rewards.append(reward)
                    break

                state = next_state


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


def remove_files():
    if os.path.isdir('test_frames'):
        os.chdir('test_frames')
        for folder in os.listdir():
            os.chdir(str(folder))
            for file in os.listdir():
                os.remove(file)
            os.chdir('..')
            os.rmdir(folder)
        os.chdir('..')


if __name__ == '__main__':
    remove_files()
    test_mod = test_DeepRL()
    test_mod.test()
    m = 0
    p = 0
    for i in test_mod.rewards:
        if i < 0:
            m += 1
        else:
            p += 1
    print(p / (m + p))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(test_mod.rewards)
    fig.savefig('test_rewards.png')  # save the figure to file
    make_gif_animation('test_frames')
    # print(test_mod.rewards)
