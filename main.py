import os
from environment import Environment
from network import DQN
import tqdm
from PIL import Image


rewards = []
net = DQN()
EPISODES = 50000
env = Environment()
MEMORY_CAPACITY = 10000


def save_fig(episode, step_counter, environment):
    if not os.path.isdir('frames'):
        os.mkdir('frames')
    os.chdir('frames')
    if not os.path.isdir('episode' + str(episode)):
        os.mkdir('episode' + str(episode))
    os.chdir('episode' + str(episode))
    fig = environment.draw_map()
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


def main():
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in tqdm.tqdm(range(EPISODES)):
        state = env.reset()
        step_counter = 0
        while True:
            if episode % 1000 == 0:
                save_fig(episode, step_counter, env)
            step_counter += 1
            action = net.choose_action(state)
            next_state, reward, done = env.step(action)

            net.store_trans(state, action, reward, next_state)
            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn()
            if done:
                step_counter_list.append(step_counter)
                rewards.append(reward)
                break

            state = next_state


if __name__ == '__main__':
    main()
    make_gif_animation('frames')


