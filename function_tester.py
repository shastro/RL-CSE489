"""File for testing functions in our environment. Use this to try and evaluate
qualititatively how the fits match our agent.

"""
import gym
from numpy import *

acos = arccos
asin = arcsin


def main():
    env = gym.make("Pendulum-v0")
    functions = []
    # functions.append(
    #     ("arccos(-0.109236055410 + sin(sqrt((x0 - ((sin(x1) - 1) - 1)))))", func1)
    # )
    functions.append(("New", func2))

    for name, testf in functions:
        for i in range(20):
            print(f"Episode {i}")
            print(f"Testing function: {name}")
            prev_state = env.reset()
            ep_reward = 0

            while True:
                env.render()

                action = testf(*prev_state)
                state, reward, done, info = env.step(array([action]))

                ep_reward += reward

                prev_state = state

                if done:
                    break

            print(f"Episodic Reward {ep_reward}")


def func1(x0, x1, x2):
    return acos(
        (sin(x0 - 0.990920662879944) + 1.38961136341095) ** 0.010417946614325
        - 0.178610488772392
    )


def func2(x0, x1, x2):
    return tan(0.001 * exp(((x2 + cos(((x2) ** (-1) + 1)))) ** (-1)))


if __name__ == "__main__":
    main()
