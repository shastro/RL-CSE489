"""File for testing functions in our environment. Use this to try and evaluate
qualititatively how the fits match our agent.

"""
import gym
import numpy as np
from sympy import symbols, sympify


def get_fn(eqn):
    eqn.replace("pi", "3.14159")
    fn = sympify(eqn, evaluate=False)

    return fn


def main():
    env = gym.make("Pendulum-v0")
    x0 = symbols("x0")
    x1 = symbols("x1")
    x2 = symbols("x2")
    functions = []

    eqn = "-3.65321298054538*x0**3 + 3.60464564450212*x0**2*x1 - 2.8576365722644*x0**2*x2 + 1.70347201314087*x0**2 + 36.7615264112664*x0*x1**2 + 6.86065152859311*x0*x1*x2 - 27.7121835635439*x0*x1 + 1.82483102990305*x0*x2**2 - 4.86407429457206*x0*x2 - 0.273210593149015*x0 + 16.4905115052767*x1**2*x2 - 27.0210909963925*x1**2 + 2.90651752869256*x1*x2**2 - 7.52160545907771*x1*x2 - 4.11304903230168*x1 + 0.765099252990031*x2**3 - 1.82976239055718*x2**2 - 4.93864101425579*x2 + 3.22746456075386"
    fn = get_fn(eqn)
    functions.append(("New", fn))

    for name, testf in functions:
        for i in range(100):
            print(f"Episode {i}")
            print(f"Testing function: {name}")
            prev_state = env.reset()
            ep_reward = 0
            # prev_state = (-0.971693, -0.236246, 0.547701)
            while True:
                env.render()

                action = np.clip(
                    testf.evalf(
                        subs={x0: prev_state[0], x1: prev_state[1], x2: prev_state[2]}
                    ),
                    -2,
                    2,
                )
                state, reward, done, info = env.step(
                    np.array([action], dtype=np.float32)
                )

                ep_reward += reward

                prev_state = state

                if done:
                    break

            print(f"Episodic Reward {ep_reward}")


if __name__ == "__main__":
    main()
