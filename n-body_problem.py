import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from time import time

G = 1
colors = list(mcolors.TABLEAU_COLORS)


def force(pos1, pos2, m1, m2):
    r = pos2 - pos1
    f = (G * m1 * m2 / (np.linalg.norm(r)**3)) * r
    return f


def gravitation(S, t, n, m):
    x = S[:n]
    y = S[n:2 * n]
    vx = S[2 * n:3 * n]
    vy = S[3 * n:4 * n]

    pos = np.array([x, y]).T
    acc = np.zeros((n, 2))

    for i in range(n):
        for j in range(i + 1, n):
            fij = force(pos[i], pos[j], m[i], m[j])
            acc[i] += fij / m[i]
            acc[j] -= fij / m[j]

    return np.concatenate((vx, vy, acc[:, 0], acc[:, 1]))


def state_to_pos(S, n):
    x = S[:n]
    y = S[n:2 * n]

    pos = np.array([x, y]).T
    return pos


def read_data(file: str):
    print('read data')
    data = pd.read_csv(file)
    n = data.shape[0]
    S0 = np.concatenate([data["x"], data["y"], data["vx"], data["vy"]],
                        dtype="float")
    m = np.array(data["m"], dtype="float")

    return n, S0, m


def main():
    file = input('enter a csv file: ')
    tmax = float(input('enter a tmax: '))
    n, S0, m = read_data(file + ".csv")
    # S0[0] += 0.001
    pos = state_to_pos(S0, n)
    dt = 1. / 30
    t = np.arange(0, tmax, dt)
    print('prepare to solve')
    sol = odeint(gravitation, S0, t, args=(n, m))
    print('solved')
    # print(sol.shape)

    print('animate')
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
    sca = ax.scatter(pos[:, 0], pos[:, 1], s=m, c=colors[:n])
    ttx = ax.text(-5, 5, '')

    def init():
        ttx.set_text('')
        return sca, ttx

    def update(i):
        sca.set_offsets(state_to_pos(sol[i], n))
        ttx.set_text('time = %.2f' % t[i])
        return sca, ttx

    t0 = time()
    update(0)
    t1 = time()

    anim = FuncAnimation(fig,
                         update,
                         frames=len(t),
                         interval=1000 * dt - (t1 - t0),
                         init_func=init)
    # plt.show()
    anim.save('sol_' + file + '.gif', fps=1 / dt)

    print('done')
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()
