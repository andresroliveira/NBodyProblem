import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from time import time

G = 1
EPS = 1e-6
colors = list(mcolors.TABLEAU_COLORS)


def force(pos1, pos2, m1, m2):
    r = pos2 - pos1
    l = np.linalg.norm(r)
    if np.abs(l) < EPS:
        raise Exception('divide by 0')
    f = (G * m1 * m2 / (l**3)) * r
    return f


def gravitation(t, S, n, m):
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
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        print("File not Found")
        input("Press Enter to continue...")
        exit(0)
    n = data.shape[0]
    S0 = np.concatenate([data["x"], data["y"], data["vx"], data["vy"]],
                        dtype="float")
    m = np.array(data["m"], dtype="float")

    return n, S0, m


def save_sol(sol, t, n):
    dic = {}
    dic['t'] = t
    for i in range(n):
        dic['x_' + str(i + 1)] = sol[:, i]
        dic['y_' + str(i + 1)] = sol[:, i + n]
        dic['vx_' + str(i + 1)] = sol[:, i + 2 * n]
        dic['vy_' + str(i + 1)] = sol[:, i + 3 * n]

    df = pd.DataFrame(dic)
    return df


def main(file: str, tmax: float, fps: float):
    """
        Parameters: 
        
        file: string
        file.csv should be a csv file with x,y,vx,vy,m collums

        tmax: float
        solve ivp problem in the interval [0, tmax] with initial conditions in file.csv. Default is 10.0

        fps: float
        frames per second of resulting imagem. Default is 26.0
    """

    n, S0, m = read_data(file + ".csv")
    print('prepare to solve')
    pos = state_to_pos(S0, n)
    t_eval = np.linspace(0, tmax, int(tmax * fps * 100))
    print('solving')

    sol_ivp = solve_ivp(gravitation,
                        t_span=[0, tmax],
                        y0=S0,
                        args=(n, m),
                        method='BDF',
                        t_eval=t_eval,
                        rtol=1e-6,
                        atol=1e-9)

    if not sol_ivp.success:
        print(sol_ivp.message)
        raise Exception("Not success solve_ivp!")

    print('solved')
    sol = sol_ivp.y.T
    t = sol_ivp.t
    dt = t[1] - t[0]

    # print(sol)
    # print(sol_ivp)
    if len(file.split('/')) > 1:
        file = file.split('/')[-1]

    df = save_sol(sol, t, n)
    df.to_csv('outputs/sol_' + file + '.csv', index=False)
    # return

    print(sol.shape)
    print(t.shape)
    print(state_to_pos(sol[0], n))
    # print('fps =', fps)

    # for i in range(1, len(t))
    #     print(t[i] - t[i-1])

    print('animate')
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
    sca = ax.scatter(pos[:, 0], pos[:, 1], s=m, c=colors[:n])
    ttx = ax.text(-5, 5, '')

    def init():
        ttx.set_text('')
        return sca, ttx

    def update(i):
        sca.set_offsets(state_to_pos(sol[100 * i], n))
        ttx.set_text('time = %.3f' % t[100 * i])
        return sca, ttx

    t0 = time()
    update(0)
    t1 = time()

    anim = FuncAnimation(fig,
                         update,
                         frames=len(t) // 100,
                         interval=1000 * dt / 100 - (t1 - t0),
                         init_func=init)

    anim.save('outputs/sol_' + file + '.gif', fps=int(100 / dt))
    plt.show()

    print('done')
    # input("Press Enter to continue...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            file = sys.argv[1]
        except:
            print('no file select')
            exit(0)

        try:
            tmax = float(sys.argv[2])
        except:
            tmax = 10.0

        try:
            fps = float(sys.argv[3])
        except:
            fps = 26.0
    else:
        file = 'ivp'
        tmax = 10.0
        fps = 26.0

    main(file, tmax, fps)
