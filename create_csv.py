import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS)


def show_points(b, n, m):
    plt.scatter(b.T[0], b.T[1], c=colors[:n], s=m)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def state_to_DataFrame(S, n, m):
    x = S[:n]
    y = S[n:2 * n]
    vx = S[2 * n:3 * n]
    vy = S[3 * n:4 * n]
    dic = {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'm': m}
    df = pd.DataFrame(dic)
    return df


def main():
    n = int(input('enter n: '))
    b = 2 * np.array([[np.cos(2 * np.pi * i / n),
                       np.sin(2 * np.pi * i / n)] for i in range(n)])
    v = np.array([[-np.sin(2 * np.pi * i / n),
                   np.cos(2 * np.pi * i / n)] for i in range(n)])
    m = np.ones(n)

    show_points(b, n, m)

    S0 = np.concatenate((b[:, 0], b[:, 1], v[:, 0], v[:, 1]))
    df = state_to_DataFrame(S0, n, m)
    print(df)
    df.to_csv(str(n) + 'th-root_of_unity.csv', index=False)


if __name__ == '__main__':
    main()