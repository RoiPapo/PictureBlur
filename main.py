import time

import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigs


def const_step(s):
    def lsearch(f, x, gf):
        return s

    return lsearch


def exact_quad(A):
    np.linalg.cholesky(A.toarray())

    def lsearch(f, x, gf):
        a = (np.linalg.norm(gf(x)) / (np.sqrt(2) * np.linalg.norm(A.toarray().dot(gf(x))))) ** 2
        return a

    return lsearch


def gradient_method(f, gf, lsearch, x0, iterations):
    x = x0
    fs = []
    gs = []
    ts = []
    gval = gf(x)
    startime = time.time()
    ts.append(0)
    for i in range(iterations):
        t = lsearch(f, x, gf)
        x = x - t * gval
        print('iter= {:2d} f(x)={:10.10f}'.format(i, f(x)))
        gval = gf(x)
        fs.append(f(x))
        gs.append(np.linalg.norm(gval))
        ts.append(time.time() - startime)
    return x, fs, gs, ts


def blur(N, band=3, sigma=0.7):
    z = np.block([np.exp(-(np.array([range(band)]) ** 2) / (2 * sigma ** 2)), np.zeros((1, N - band))])
    A = toeplitz(z)
    A = csr_matrix(A)
    A = (1 / (2 * scipy.pi * sigma ** 2)) * kron(A, A)

    x = np.zeros((N, N))
    N2 = round(N / 2)
    N3 = round(N / 3)
    N6 = round(N / 6)
    N12 = round(N / 12)

    # Large elipse
    T = np.zeros((N6, N3))
    for i in range(1, N6 + 1):
        for j in range(1, N3 + 1):
            if (i / N6) ** 2 + (j / N3) ** 2 < 1:
                T[i - 1, j - 1] = 1

    T = np.block([np.fliplr(T), T])
    T = np.block([[np.flipud(T)], [T]])
    x[2:2 + 2 * N6, N3 - 1:3 * N3 - 1] = T

    # Small elipse
    T = np.zeros((N6, N3))
    for i in range(1, N6 + 1):
        for j in range(1, N3 + 1):
            if (i / N6) ** 2 + (j / N3) ** 2 < 0.6:
                T[i - 1, j - 1] = 1

    T = np.block([np.fliplr(T), T])
    T = np.block([[np.flipud(T)], [T]])
    x[N6:3 * N6, N3 - 1:3 * N3 - 1] = x[N6:3 * N6, N3 - 1:3 * N3 - 1] + 2 * T
    x[x == 3] = 2 * np.ones((x[x == 3]).shape)

    T = np.triu(np.ones((N3, N3)))
    mT, nT = T.shape
    x[N3 + N12:N3 + N12 + nT, 1:mT + 1] = 3 * T

    T = np.zeros((2 * N6 + 1, 2 * N6 + 1))
    mT, nT = T.shape
    T[N6, :] = np.ones((1, nT))
    T[:, N6] = np.ones((mT))
    x[N2 + N12:N2 + N12 + mT, N2:N2 + nT] = 4 * T

    x = x[:N, :N].reshape(N ** 2, 1)
    b = A @ x

    return A, b, x


def fista(f, g, L, x0, eps, j):
    y = x0
    x_k = y
    x_k_m1 = y
    t_k = 1
    t_k_1 = 0
    fs = []
    gs = []
    ts = []
    # gval = g(y)
    gs.append(np.linalg.norm(g(y)))
    i = 1
    startime = time.time()
    ts.append(0)
    while i < j and gs[-1] > eps:
        x_k = y - (1 / L) * g(y)
        print('iter= {:2d} f(x)={:10.10f}'.format(i, f(x_k)))
        t_k_1 = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y = x_k + ((t_k - 1) / t_k_1) * (x_k - x_k_m1)
        x_k_m1 = x_k
        t_k=t_k_1
        fs.append(f(x_k))
        gs.append(np.linalg.norm(g(x_k)))
        i += 1
        ts.append(time.time() - startime)
    return x_k, fs, gs, ts


def main():
    A, b, x = blur(128, 5, 1)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(x.reshape(256, 256), cmap='gray')
    # plt.show()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(b.reshape(256, 256), cmap='gray')
    # plt.show()
    f = lambda x: np.linalg.norm(A @ x - b) ** 2
    g = lambda x: 2 * A.T @ (A @ x - b)
    x0 = np.zeros([128 * 128, 1])
    AtA = (A.T.dot(A)).toarray()
    vals, vecs = eigs(AtA, k=1)

    max_eign = np.real(vals[0])

    # for j in [1, 10, 100, 1000]:
    #     x_output1, f_s_output1, g_s_output1,ts_1 = gradient_method(f, g, exact_quad(A), x0, j)
    #     x_output2, f_s_output2, g_s_output2,ts_2 = gradient_method(f, g, const_step(1 / (2 * max_eign)), x0, j)
    #     x_fista, f_fista, g_fista, ts_fista = fista(f, g, 2 * max_eign, x0,10**-5, j)
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(x_fista.reshape(128, 128), cmap='gray')
    #     plt.show()

    # x_output1, f_s_output1, g_s_output1,ts_1 = gradient_method(f, g, exact_quad(A), x0, j)
    x_output2, f_s_output2, g_s_output2, ts_2 = gradient_method(f, g, const_step(1 / (2 * max_eign)), x0, 1000)
    x_fista, f_fista, g_fista, ts_fista = fista(f, g, 2 * max_eign, x0, 10 ** -5, 1000)
    # plt.title('Gradient descend - gf value per second')
    p1 = plt.semilogy(range(len(ts_2)), ts_2)
    p2 = plt.semilogy(range(len(ts_fista)), ts_fista)
    p3 = plt.semilogy(range(len(f_s_output2)), f_s_output2)
    p4 = plt.semilogy(range(len(f_fista)), f_fista)
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ( 'time fista','time const 2', 'fs method2', 'fs fista'))
    plt.show()


if __name__ == "__main__":
    main()
