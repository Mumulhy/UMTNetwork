# -*- coding: utf-8 -*-
# UMT模型及曲线拟合实现

"""
Created on Fri Sept 16 10:17 2022

@author: _Mumu
Environment: py39
"""

import numpy as np
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from random import choice
from scipy.optimize import curve_fit
from tqdm import tqdm
from BallsSet import BallsSet

MAX_SIZE = 10 ** 9
BALLS_SET = BallsSet(MAX_SIZE)


class UMT:
    def __init__(self, M0, rho, nu):
        self.U = BALLS_SET.extract(M0)
        self.U_social = self.U.copy()
        self.u_social = M0
        self.S = Counter()
        self.D = 0
        self.rho = rho
        self.nu = nu
        self.reinforcement = []
        self.trigger = []
        self.social_trigger = []

    def discover(self):
        ball = choice(self.U_social)
        self.S[ball] += 1
        self.reinforcement += [ball] * self.rho
        if self.S[ball] == 1:
            self.trigger += BALLS_SET.extract(self.nu)
            self.D += 1
        return

    def step(self):
        self.U += self.reinforcement + self.trigger
        candidates = self.reinforcement + self.trigger + self.social_trigger
        self.U_social += candidates
        self.u_social += len(candidates)
        self.reinforcement.clear()
        self.trigger.clear()
        self.social_trigger.clear()
        return


class UMTNetwork:
    def __init__(self, M0s, rhos, nus, n, edges):
        self.n = n
        # self.A = [[0] * n for _ in range(n)]
        self.graph = defaultdict(set)
        for x, y in edges:
            # self.A[x][y] = 1
            self.graph[x].add(y)
        if isinstance(M0s, int):
            M0s = [M0s] * n
        if isinstance(rhos, int):
            rhos = [rhos] * n
        if isinstance(nus, int):
            nus = [nus] * n
        self.UMTs = [UMT(M0, rho, nu) for M0, rho, nu in zip(M0s, rhos, nus)]
        for i in self.graph.keys():
            for j in self.graph[i]:
                self.UMTs[i].U_social += self.UMTs[j].U
                self.UMTs[i].u_social += M0s[j]
        self.dynamics = [[] for _ in range(n)]

    def discover(self):
        for i in range(self.n):
            self.UMTs[i].discover()
        return

    def step(self):
        for i in self.graph.keys():
            for j in self.graph[i]:
                self.UMTs[i].social_trigger += self.UMTs[j].trigger
        for i in range(self.n):
            self.UMTs[i].step()
            self.dynamics[i].append(self.UMTs[i].D)
        return

    def print_state(self):
        for umt in self.UMTs:
            print(umt.D, umt.S, umt.U, umt.U_social)
        return

    def print_dynamics(self):
        for i, dynamic in enumerate(self.dynamics):
            print(i, dynamic)
        return


class CurveFitter:
    def __init__(self, rhos, nus, n, t0=1, with_bias=True):
        self.n = n
        self.t0 = t0
        if isinstance(rhos, int):
            rhos = [rhos] * n
        if isinstance(nus, int):
            nus = [nus] * n
        self.with_bias = with_bias
        if with_bias:
            self.funcs_to_fit = [lambda t, bias, *args: bias + sum(args[k] * np.log(t) ** k
                                                                   for k in range(n)) * t ** (nus[i] / rhos[i])
                                 for i in range(n)]
            self.funcs_to_fit_log = [lambda t, bias, *args: bias + sum(args[k] * t ** k
                                                                       for k in range(n)) * np.exp(t * nus[i] / rhos[i])
                                     for i in range(n)]
        else:
            self.funcs_to_fit = [lambda t, *args: sum(args[k] * np.log(t) ** k
                                                      for k in range(n)) * t ** (nus[i] / rhos[i])
                                 for i in range(n)]
            self.funcs_to_fit_log = [lambda t, *args: sum(args[k] * t ** k
                                                          for k in range(n)) * np.exp(t * nus[i] / rhos[i])
                                     for i in range(n)]

    def fit(self, dynamics):
        t = list(range(self.t0, len(dynamics[0]) + 1))
        params = []
        for func_to_fit, dynamic in zip(self.funcs_to_fit, dynamics):
            if self.with_bias:
                param, _ = curve_fit(func_to_fit, t, dynamic[self.t0 - 1:], [0] + [1] * self.n)
            else:
                param, _ = curve_fit(func_to_fit, t, dynamic[self.t0 - 1:], [1] * self.n)
            params.append(param)
        return params

    def fit_log(self, dynamics):
        t = np.log(np.arange(self.t0, len(dynamics[0]) + 1))
        params = []
        for func_to_fit, dynamic in zip(self.funcs_to_fit, dynamics):
            if self.with_bias:
                param, _ = curve_fit(func_to_fit, t, dynamic[self.t0 - 1:], [0] + [1] * self.n)
            else:
                param, _ = curve_fit(func_to_fit, t, dynamic[self.t0 - 1:], [1] * self.n)
            params.append(param)
        return params


if __name__ == '__main__':
    M0s = 500
    rhos = 50
    nus = 2
    n = 1
    edges = []
    runs = 50
    t0 = 100000
    T = 1000000

    avg_dynamics = None
    p_bar = tqdm(range(runs))
    p_bar.set_description('Simulating runs: ')
    for _ in p_bar:
        net = UMTNetwork(M0s, rhos, nus, n, edges)
        for _ in range(T):
            net.discover()
            net.step()
        if avg_dynamics is None:
            avg_dynamics = net.dynamics
        else:
            for i in range(n):
                for j in range(T):
                    avg_dynamics[i][j] += net.dynamics[i][j]
        # net.print_dynamics()
    for i in range(n):
        for j in range(T):
            avg_dynamics[i][j] /= runs
    print('Fitting...')
    curve_fitter_bias = CurveFitter(rhos, nus, n, t0)
    curve_fitter_no_bias = CurveFitter(rhos, nus, n, t0, with_bias=False)
    params_bias = curve_fitter_bias.fit(avg_dynamics)
    # params_no_bias = curve_fitter_no_bias.fit(avg_dynamics)
    # params_bias_log = curve_fitter_bias.fit_log(avg_dynamics)
    # params_no_bias_log = curve_fitter_no_bias.fit_log(avg_dynamics)
    print(params_bias)
    # print(params_no_bias)
    # print(params_bias_log)
    # print(params_no_bias_log)
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # y_true = avg_dynamics[0][t0:T]
    # x = np.arange(t0, T)
    # y_bias = curve_fitter_bias.funcs_to_fit[0](x, *params_bias[0])
    # y_no_bias = curve_fitter_no_bias.funcs_to_fit[0](x, *params_no_bias[0])
    # plt.plot(x, y_true, '.k', x, y_bias, '-r', x, y_no_bias, '--g')
    # plt.xlabel('t')
    # plt.ylabel('D_0(t)')
    # plt.subplot(2, 1, 2)
    # x_log = np.log(np.arange(t0, T))
    # y_bias_log = curve_fitter_bias.funcs_to_fit_log[0](x_log, *params_bias_log[0])
    # y_no_bias_log = curve_fitter_no_bias.funcs_to_fit_log[0](x_log, *params_no_bias_log[0])
    # plt.plot(x_log, y_true, '.k', x_log, y_bias_log, '-r', x_log, y_no_bias_log, '--g')
    # plt.xlabel('ln(t)')
    # plt.ylabel('D_0(t)')
    # plt.show()
    plt.figure(figsize=(10, 6))
    y_true = avg_dynamics[0][t0:T]
    x = np.arange(t0, T)
    y_bias = curve_fitter_bias.funcs_to_fit[0](x, *params_bias[0])
    plt.plot(x, y_true, '.k', x, y_bias, '--r')
    plt.xlabel('t')
    plt.ylabel('D_0(t)')
    plt.legend(['y_true', 'y_fit'])
    plt.grid()
    plt.show()
