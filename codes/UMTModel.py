# -*- coding: utf-8 -*-
# UMT模型及曲线拟合实现

"""
Created on Fri Sept 16 10:17 2022

@author: _Mumu
Environment: py39
"""

import numpy as np
from collections import Counter, defaultdict
from random import choice
from scipy.optimize import curve_fit
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


class CurveFit:
    def __init__(self, rhos, nus, n):
        self.n = n
        if isinstance(rhos, int):
            rhos = [rhos] * n
        if isinstance(nus, int):
            nus = [nus] * n
        self.funcs_to_fit = [lambda t, bias, *args: bias + sum(args[k] * np.log(t) ** k
                                                               for k in range(n)) * t ** (nus[i] / rhos[i])
                             for i in range(n)]

    def fit(self, dynamics, t=None):
        if t is None:
            t = list(range(1, len(dynamics[0]) + 1))
        params = []
        for func_to_fit, dynamic in zip(self.funcs_to_fit, dynamics):
            param, _ = curve_fit(func_to_fit, t, dynamic, [1] * (self.n + 1))
            params.append(param)
        return params


if __name__ == '__main__':
    # umt = UMT(10, 2, 2)
    # print(umt.U)
    # print(umt.U_social)
    # print(umt.S)
    # print(umt.D)
    # for _ in range(5):
    #     umt.discover()
    #     umt.step()
    #     print(umt.U)
    #     print(umt.U_social)
    #     print(umt.S)
    #     print(umt.D)
    net = UMTNetwork(500, 50, 2, 2, [[0, 1]])
    for _ in range(10000):
        net.discover()
        net.step()
    net.print_dynamics()
