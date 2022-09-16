# -*- coding: utf-8 -*-
# 实现全体小球集合

"""
Created on Fri Sept 16 10:17 2022

@author: _Mumu
Environment: py39
"""

from random import randint


class BallsSet:
    def __init__(self, num):
        self.map = {}
        self.left = num
        self.ori = num

    def extract(self, num=1):
        if num == 0:
            return []
        if self.left == 0:
            raise ValueError('Can\'t extract more balls!')
        idx = randint(1, self.left)
        ball = self.map.get(idx, idx)
        self.map[idx] = self.map.get(self.left, self.left)
        self.left -= 1
        return [ball] + self.extract(num - 1)

    def reset(self):
        self.map.clear()
        self.left = self.ori
        return


if __name__ == '__main__':
    b = BallsSet(5)
    print(b.extract(5))
