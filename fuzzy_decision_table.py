#!/usr/bin/python
# -*- coding: utf-8 -*-

# 功能: 生成模糊决策表
# 作者: 王丰
# 时间: 2016-12-8
import numpy as np
from  sklearn.datasets import load_iris

class FuzzyDecisionTable(object):
    # table: input data set
    # table type: numpy.ndarray
    def __init__(self,table):
        self.table = table
    
    # 模糊相似关系
    # input: two real numbers
    def f(self,x,y):
        if abs(x-y)<0.1:
            return 1
        else:
            return 0
    # 选择一列数=数据，用f计算出一个模糊相似矩阵
    # col: 某一列，类型为integer
    # return type: np.ndarry
    def fuzzy_matrix(self,col):
        (rows,cols) = self.table.shape
        assert type(col) == int and col >=0 and col < cols
        data = self.table[:,col] # 第col列数据取出来
        # 两两计算f，存到一个矩阵中，矩阵大小为rows*rows
        f_matrix = np.identity(rows)
        for i in range(0,rows):
            for j in range(0,rows):
                if i != j:
                    f_matrix[i][j] = self.f(data[i],data[j])
        print "f_matrix = ", f_matrix
        print '\n'
        return np.multiply(f_matrix,f_matrix)


if __name__ == '__main__':
    iris = load_iris()
    fdt = FuzzyDecisionTable(iris.data[0:4])
    print fdt.f(1,0)
    print fdt.fuzzy_matrix(2)

