#!/usr/bin/python
# -*- coding: utf-8 -*-

# 功能: 生成模糊决策表
# 作者: 王丰
# 时间: 2016-12-8
import numpy as np
from  sklearn.datasets import load_iris
from math import sqrt
from DFS import dfs
class FuzzyDecisionTable(object):
    # table: input data set
    # table type: numpy.ndarray
    def __init__(self,table):
        self.table = table
    
    # 模糊相似关系
    # input: two real vectors
    # x,y: ndarry
    def f(self,x,y):
        assert x.shape[0] == y.shape[0]
        nums  = x.shape[0] # 包含元素个数
        print "x = ",x
        print "y = ",y
        print x*y
        print "sum = ",sum(x*y)
        numerator = sum(x*y)
        sum_x = 0 # x 平方和
        sum_y = 0 # y 平方和 
        for i in range(0,nums):
            
            sum_x += x[i]**2
            sum_y += y[i]**2
        length_of_x = sqrt(sum_x)
        length_of_y = sqrt(sum_y)
        denomimator = length_of_y*length_of_x
        return numerator/denomimator

    # 用f计算出一个模糊相似矩阵
    def get_fuzzy_matrix(self):
        (rows,cols) = self.table.shape
        fuzzy_matrix = np.identity(rows)
        # 向量两两计算f
        for i in range(0,rows):
            for j in range(0,rows):
                if i != j:
                    fuzzy_matrix[i][j] = self.f(self.table[i],self.table[j])
        temp_fuzzy_matrix = fuzzy_matrix
        print fuzzy_matrix
        print '\n'
        for _ in range(20-1):
            fuzzy_matrix = np.dot(temp_fuzzy_matrix,fuzzy_matrix)
        return fuzzy_matrix

    # 根据模糊相似矩阵进行聚类，返回聚类后的矩阵
    def get_set(self,threshold):
        fuzzy_matrix = self.get_fuzzy_matrix()
        assert fuzzy_matrix.shape[0] == fuzzy_matrix.shape[1]
        r = fuzzy_matrix.shape[0]
        sets = [] # 分类集合
        set_matrix = np.dnarray(shape=fuzzy_matrix.shape,dtype=int)
        
        for i in range(r):
            for j in range(r):
                if i ==j:
                    set_matrix[i][j] = 1
                elif fuzzy_matrix[i][j] < threshold:
                    set_matrix[i][j] = 0
                else:
                    fuzzy_matrix[i][j] = 1

        # set_matrix中，值为1说明属于一个集合,否则属于不同集合
        print "set matrix = ",set_matrix
        # 把set_matrix看成一个邻接矩阵，DFS就可以得到的联通分量的个数，也就是集合的个数
        # 以上得到两两的集合，下面合并集合
        # set_num 集合的个数
        # sets 聚类后的集合
        sets = dfs(set_matrix)
        return sets

    # some_set: integer set
    def get_mid_point(self,some_set):
        MIN = 1e9
        n =  len(some_set)
        for i in range(0,n):
            temp = 0
            # 计算some_set[i]和剩余的点的模糊相似关系的和
            for j in range(0,n):
                if i != j:
                    temp += self.f(self.table[some_set[i]],self.table[some_set[j]])
                
            if temp < MIN:
                MIN = temp
                min_point = some_set[i]
        return min_point

    # 生成每个集合的中心点
    def get_mid_points(self):
        sets = self.get_set()
        self.sets = sets
        n = len(sets) # n 个聚类
        ret = [] # store the indexs of data 
        for s in sets:
            ret.append(self.get_mid_point(s))
        return ret
    # 计算每项与聚类的得到的集合的隶属度
    # 返回类型为[[],[],[]] def li_shu_du(self):
        mid_points = self.get_mid_points()
        num_of_sets = len(mid_points) # 集合个数
        ret = []
        for i in range(self.table.shape[0]):
            # i is the index
            # 分别计算隶属度
            li_shu_du_temp = []
            for index in range(num_of_sets):
                if i == mid_points[index]:
                    # 若i是中心\
                    li_shu_du_temp[index] = 1
                else:
                    li_shu_du_temp[index] = self.f(self.table[mid_points[index]],self.table[i])
            ret.append(li_shu_du_temp)
        return ret



if __name__ == '__main__':
    iris = load_iris()
    fdt = FuzzyDecisionTable(iris.data[0:4])
    print fdt.get_fuzzy_matrix()

