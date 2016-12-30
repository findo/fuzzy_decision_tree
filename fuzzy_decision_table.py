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
    def __init__(self,table,threshold):
        self.table = table
        self.threshold = threshold
    
    # 某个属性下的最大值和最小值
    def get_max_and_min(self,index):
        data = self.table[:,index]
        return max(data),min(data)
    # 模糊相似关系函数
    # input: real numbers
    def f(self,x,y,x_max,x_min):
        assert x_max > x_min
        return 1 - abs(x-y)/(x_max - x_min)


    # 计算某个属性下的模糊相似矩阵
    # input: 属性的index
    def get_fuzzy_matrix(self,index):
        data = self.table[:,index] # 第index列数据
        (x_max,x_min) = self.get_max_and_min(index)
        num = data.shape[0]
        # 两两计算相似关系
        # ret: 最后的相似矩阵
        ret = np.ndarray(shape=(num,num),dtype=float)
        for i in range(num):
            for j in range(num):
                if i == j:
                    ret[i][j] = 1
                else:
                    ret[i][j] = self.f(data[i],data[j],x_max,x_min)
                    
        temp = ret
        #print '相似关系矩阵 = '
        #print ret
        #print '\n'
        # 自乘15次
        for i in range(8-1): 
            ret = np.dot(ret,temp)
        return ret


    
    # def get_fuzzy_matrix(self):
    #     (rows,cols) = self.table.shape
    #     fuzzy_matrix = np.identity(rows)
    #     # 向量两两计算f
    #     for i in range(0,rows):
    #         for j in range(0,rows):
    #             if i != j:
    #                 fuzzy_matrix[i][j] = self.f(self.table[i],self.table[j])
    #     temp_fuzzy_matrix = fuzzy_matrix
    #     print fuzzy_matrix
    #     print '\n'
    #     for _ in range(20-1):
    #         fuzzy_matrix = np.dot(temp_fuzzy_matrix,fuzzy_matrix)
    #     return fuzzy_matrix


    # 根据模糊相似矩阵进行聚类，返回聚类后的集合
    # 针对的是特定的属性
    def get_set(self,index):
        fuzzy_matrix = self.get_fuzzy_matrix(index)
        #print "fuzzy_matrix %d " % index
        # print fuzzy_matrix
        assert fuzzy_matrix.shape[0] == fuzzy_matrix.shape[1]
        r = fuzzy_matrix.shape[0]
        sets = [] # 分类集合
        set_matrix = np.ndarray(shape=fuzzy_matrix.shape,dtype=int)
        
        for i in range(r):
            for j in range(r):
                if i ==j:
                    set_matrix[i][j] = 1
                elif fuzzy_matrix[i][j] < self.threshold:
                    set_matrix[i][j] = 0
                else:
                    set_matrix[i][j] = 1

        # set_matrix中，值为1说明属于一个集合,否则属于不同集合
        # print "set matrix = ",set_matrix
        # 把set_matrix看成一个邻接矩阵，DFS就可以得到的联通分量的个数，也就是集合的个数
        # 以上得到两两的集合，下面合并集合
        # set_num 集合的个数
        # sets 聚类后的集合
        sets = dfs(set_matrix)
        return sets

    # some_set: integer set
    # some_set里面存的是数据的下标 （get_set会返回这个）
    # index是some_set对应的属性下标
    # return: 返回的也是下标
    def get_mid_point(self,some_set,index):
        (x_max,x_min) = self.get_max_and_min(index)
        MIN = 1e9
        n =  len(some_set)
        assert n > 0
        if n == 1:
            return some_set[0]

        for i in range(0,n):
            sum_i = 0.0
            # 计算some_set[i]和剩余的数据的模糊相似关系的和
            for j in range(0,n):
                if i != j:
                    xi = self.table[:,index][some_set[i]]
                    xj = self.table[:,index][some_set[j]]
                    # print self.f(xi,xj,x_max,x_min)
                    sum_i += self.f(xi,xj,x_max,x_min)
            # print "sum_i = ",sum_i
            # assert type(sum_i) == float
            if sum_i < MIN:
                MIN = sum_i
                min_point = some_set[i]

        return min_point

    # 生成某个属性下的中心点
    def get_mid_points(self, index):
        sets = self.get_set(index)
        # print "index = %d" % index
        # print "sets = ",sets
        n = len(sets) # n 个聚类
        ret = [] # store the indexs of data 
        for s in sets:
            ret.append(self.get_mid_point(s,index))
        return ret

    # 计算每个属性小的隶属度
    # 返回类型为[[],[],[],...] 
    def some_li_shu_du(self, index):
        mid_points = self.get_mid_points(index)
        # print "%d ->" % index,
        # print "mid points = ",mid_points
        # 聚类的个数n
        num_set = len(mid_points)
        (x_max, x_min) = self.get_max_and_min(index)
        data = self.table[:,index]
        #print "data = "
        #print data
        # 数据的个数
        num_data = len(data)
        ret = np.ndarray(shape=(num_data,num_set),dtype=float)

        for i in range(num_data):
            # data[i]
            for j in range(num_set):
                # 如果data[i]是聚类的中心，则为1
                if i == mid_points[j]:
                    ret[i][j] = 1
                else:
                    ret[i][j] = self.f(data[i], data[mid_points[j]], x_max, x_min)
                    # print "ret "+str(i)+str(j)+" = ",ret[i][j]
        return ret

    # 返回所有属性的隶属度
    def run(self):
        (_,c) = self.table.shape
        ret = []

        for index in range(c):
            #print self.some_li_shu_du(index)
            #print "index = ",index
            ret.append(self.some_li_shu_du(index))

        return ret 

if __name__ == '__main__':
    iris = load_iris()
    fdt = FuzzyDecisionTable(iris.data[0:7],threshold=170000)
    # print fdt.some_li_shu_du(0)
    print fdt.run()

