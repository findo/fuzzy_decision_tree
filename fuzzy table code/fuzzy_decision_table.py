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


    # 一个矩阵的最大值和最小值
    def get_matrix_max_and_min(self, M):
        MIN = 1e9
        MAX = -MIN
        (r,c) = M.shape
        for i in range(r):
            for j in range(c):
                if M[i][j] == 1:
                    continue
                else:
                    if MIN > M[i][j]:
                        MIN = M[i][j]
                    elif M[i][j] > MAX:
                        MAX = M[i][j]
        return MAX, MIN
    # 模糊相似关系函数
    # input: real numbers
    def f(self,x,y,x_max,x_min):
        assert x_max > x_min
        return 1 - abs(x-y)/(x_max - x_min)


    # 关系合成运算: A◦B
    def get_relation_matrix(self,A,B):
        # A: n*n B: n*n
        assert A.shape[0] == A.shape[1] and A.shape[0] == B.shape[0] and A.shape[0] == B.shape[1]
        n = A.shape[0]
        ret = np.ndarray(shape=(n,n), dtype=float)
        # ret[i][j] = max {min{A[i][k],B[k][j]}} (k: 0 -> n-1)
        for i in range(n):
            for j in range(n):
                s = []
                for k in range(n):
                    s.append(min(A[i][k], B[k][j]))
                #ret[i][j] = max([min(A[i][k], B[k][j]) for k in range(n)])
                ret[i][j] = max(s)
        return ret
    # 计算 A^n
    def A_n(self, A, n):
        assert n >= 1
        temp = A
        for i in range(n-1):
            A = self.get_relation_matrix(temp,A)
        return A

    # 两个矩阵相同
    def is_same(self,A,B):
        assert A.shape[0] == A.shape[1] and A.shape[0] == B.shape[0] and A.shape[0] == B.shape[1]
        n = A.shape[0]

        for i in range(n):
            for j in range(n):
                if abs(A[i][j] - B[i][j]) > 0.000001:
                    return False
        return True




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
        #print "模糊相似矩阵 = "
        #print ret
        A = ret 
        # 如果A^2k == A^k，则A^k就是我们想要的结果
        k = 1
        store = {} # 把A^k存下来
        
        A2 = self.A_n(A,2) # A^2
        left_2k = A2

        right_k = self.A_n(A,1)

        while(self.is_same(left_2k,right_k) == False):
            store[k] = right_k # = A^k

            # 计算 : A^(k+1) ◦ A^(k+1) 
            # 下一个
            left_2k = self.get_relation_matrix(self.get_relation_matrix(store[k], store[k]), A2)
            right_k = self.get_relation_matrix(store[k],A)
            k = k+1

            print "k = ",k

        return right_k

    # 返回自乘很多次之前的模糊相似矩阵
    def get_fuzzy_similay_matrix(self,index):
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
                    
        return ret


    # 根据模糊相似矩阵进行聚类，返回聚类后的集合
    # 针对的是特定的属性
    def get_set(self,index):
        fuzzy_matrix = self.get_fuzzy_matrix(index)
        # print "fuzzy_matrix %d " % index
        # print fuzzy_matrix
        assert fuzzy_matrix.shape[0] == fuzzy_matrix.shape[1]
        r = fuzzy_matrix.shape[0]
        sets = [] # 分类集合
        set_matrix = np.ndarray(shape=fuzzy_matrix.shape,dtype=int)
        
        for i in range(r):
            for j in range(r):
                if i ==j:
                    set_matrix[i][j] = 1
                elif fuzzy_matrix[i][j] < self.threshold[index]:
                    set_matrix[i][j] = 0
                else:
                    set_matrix[i][j] = 1

        # set_matrix中，值为1说明属于一个集合,否则属于不同集合
        # print "set matrix = ",set_matrix
        # 把set_matrix看成一个邻接矩阵，DFS就可以得到的联通分量的个数，也就是集合的个数
        # 以上得到两两的集合，下面合并集合
        # set_num 集合的个数
        # sets 聚类后的集合
        file_object = open('fuzzy_matrix.txt','w')
       # print "set matrix = "
        #print set_matrix
        file_object.write(str(fuzzy_matrix.tolist()))
        #print "set matrix shape = ",set_matrix.shape
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

    # 计算每个属性的隶属度
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
        fuzzy_similary_matrix = []
        ret = []
        mid_points = []
        for index in range(c):
            #print self.some_li_shu_du(index)
            #print "index = ",index
            mid_points.append(self.get_mid_points(index))
            print "属性%d的聚类个数 = %d" % (index, len(self.get_set(index)))
            ret.append(self.some_li_shu_du(index))
            fuzzy_similary_matrix.append(self.get_fuzzy_similay_matrix(index))
        return fuzzy_similary_matrix, ret, mid_points

if __name__ == '__main__':
    iris = load_iris()
    #fdt = FuzzyDecisionTable(iris.data[0:100],threshold=0.8)
    fdt = FuzzyDecisionTable(np.concatenate((iris.data[0:25],iris.data[50:75])),threshold=[0.95, 0.91, 0.91, 0.86])
    # print fdt.some_li_shu_du(0)
    # print fdt.get_fuzzy_matrix(0)
    (fuzzy_similary_matrix, final_table, mid_points) = fdt.run()

    fuzzy_similary_matrix_out = open('fuzzy_similary_matrix.txt','w')
    fuzzy_similary_matrix_out.write(str(fuzzy_similary_matrix))

    final_table_out = open('final_table.txt','w')
    final_table_out.write(str(final_table))

    print final_table
    #print fuzzy_similary_matrix
    # final_table 是一个list，里面有若干个np.ndarray()类型的矩阵
    # 即每个属性下的相似关系矩阵
    # r1 = [1, 0.88, 0.49, 0.88, 0.30, 0.24, 0.20, 0.93, 0.77]
    # r2 = [0.88, 1, 0.38, 0.94, 0.06, 0.05, 0.01, 0.95, 0.93]
    # r3 = [0.49, 0.38, 1, 0.67, 0.76, 0.80, 0.71, 0.45, 0.55]
    # r4 = [0.88, 0.94, 0.67, 1, 0.30, 0.30, 0.24, 0.92, 0.95]
    # r5 = [0.30, 0.06, 0.76, 0.30, 1, 0.99, 0.98, 0.21, 0.21]
    # r6 = [0.24, 0.05, 0.80, 0.30, 0.99, 1, 0.99, 0.18, 0.23]
    # r7 = [0.20, 0.01, 0.71, 0.24, 0.98, 0.99, 1, 0.14, 0.19]
    # r8 = [0.93, 0.95, 0.45, 0.92, 0.21, 0.18, 0.14, 1, 0.90]
    # r9 = [0.77, 0.93, 0.55, 0.95, 0.21, 0.23, 0.19, 0.90, 1]
    # r = [r1,r2,r3,r4,r5,r6,r7,r8,r9]
    # print "A = "
    # A = np.array(r)
    # # print A
    # # print '\n'
    # # A2 = fdt.get_relation_matrix(A,A)
    # # A3 = fdt.get_relation_matrix(A,A2)
    # # A4 = fdt.get_relation_matrix(A2,A2)
    # # print A4
    # # print fdt.A_n(A,4)
    # # print "is A8 == A4 ? ",
    # # print fdt.is_same(fdt.A_n(A,4), fdt.A_n(A,8))
    # k = 1
    # while(fdt.is_same(fdt.A_n(A,k*2), fdt.A_n(A,k)) == False):
    #     k = k+1
    # print "k = ",k
    # print fdt.A_n(A,k)
    # print "A4 = "
    # print fdt.A_n(A,4)
    # 找第一个属性的threshold

    # 0.96 : 4
    # 0.95 : 4
    # 0.93 : 4
    # 0.9 :  1
    # 0.92 : 1
    # index = 0
    # fdt.threshold = 0.951233123
    # (MAX, MIN) = fdt.get_matrix_max_and_min(fdt.get_fuzzy_matrix(index))
    # print "属性%d的聚类个数 = %d" % (index, len(fdt.get_set(index)))
    # index = 1
    # fdt.threshold = 0.91
    # print "属性%d的聚类个数 = %d" % (index, len(fdt.get_set(index)))
    # index = 2
    # fdt.threshold = 0.91
    # print "属性%d的聚类个数 = %d" % (index, len(fdt.get_set(index)))
    # index = 3
    # fdt.threshold = 0.86
    # print "属性%d的聚类个数 = %d" % (index, len(fdt.get_set(index)))

