# -*- coding: utf-8 -*-
# 返回连通分量的个数
# G: np.ndarray
import numpy as np
# 从s出发dfs
visited = [0 for i in range(2000)]
def dfs_once(G,s,visited_nodes):
    n = G.shape[0] # 顶点个数
    assert G.shape[0] == G.shape[1] and type(s) == int and s >=0 and s < n
    visited[s] = 1 # 当前节点访问结束
    visited_nodes.append(s)
    #print s
    for i in range(n):
        if G[i][s] == 1 and visited[i] == 0:
            dfs_once(G,i,visited_nodes)
    return
    
def dfs(G):
    #count = 0
    n = G.shape[0] # 顶点个数
    connected_component = []
    assert G.shape[0] == G.shape[1]
    for i in range(n):
        if visited[i] == 0:
            visited_nodes = []
            dfs_once(G,i,visited_nodes)
            #count += 1
            connected_component.append(visited_nodes)
            #print visited_nodes
    return connected_component
if __name__ == '__main__':
    G = np.array([[1,1,1,0],[1,1,0,0],[1,0,1,0],[0,0,0,1]])
    #dfs_once(G,0)
    num,cc = dfs(G)
    print num

    
    print cc