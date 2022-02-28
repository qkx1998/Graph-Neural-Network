'''
将招聘者，求职者的投递数据从 dataframe 转化为 Graph，并调用 louvain 社区发现算法进行模拟
'''
import pandas as pd
import networkx as nx
import itertools
import community

df = pd.read_csv('recruit_folder.csv')

'''
RECRUIT_ID	PERSON_ID	 LABEL
 825081	    6256839	    0
 772899	    5413605	    0
 795668	    5219796	    0
 769754	    5700693	    0
 773645	    6208645	    0
'''

G = nx.Graph()

'''
注意：
下面创建的图所有的节点都是 PERSON_ID，不同的 PERSON_ID 之间通过有过相同的 RECRUIT_ID 投递记录从而建立“边”。
如果两个人有过多次的相同投递记录，则权重乘 2。
'''
for p, a in df.groupby('RECRUIT_ID')['PERSON_ID']: 
    '''
    可以创建一个迭代器，返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序。例如：
    list1 = [1, 3, 4, 5]
    list2 = list(itertools.combinations(list1, 2))
    返回的list2为：[(1, 3), (1, 4), (1, 5), (3, 4), (3, 5), (4, 5)]
    '''
    for u, v in itertools.combinations(a, 2):
        if G.has_edge(u, v):
            G[u][v]['weight'] *= 2
        else:
            G.add_edge(u, v, weight=1)
            
print('\nSize of graph, i.e. number of edges:', G.size())

partition = community.best_partition(G)
print('Modularity: ', community.modularity(partition, G))
