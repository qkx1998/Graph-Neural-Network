import pandas as pd
import networkx as nx
import itertools
import community

df = pd.read_csv('recruit_folder.csv')
df.head()

'''
RECRUIT_ID	PERSON_ID	LABEL
825081	    6256839	    0
772899	    5413605	    0
795668	    5219796	    0
769754	    5700693	    0
773645	    6208645	    0
'''

G = nx.Graph()

# 添加节点和边
G.add_edges_from(
    [(row['RECRUIT_ID'], row['PERSON_ID']) for idx, row in df.iterrows()])

# 添加节点和附带权重的边
# G.add_weighted_edges_from(
#     [(row['RECRUIT_ID'], row['PERSON_ID'], 1) for idx, row in df.iterrows()], 
#     weight='weight')

len(G.nodes()) == df['RECRUIT_ID'].nunique() + df['PERSON_ID'].nunique() # 返回True
