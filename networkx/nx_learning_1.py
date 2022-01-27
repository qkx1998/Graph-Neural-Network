import networkx as nx
from matplotlib import pyplot as plt

G = nx.Graph() 

# 一次添加一个节点
G.add_node('A') 
# 一次添加多个节点
G.add_nodes_from(['B','C']) 

# 一次添加一条边
G.add_edge('A','B') 
# 一次添加多条边
G.add_edges_from([('B','C'),('A','C')]) 
# 在不存在的节点基础上添加多条边，同时创建节点
G.add_edges_from([('B', 'D'), ('C', 'E')])

plt.figure() 
nx.draw_networkx(G)
plt.show()

# 获取节点 和 边的属性
list(G.nodes)
list(G.edges)

# 判断节点或边是否存在
uid = 'A'
uid in G 
G.has_node(uid) 

pid = 'C'
(uid,pid) in G.edges #True
G.has_edge(uid,pid) #True

# 获取节点的邻居
list(G.neighbors(uid))


# 空手道俱乐部数据
G = nx.karate_club_graph() 

# 为节点和边添加属性， 遍历节点，添加club属性
member_club = [
              0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
              0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
              1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1]
for node_id in G.nodes:
    G.nodes[node_id]["club"] = member_club[node_id]
    
# 定制节点的颜色
node_color = ['#1f78b4' if G.nodes[v]["club"] == 0 else '#33a02c' for v in G]
nx.draw_networkx(G, label=True, node_color=node_color)

# 遍历边
for v, w in G.edges:

    # 判断两个节点的club属性是否相同
    if G.nodes[v]["club"] == G.nodes[w]["club"]: 
        
        # 为边添加 internal 属性
        G.edges[v, w]["internal"] = True
    else:
        G.edges[v, w]["internal"] = False
        
# 存在联系的数组
internal = [e for e in G.edges if G.edges[e]["internal"]] 
# 不存在联系的数组
external = [e for e in G.edges if not G.edges[e]["internal"]]

karate_pos = nx.spring_layout(G,k = 0.3) 

# 多个线样式，需要绘制多次
# 绘制节点和标签  
nx.draw_networkx_nodes(G, karate_pos, node_color=node_color)
nx.draw_networkx_labels(G, karate_pos)
# 内边 绘制成实线
nx.draw_networkx_edges(G, karate_pos, edgelist=internal)
# 外边 绘制成虚线
nx.draw_networkx_edges(G, karate_pos,edgelist=external, style="dashed")

# 为边增加权重
# 计算边权重的函数
def tie_strength(G, v, w):
    v_neighbors = set(G.neighbors(v))
    w_neighbors = set(G.neighbors(w))
    # 两个节点的邻居节点交集大小
    return 1 + len(v_neighbors & w_neighbors) # 交集大小

# 遍历每条边，计算权重
for v, w in G.edges: 
    G.edges[v, w]["weight"] = tie_strength(G, v, w)
    
# 吧权重存储在列表里
edge_weights = [G.edges[v, w]["weight"] for v, w in G.edges]

# 将边权值传递给spring_layout()，将强连接节点推得更近
weighted_pos = nx.spring_layout(G, pos=karate_pos, k=0.3, weight="weight")

nx.draw_networkx(
    G, weighted_pos, width=8, node_color=node_color,
    edge_color=edge_weights, edge_vmin=0, edge_vmax=6, edge_cmap=plt.cm.Blues)
nx.draw_networkx_edges(G, weighted_pos, edgelist=internal, edge_color="gray")
nx.draw_networkx_edges(G, weighted_pos, edgelist=external, edge_color="gray", style="dashed")

# 创建有向图
G2 = nx.DiGraph()
G2.add_node('A') 

# 创建由B指向C的边
G2.add_nodes_from(['B','C']) 

G2.add_edge('A','B') 
G2.add_edges_from([('B','C'),('A','C')]) 
G2.add_edges_from([('B', 'D'), ('C', 'E')])

plt.figure() 
nx.draw_networkx(G2)
plt.show()
