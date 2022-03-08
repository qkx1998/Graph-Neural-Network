import pandas as pd
import csv
from py2neo import Graph
import py2neo
py2neo.__version__ # 2021.2.3

df = pd.read_csv('bs140513_032310.csv')

'''
py2neo的安装需要java和neo4j的支持,二者的安装参考自：
https://blog.csdn.net/qq_40642546/article/details/107401304
https://blog.csdn.net/qq_35902726/article/details/118890218
'''
graph = Graph("neo4j://localhost:7687", auth = ('neo4j', 'Qkx7592759#'))
graph.delete_all() # 清除neo4j中原有的结点等所有信息


'''
将本地的csv文件写入neo4j 
参考自：
https://blog.csdn.net/qq_54531415/article/details/116176421
https://blog.csdn.net/weixin_43788143/article/details/108388520

打开neo4j交互网站的方式：http://127.0.0.1:7474/browser/  复制到浏览器
'''
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

with open('bs140513_032310.csv','r',encoding='gbk') as f:
    reader=csv.reader(f)
    for item in tqdm(reader):
        if reader.line_num == 1:
            continue
            
        #print("当前行数：",reader.line_num,"当前内容：",item)
        customer_node = Node("customer",name=item[0])
        merchant_node = Node("merchant",name=item[4])
        graph.create(customer_node)
        graph.create(merchant_node)
        
        relation = Relationship(customer_node, merchant_node)
        graph.create(relation)
        
def add_degree(x):
    return valueDict[x]['degree']

def add_community(x):
    return str(valueDict[x]['community'])

def add_pagerank(x):
    return valueDict[x]['pagerank']

query = """
        MATCH (n)
        RETURN n.name AS name, n.degree AS degree, n.pagerank as pagerank, n.community AS community 
        """
        
data = graph.run(query) #不知道为什么这里查询到的除了name都是Null。 要么是导入数据出了错，要么是查询方式出了错。

valueDict = {}
for d in data:
    valueDict[d['name']] = {'degree': d['degree'], 'pagerank': d['pagerank'], 'community': d['community']}

df['merchDegree'] = df.merchant.apply(add_degree)
df['custDegree'] = df.customer.apply(add_degree)
df['custPageRank'] = df.customer.apply(add_pagerank)
df['merchPageRank'] = df.merchant.apply(add_pagerank)
df['merchCommunity'] = df.merchant.apply(add_community)
df['custCommunity'] = df.customer.apply(add_community)


# 另一种构图方式
# import networkx as nx
# G = nx.Graph(df[["customer", "merchant"]].values.tolist(), directed=False, weighted=False)

# 根据客户和商家构建二部图。返回degree, pagerank
# degree_dict = dict(G.degree())
# pagerank_dict = dict(nx.pagerank(G)) #但是这里的pagerank和上面的代码运算的结果不一致，暂不清楚原因。

# df['merchDegree'] = df['merchant'].map(degree_dict)
# df['custDegree'] = df['customer'].map(degree_dict)
# df['merchPageRank'] = df['merchant'].map(pagerank_dict) 
# df['custPageRank'] = df['customer'].map(pagerank_dict)

# df = df.drop(['customer', 'zipcodeOri', 'zipMerchant'], axis = 1)


df['age'] = df['age'].apply(lambda x: x[1]).replace('U', 7).astype(int)
df['gender'] = df['gender'].apply(lambda x: x[1])
df['merchant'] = df['merchant'].apply(lambda x: x[1:-1])
df['category'] = df['category'].apply(lambda x: x[1:-1])
df = df.sample(frac = 1).reset_index(drop = True)


features = df.drop('fraud', axis = 1)
label = df.fraud
features = features[['amount', 'age', 'gender', 'merchant', 'category', 'merchDegree', 'custDegree', 'merchPageRank', 'custPageRank']]
features = pd.get_dummies(features, columns = ['age', 'gender', 'merchant', 'category'])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size = 0.8, random_state = 42, stratify = label)


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
x_train['amount'] = min_max_scaler.fit_transform(x_train['amount'].values.reshape(-1, 1))
x_test['amount'] = min_max_scaler.fit_transform(x_test['amount'].values.reshape(-1, 1))


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_lr = LogisticRegression(C = 1, class_weight = {1: 0.81, 0: 0.1}, penalty = 'l1', solver = 'liblinear')
model_dt = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
model_df = RandomForestClassifier(n_estimators = 15, max_depth = 19, criterion = "entropy", random_state = 0, min_samples_split = 20)

models = [model_lr, model_dt, model_df]
from sklearn import metrics

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('----------------------------------')
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))
    
    
