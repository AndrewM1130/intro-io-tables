import pandas as pd

# FIXME: this isn't working right now

class IO:
    def __init__(self):
        self.data = []

    def to_weight_matrix(self, x: pd.DataFrame):
        test = x.set_index('Name')
        rows = test.index.to_list(); cols = test.columns.tolist()
        diff = list(set(cols) - set(rows)); diff2 = list(set(rows) - set(cols))
        weight = test.drop(diff, axis = 1).drop(diff2, axis = 0)
        weight.update(weight.div(weight.sum(axis=0),axis=1))
        return weight

    def to_adj_matrix(self, x : pd.DataFrame):
        res = self.to_weight_matrix(x)
        res[res!= 0] = 1
        return res

    def to_adj_list(self, x : pd.DataFrame):
        res = self.to_adj_matrix(x)
        industries = res.index.to_list()
        for i in range(0,res.shape[0]):
            for j in range(0,res.shape[1]):
                if i == j:
                    res.iloc[i,j] = 0
                if str(int(res.iloc[i,j])) == '1':
                    res.iloc[i,j] = res.columns[j]

        master = res.values.tolist()
        temp = [[y for y in x if y != 0] for x in master]
        edges = {industries[i]: temp[i] for i in range(0,len(temp))}
        return edges

    def to_edge_list(self, x : pd.DataFrame):
        edge = self.to_adj_list(x)
        weights = self.to_weight_matrix(x)
        industries = [*edge.keys()]
        temp = []
        for i in range(len(industries)):
            for j in range(len(edge[industries[i]])):
                if weights.iloc[i,j] != 0:
                    k = (industries[i], edge[industries[i]][j], weights.iloc[i,j])
                    temp.append(k)
        return pd.DataFrame(temp, columns = ['from','to','weights'])

    def to_network(self, x):
        network = self.to_adj_matrix(x)
        G = nx.from_pandas_adjacency(network)
        G = nx.relabel_nodes(G, mapping)
        return(G)
