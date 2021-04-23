import numpy as np

nodes = [(i+1) for i in range(9)]
arcs1 = [(1,3,5),(1,2,14),(1,4,2),(2,3,9),(2,4,8),(3,5,13),(2,5,15),(4,5,10),(3,6,8),(4,8,11),(5,6,1),(5,7,7),(5,8,5),(7,6,10),(7,8,0),(6,9,11),(7,9,12),(8,9,6)]

arcs2 = [(1,2,20),(1,4,35),(1,5,30),(2,3,7),(2,5,18),(2,6,9),(3,6,20),(4,5,23),(4,7,25),(5,6,11),(5,7,12),(5,8,15),(5,9,25),(6,9,4),(7,8,22),(8,9,20)]

def get_adjacency_mat(nodes, arcs):
    n = len(nodes)
    mat = np.matrix(np.ones((n, n)) * np.Infinity)
    for arc in arcs:
        mat[arc[0]-1, arc[1]-1] = arc[2]
        mat[arc[1]-1, arc[0]-1] = arc[2]
    for i in range(n):
        mat[i,i] = 0
    return mat

def find_ancestor(genealogy, i):
    while genealogy[i] != i:
        i = genealogy[i]
    return i

def kruskal(vertices, edges):
    edges.sort(key = lambda x: x[2])
    genealogy = {v:v for v in vertices}
    mst = list()
    for e in edges:
        i = e[0]
        j = e[1]
        ans_i = find_ancestor(genealogy, i)
        ans_j = find_ancestor(genealogy, j)
        if ans_i != ans_j:
            mst.append(e)
            genealogy[ans_i] = ans_j
    return mst

def find_odd_nodes(mst, nodes):
    n = len(nodes)
    degree = {node:0 for node in nodes}
    for e in mst:
        degree[e[0]] += 1
        degree[e[1]] += 1
    return [i for i in degree.keys() if degree[i]%2 == 1]

def floyd_warshall(mat):
    n = mat.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                candidate1 = mat[i,j]
                candidate2 = mat[i,k] + mat[k,j]
                mat[i,j] = mat[j,i] = min(candidate1, candidate2)
    return mat

def dijkstra(vertices, edges, s):
    distance = {v: np.Infinity for v in vertices}
    distance[s] = 0
    undetermined = {v: np.Infinity for v in vertices}
    undetermined[s] = 0
    pred = {v:v for v in vertices}
    while bool(undetermined):
        new_v = min(undetermined, key = undetermined.get)
        for e in edges:
            if e[0] == new_v:
                if distance[e[1]] > distance[new_v] + e[2]:
                    distance[e[1]] = distance[new_v] + e[2]
                    pred[e[1]] = new_v
            elif e[1] == new_v:
                if distance[e[0]] > distance[new_v] + e[2]:
                    distance[e[0]] = distance[new_v] + e[2]
                    pred[e[0]] = new_v
        del undetermined[new_v]
    return pred

# shortest path between any pair of nodes
# dict of dict, key is starting vertex, value is the pred
def all_pairs_sp(vertices, edges):
    preds = dict()
    for v in vertices:
        preds[v] = dijkstra(vertices, edges, v)
    return preds

def find_pairs(odd_nodes):
    if len(odd_nodes) == 2:
        return [[tuple(odd_nodes)]]
    else:
        remaining_pairing = list()
        for i in range(1,len(odd_nodes)):
            pair1 = (odd_nodes[0], odd_nodes[i])
            remaining = odd_nodes[1:i] + odd_nodes[i+1:]
            sub_pairing = find_pairs(remaining)
            for j in range(len(sub_pairing)):
                sub_pairing[j].append(pair1)
            remaining_pairing = remaining_pairing + sub_pairing
        return remaining_pairing

def pairing_cost(pairing, mat):
    cost = 0
    for t in pairing:
        cost += mat[t[0]-1, t[1]-1]
    return cost

# Question 3b
mat1 = get_adjacency_mat(nodes, arcs1)
sp1 = floyd_warshall(mat1)

mat2 = get_adjacency_mat(nodes, arcs2)
sp2 = floyd_warshall(mat2)


# Question 3c
mst1 = kruskal(nodes, arcs1)
odd_nodes1 = find_odd_nodes(mst1, nodes)
all_possible_pairing = find_pairs(odd_nodes1)

min_cost_pairing1 = None
min_cost1 = np.Infinity
for pairing in all_possible_pairing:
    cost = pairing_cost(pairing, mat1)
    if cost < min_cost1:
        min_cost1 = cost
        min_cost_pairing1 = pairing


mst2 = kruskal(nodes, arcs2)
odd_nodes2 = find_odd_nodes(mst2, nodes)
all_possible_pairing = find_pairs(odd_nodes2)

min_cost_pairing2 = None
min_cost2 = np.Infinity
for pairing in all_possible_pairing:
    cost = pairing_cost(pairing, mat2)
    if cost < min_cost2:
        min_cost2 = cost
        min_cost_pairing2 = pairing

for p in min_cost_pairing2:
    print(mat2[p[0]-1,p[1]-1])