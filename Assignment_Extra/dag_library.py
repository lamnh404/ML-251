from collections import deque

class DAG:
    def __init__(self):
        self.nodes = set()
        self.adj = {}
        self.in_degree = {}

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.adj[node] = []
            self.in_degree[node] = 0

    def add_edge(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Các nút phải tồn tại trong đồ thị trước khi thêm cạnh.")
        self.adj[u].append(v)
        self.in_degree[v] += 1
    
    def get_all_nodes(self):
        return self.nodes

    def topological_sort(self):
        q = [node for node in self.nodes if self.in_degree[node] == 0]
        topo_order = []
        in_degree_copy = self.in_degree.copy()
        
        while q:
            u = q.pop(0)
            topo_order.append(u)
            for v in self.adj[u]:
                in_degree_copy[v] -= 1
                if in_degree_copy[v] == 0:
                    q.append(v)
        
        if len(topo_order) != len(self.nodes):
            raise ValueError("Đồ thị có chu trình. Không thể sắp xếp topo.")
        return topo_order