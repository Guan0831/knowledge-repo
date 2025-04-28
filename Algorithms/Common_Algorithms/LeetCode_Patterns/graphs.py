# algorithms/common_algorithms/leetcode_patterns/graphs/graph_algorithms_code.py

"""
图算法实现示例 (Python)

本文件包含了图的基本表示方法以及一些常见的图算法的 Python 实现，
包括遍历、最短路径、最小生成树和拓扑排序。
"""

import collections # 常用数据结构如 deque 和 defaultdict
import heapq       # 用于实现优先队列，Dijkstra 算法需要用到

# --- 图表示 (Graph Representation) ---

class Graph:
    """
    使用邻接表表示图

    适用于有向图或无向图。
    无向图：边 (u, v) 会同时在 u 的邻接列表中添加 v，以及在 v 的邻接列表中添加 u。
    带权图：邻接表可以存储 (邻居节点, 权重) 的元组。
    """
    def __init__(self, directed=False):
        # 邻接表：使用 defaultdict(list)，键是节点，值是其邻居节点的列表
        self.adj = collections.defaultdict(list)
        self.directed = directed # 标记是否为有向图
        self.nodes = set()       # 存储所有节点

    def add_edge(self, u, v, weight=1):
        """
        添加一条从节点 u 到节点 v 的边

        u, v: 边的起点和终点
        weight: 边的权重 (默认为 1，适用于无权图)
        """
        self.adj[u].append((v, weight)) # 邻接表存储 (邻居，权重) 元组
        self.nodes.add(u)
        self.nodes.add(v)
        if not self.directed:
            # 如果是无向图，反向边也需要添加
            self.adj[v].append((u, weight))
            self.nodes.add(v) # 确保 v 节点也被加入 (虽然无向图时已经在 v->u 这步加了)


# --- 图遍历 (Graph Traversal) ---

def bfs(graph, start_node):
    """
    广度优先搜索 (BFS - Breadth-First Search)

    原理：从起始节点开始，优先访问所有距离它为 1 的节点，然后是距离为 2 的节点，以此类推。
    使用数据结构：队列 (Queue)。
    """
    visited = set() # 记录已访问的节点
    queue = collections.deque([start_node]) # 初始化队列，放入起始节点
    visited.add(start_node)

    bfs_order = [] # 记录 BFS 遍历的节点顺序

    while queue:
        current_node = queue.popleft() # 从队列头部取出节点
        bfs_order.append(current_node)

        # 遍历当前节点的所有邻居
        # 注意：如果图是带权的，adj[current_node] 存储的是 (邻居, 权重) 元组
        # 我们只需要邻居节点，所以取 pair[0]
        for neighbor, weight in graph.adj.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor) # 将未访问的邻居加入队列

    return bfs_order # 返回 BFS 遍历顺序

def dfs_recursive(graph, start_node, visited=None, dfs_order=None):
    """
    深度优先搜索 (DFS - Depth-First Search) - 递归实现

    原理：从起始节点开始，沿着一条路径尽可能深地访问节点，直到不能再深入，然后回溯，尝试其他路径。
    使用数据结构：隐式的递归调用栈。
    """
    if visited is None:
        visited = set() # 初始化访问集合
        dfs_order = []  # 初始化 DFS 顺序列表

    if start_node not in visited:
        visited.add(start_node)
        dfs_order.append(start_node) # 记录 DFS 遍历顺序

        # 遍历当前节点的所有邻居
        for neighbor, weight in graph.adj.get(start_node, []):
            # 对未访问的邻居进行递归调用
            dfs_recursive(graph, neighbor, visited, dfs_order)

    return dfs_order # 返回 DFS 遍历顺序

def dfs_iterative(graph, start_node):
    """
    深度优先搜索 (DFS - Depth-First Search) - 迭代实现

    原理：使用栈来模拟递归过程。
    使用数据结构：栈 (Stack)。
    """
    visited = set() # 记录已访问的节点
    stack = [start_node] # 初始化栈，放入起始节点
    # 注意：这里的 visited 集合和堆栈入栈顺序需要仔细处理，
    # 通常第一次遇到就标记 visited 并入栈，避免重复处理；或者入栈时标记 visited。
    # 以下是一种常见的实现方式：入栈即标记 visited
    visited.add(start_node)

    dfs_order = [] # 记录 DFS 遍历顺序

    while stack:
        current_node = stack.pop() # 从栈顶取出节点 (后进先出)
        dfs_order.append(current_node)

        # 遍历当前节点的所有邻居
        # 注意：为了让输出顺序与递归版本类似，通常会倒序遍历邻居，
        # 因为栈是后进先出，最后入栈的会先被处理。
        for neighbor, weight in reversed(graph.adj.get(current_node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor) # 将未访问的邻居加入栈

    return dfs_order # 返回 DFS 遍历顺序

# --- 最短路径算法 (Shortest Path Algorithms) ---

def dijkstra(graph, start_node):
    """
    Dijkstra 算法 (适用于带非负权边的图)

    原理：从起始节点开始，逐步扩展最短路径。使用优先队列优先处理当前距离最短的节点。
    数据结构：优先队列 (Priority Queue)，通常用二叉堆实现。
    """
    # distances: 字典，存储从 start_node 到各节点的当前最短距离，初始化为无穷大
    # predecessors: 字典，可选，存储最短路径上的前驱节点，用于重建路径
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start_node] = 0 # 起始节点到自身的距离为 0

    # priority_queue: 存储 (距离, 节点) 元组，优先队列会按第一个元素 (距离) 排序
    # 初始只包含起始节点
    priority_queue = [(0, start_node)]

    while priority_queue:
        # 从优先队列中取出当前距离最小的节点
        current_distance, current_node = heapq.heappop(priority_queue)

        # 如果取出的距离比已记录的距离大，说明已经找到更短路径，跳过 (处理旧的/重复的队列项)
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的所有邻居
        for neighbor, weight in graph.adj.get(current_node, []):
            # 计算通过当前节点到达邻居的新距离
            distance = current_distance + weight

            # 如果新距离比当前记录的到邻居的距离更短
            if distance < distances[neighbor]:
                distances[neighbor] = distance # 更新最短距离
                # 将 (新距离, 邻居) 加入优先队列
                heapq.heappush(priority_queue, (distance, neighbor))
                # predecessors[neighbor] = current_node # 如果需要重建路径

    return distances # 返回从起始节点到所有节点的最短距离字典

# --- 并查集 (Union-Find) - Kruskal 算法需要用到 ---

class UnionFind:
    """
    并查集数据结构

    用于处理集合的合并和查询是否在同一集合中。
    主要操作：
    - find(x): 查找元素 x 所在的集合的代表元素 (根节点)。使用路径压缩优化。
    - union(x, y): 合并包含元素 x 和 y 的两个集合。使用按秩/大小合并优化。
    """
    def __init__(self, elements):
        # parent: 字典，存储每个元素的父节点。初始化时，每个元素是自己的父节点。
        self.parent = {e: e for e in elements}
        # rank/size: 字典，用于按秩或按大小合并优化。这里使用 size。
        self.size = {e: 1 for e in elements}

    def find(self, x):
        """
        查找元素 x 所在的集合的代表元素 (带路径压缩)
        """
        if self.parent[x] != x:
            # 如果 x 不是根节点，递归查找根节点，并在回溯时进行路径压缩
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        合并包含元素 x 和 y 的两个集合 (按大小合并)

        如果 x 和 y 不在同一个集合中，则合并它们所在的集合，返回 True。
        如果已经在同一个集合中，不做任何操作，返回 False。
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # 按大小合并：将小集合的根节点连接到大集合的根节点
            if self.size[root_x] < self.size[root_y]:
                self.parent[root_x] = root_y
                self.size[root_y] += self.size[root_x]
            else:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
            return True # 发生了合并
        return False # 没有发生合并 (已经在同一个集合中)

# --- 最小生成树算法 (Minimum Spanning Tree - MST) ---

def kruskal(graph):
    """
    Kruskal 算法 (用于计算图的最小生成树)

    原理：将图的边按权重从小到大排序，依次考虑每条边。如果加入这条边不会形成环，则将其加入 MST。
         使用并查集来判断是否形成环。
    适用于：连通的无向图。
    """
    # 1. 获取图中所有的边，格式为 (权重, 节点u, 节点v)
    edges = []
    # 需要注意无向图的边 (u,v) 和 (v,u) 在邻接表中会出现两次，只取一次
    processed_edges = set() # 记录已经处理过的边，避免重复
    for u in graph.nodes:
        for v, weight in graph.adj.get(u, []):
            # 对于无向图，只添加一次边 (u, v) 或 (v, u)
            if graph.directed or (u, v) not in processed_edges and (v, u) not in processed_edges:
                 edges.append((weight, u, v))
                 processed_edges.add((u, v))
                 processed_edges.add((v, u))


    # 2. 对边按权重进行排序 (从小到大)
    edges.sort()

    # 3. 初始化并查集，包含图中的所有节点
    uf = UnionFind(graph.nodes)

    # 4. 遍历排序后的边，构建 MST
    mst_edges = [] # 存储 MST 的边
    mst_weight = 0   # 存储 MST 的总权重

    for weight, u, v in edges:
        # 如果边 (u, v) 的两个端点不在同一个集合中 (加入这条边不会形成环)
        if uf.find(u) != uf.find(v):
            uf.union(u, v) # 合并 u 和 v 所在的集合
            mst_edges.append((u, v, weight)) # 将这条边加入 MST
            mst_weight += weight

            # 优化：如果 MST 的边数等于节点数-1，说明已经构建完成 (对于连通图)
            if len(mst_edges) == len(graph.nodes) - 1 and len(graph.nodes) > 0:
                 break


    # 检查图是否连通 (如果 MST 的边数小于节点数-1，且节点数>1，则不连通)
    # 这是一个简单的检查，更严谨需要判断并查集最终集合的数量是否为 1
    if len(graph.nodes) > 1 and len(mst_edges) != len(graph.nodes) - 1:
         print("Warning: Graph is not connected, MST cannot include all nodes.")


    return mst_edges, mst_weight # 返回 MST 的边列表和总权重

# --- 拓扑排序 (Topological Sort) ---

def topological_sort_kahn(graph):
    """
    拓扑排序 - Kahn 算法 (基于入度)

    原理：适用于有向无环图 (DAG)。
         1. 计算每个节点的入度 (指向该节点的边的数量)。
         2. 将所有入度为 0 的节点加入队列。
         3. 从队列中取出一个节点，将其加入结果列表，并减少其所有邻居的入度。
         4. 如果邻居的入度变为 0，将其加入队列。
         5. 重复步骤 3-4 直到队列为空。
    如果结果列表中的节点数量等于图中的节点数量，则拓扑排序成功；否则图中存在环。
    """
    # 1. 计算入度
    in_degree = {node: 0 for node in graph.nodes}
    for u in graph.nodes:
        for v, weight in graph.adj.get(u, []):
            in_degree[v] += 1

    # 2. 初始化队列，加入所有入度为 0 的节点
    queue = collections.deque([node for node in graph.nodes if in_degree[node] == 0])

    # 存储拓扑排序结果
    topological_order = []

    # 3. 处理队列中的节点
    while queue:
        current_node = queue.popleft()
        topological_order.append(current_node)

        # 遍历当前节点的所有邻居
        for neighbor, weight in graph.adj.get(current_node, []):
            in_degree[neighbor] -= 1 # 邻居的入度减 1
            # 如果邻居的入度变为 0，加入队列
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 检查图中是否存在环
    if len(topological_order) == len(graph.nodes):
        return topological_order # 成功，返回拓扑排序结果
    else:
        # 图中存在环，无法进行拓扑排序
        print("Warning: Graph has a cycle, topological sort is not possible.")
        return None


# --- 示例用法 ---
if __name__ == "__main__":
    # 示例图 (无向带权图 for Dijkstra, Kruskal)
    g_undirected_weighted = Graph(directed=False)
    g_undirected_weighted.add_edge('A', 'B', 4)
    g_undirected_weighted.add_edge('A', 'C', 2)
    g_undirected_weighted.add_edge('B', 'E', 3)
    g_undirected_weighted.add_edge('C', 'D', 2)
    g_undirected_weighted.add_edge('C', 'F', 4)
    g_undirected_weighted.add_edge('D', 'E', 3)
    g_undirected_weighted.add_edge('D', 'F', 1)
    g_undirected_weighted.add_edge('E', 'F', 1)

    # 示例图 (有向无权图 for BFS, DFS, Topological Sort)
    g_directed_unweighted = Graph(directed=True)
    g_directed_unweighted.add_edge('A', 'B')
    g_directed_unweighted.add_edge('A', 'C')
    g_directed_unweighted.add_edge('B', 'D')
    g_directed_unweighted.add_edge('C', 'D')
    g_directed_unweighted.add_edge('C', 'E')
    g_directed_unweighted.add_edge('D', 'F')
    g_directed_unweighted.add_edge('E', 'F')
    g_directed_unweighted.add_edge('G', 'H') # 一个不连通的部分

    # 示例图 (有向图带环，用于测试拓扑排序环检测)
    g_cyclic = Graph(directed=True)
    g_cyclic.add_edge('A', 'B')
    g_cyclic.add_edge('B', 'C')
    g_cyclic.add_edge('C', 'A') # 形成环


    print("--- 图遍历示例 ---")
    print("使用有向无权图:")
    start_node_traversal = 'A'
    print(f"BFS 从节点 {start_node_traversal} 开始:")
    print(bfs(g_directed_unweighted, start_node_traversal)) # 期望: ['A', 'B', 'C', 'D', 'E', 'F'] (顺序可能不同)

    print(f"\nDFS (递归) 从节点 {start_node_traversal} 开始:")
    print(dfs_recursive(g_directed_unweighted, start_node_traversal)) # 期望: ['A', 'B', 'D', 'F', 'C', 'E'] (顺序取决于邻居遍历顺序)

    print(f"\nDFS (迭代) 从节点 {start_node_traversal} 开始:")
    print(dfs_iterative(g_directed_unweighted, start_node_traversal)) # 期望: ['A', 'C', 'E', 'F', 'D', 'B'] (顺序取决于栈的后进先出)


    print("\n--- 最短路径示例 (Dijkstra) ---")
    print("使用无向带权图:")
    start_node_dijkstra = 'A'
    distances = dijkstra(g_undirected_weighted, start_node_dijkstra)
    print(f"\n从节点 {start_node_dijkstra} 到所有节点的最短距离:")
    # 打印时按节点字母顺序排序
    for node in sorted(distances.keys()):
        print(f"到 {node}: {distances[node]}")
    # 期望: 到 A: 0, 到 B: 4, 到 C: 2, 到 D: 4, 到 E: 5, 到 F: 3 (路径 A->C->D->F)


    print("\n--- 最小生成树示例 (Kruskal) ---")
    print("使用无向带权图:")
    mst_edges, mst_weight = kruskal(g_undirected_weighted)
    print(f"\nMST 的边和权重:")
    for u, v, w in mst_edges:
        print(f"边 ({u}, {v}), 权重 {w}")
    print(f"MST 总权重: {mst_weight}")
    # 期望边: (C, D, 2), (D, F, 1), (E, F, 1), (A, C, 2), (B, E, 3) (顺序可能不同)
    # 期望总权重: 1+1+2+2+3 = 9


    print("\n--- 拓扑排序示例 (Kahn 算法) ---")
    print("使用有向无权图 (无环):")
    topological_order_acyclic = topological_sort_kahn(g_directed_unweighted)
    print(f"\n拓扑排序结果:")
    print(topological_order_acyclic)
    # 期望是有效拓扑序之一，如 ['A', 'G', 'B', 'C', 'H', 'D', 'E', 'F'] (顺序不唯一)

    print("\n使用有向图 (有环):")
    topological_order_cyclic = topological_sort_kahn(g_cyclic)
    print(f"\n拓扑排序结果:")
    print(topological_order_cyclic)
    # 期望: None (并打印警告信息)