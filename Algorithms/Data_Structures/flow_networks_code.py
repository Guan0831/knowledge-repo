# algorithms/common_algorithms/beyond_basics/flow_networks/flow_networks_code.py

"""
网络流 (Flow Networks) - Edmonds-Karp 算法实现 (Python)

本文件实现了 Edmonds-Karp 算法，用于计算网络的最大流。
最大流最小割定理指出，网络中的最大流值等于其最小割的容量。

Edmonds-Karp 算法是 Ford-Fulkerson 算法的一种具体实现，通过使用 BFS
在残差图 (Residual Graph) 中寻找从源点 (Source, S) 到汇点 (Sink, T) 的增广路径 (Augmenting Path)。

概念:
- 流网络 (Flow Network): 带容量的有限有向图。
- 容量 (Capacity): 边的最大允许流量。
- 流 (Flow): 边的实际流量，不能超过容量，且满足流量守恒 (除源汇点)。
- 残差图 (Residual Graph): 表示每条边还可以承载多少额外流量，以及反向边的概念 (用于减少流量)。
- 增广路径 (Augmenting Path): 残差图中一条从源点到汇点的路径，沿途每条边都有正的剩余容量。
- 瓶颈容量 (Bottleneck Capacity): 增广路径上剩余容量最小的边的容量。

Edmonds-Karp 算法步骤:
1. 初始化所有边的流量为 0，残差图与原图相同 (反向边容量为 0)。
2. 在残差图中用 BFS 寻找一条从 S 到 T 的增广路径。
3. 如果找到增广路径，计算路径上的瓶颈容量。
4. 沿增广路径增加流量 (正向边减少剩余容量，反向边增加剩余容量)，并将瓶颈容量加到总流量上。
5. 重复步骤 2-4，直到残差图中不存在从 S 到 T 的路径。
"""

import collections # 用于实现队列

# 图的表示方法：邻接表
# 对于每条边 (u, v, capacity)，我们在 u 的邻接表中添加一个元组/列表：
# [v, capacity, index_of_reverse_edge_in_v's_adj_list]
# 同时，在 v 的邻接表中添加对应的反向边：
# [u, 0, index_of_forward_edge_in_u's_adj_list]
# 初始时，反向边的容量为 0。在增广过程中，残差图中的容量会发生变化。

def add_edge(adj, u, v, capacity):
    """
    向图中添加一条有向边及其对应的反向边。

    adj: 邻接表，一个 defaultdict(list)。
    u, v: 边的起点和终点。
    capacity: 边的容量。
    """
    # 获取当前 u 和 v 的邻接表长度，用于记录反向边的索引
    len_u_adj = len(adj[u])
    len_v_adj = len(adj[v])

    # 添加正向边 [邻居节点, 容量, 对应反向边在邻居邻接表中的索引]
    adj[u].append([v, capacity, len_v_adj])
    # 添加反向边 [原始起点, 初始容量 (0), 对应正向边在原始起点邻接表中的索引]
    adj[v].append([u, 0, len_u_adj])


def bfs_find_augmenting_path(adj, s, t, parent_edge):
    """
    在残差图中用 BFS 寻找一条从源点 s 到汇点 t 的增广路径。

    adj: 图的邻接表 (表示残差图的容量)。
    s: 源点。
    t: 汇点。
    parent_edge: 一个列表，用于存储 BFS 过程中每个节点的前驱信息。
                 parent_edge[v] = (u, edge_index_in_u_adj) 表示到达 v 是从 u 经过 u 的邻接表中索引为 edge_index_in_u_adj 的边。
                 初始化为 [-1] * num_nodes 或类似。
    """
    # 将 parent_edge 列表初始化为 -1，表示节点未被访问
    # 注意：这里的 parent_edge 应该在 Edmonds-Karp 主函数中初始化并传入
    # size = len(adj) # 如果节点是 0 到 size-1 的整数
    # parent_edge = [-1] * size

    # visited 集合或列表，用于 BFS 避免重复访问
    visited = {s} # 从源点开始访问

    # 队列用于 BFS，存储待访问的节点
    queue = collections.deque([s])

    # path_bottleneck 列表，存储从源点到该节点路径上的最小剩余容量
    # size = len(adj)
    # path_bottleneck = [float('inf')] * size # 初始化为无穷大
    # path_bottleneck[s] = float('inf') # 源点到自己的瓶颈容量无穷大

    # 在 Edmonds-Karp 中初始化 parent_edge 和 path_bottleneck 并传递
    # 这里只负责 BFS 逻辑

    while queue:
        u = queue.popleft() # 从队列头部取出节点

        # 如果已经到达汇点，停止 BFS
        if u == t:
            break

        # 遍历节点 u 的所有邻居边
        # edge 是一个列表 [v, capacity, reverse_edge_index]
        for edge_index, edge in enumerate(adj[u]):
            v, capacity, _ = edge # 邻居节点 v 和边 (u, v) 在残差图中的容量 capacity

            # 如果邻居节点 v 未被访问 且 边 (u, v) 在残差图中有剩余容量 (capacity > 0)
            if v not in visited and capacity > 0:
                visited.add(v) # 标记 v 已访问
                queue.append(v) # 将 v 加入队列

                # 记录到达 v 的前驱是 u，经过 u 的邻接表中的 edge_index 条边
                parent_edge[v] = (u, edge_index)

                # 计算从源点到 v 的路径上的瓶颈容量
                # 从源点到 u 的瓶颈容量 和 边 (u, v) 的容量 中的最小值
                # path_bottleneck[v] = min(path_bottleneck[u], capacity) # 如果需要返回瓶颈容量

    # BFS 结束后，检查是否找到了从 s 到 t 的路径 (即 t 是否可达)
    # 如果 parent_edge[t] 仍然是初始值 (-1), 说明 t 不可达
    if parent_edge[t] == -1:
        return 0 # 没有找到增广路径，返回 0

    # 如果找到了路径，计算这条路径上的瓶颈容量
    bottleneck = float('inf')
    current = t # 从汇点开始，沿 parent_edge 回溯到源点
    while current != s:
        u, edge_index = parent_edge[current] # 获取当前节点的前驱和到达它的边的信息
        v, capacity, reverse_edge_index = adj[u][edge_index] # 获取这条边的详细信息
        # 找到路径上最小的剩余容量 (瓶颈)
        bottleneck = min(bottleneck, capacity)
        current = u # 移动到前驱节点，继续回溯

    return bottleneck # 返回找到的增广路径的瓶颈容量


def edmonds_karp(graph_edges, num_nodes, s, t):
    """
    Edmonds-Karp 算法实现，计算最大流。

    graph_edges: 边的列表，每条边格式为 (u, v, capacity)。节点通常为 0 到 num_nodes-1 的整数。
    num_nodes: 图中的节点数量。
    s: 源点索引。
    t: 汇点索引。
    """
    # 使用 defaultdict(list) 作为邻接表，键是节点索引，值是边列表
    adj = collections.defaultdict(list)

    # 根据输入的边列表构建邻接表，包括正向边和初始容量为 0 的反向边
    for u, v, capacity in graph_edges:
        add_edge(adj, u, v, capacity)

    max_flow = 0 # 初始化最大流为 0

    # 用于记录 BFS 过程中每个节点的前驱信息，以便重建路径
    # parent_edge[i] = (前驱节点, 前驱节点邻接表中到达当前节点的边的索引)
    parent_edge = [-1] * num_nodes

    # 循环寻找增广路径并增加流量，直到找不到为止
    while True:
        # 初始化 parent_edge 列表，每次新的 BFS 都需要重置
        parent_edge = [-1] * num_nodes
        # 使用 BFS 寻找增广路径，并获取这条路径的瓶颈容量
        bottleneck = bfs_find_augmenting_path(adj, s, t, parent_edge)

        # 如果瓶颈容量为 0，说明找不到增广路径了，算法结束
        if bottleneck == 0:
            break

        # 如果找到了增广路径 (瓶颈容量 > 0)，则沿路径增加流量
        max_flow += bottleneck # 将瓶颈容量加到总流量上

        # 沿增广路径回溯，更新残差图的容量
        current = t # 从汇点开始回溯
        while current != s:
            u, edge_index = parent_edge[current] # 获取当前节点的前驱和到达它的边的信息
            v, capacity, reverse_edge_index = adj[u][edge_index] # 获取这条边的详细信息

            # 更新正向边的残差容量：容量减少瓶颈值
            adj[u][edge_index][1] -= bottleneck

            # 更新反向边的残差容量：容量增加瓶颈值
            # 反向边的索引保存在正向边的第三个位置
            adj[v][reverse_edge_index][1] += bottleneck

            current = u # 移动到前驱节点，继续回溯

    return max_flow # 返回计算出的最大流值

"""
# --- 示例用法 ---
if __name__ == "__main__":
    # 在 main 块中定义测试数据

    # 示例流网络 1 (简单的二源二汇)
    # 节点 0 (S), 1, 2, 3 (T)
    # 边: (0, 1, 10), (0, 2, 10), (1, 2, 2), (1, 3, 4), (2, 3, 9)
    edges1 = [
        (0, 1, 10),
        (0, 2, 10),
        (1, 2, 2),
        (1, 3, 4),
        (2, 3, 9)
    ]
    num_nodes1 = 4
    s1 = 0
    t1 = 3
    print(f"--- 示例网络 1 ---")
    print(f"节点数: {num_nodes1}, 源点: {s1}, 汇点: {t1}")
    print(f"边和容量: {edges1}")
    max_flow1 = edmonds_karp(edges1, num_nodes1, s1, t1)
    print(f"计算出的最大流: {max_flow1}\n") # 期望: 19 (路径 S->0->1->3 (容量4), S->0->2->3 (容量9), S->0->1->2->3 (容量 2+2))

    # 示例流网络 2 (稍微复杂一点，来自常见教程)
    # 节点 0 (S), 1, 2, 3, 4, 5 (T)
    edges2 = [
        (0, 1, 10), (0, 2, 10),
        (1, 2, 2), (1, 3, 4), (1, 4, 8),
        (2, 4, 9),
        (3, 5, 10),
        (4, 3, 6), (4, 5, 10)
    ]
    num_nodes2 = 6
    s2 = 0
    t2 = 5
    print(f"--- 示例网络 2 ---")
    print(f"节点数: {num_nodes2}, 源点: {s2}, 汇点: {t2}")
    print(f"边和容量: {edges2}")
    max_flow2 = edmonds_karp(edges2, num_nodes2, s2, t2)
    print(f"计算出的最大流: {max_flow2}\n") # 期望: 19

    # 示例网络 3 (不连通或容量为 0)
    edges3 = [
        (0, 1, 5),
        (2, 3, 5) # S=0, T=3 不连通
    ]
    num_nodes3 = 4
    s3 = 0
    t3 = 3
    print(f"--- 示例网络 3 (不连通) ---")
    print(f"节点数: {num_nodes3}, 源点: {s3}, 汇点: {t3}")
    print(f"边和容量: {edges3}")
    max_flow3 = edmonds_karp(edges3, num_nodes3, s3, t3)
    print(f"计算出的最大流: {max_flow3}\n") # 期望: 0

    # 示例网络 4 (容量为 0)
    edges4 = [
        (0, 1, 0),
        (1, 2, 5)
    ]
    num_nodes4 = 3
    s4 = 0
    t4 = 2
    print(f"--- 示例网络 4 (容量为 0) ---")
    print(f"节点数: {num_nodes4}, 源点: {s4}, 汇点: {t4}")
    print(f"边和容量: {edges4}")
    max_flow4 = edmonds_karp(edges4, num_nodes4, s4, t4)
    print(f"计算出的最大流: {max_flow4}\n") # 期望: 0
"""