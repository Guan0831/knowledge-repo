# algorithms/common_algorithms/leetcode_patterns/union_find/union_find_code.py

"""
并查集 (Union-Find) 数据结构实现 (Python)

本文件包含了并查集数据结构的实现，使用了路径压缩 (Path Compression)
和按大小合并 (Union by Size) 两种优化方法，以提高操作效率。

并查集用于维护一组不相交的集合，支持以下操作：
- 查找 (Find): 确定元素属于哪个集合。
- 合并 (Union): 将两个集合合并为一个集合。

常用于解决连通性问题，如判断图中是否存在环、计算连通分量等。
"""

class UnionFind:
    """
    并查集数据结构实现 (带路径压缩和按大小合并)
    """
    def __init__(self, elements):
        """
        初始化并查集。

        每个元素初始时位于一个独立的集合中，自身是自己的父节点，集合大小为 1。
        elements: 可迭代对象，包含所有需要加入并查集的元素。
        """
        # parent 字典: 存储每个元素的父节点。parent[i] = j 表示 j 是 i 的父节点。
        # 初始化时 parent[i] = i
        self.parent = {e: e for e in elements}

        # size 字典: 存储以某个元素为根节点的集合的大小。
        # 初始化时 size[i] = 1
        self.size = {e: 1 for e in elements}

        # 集合数量 (可选): 记录当前不相交集合的数量
        self._num_sets = len(elements)


    def find(self, x):
        """
        查找元素 x 所在的集合的代表元素 (根节点)。

        同时实现路径压缩：在查找过程中，将路径上的所有节点的父节点直接指向根节点。
        """
        # 如果 x 不是根节点 (x 的父节点不是自身)
        if self.parent[x] != x:
            # 递归查找根节点，并将 x 的父节点直接指向找到的根节点 (路径压缩)
            self.parent[x] = self.find(self.parent[x])
        # 返回根节点
        return self.parent[x]


    def union(self, x, y):
        """
        合并包含元素 x 和 y 的两个集合。

        使用按大小合并优化：将较小集合的根节点连接到较大集合的根节点。
        如果 x 和 y 已经在同一个集合中，则不进行合并。

        返回值: 如果发生了合并，返回 True；如果 x 和 y 已在同一个集合中，返回 False。
        """
        # 找到 x 和 y 所在的集合的根节点
        root_x = self.find(x)
        root_y = self.find(y)

        # 如果它们不在同一个集合中
        if root_x != root_y:
            # 按大小合并：将大小较小的集合合并到大小较大的集合
            if self.size[root_x] < self.size[root_y]:
                self.parent[root_x] = root_y  # 将 root_x 连接到 root_y
                self.size[root_y] += self.size[root_x] # 更新新根节点的大小
            else:
                self.parent[root_y] = root_x  # 将 root_y 连接到 root_x
                self.size[root_x] += self.size[root_y] # 更新新根节点的大小

            self._num_sets -= 1 # 合并成功，不相交集合数量减一
            return True # 发生了合并
        else:
            # 已经在同一个集合中，不进行合并
            return False


    def is_connected(self, x, y):
        """
        检查元素 x 和 y 是否在同一个集合中。

        只需判断它们的根节点是否相同。
        """
        return self.find(x) == self.find(y)

    # 可选方法：获取集合数量
    def get_num_sets(self):
         return self._num_sets

"""

# --- 示例用法 ---
if __name__ == "__main__":
    # 创建一个包含 0 到 9 的并查集
    elements = list(range(10))
    uf = UnionFind(elements)

    print(f"初始状态下的父节点: {list(uf.parent.items())}")
    print(f"初始状态下的集合大小: {list(uf.size.items())}")
    print(f"初始集合数量: {uf.get_num_sets()}\n") # 期望: 10


    print("--- 进行合并操作 ---")
    print("合并 0 和 1:", uf.union(0, 1)) # 期望: True
    print("合并 2 和 3:", uf.union(2, 3)) # 期望: True
    print("合并 0 和 2:", uf.union(0, 2)) # 期望: True (0和2现在同属于一个大集合)
    print("合并 4 和 5:", uf.union(4, 5)) # 期望: True
    print("合并 1 和 3:", uf.union(1, 3)) # 期望: False (1和3已经在同一个集合，都是0或2的后代)
    print("合并 7 和 8:", uf.union(7, 8)) # 期望: True

    print("\n--- 进行查找操作 ---")
    print("查找 1 的根节点:", uf.find(1)) # 期望: 0 或 2 (取决于合并顺序，但应该是 0, 1, 2, 3 所在集合的根)
    print("查找 3 的根节点:", uf.find(3)) # 期望: 同上
    print("查找 5 的根节点:", uf.find(5)) # 期望: 4 或 5 (4和5所在集合的根)
    print("查找 9 的根节点:", uf.find(9)) # 期望: 9 (9 아직 没有合并)

    # 再次查找以展示路径压缩的效果 (父节点可能会变)
    print("\n--- 再次查找 (展示路径压缩) ---")
    print("查找 1 的根节点:", uf.find(1)) # 期望: 同上，但 uf.parent[1] 现在直接指向根了
    print(f"1 的父节点现在是: {uf.parent[1]}") # 期望: 0 或 2 (之前的根)

    print("\n--- 检查连通性 ---")
    print("检查 0 和 3 是否连接:", uf.is_connected(0, 3)) # 期望: True
    print("检查 1 和 4 是否连接:", uf.is_connected(1, 4)) # 期望: False
    print("检查 7 和 9 是否连接:", uf.is_connected(7, 9)) # 期望: False
    print("检查 7 和 8 是否连接:", uf.is_connected(7, 8)) # 期望: True

    print(f"\n当前不相交集合数量: {uf.get_num_sets()}") # 期望: 10 - 1 - 1 - 1 - 1 - 1 = 5
    # 初始 10 个集合 {0}..{9}
    # union(0,1): {0,1}, {2}..{9} -> 9 个集合
    # union(2,3): {0,1}, {2,3}, {4}..{9} -> 8 个集合
    # union(0,2): {0,1,2,3}, {4}..{9} -> 7 个集合
    # union(4,5): {0,1,2,3}, {4,5}, {6}..{9} -> 6 个集合
    # union(1,3): 不合并
    # union(7,8): {0,1,2,3}, {4,5}, {6}, {7,8}, {9} -> 5 个集合
"""