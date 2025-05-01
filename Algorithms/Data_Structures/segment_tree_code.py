# algorithms/common_algorithms/beyond_basics/segment_tree/segment_tree_code.py

"""
线段树 (Segment Tree) 实现示例 (用于区间求和和点更新) (Python)

线段树是一种树形数据结构，用于高效地处理数组上的区间查询和点更新/区间更新操作。
每个节点代表数组的一个区间。

核心操作:
- 构建 (Build): O(n)
- 查询 (Query): O(log n)
- 更新 (Update): O(log n)
"""

import math # 用于计算树数组大小，虽然通常 4*n 就足够安全

class SegmentTree:
    """
    线段树实现类 (用于区间求和和点更新)
    """
    def __init__(self, data, operation=lambda x, y: x + y, default_value=0):
        """
        初始化线段树。

        data: 用于构建线段树的原始数组。
        operation: 用于合并子区间结果的操作 (如 lambda x, y: x + y)。
        default_value: 操作的幺元 (Identity Element)，用于处理空区间或边界情况 (如求和是 0，求最小值是正无穷)。
        """
        self.data = data # 原始数据
        self.n = len(data) # 原始数据大小
        self.operation = operation # 合并操作
        self.default_value = default_value # 操作的幺元

        # 线段树数组的大小通常是原始数据大小的 4 倍，以确保不会越界
        # 或者更精确地计算：2 * (2^ceil(log2(n)))，但 4*n 简单且安全
        tree_size = 4 * self.n
        self.tree = [None] * tree_size # 存储线段树节点的值

        # 从根节点 (索引 1) 开始构建线段树，根节点代表整个区间 [0, n-1]
        self._build(1, 0, self.n - 1)


    def _build(self, v, tl, tr):
        """
        递归构建线段树。

        v: 当前节点的索引 (在线段树数组 self.tree 中)。通常根节点从 1 开始。
        tl: 当前节点代表区间的左边界 (inclusive)。
        tr: 当前节点代表区间的右边界 (inclusive)。
        """
        # Base Case: 如果是叶子节点 (区间只有一个元素)
        if tl == tr:
            # 叶子节点的值就是原始数据对应位置的值
            self.tree[v] = self.data[tl]
        else:
            # 递归构建左右子树
            tm = (tl + tr) // 2 # 中间点，分割成左右两个子区间 [tl, tm] 和 [tm+1, tr]
            self._build(2 * v, tl, tm)       # 构建左子节点 (索引 2*v)
            self._build(2 * v + 1, tm + 1, tr) # 构建右子节点 (索引 2*v+1)

            # 当前节点的值是其左右子节点的值通过指定操作合并的结果
            self.tree[v] = self.operation(self.tree[2 * v], self.tree[2 * v + 1])


    def query(self, v, tl, tr, l, r):
        """
        递归查询给定区间 [l, r] 的结果。

        v: 当前节点的索引。
        tl: 当前节点代表区间的左边界。
        tr: 当前节点代表区间的右边界。
        l: 查询区间的左边界。
        r: 查询区间的右边界。
        """
        # 如果查询区间 [l, r] 是一个无效区间 (左边界大于右边界)
        if l > r:
            return self.default_value # 返回操作的幺元

        # 如果当前节点代表的区间 [tl, tr] 完全包含在查询区间 [l, r] 内
        # 即 l <= tl 且 tr <= r
        if l <= tl and tr <= r:
            return self.tree[v] # 直接返回当前节点存储的值

        # 如果当前节点代表的区间与查询区间没有交集
        # 即当前区间的右边界在查询区间的左边界左边 (tr < l)
        # 或者当前区间的左边界在查询区查询的右边界右边 (tl > r)
        if tr < l or tl > r:
             return self.default_value # 返回操作的幺元

        # 如果当前节点代表的区间与查询区间有部分交集
        # 递归查询左右子树，并将结果合并
        tm = (tl + tr) // 2 # 中间点
        # 查询左子树 (区间 [tl, tm]) 与查询区间 [l, r] 的交集
        left_result = self.query(2 * v, tl, tm, l, r)
        # 查询右子树 (区间 [tm+1, tr]) 与查询区间 [l, r] 的交集
        right_result = self.query(2 * v + 1, tm + 1, tr, l, r)

        # 返回左右子树查询结果的合并
        return self.operation(left_result, right_result)


    def update(self, v, tl, tr, pos, new_val):
        """
        递归更新原始数组中位置 pos 的值为 new_val，并同步更新线段树。

        v: 当前节点的索引。
        tl: 当前节点代表区间的左边界。
        tr: 当前节点代表区间的右边界。
        pos: 需要更新的原始数组中的位置索引。
        new_val: 位置 pos 的新值。
        """
        # Base Case: 如果是叶子节点 (找到了需要更新的位置 pos 对应的叶子节点)
        if tl == tr:
            # 更新叶子节点的值
            self.tree[v] = new_val
            # 同时更新原始数据（可选，但通常保持同步）
            # self.data[pos] = new_val
        else:
            # 递归更新左右子树
            tm = (tl + tr) // 2 # 中间点
            # 如果更新位置 pos 在左子树的区间内
            if pos <= tm:
                self.update(2 * v, tl, tm, pos, new_val)
            # 如果更新位置 pos 在右子树的区间内
            else:
                self.update(2 * v + 1, tm + 1, tr, pos, new_val)

            # 左右子树更新完成后，更新当前节点的值 (合并左右子节点的新值)
            self.tree[v] = self.operation(self.tree[2 * v], self.tree[2 * v + 1])

    # --- 外部调用的接口 ---

    def query_range(self, l, r):
        """
        外部调用接口：查询原始数组区间 [l, r] 的结果。
        """
        # 调用内部递归方法，从根节点 (索引 1) 开始，根节点代表整个区间 [0, n-1]
        # 确保查询区间 [l, r] 在原始数据范围内
        l = max(0, l)
        r = min(self.n - 1, r)
        if l > r: # 处理无效区间
             return self.default_value
        return self.query(1, 0, self.n - 1, l, r)

    def update_value(self, pos, new_val):
        """
        外部调用接口：更新原始数组位置 pos 的值为 new_val。
        """
        # 确保更新位置 pos 在原始数据范围内
        if 0 <= pos < self.n:
             self.update(1, 0, self.n - 1, pos, new_val)
        else:
             print(f"Warning: Update position {pos} is out of bounds.")

"""
# --- 示例用法 ---
if __name__ == "__main__":
    # 在 main 块中定义测试数据
    initial_array = [1, 3, 5, 7, 9, 11]
    print(f"原始数组: {initial_array}\n")

    # 构建一个用于区间求和的线段树
    # operation=lambda x, y: x + y (默认值)
    # default_value=0 (默认值)
    sum_segment_tree = SegmentTree(initial_array)

    print("--- 区间求和查询 ---")
    # 查询整个区间 [0, 5]
    query_l1, query_r1 = 0, 5
    print(f"查询区间 [{query_l1}, {query_r1}] 的和: {sum_segment_tree.query_range(query_l1, query_r1)}") # 期望: 36 (1+3+5+7+9+11)

    # 查询子区间 [1, 4]
    query_l2, query_r2 = 1, 4
    print(f"查询区间 [{query_l2}, {query_r2}] 的和: {sum_segment_tree.query_range(query_l2, query_r2)}") # 期望: 24 (3+5+7+9)

    # 查询单元素区间 [2, 2]
    query_l3, query_r3 = 2, 2
    print(f"查询区间 [{query_l3}, {query_r3}] 的和: {sum_segment_tree.query_range(query_l3, query_r3)}") # 期望: 5

    # 查询无效区间
    query_l4, query_r4 = 4, 1
    print(f"查询无效区间 [{query_l4}, {query_r4}] 的和: {sum_segment_tree.query_range(query_l4, query_r4)}") # 期望: 0 (default_value)

    # 查询越界区间
    query_l5, query_r5 = -1, 10
    print(f"查询越界区间 [{query_l5}, {query_r5}] 的和: {sum_segment_tree.query_range(query_l5, query_r5)}") # 期望: 36 (会被限制在 [0, 5])


    print("\n--- 点更新示例 ---")
    update_pos = 2
    new_value = 100
    print(f"更新位置 {update_pos} 的值为 {new_value}")
    sum_segment_tree.update_value(update_pos, new_value) # 原数组变成 [1, 3, 100, 7, 9, 11]

    print("\n--- 更新后再次查询 ---")
    # 再次查询整个区间 [0, 5]
    print(f"查询区间 [{query_l1}, {query_r1}] 的和: {sum_segment_tree.query_range(query_l1, query_r1)}") # 期望: 1 + 3 + 100 + 7 + 9 + 11 = 131

    # 再次查询子区间 [1, 4]
    print(f"查询区间 [{query_l2}, {query_r2}] 的和: {sum_segment_tree.query_range(query_l2, query_r2)}") # 期望: 3 + 100 + 7 + 9 = 119

    # 再次查询单元素区间 [2, 2]
    print(f"查询区间 [{query_l3}, {query_r3}] 的和: {sum_segment_tree.query_range(query_l3, query_r3)}") # 期望: 100

    # 更新越界位置 (会打印警告)
    sum_segment_tree.update_value(10, 999)

    print("\n--- 示例：构建一个用于区间求最小值的线段树 ---")
    import sys # 导入 sys 以使用 sys.maxsize 作为无穷大

    min_array = [5, 2, 8, 1, 6, 3]
    print(f"原始数组: {min_array}")

    # 构建一个用于区间求最小值的线段树
    min_segment_tree = SegmentTree(min_array, operation=lambda x, y: min(x, y), default_value=sys.maxsize)

    # 查询区间 [0, 5] 的最小值
    print(f"查询区间 [0, 5] 的最小值: {min_segment_tree.query_range(0, 5)}") # 期望: 1

    # 查询区间 [1, 4] 的最小值
    print(f"查询区间 [1, 4] 的最小值: {min_segment_tree.query_range(1, 4)}") # 期望: 1

    # 更新位置 3 的值为 10
    min_segment_tree.update_value(3, 10) # 原数组变成 [5, 2, 8, 10, 6, 3]

    # 再次查询区间 [1, 4] 的最小值
    print(f"更新后查询区间 [1, 4] 的最小值: {min_segment_tree.query_range(1, 4)}") # 期望: 2
"""