# algorithms/common_algorithms/beyond_basics/fenwick_tree_bit/fenwick_tree_bit_code.py

"""
树状数组 / 二叉索引树 (Fenwick Tree / Binary Indexed Tree - BIT) 实现示例

树状数组是一种数据结构，用于高效地处理数组上的点更新和前缀和查询。
相比线段树，它在实现这类问题时更简洁高效。

核心思想：利用数的二进制表示和低位函数 (i & -i) 来确定节点管辖的范围和依赖关系。

核心操作:
- 构建 (Build): O(n log n) (通过 n 次点更新实现)
- 更新 (Update): O(log n)
- 查询 (Query - 前缀和): O(log n)
- 区间查询 (Range Query): O(log n)
"""

class FenwickTree:
    """
    树状数组 / 二叉索引树 (BIT) 实现 (用于前缀求和和点更新)
    """
    def __init__(self, data):
        """
        初始化树状数组。

        data: 用于构建 BIT 的原始数组。BIT 的大小会比原始数组大 1。
        """
        self.size = len(data) # 原始数组的大小
        # BIT 数组，通常从索引 1 开始使用，所以大小是原始数组大小 + 1
        # 初始化为 0
        self.bit = [0] * (self.size + 1)

        # 通过对原始数组的每个元素进行更新来构建 BIT
        for i in range(self.size):
            self.update(i, data[i]) # 注意：这里的 update 接收的是 0-based 索引

    def _lowbit(self, i):
        """
        辅助函数：计算 i 的最低设置位 (Lowest Set Bit) 对应的值。
        等价于 i & (~i + 1)
        """
        return i & (-i)

    def update(self, index, delta):
        """
        更新原始数组中位置 index 的值，增加 delta，并同步更新 BIT。

        index: 原始数组中的位置索引 (0-based)。
        delta: 需要增加到原始数组 data[index] 的值。
        """
        # 将 0-based 索引转换为 BIT 使用的 1-based 索引
        idx = index + 1

        # 从 idx 开始，向上更新 BIT 节点，直到超出范围
        # 每次跳跃的步长是 _lowbit(idx)
        while idx <= self.size:
            self.bit[idx] += delta # 当前节点的值增加 delta
            idx += self._lowbit(idx) # 跳到下一个需要更新的父节点


    def query(self, index):
        """
        查询原始数组从开始到位置 index (包含) 的前缀和。

        index: 原始数组中的位置索引 (0-based)。
        """
        # 将 0-based 索引转换为 BIT 使用的 1-based 索引
        idx = index + 1
        total = 0 # 累积前缀和

        # 从 idx 开始，向下累加 BIT 节点的值，直到回到根 (索引 0)
        # 每次跳跃的步长是 _lowbit(idx)
        while idx > 0:
            total += self.bit[idx] # 累加当前节点存储的值
            idx -= self._lowbit(idx) # 跳到下一个需要累加的祖先节点 (其范围包含当前范围)

        return total


    def range_query(self, l, r):
        """
        查询原始数组区间 [l, r] (包含) 的和。

        l: 区间左边界 (0-based)。
        r: 区间右边界 (0-based)。
        """
        # 区间 [l, r] 的和 = 前缀和 query(r) - 前缀和 query(l-1)
        # 需要处理 l=0 的情况，此时 query(l-1) 无效，前缀和就是 query(r)
        if l > r:
             return 0 # 无效区间
        if l < 0 or r >= self.size:
             print(f"Warning: Query range [{l}, {r}] might be partially out of bounds.")

        # 确保 l 和 r 在有效范围内（即使有警告也尝试在范围内查询）
        l = max(0, l)
        r = min(self.size - 1, r)

        return self.query(r) - (self.query(l - 1) if l > 0 else 0)


    # 辅助方法：获取原始数组某个位置的当前值
    # BIT 本身不直接存储原始值，但可以通过两次前缀和查询或一次查询加一次更新来获取
    # 简单起见，如果要频繁获取原始值，建议在类中同步维护一份原始数组，
    # 或者这里提供一个通过 BIT 计算原始值的复杂方法（不常用，所以省略）
    # def get_original_value(self, index):
    #     # 获取 query(index) 和 query(index-1) 的差
    #     return self.query(index) - (self.query(index - 1) if index > 0 else 0)

"""
# --- 示例用法 ---
if __name__ == "__main__":
    # 在 main 块中定义测试数据
    initial_array = [1, 3, 5, 7, 9, 11]
    print(f"原始数组: {initial_array}")

    # 构建一个用于前缀求和和点更新的 BIT
    bit = FenwickTree(initial_array)
    # BIT 内部的 bit 数组 (1-based index)
    # print(f"构建后的 BIT 数组 (1-based): {bit.bit}")


    print("\n--- 前缀求和查询 ---")
    # 查询到索引 0 的前缀和 (即原始数组第一个元素)
    query_idx1 = 0
    print(f"查询到索引 {query_idx1} 的前缀和: {bit.query(query_idx1)}") # 期望: 1

    # 查询到索引 2 的前缀和 (1+3+5)
    query_idx2 = 2
    print(f"查询到索引 {query_idx2} 的前缀和: {bit.query(query_idx2)}") # 期望: 9

    # 查询到索引 5 的前缀和 (整个数组的和)
    query_idx3 = 5
    print(f"查询到索引 {query_idx3} 的前缀和: {bit.query(query_idx3)}") # 期望: 36


    print("\n--- 区间求和查询 ---")
    # 查询区间 [1, 4] (3+5+7+9)
    query_l1, query_r1 = 1, 4
    print(f"查询区间 [{query_l1}, {query_r1}] 的和: {bit.range_query(query_l1, query_r1)}") # 期望: 24

    # 查询单元素区间 [2, 2] (5)
    query_l2, query_r2 = 2, 2
    print(f"查询区间 [{query_l2}, {query_r2}] 的和: {bit.range_query(query_l2, query_r2)}") # 期望: 5

    # 查询整个区间 [0, 5]
    query_l3, query_r3 = 0, 5
    print(f"查询区间 [{query_l3}, {query_r3}] 的和: {bit.range_query(query_l3, query_r3)}") # 期望: 36


    print("\n--- 点更新示例 ---")
    update_index = 2 # 更新原始数组索引 2 的位置 (值为 5)
    delta_value = 100 - 5 # 将 5 改变为 100，所以 delta 是 95
    print(f"更新原始数组索引 {update_index} 的值，增加 {delta_value} (原值 5 变为 100)")
    bit.update(update_index, delta_value) # 更新 BIT

    # BIT 内部的 bit 数组会发生变化
    # print(f"更新后的 BIT 数组 (1-based): {bit.bit}")


    print("\n--- 更新后再次查询 ---")
    # 再次查询到索引 2 的前缀和 (1+3+100)
    print(f"查询到索引 {query_idx2} 的前缀和: {bit.query(query_idx2)}") # 期望: 104

    # 再次查询区间 [1, 4] 的和 (3+100+7+9)
    print(f"查询区间 [{query_l1}, {query_r1}] 的和: {bit.range_query(query_l1, query_r1)}") # 期望: 119

    # 再次查询整个区间 [0, 5]
    print(f"查询区间 [{query_l3}, {query_r3}] 的和: {bit.range_query(query_l3, query_r3)}") # 期望: 131


    print("\n--- 示例：处理越界或无效查询 ---")
    print(f"查询区间 [-1, 3] 的和: {bit.range_query(-1, 3)}") # 期望: query(3)-query(-1) -> query(3) - 0 = 1+3+100+7 = 111 (会打印警告)
    print(f"查询区间 [3, 8] 的和: {bit.range_query(3, 8)}") # 期望: query(5)-query(2) = 131 - 104 = 27 (会打印警告)
    print(f"查询区间 [5, 4] 的和: {bit.range_query(5, 4)}") # 期望: 0 (无效区间)
"""