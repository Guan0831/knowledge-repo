# algorithms/common_algorithms/leetcode_patterns/heap_priority_queue/heap_priority_queue_code.py

"""
堆 (Heap) 与优先队列 (Priority Queue) 示例 (Python)

本文件演示如何使用 Python 标准库中的 heapq 模块来实现最小堆和模拟最大堆，
heapq 模块提供了基于堆的优先队列算法。

heapq 默认实现的是一个**最小堆**，堆顶元素总是最小的。
操作的时间复杂度：
- heapify (构建堆): O(n)
- heappush (插入): O(log n)
- heappop (弹出堆顶): O(log n)
"""

import heapq
import copy # 用于复制列表，避免修改原始测试数据

# --- 最小堆示例 (Min-Heap Example) ---

def min_heap_example(initial_data):
    """
    使用 heapq 实现最小堆的基本操作。

    initial_data: 包含初始元素的列表。
    """
    print("--- 最小堆示例 ---")

    # 复制一份数据，避免修改原始输入列表
    data = copy.copy(initial_data)
    print(f"原始列表: {initial_data}")

    if not data:
        print("初始数据为空，跳过堆操作。\n")
        return

    # 1. 将一个列表原地转换为最小堆 O(n)
    heapq.heapify(data) # 列表现在是一个最小堆
    print(f"Heapify 后 (列表表示): {data}") # 列表表示不是严格的树结构，但满足堆属性

    # 2. 向堆中添加元素 O(log n)
    print("添加元素:")
    elements_to_add = [0, 7]
    for elem in elements_to_add:
        heapq.heappush(data, elem)
        print(f"  添加 {elem} 后: {data}")

    # 3. 从堆中弹出最小元素 O(log n)
    print("\n弹出最小元素:")
    while data:
        min_val = heapq.heappop(data) # 弹出并移除堆顶的最小元素
        print(f"  弹出: {min_val}, 剩余: {data}")

    print("堆已为空。\n")
    print("-" * 20)

# --- 最大堆模拟示例 (Max-Heap Simulation Example) ---
# heapq 只有最小堆，可以通过一些技巧来模拟最大堆

def max_heap_example_negate(initial_data):
    """
    使用 heapq 模拟最大堆 - 通过存储元素的负数。

    initial_data: 包含初始元素的列表 (假定元素为数值类型)。
    """
    print("--- 最大堆模拟示例 (通过负数) ---")

    # 复制一份数据
    data = copy.copy(initial_data)
    print(f"原始列表: {initial_data}")

    if not data:
         print("初始数据为空，跳过堆操作。\n")
         return

    # 存储元素的负数构建最小堆
    max_heap_data = [-x for x in data]
    heapq.heapify(max_heap_data)
    print(f"构建模拟最大堆后 (内部存储负数): {max_heap_data}") # 堆顶是最小负数 (原最大值的负数)

    # 添加元素 (添加负数)
    print("\n添加元素 (存为负数):")
    elements_to_add = [10, -5] # 原值 10, -5
    for elem in elements_to_add:
        heapq.heappush(max_heap_data, -elem) # 插入元素的负数
        print(f"  添加 {elem} (存为 {-elem}) 后: {max_heap_data}")

    # 弹出最大元素 (弹出负数并取反)
    print("\n弹出最大元素:")
    while max_heap_data:
        max_val_neg = heapq.heappop(max_heap_data)
        print(f"  弹出: {-max_val_neg}, 剩余 (内部存储): {max_heap_data}") # 弹出时取反优先级

    print("模拟最大堆已为空。\n")
    print("-" * 20)

def max_heap_example_tuple(initial_tasks):
    """
    使用 heapq 模拟最大堆 - 通过存储 (负优先级, 元素) 元组。

    initial_tasks: 包含 (优先级, 任务名称) 元组的列表。
    """
    print("--- 最大堆模拟示例 (通过元组) ---")

    # 复制一份数据
    tasks = copy.copy(initial_tasks)
    print(f"原始任务列表 (优先级, 任务): {tasks}")

    if not tasks:
         print("初始数据为空，跳过堆操作。\n")
         return

    # 存储 (负优先级, 任务名称) 元组构建最小堆
    # 元组比较时，先比较第一个元素 (-优先级)，负优先级小的元组排在前面
    # 当负优先级相同时，比较第二个元素 (任务名称)，按字母顺序排，这里我们不需要关心第二个元素的顺序
    max_heap_tasks = [(-priority, task) for priority, task in tasks]
    heapq.heapify(max_heap_tasks)
    print(f"构建模拟最大堆后 (内部存储): {max_heap_tasks}") # 堆顶是负优先级最小的元组

    # 添加任务 (添加负优先级元组)
    print("\n添加任务 (存为负优先级元组):")
    tasks_to_add = [(5, 'sleep'), (2, 'read book')]
    for priority, task in tasks_to_add:
        heapq.heappush(max_heap_tasks, (-priority, task))
        print(f"  添加 ({priority}, '{task}') (存为 {(-priority, task)}) 后: {max_heap_tasks}")

    # 按优先级从高到低弹出任务
    print("\n按优先级从高到低弹出任务:")
    while max_heap_tasks:
        # 弹出并移除堆顶的元组
        neg_priority, task = heapq.heappop(max_heap_tasks)
        print(f"  弹出任务: {task}, 优先级: {-neg_priority}, 剩余 (内部存储): {max_heap_tasks}") # 弹出时取反优先级

    print("模拟最大堆已为空。\n")
    print("-" * 20)

# --- heapq 的其他常用功能 ---

def other_heapq_features(initial_data):
     """
     演示 heapq 的 nsmallest 和 nlargest 功能。

     initial_data: 包含元素的列表。
     """
     print("--- heapq 其他常用功能 ---")
     data = copy.copy(initial_data) # 复制一份数据
     print(f"原始列表: {data}")

     if not data:
         print("初始数据为空，跳过操作。\n")
         return

     # 找到最小的 n 个元素 (不修改原列表，返回一个新列表) O(n log k) 或 O(n + k log n)
     n_smallest = 3
     if len(data) >= n_smallest:
         smallest_n = heapq.nsmallest(n_smallest, data)
         print(f"最小的 {n_smallest} 个元素: {smallest_n}") # 期望: [1, 1, 2]

     # 找到最大的 n 个元素 (通过在内部使用大小为 n 的最小堆实现) O(n log k) 或 O(n + k log n)
     n_largest = 3
     if len(data) >= n_largest:
         largest_n = heapq.nlargest(n_largest, data)
         print(f"最大的 {n_largest} 个元素: {largest_n}") # 期望: [9, 8, 7]

     print("-" * 20)

"""
# --- 运行示例 ---
if __name__ == "__main__":
    # --- 定义测试数据 ---
    min_heap_data = [3, 1, 4, 1, 5, 9, 2, 6]
    max_heap_negate_data = [3, 1, 4, 1, 5, 9, 2, 6]
    max_heap_tuple_tasks = [(3, 'write code'), (1, 'eat food'), (4, 'meditate'), (1, 'go for a walk')]
    other_features_data = [3, 1, 4, 1, 5, 9, 2, 6, 8, 7] # 可以用不同的数据集合

    # --- 调用示例函数，将数据作为参数传入 ---
    min_heap_example(min_heap_data)
    max_heap_example_negate(max_heap_negate_data)
    max_heap_example_tuple(max_heap_tuple_tasks)
    other_heapq_features(other_features_data)

    # 测试空列表情况
    # min_heap_example([])
    # max_heap_example_negate([])
    # max_heap_example_tuple([])
    # other_heapq_features([])
"""