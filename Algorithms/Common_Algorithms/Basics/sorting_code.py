# algorithms/common-algorithms/01-basics/sorting_code.py

"""
常见的排序算法实现 (Python)

本文件包含了几种经典的排序算法的 Python 实现代码，用于学习和参考。
每种算法都有其特定的时间复杂度和空间复杂度，适用于不同的场景。
"""

import heapq # 堆排序需要用到 heapq 模块

def bubble_sort(arr):
    """
    冒泡排序 (Bubble Sort)

    原理：重复遍历列表，比较相邻元素，如果顺序错误则交换，直到没有元素需要交换。
    时间复杂度：O(n^2) - 最好、平均、最坏情况都是二次方。
    空间复杂度：O(1) - 原地排序。
    稳定性：稳定排序。
    """
    n = len(arr)
    for i in range(n):
        # 标记在本次遍历中是否发生了交换，如果一轮遍历没有交换，说明已经有序
        swapped = False
        # 最后 i 个元素已经排好序，不需要再比较
        for j in range(n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] # 交换元素
                swapped = True
        # 如果本次遍历没有发生交换，提前结束
        if not swapped:
            break
    return arr

def selection_sort(arr):
    """
    选择排序 (Selection Sort)

    原理：每次遍历从未排序的部分找到最小（或最大）元素，放到已排序部分的末尾。
    时间复杂度：O(n^2) - 最好、平均、最坏情况都是二次方。
    空间复杂度：O(1) - 原地排序。
    稳定性：不稳定排序。
    """
    n = len(arr)
    for i in range(n):
        # 假设当前位置 i 是最小元素的位置
        min_idx = i
        # 在未排序部分 (从 i+1 到末尾) 查找最小元素
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # 如果最小元素不在当前位置 i，则交换
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """
    插入排序 (Insertion Sort)

    原理：遍历列表，将当前元素插入到已排序的左侧部分的正确位置。
    时间复杂度：O(n^2) - 平均和最坏情况。O(n) - 最好情况 (列表已基本有序)。
    空间复杂度：O(1) - 原地排序。
    稳定性：稳定排序。
    """
    for i in range(1, len(arr)):
        # 当前要插入的元素
        key = arr[i]
        # 已排序部分的最后一个元素的索引
        j = i - 1
        # 将已排序部分中大于 key 的元素向右移动，为 key 腾出位置
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        # 将 key 插入到正确的位置
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """
    归并排序 (Merge Sort)

    原理：分治法。将列表递归地分成两半，直到只剩一个元素（有序），然后将有序的子列表合并。
    时间复杂度：O(n log n) - 稳定且高效。
    空间复杂度：O(n) - 需要额外的空间来存储合并过程中的临时列表。
    稳定性：稳定排序。
    """
    if len(arr) <= 1:
        return arr # 列表只有一个或零个元素，已经有序

    # 分割点
    mid = len(arr) // 2
    # 递归地分割左右两半
    left_half = arr[:mid]
    right_half = arr[mid:]

    # 递归排序左右两半
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    # 合并排序好的两半
    return _merge(left_half, right_half)

def _merge(left, right):
    """
    辅助函数：合并两个已排序的列表
    """
    merged = []
    i = 0 # 左列表指针
    j = 0 # 右列表指针

    # 比较左右列表的元素，将较小的添加到结果列表
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # 将剩余的元素添加到结果列表 (只有一个列表还有剩余)
    while i < len(left):
        merged.append(left[i])
        i += 1
    while j < len(right):
        merged.append(right[j])
        j += 1

    return merged

def quick_sort(arr):
    """
    快速排序 (Quick Sort)

    原理：分治法。选择一个“基准”(pivot) 元素，将列表分割成两部分：
         一部分所有元素都小于基准，另一部分所有元素都大于基准。
         然后递归地对这两部分进行快速排序。
    时间复杂度：O(n log n) - 平均情况。O(n^2) - 最坏情况 (取决于基准的选择，如选择已排序列表的第一个或最后一个元素)。
    空间复杂度：O(log n) - 平均情况（递归栈空间）。O(n) - 最坏情况。
    稳定性：不稳定排序。
    """
    # 调用辅助函数进行原地排序
    _quick_sort_recursive(arr, 0, len(arr) - 1)
    return arr

def _quick_sort_recursive(arr, low, high):
    """
    辅助函数：快速排序的递归实现 (原地排序)
    """
    if low < high:
        # 找到分割点 (基准的最终位置)
        pi = _partition(arr, low, high)

        # 递归排序分割点左右两边的部分
        _quick_sort_recursive(arr, low, pi - 1)
        _quick_sort_recursive(arr, pi + 1, high)

def _partition(arr, low, high):
    """
    辅助函数：分割操作 (选择最后一个元素作为基准)
    """
    pivot = arr[high] # 选择最后一个元素作为基准
    i = low - 1       # 小于基准元素的区域的边界

    # 遍历从 low 到 high-1 的元素
    for j in range(low, high):
        # 如果当前元素小于或等于基准
        if arr[j] <= pivot:
            i += 1 # 将边界向右移动
            arr[i], arr[j] = arr[j], arr[i] # 交换当前元素与 i 位置的元素

    # 将基准元素放到正确的位置 (i+1)
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    # 返回分割点索引
    return i + 1

def heap_sort(arr):
    """
    堆排序 (Heap Sort)

    原理：利用堆这种数据结构。首先将列表构造成一个最大堆（或最小堆），
         然后重复地将堆顶元素（最大或最小）与堆末尾元素交换，并调整剩余元素以维护堆属性。
    时间复杂度：O(n log n) - 构建堆 O(n)，n 次提取最大值 O(log n)。
    空间复杂度：O(1) - 原地排序（不考虑递归或系统栈空间，如果使用库函数则看库函数实现）。
             注意：使用 heapq 在 Python 中并非严格原地，因为它操作的是列表。
    稳定性：不稳定排序。
    """
    n = len(arr)

    # 1. 构建最大堆 (从最后一个非叶子节点开始，向上调整)
    # 非叶子节点的索引范围是 n//2 - 1 到 0
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # 2. 一个个地从堆中取出元素 (将堆顶最大元素放到末尾)
    for i in range(n - 1, 0, -1):
        # 将当前堆顶 (最大元素) 与末尾元素交换
        arr[i], arr[0] = arr[0], arr[i]
        # 对剩余的堆 (大小为 i) 重新进行堆调整
        _heapify(arr, i, 0)
        
    # heapq 默认实现是最小堆，如果需要最大堆，可以存入元素的负数或者自定义比较函数
    # 下面是基于Python heapq的另一种实现思路（可能不是严格原地，取决于用法）
    # temp_arr = [-x for x in arr] # 使用负数构建最大堆
    # heapq.heapify(temp_arr)
    # sorted_arr = [-heapq.heappop(temp_arr) for _ in range(n)]
    # arr[:] = sorted_arr # 将排序结果放回原列表 (如果需要修改原列表的话)
    # return arr # 或者直接返回 sorted_arr


    return arr

def _heapify(arr, heap_size, root_index):
    """
    辅助函数：堆调整 (Maintain heap property)

    arr: 列表
    heap_size: 当前堆的大小 (未排序部分的元素个数)
    root_index: 需要进行堆调整的子树的根节点索引
    """
    largest = root_index  # 初始化 largest 为根节点
    left_child = 2 * root_index + 1 # 左子节点索引
    right_child = 2 * root_index + 2 # 右子节点索引

    # 如果左子节点存在且大于根节点
    if left_child < heap_size and arr[left_child] > arr[largest]:
        largest = left_child

    # 如果右子节点存在且大于目前 largest
    if right_child < heap_size and arr[right_child] > arr[largest]:
        largest = right_child

    # 如果 largest 不是根节点 (说明发生了变化)
    if largest != root_index:
        # 交换根节点与 largest 元素
        arr[root_index], arr[largest] = arr[largest], arr[root_index]
        # 递归地对受影响的子树进行堆调整
        _heapify(arr, heap_size, largest)

# --- 示例用法 ---
"""
if __name__ == "__main__":
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5], # 已排序
        [5, 4, 3, 2, 1], # 逆序
        [1], # 单元素
        [] # 空列表
    ]

    print("--- 冒泡排序 ---")
    for arr in test_cases:
        original = list(arr) # 复制列表，避免修改原列表影响下一次测试
        print(f"原始列表: {original}")
        sorted_arr = bubble_sort(arr)
        print(f"排序结果: {sorted_arr}\n")

    print("--- 选择排序 ---")
    # 重新加载测试用例，因为上面的排序会修改原列表
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1],
        []
    ]
    for arr in test_cases:
        original = list(arr)
        print(f"原始列表: {original}")
        sorted_arr = selection_sort(arr)
        print(f"排序结果: {sorted_arr}\n")

    print("--- 插入排序 ---")
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1],
        []
    ]
    for arr in test_cases:
        original = list(arr)
        print(f"原始列表: {original}")
        sorted_arr = insertion_sort(arr)
        print(f"排序结果: {sorted_arr}\n")

    print("--- 归并排序 ---")
    # 注意：归并排序返回新列表，不修改原列表，所以这里不需要复制
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1],
        []
    ]
    for arr in test_cases:
        print(f"原始列表: {arr}")
        sorted_arr = merge_sort(arr)
        print(f"排序结果: {sorted_arr}\n")

    print("--- 快速排序 ---")
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1],
        []
    ]
    for arr in test_cases:
        original = list(arr)
        print(f"原始列表: {original}")
        sorted_arr = quick_sort(arr)
        print(f"排序结果: {sorted_arr}\n")

    print("--- 堆排序 ---")
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1],
        []
    ]
    for arr in test_cases:
        original = list(arr)
        print(f"原始列表: {original}")
        sorted_arr = heap_sort(arr)
        print(f"排序结果: {sorted_arr}\n")
"""