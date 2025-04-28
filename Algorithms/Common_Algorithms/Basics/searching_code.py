# algorithms/common_algorithms/basics/searching_code.py

"""
常见的搜索算法实现 (Python)

本文件包含了线性搜索和二分搜索两种基本搜索算法的 Python 实现。
二分搜索要求输入列表是已排序的。
"""

def linear_search(arr, target):
    """
    线性搜索 (Linear Search)

    原理：从列表的第一个元素开始，逐个检查每个元素是否等于目标值。
    时间复杂度：O(n) - 最坏情况下需要遍历整个列表。O(1) - 最好情况 (目标是第一个元素)。
    空间复杂度：O(1) - 原地搜索。
    返回值：如果找到目标，返回其索引；如果找不到，返回 -1。
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # 找到目标，返回索引
    return -1 # 遍历完都没找到

def binary_search_iterative(arr, target):
    """
    二分搜索 - 迭代实现 (Binary Search - Iterative)

    原理：适用于已排序的列表。每次比较目标值与中间元素，根据比较结果将搜索范围缩小到左半部分或右半部分，直到找到目标或搜索范围为空。
    先决条件：输入列表 arr 必须是已排序的。
    时间复杂度：O(log n) - 每次搜索范围减半。
    空间复杂度：O(1) - 迭代实现不需要额外的递归栈空间。
    返回值：如果找到目标，返回其索引；如果找不到，返回 -1。
    """
    low = 0            # 搜索范围的起始索引
    high = len(arr) - 1 # 搜索范围的结束索引

    while low <= high:
        mid = (low + high) // 2 # 计算中间元素的索引 (注意整数除法)
        mid_val = arr[mid]      # 获取中间元素的值

        if mid_val == target:
            return mid  # 找到目标，返回索引
        elif mid_val < target:
            low = mid + 1 # 目标在右半部分，缩小搜索范围到 mid+1 到 high
        else: # mid_val > target
            high = mid - 1 # 目标在左半部分，缩小搜索范围到 low 到 mid-1

    return -1 # 循环结束，说明目标不在列表中

def binary_search_recursive(arr, target, low, high):
    """
    二分搜索 - 递归实现 (Binary Search - Recursive)

    原理：与迭代实现相同，但使用递归调用来缩小搜索范围。
    先决条件：输入列表 arr 必须是已排序的。
    时间复杂度：O(log n)。
    空间复杂度：O(log n) - 递归调用栈会占用空间。
    返回值：如果找到目标，返回其索引；如果找不到，返回 -1。
    """
    # 递归终止条件：搜索范围为空
    if high < low:
        return -1

    mid = (low + high) // 2
    mid_val = arr[mid]

    if mid_val == target:
        return mid # 找到目标，返回索引
    elif mid_val < target:
        # 目标在右半部分，递归搜索 mid+1 到 high
        return binary_search_recursive(arr, target, mid + 1, high)
    else: # mid_val > target
        # 目标在左半部分，递归搜索 low 到 mid-1
        return binary_search_recursive(arr, target, low, mid - 1)

# 为了方便调用递归版本，可以提供一个封装函数
def binary_search_recursive_wrapper(arr, target):
    """
    二分搜索 - 递归实现封装函数
    """
    return binary_search_recursive(arr, target, 0, len(arr) - 1)

"""
# --- 示例用法 ---
if __name__ == "__main__":
    my_list = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] # 二分搜索需要已排序列表
    unsorted_list = [5, 2, 8, 1, 9, 4] # 线性搜索可以使用未排序列表

    target1 = 23
    target2 = 100
    target3 = 1

    print("--- 线性搜索 ---")
    print(f"在列表 {unsorted_list} 中搜索 {target1}:")
    index1_linear = linear_search(unsorted_list, target1)
    print(f"找到索引: {index1_linear} (期望: -1)\n")

    print(f"在列表 {unsorted_list} 中搜索 {target3}:")
    index3_linear = linear_search(unsorted_list, target3)
    print(f"找到索引: {index3_linear} (期望: 3)\n")


    print("--- 二分搜索 (迭代实现) ---")
    print(f"在已排序列表 {my_list} 中搜索 {target1}:")
    index1_binary_iter = binary_search_iterative(my_list, target1)
    print(f"找到索引: {index1_binary_iter} (期望: 5)\n")

    print(f"在已排序列表 {my_list} 中搜索 {target2}:")
    index2_binary_iter = binary_search_iterative(my_list, target2)
    print(f"找到索引: {index2_binary_iter} (期望: -1)\n")


    print("--- 二分搜索 (递归实现) ---")
    print(f"在已排序列表 {my_list} 中搜索 {target1}:")
    index1_binary_rec = binary_search_recursive_wrapper(my_list, target1)
    print(f"找到索引: {index1_binary_rec} (期望: 5)\n")

    print(f"在已排序列表 {my_list} 中搜索 {target2}:")
    index2_binary_rec = binary_search_recursive_wrapper(my_list, target2)
    print(f"找到索引: {index2_binary_rec} (期望: -1)\n")

    # 注意：对未排序列表使用二分搜索会得到错误结果！
    print("--- 注意：对未排序列表使用二分搜索会出错！ ---")
    print(f"在未排序列表 {unsorted_list} 中搜索 {target3} (错误示例):")
    index_error_example = binary_search_iterative(unsorted_list, target3)
    print(f"找到索引 (可能错误): {index_error_example}\n")
"""