from Data_Analysis.Common_Algorithms.Basics import sorting_code
import heapq

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
        sorted_arr = sorting_code.bubble_sort(arr)
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
        sorted_arr = sorting_code.selection_sort(arr)
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
        sorted_arr = sorting_code.insertion_sort(arr)
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
        sorted_arr = sorting_code.merge_sort(arr)
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
        sorted_arr = sorting_code.quick_sort(arr)
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
        sorted_arr = sorting_code.heap_sort(arr)
        print(f"排序结果: {sorted_arr}\n")