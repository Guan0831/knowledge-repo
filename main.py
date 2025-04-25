from Data_Analysis.Common_Algorithms.Basics import searching_code
if __name__ == "__main__":
    my_list = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] # 二分搜索需要已排序列表
    unsorted_list = [5, 2, 8, 1, 9, 4] # 线性搜索可以使用未排序列表

    target1 = 23
    target2 = 100
    target3 = 1

    print("--- 线性搜索 ---")
    print(f"在列表 {unsorted_list} 中搜索 {target1}:")
    index1_linear = searching_code.linear_search(unsorted_list, target1)
    print(f"找到索引: {index1_linear} (期望: -1)\n")

    print(f"在列表 {unsorted_list} 中搜索 {target3}:")
    index3_linear = searching_code.linear_search(unsorted_list, target3)
    print(f"找到索引: {index3_linear} (期望: 3)\n")


    print("--- 二分搜索 (迭代实现) ---")
    print(f"在已排序列表 {my_list} 中搜索 {target1}:")
    index1_binary_iter = searching_code.binary_search_iterative(my_list, target1)
    print(f"找到索引: {index1_binary_iter} (期望: 5)\n")

    print(f"在已排序列表 {my_list} 中搜索 {target2}:")
    index2_binary_iter = searching_code.binary_search_iterative(my_list, target2)
    print(f"找到索引: {index2_binary_iter} (期望: -1)\n")


    print("--- 二分搜索 (递归实现) ---")
    print(f"在已排序列表 {my_list} 中搜索 {target1}:")
    index1_binary_rec = searching_code.binary_search_recursive_wrapper(my_list, target1)
    print(f"找到索引: {index1_binary_rec} (期望: 5)\n")

    print(f"在已排序列表 {my_list} 中搜索 {target2}:")
    index2_binary_rec = searching_code.binary_search_recursive_wrapper(my_list, target2)
    print(f"找到索引: {index2_binary_rec} (期望: -1)\n")

    # 注意：对未排序列表使用二分搜索会得到错误结果！
    print("--- 注意：对未排序列表使用二分搜索会出错！ ---")
    print(f"在未排序列表 {unsorted_list} 中搜索 {target3} (错误示例):")
    index_error_example = searching_code.binary_search_iterative(unsorted_list, target3)
    print(f"找到索引 (可能错误): {index_error_example}\n")