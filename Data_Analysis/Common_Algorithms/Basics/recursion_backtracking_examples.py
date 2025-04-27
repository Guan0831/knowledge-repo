# algorithms/common_algorithms/basics/recursion_backtracking_examples.py

"""
递归与回溯算法示例 (Python)

本文件包含了递归和回溯两种算法思想的经典示例实现。
回溯通常是递归的一种应用，用于在解决问题时探索所有可能的路径。
"""

# --- 递归示例 (Recursion Examples) ---

def factorial_recursive(n):
    """
    计算阶乘 (递归实现) - n!

    原理：n! = n * (n-1)!，直到 n=0 时 0! = 1。
    Base Case: 当 n <= 1 时，返回 1。
    Recursive Step: 返回 n * factorial_recursive(n-1)。
    """
    if n < 0:
        raise ValueError("阶乘不接受负数")
    if n == 0 or n == 1:
        return 1  # Base Case
    else:
        return n * factorial_recursive(n - 1) # Recursive Step

def fibonacci_recursive(n):
    """
    计算斐波那契数列的第 n 项 (递归实现)

    原理：F(n) = F(n-1) + F(n-2)。
    Base Cases: F(0) = 0, F(1) = 1。
    Recursive Step: 返回 fibonacci_recursive(n-1) + fibonacci_recursive(n-2)。
    注意：这种朴素递归实现存在大量重复计算，效率较低，通常需要记忆化或动态规划优化。
    """
    if n < 0:
        raise ValueError("斐波那契数列项数不能为负")
    if n == 0:
        return 0  # Base Case 1
    elif n == 1:
        return 1  # Base Case 2
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2) # Recursive Step

# --- 回溯示例 (Backtracking Examples) ---

def permutations(items):
    """
    生成列表的所有全排列 (回溯实现)

    原理：在每一步，从未选择的元素中选择一个加入当前排列，然后递归地对剩余元素进行同样操作。
         当当前排列长度等于原始列表长度时，得到一个完整的排列，记录下来。
         递归返回时，撤销当前的选择，尝试其他可能性 (回溯)。
    """
    result = [] # 存放所有生成的排列
    n = len(items)
    used = [False] * n # 标记哪些元素已经被使用过

    def backtrack(current_permutation):
        # Base Case: 当当前排列的长度等于原始列表长度时，说明生成了一个完整的排列
        if len(current_permutation) == n:
            result.append(list(current_permutation)) # 记录当前排列 (注意复制一份)
            return

        # Recursive Step (Choices, Constraints, Explore, Un-choose)
        for i in range(n):
            # 选择 (Choice): 如果第 i 个元素还没有被使用过
            if not used[i]:
                # 标记为已使用，并将元素加入当前排列
                used[i] = True
                current_permutation.append(items[i])

                # 探索 (Explore): 递归调用，继续构建下一个位置的排列
                backtrack(current_permutation)

                # 撤销选择 (Un-choose/Backtrack): 递归返回后，撤销刚刚的选择
                # 1. 将元素从当前排列中移除
                current_permutation.pop()
                # 2. 将元素标记为未使用，以便其他路径可以选择它
                used[i] = False

    # 从空的排列开始回溯
    backtrack([])
    return result

def combinations(items, k):
    """
    生成列表 items 中选择 k 个元素的所有组合 (回溯实现)

    原理：在每一步，从剩余可选的元素中选择一个加入当前组合，然后递归地对剩余元素进行同样操作。
         为了避免重复的组合 (如 [1, 2] 和 [2, 1] 是同一个组合)，我们在选择时限定必须从当前选择的元素之后开始选择。
         当当前组合包含 k 个元素时，记录下来。
         递归返回时，撤销当前的选择，尝试其他可能性 (回溯)。
    """
    result = [] # 存放所有生成的组合
    n = len(items)

    # start_index: 本次选择可以从原始列表的哪个索引开始，确保不重复
    def backtrack(start_index, current_combination):
        # Base Case: 当当前组合的长度等于 k 时，说明生成了一个完整的组合
        if len(current_combination) == k:
            result.append(list(current_combination)) # 记录当前组合 (注意复制一份)
            return

        # Recursive Step (Choices, Constraints, Explore, Un-choose)
        # 从 start_index 开始遍历可选的元素
        for i in range(start_index, n):
            # 选择 (Choice): 将第 i 个元素加入当前组合
            current_combination.append(items[i])

            # 探索 (Explore): 递归调用，继续选择下一个元素
            # 注意：下一次选择必须从 i+1 开始，避免重复组合和考虑已经选择的元素
            backtrack(i + 1, current_combination)

            # 撤销选择 (Un-choose/Backtrack): 递归返回后，撤销刚刚的选择
            # 将元素从当前组合中移除
            current_combination.pop()

    # 从索引 0 开始，空组合开始回溯
    backtrack(0, [])
    return result

# --- 示例用法 ---
"""
if __name__ == "__main__":
    print("--- 递归示例 ---")
    num_factorial = 5
    print(f"{num_factorial}! = {factorial_recursive(num_factorial)}") # 期望: 120

    num_fib = 10
    print(f"斐波那契数列第 {num_fib} 项 (递归): {fibonacci_recursive(num_fib)}") # 期望: 55
    # 注意：fibonacci_recursive(40) 以上可能会非常慢！

    print("\n--- 回溯示例 ---")
    test_items_perm = [1, 2, 3]
    print(f"列表 {test_items_perm} 的全排列:")
    all_permutations = permutations(test_items_perm)
    print(all_permutations)
    # 期望: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]] (顺序可能不同)

    test_items_comb = [1, 2, 3, 4]
    comb_size = 2
    print(f"\n从列表 {test_items_comb} 中选择 {comb_size} 个元素的组合:")
    all_combinations = combinations(test_items_comb, comb_size)
    print(all_combinations)
    # 期望: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] (顺序可能不同)

    test_items_comb_2 = ["a", "b", "c"]
    comb_size_2 = 3
    print(f"\n从列表 {test_items_comb_2} 中选择 {comb_size_2} 个元素的组合:")
    all_combinations_2 = combinations(test_items_comb_2, comb_size_2)
    print(all_combinations_2)
    # 期望: [['a', 'b', 'c']]
"""    