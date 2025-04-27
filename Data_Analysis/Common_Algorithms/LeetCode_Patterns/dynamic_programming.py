# algorithms/common_algorithms/leetcode_patterns/dp_code_examples.py

"""
动态规划 (Dynamic Programming) 经典问题实现 (Python)

本文件包含了几个经典的动态规划问题的 Python 实现，
旨在演示 DP 的核心思想：重叠子问题和最优子结构，以及记忆化和列表法两种实现方式。
"""

import functools # 用于使用 @lru_cache 装饰器进行记忆化

# --- 示例 1: 斐波那契数列 (展示记忆化和列表法) ---

def fibonacci_memoization(n):
    """
    斐波那契数列 - 记忆化搜索 (Top-Down DP)

    原理：使用一个字典或列表来存储已经计算过的斐波那契项，避免重复计算。
    状态定义：dp[i] 表示斐波那契数列的第 i 项。
    状态转移方程：dp[i] = dp[i-1] + dp[i-2]。
    基本情况：dp[0] = 0, dp[1] = 1。
    """
    if n < 0:
        raise ValueError("斐波那契数列项数不能为负")

    # 使用字典作为缓存 (或者使用 functools.lru_cache)
    memo = {}

    def fib_helper(k):
        if k in memo:
            return memo[k] # 如果已经计算过，直接返回

        # 基本情况
        if k == 0:
            return 0
        elif k == 1:
            return 1

        # 递归计算并存储结果
        memo[k] = fib_helper(k - 1) + fib_helper(k - 2)
        return memo[k]

    return fib_helper(n)

@functools.lru_cache(None) # Python 3.2+ 内置的记忆化装饰器，非常方便
def fibonacci_memoization_lru(n):
    """
    斐波那契数列 - 记忆化搜索 (@lru_cache 实现)

    与上面手动记忆化相同原理，使用 Python 内置装饰器更简洁。
    """
    if n < 0:
        raise ValueError("斐波那契数列项数不能为负")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_memoization_lru(n - 1) + fibonacci_memoization_lru(n - 2)


def fibonacci_tabulation(n):
    """
    斐波那契数列 - 列表法 (Bottom-Up DP)

    原理：创建一个 DP 表，从基本情况开始，逐步计算到目标项。
    状态定义：dp[i] 表示斐波那契数列的第 i 项。
    状态转移方程：dp[i] = dp[i-1] + dp[i-2]。
    基本情况：dp[0] = 0, dp[1] = 1。
    """
    if n < 0:
        raise ValueError("斐波那契数列项数不能为负")
    if n == 0:
        return 0
    if n == 1:
        return 1

    # 创建 DP 表，大小为 n+1
    dp = [0] * (n + 1)

    # 设置基本情况
    dp[0] = 0
    dp[1] = 1

    # 从基本情况开始，迭代计算到 n
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# --- 示例 2: 爬楼梯 (Climbing Stairs) ---

def climb_stairs(n):
    """
    爬楼梯

    LeetCode 问题：假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶？

    原理：到达第 n 阶的方法数等于到达第 n-1 阶的方法数加上到达第 n-2 阶的方法数。
    状态定义：dp[i] 表示到达第 i 阶的方法数。
    状态转移方程：dp[i] = dp[i-1] + dp[i-2]。
    基本情况：dp[0] = 1 (到达 0 阶有 1 种方法，即不爬), dp[1] = 1 (到达 1 阶有 1 种方法，爬一步)。
             (注意基本情况定义可以有多种方式，这里 dp[0]=1 是为了方便后续迭代)
    实现方式：列表法 (Bottom-Up DP)
    """
    if n < 0:
        return 0
    if n == 0: # 到达 0 阶的方法数
        return 1
    if n == 1: # 到达 1 阶的方法数
        return 1

    # 创建 DP 表
    dp = [0] * (n + 1)

    # 设置基本情况
    dp[0] = 1
    dp[1] = 1

    # 迭代计算到第 n 阶
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# --- 示例 3: 零钱兑换 II (组合数) (Coin Change II) ---

def change(amount, coins):
    """
    零钱兑换 II (计算组合数)

    LeetCode 问题：给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每种硬币的数量是无限的。

    原理：考虑每种硬币的使用，更新能凑成的金额数。这是一个背包问题的变种 (完全背包)。
    状态定义：dp[i] 表示凑成总金额 i 的组合数。
    状态转移方程：dp[i] += dp[i - coin] 对于每种硬币 coin。
                   这里的关键是遍历硬币在外层循环，金额在内层循环，保证每种组合只计算一次。
    基本情况：dp[0] = 1 (凑成金额 0 的组合数是 1 种，即不选任何硬币)。
    实现方式：列表法 (Bottom-Up DP)
    """
    # 创建 DP 表，dp[i] 表示凑成金额 i 的组合数
    dp = [0] * (amount + 1)

    # 基本情况：凑成金额 0 的组合数是 1 种
    dp[0] = 1

    # 遍历每种硬币
    for coin in coins:
        # 遍历从当前硬币面额到总金额的所有金额
        for i in range(coin, amount + 1):
            # 如果当前金额 i 可以通过减去 coin 得到 (i - coin >= 0)
            # 那么凑成金额 i 的组合数就加上凑成金额 i - coin 的组合数
            dp[i] += dp[i - coin]

    return dp[amount]

# --- 示例 4: 0/1 背包问题 (0/1 Knapsack Problem) ---

def knapsack_01(weights, values, capacity):
    """
    0/1 背包问题

    问题：给定 n 个物品，每个物品有重量 w[i] 和价值 v[i]。
          一个背包有最大承载重量 capacity。
          每个物品只能选择一次 (0 或 1 次)。
          问：在不超过背包容量的前提下，能放入背包的最大总价值是多少？

    原理：考虑每个物品是否放入背包。
    状态定义：dp[i][w] 表示考虑前 i 个物品，背包容量为 w 时的最大价值。
    状态转移方程：
        如果第 i 个物品的重量 w[i-1] > 当前容量 w：
            dp[i][w] = dp[i-1][w] (第 i 个物品放不进，最大价值等于考虑前 i-1 个物品时的最大价值)
        如果第 i 个物品的重量 w[i-1] <= 当前容量 w：
            dp[i][w] = max(dp[i-1][w],  # 不放第 i 个物品
                           dp[i-1][w - weights[i-1]] + values[i-1]) # 放第 i 个物品
    基本情况：dp[0][w] = 0 (没有物品时，价值为 0)
              dp[i][0] = 0 (背包容量为 0 时，价值为 0)
    实现方式：列表法 (Bottom-Up DP)
    空间优化：可以优化到 O(capacity) 空间，但这里先展示 O(n * capacity) 的二维表。
    """
    n = len(weights)
    # 创建 DP 表，dp[i][w] 表示考虑前 i 个物品，容量 w 时的最大价值
    # dp 大小为 (n+1) * (capacity+1)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # 遍历物品 (从第 1 个物品到第 n 个物品)
    for i in range(1, n + 1):
        # 遍历背包容量
        for w in range(capacity + 1):
            # 第 i 个物品的索引是 i-1 (因为物品是从 1 计数，列表从 0 计数)
            current_weight = weights[i - 1]
            current_value = values[i - 1]

            # 如果当前物品的重量大于当前背包容量，则放不进
            if current_weight > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # 否则，考虑放或不放，取最大价值
                # 不放：dp[i-1][w]
                # 放：dp[i-1][w - current_weight] + current_value
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - current_weight] + current_value)

    # 最终结果在 dp[n][capacity]
    return dp[n][capacity]

# --- 示例 5: 最长递增子序列 (LIS) ---

def length_of_lis(nums):
    """
    最长递增子序列 (Longest Increasing Subsequence)

    LeetCode 问题：给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

    原理：对于列表中的每个元素，找到以该元素结尾的最长递增子序列的长度。
    状态定义：dp[i] 表示以 nums[i] 结尾的最长严格递增子序列的长度。
    状态转移方程：dp[i] = max(dp[j] + 1) 对于所有 j < i 且 nums[j] < nums[i]。
                  如果 nums[i] 不能接在任何前面的元素之后，则 dp[i] = 1 (以自身构成长度为 1 的子序列)。
    基本情况：dp[i] 初始都设为 1 (每个元素自身都可以构成长度为 1 的递增子序列)。
    最终结果：整个 dp 数组中的最大值。
    实现方式：列表法 (Bottom-Up DP)
    时间复杂度优化：可以使用二分查找将 O(n^2) 优化到 O(n log n)，这里先展示 O(n^2)。
    """
    if not nums:
        return 0

    n = len(nums)
    # 创建 DP 表，dp[i] 表示以 nums[i] 结尾的最长递增子序列的长度
    dp = [1] * n # 初始化每个元素自己的 LIS 长度都是 1

    # 遍历每个元素 i (从第二个元素开始)
    for i in range(1, n):
        # 遍历 i 之前的所有元素 j
        for j in range(i):
            # 如果 nums[i] 大于 nums[j] (可以接在以 nums[j] 结尾的递增子序列后面)
            if nums[i] > nums[j]:
                # 更新 dp[i]，取当前值和 (以 nums[j] 结尾的 LIS 长度 + 1) 中的最大值
                dp[i] = max(dp[i], dp[j] + 1)

    # 最长递增子序列的长度是 dp 数组中的最大值
    return max(dp)

"""
# --- 示例用法 ---
if __name__ == "__main__":
    print("--- 斐波那契数列 ---")
    n_fib = 15
    print(f"斐波那契数列第 {n_fib} 项 (记忆化): {fibonacci_memoization(n_fib)}")
    # 使用 lru_cache 时，第一次运行时会计算并缓存，后续调用如果参数相同会很快
    print(f"斐波那契数列第 {n_fib} 项 (@lru_cache 记忆化): {fibonacci_memoization_lru(n_fib)}")
    print(f"斐波那契数列第 {n_fib} 项 (列表法): {fibonacci_tabulation(n_fib)}\n")

    print("--- 爬楼梯 ---")
    n_stairs = 10
    print(f"爬 {n_stairs} 阶楼梯的方法数: {climb_stairs(n_stairs)}\n") # 期望: 89

    print("--- 零钱兑换 II (组合数) ---")
    amount_change = 5
    coins_change = [1, 2, 5]
    print(f"凑成金额 {amount_change} 的组合数 (硬币: {coins_change}): {change(amount_change, coins_change)}\n") # 期望: 4 (1+1+1+1+1, 1+1+1+2, 1+2+2, 5)

    print("--- 0/1 背包问题 ---")
    weights_knapsack = [1, 4, 3]
    values_knapsack = [1500, 3000, 2000]
    capacity_knapsack = 4
    print(f"背包容量 {capacity_knapsack}, 物品重量 {weights_knapsack}, 物品价值 {values_knapsack}")
    print(f"最大总价值: {knapsack_01(weights_knapsack, values_knapsack, capacity_knapsack)}\n") # 期望: 3500 (选择物品 1 和 3)

    print("--- 最长递增子序列 (LIS) ---")
    nums_lis = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"列表 {nums_lis} 的最长递增子序列长度: {length_of_lis(nums_lis)}\n") # 期望: 4 (序列如 2, 3, 7, 18 或 2, 5, 7, 18 等)

    nums_lis_2 = [0, 1, 0, 3, 2, 3]
    print(f"列表 {nums_lis_2} 的最长递增子序列长度: {length_of_lis(nums_lis_2)}\n") # 期望: 4 (序列如 0, 1, 2, 3)
    
"""