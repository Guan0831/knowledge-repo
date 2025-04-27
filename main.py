from Data_Analysis.Common_Algorithms.Basics import recursion_backtracking_examples
if __name__ == "__main__":
    print("--- 递归示例 ---")
    num_factorial = 5
    print(f"{num_factorial}! = {recursion_backtracking_examples.factorial_recursive(num_factorial)}") # 期望: 120

    num_fib = 10
    print(f"斐波那契数列第 {num_fib} 项 (递归): {recursion_backtracking_examples.fibonacci_recursive(num_fib)}") # 期望: 55
    # 注意：fibonacci_recursive(40) 以上可能会非常慢！

    print("\n--- 回溯示例 ---")
    test_items_perm = [1, 2, 3]
    print(f"列表 {test_items_perm} 的全排列:")
    all_permutations = recursion_backtracking_examples.permutations(test_items_perm)
    print(all_permutations)
    # 期望: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]] (顺序可能不同)

    test_items_comb = [1, 2, 3, 4]
    comb_size = 2
    print(f"\n从列表 {test_items_comb} 中选择 {comb_size} 个元素的组合:")
    all_combinations = recursion_backtracking_examples.combinations(test_items_comb, comb_size)
    print(all_combinations)
    # 期望: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] (顺序可能不同)

    test_items_comb_2 = ["a", "b", "c"]
    comb_size_2 = 3
    print(f"\n从列表 {test_items_comb_2} 中选择 {comb_size_2} 个元素的组合:")
    all_combinations_2 = recursion_backtracking_examples.combinations(test_items_comb_2, comb_size_2)
    print(all_combinations_2)
    # 期望: [['a', 'b', 'c']]
