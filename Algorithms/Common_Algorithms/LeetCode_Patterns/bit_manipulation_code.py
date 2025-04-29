# algorithms/common_algorithms/leetcode_patterns/bit_manipulation/bit_manipulation_code.py

"""
位运算 (Bit Manipulation) 示例 (Python)

本文件包含了 Python 中位运算的基本概念、常用技巧以及经典问题的实现示例。
位运算直接操作数字的二进制位，通常非常高效。

基本位运算符:
& (AND): 按位与。只有两个对应的二进制位都为 1 时，结果位才为 1。
| (OR):  按位或。两个对应的二进制位中只要有一个为 1 时，结果位就为 1。
^ (XOR): 按位异或。两个对应的二进制位不相同时，结果位才为 1。
~ (NOT): 按位取反。将对应的二进制位翻转 (0 变 1，1 变 0)。在 Python 中是按位取反并取其补码。
<< (左移): 将二进制位向左移动指定的位数，右边空出的位用 0 填充。相当于乘以 2 的幂。
>> (右移): 将二进制位向右移动指定的位数。左边空出的位用最高有效位填充 (有符号数) 或用 0 填充 (无符号数)。对于正整数，相当于除以 2 的幂并取整。
"""

# --- 基本位操作技巧 (Common Bit Manipulation Techniques) ---

def is_even(n):
    """
    检查一个整数是否为偶数 (使用位运算)

    原理：一个数的二进制表示的最低位 (最右边的位) 是 0 则为偶数，是 1 则为奇数。
    将数与 1 (二进制为 ...001) 进行按位与操作，结果为 0 则最低位是 0 (偶数)，结果为 1 则最低位是 1 (奇数)。
    """
    # 确保输入是整数
    if not isinstance(n, int):
        raise TypeError("输入必须是整数")
    return (n & 1) == 0 # 如果 n & 1 的结果为 0，说明最低位是 0，是偶数

def get_bit(n, k):
    """
    获取整数 n 的第 k 位 (从右往左，最低位是第 0 位)

    原理：将 1 向左移动 k 位 (1 << k)，得到一个只有第 k 位是 1 的掩码。
    将 n 与这个掩码进行按位与操作。如果结果不为 0，说明 n 的第 k 位是 1；如果结果为 0，说明 n 的第 k 位是 0。
    """
    if k < 0:
        raise ValueError("位索引 k 不能为负数")
    # (n >> k) 将 n 的第 k 位移动到最低位，然后与 1 进行按位与，只保留最低位
    return (n >> k) & 1

def set_bit(n, k):
    """
    设置整数 n 的第 k 位为 1

    原理：将 1 向左移动 k 位 (1 << k)，得到一个只有第 k 位是 1 的掩码。
    将 n 与这个掩码进行按位或操作。按位或可以确保如果原第 k 位是 0，会变成 1；如果原第 k 位是 1，仍然是 1。
    """
    if k < 0:
        raise ValueError("位索引 k 不能为负数")
    return n | (1 << k)

def clear_bit(n, k):
    """
    清除整数 n 的第 k 位 (设为 0)

    原理：将 1 向左移动 k 位 (1 << k)，得到一个只有第 k 位是 1 的掩码。
    对这个掩码取反 (~(1 << k))，得到一个除了第 k 位是 0 外，其他位都是 1 的掩码。
    将 n 与这个取反后的掩码进行按位与操作。按位与可以确保如果原第 k 位是 1，会变成 0；如果原第 k 位是 0，仍然是 0。
    注意：Python 的按位取反 `~` 是对该数的补码取反，结果会包括负号。但在与正整数进行 `&` 操作时，通常能得到想要的结果。对于非负整数 n 和 k，~(1 << k) 产生的掩码在低位是正确的。
    """
    if k < 0:
        raise ValueError("位索引 k 不能为负数")
    mask = 1 << k
    return n & (~mask)

def toggle_bit(n, k):
    """
    翻转整数 n 的第 k 位 (0 变 1，1 变 0)

    原理：将 1 向左移动 k 位 (1 << k)，得到一个只有第 k 位是 1 的掩码。
    将 n 与这个掩码进行按位异或操作。按位异或可以确保如果原第 k 位与掩码的第 k 位不同 (0^1=1, 1^1=0)，结果是翻转；其他位与掩码的 0 异或保持不变 (0^0=0, 1^0=1)。
    """
    if k < 0:
        raise ValueError("位索引 k 不能为负数")
    return n ^ (1 << k)

def clear_lowest_set_bit(n):
    """
    清除整数 n 的最低设置位 (最右边的 1)

    原理：n - 1 会将 n 最右边的 1 变成 0，并将该 1 右边的所有 0 变成 1。
    将 n 与 n - 1 进行按位与操作。在 n 的最低设置位及其左边，n 和 n-1 的二进制位是相同的。在最低设置位右边，n 是 0，n-1 是 1，按位与结果是 0。最低设置位本身在 n 是 1，在 n-1 是 0，按位与结果是 0。
    因此，只有最低设置位被清除。
    如果 n 为 0，结果仍为 0。
    """
    # 确保输入是非负整数
    if not isinstance(n, int) or n < 0:
        raise ValueError("输入必须是非负整数")
    return n & (n - 1) if n > 0 else 0


def get_lowest_set_bit(n):
    """
    获取整数 n 的最低设置位 (最右边的 1) 对应的数值

    原理：利用二进制补码的性质。-n 的补码等于 n 的补码取反加一。
          对于正整数 n，它的补码就是其本身。
          ~n 会将所有位翻转。~n + 1 会将最右边的 0 及其右边的所有 1 翻转，并将最低设置位左边的位保持不变。
          当 n 与 (-n) 或 (~n + 1) 进行按位与操作时，只有 n 的最低设置位及其右边的 0 会保留，其余位都变成 0。
          例如： n = 12 (00001100)
                ~n =   (11110011) (Python中对正数取反会变成负数，这里只看二进制位)
             ~n + 1 = (11110100) (这是 -12 的补码表示)
                n & (-n) 或 n & (~n + 1) = 00000100 (结果就是 4)
    如果 n 为 0，结果为 0。
    """
    # 确保输入是非负整数
    if not isinstance(n, int) or n < 0:
        raise ValueError("输入必须是非负整数")
    return n & (-n) if n > 0 else 0
    # 或者 return n & (~n + 1) if n > 0 else 0 # 在Python中对于正整数是等价的


# --- 经典问题示例 (Classic Problem Example) ---

def single_number(nums):
    """
    找出数组中唯一出现一次的数字

    LeetCode 问题：给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现一次的元素。

    原理：利用按位异或 (XOR) 的性质：
          - 任何数和 0 异或，结果是其本身 (a ^ 0 = a)。
          - 任何数和自身异或，结果是 0 (a ^ a = 0)。
          - 异或操作满足交换律和结合律 (a ^ b ^ c = a ^ (b ^ c) = (a ^ b) ^ c)。
    将数组中所有元素进行异或。所有出现两次的元素都会两两异或为 0，最终结果就是那个只出现一次的数字与 0 的异或，即它本身。
    时间复杂度：O(n) - 只需遍历一次数组。
    空间复杂度：O(1) - 不需要额外的存储空间。
    """
    result = 0 # 异或的初始值是 0

    for num in nums:
        result ^= num # 将当前元素与结果进行异或

    return result

"""
# --- 运行示例 ---
if __name__ == "__main__":
    print("--- 基本位运算符示例 ---")
    a = 60  # Binary: 0011 1100
    b = 13  # Binary: 0000 1101
    print(f"a = {a} (Binary: {bin(a)[2:]})")
    print(f"b = {b} (Binary: {bin(b)[2:]})")

    print(f"a & b (按位与): {a & b} (Binary: {bin(a & b)[2:]})")   # 12 (0000 1100)
    print(f"a | b (按位或): {a | b} (Binary: {bin(a | b)[2:]})")   # 61 (0011 1101)
    print(f"a ^ b (按位异或): {a ^ b} (Binary: {bin(a ^ b)[2:]})") # 49 (0011 0001)
    # ~a (按位取反) 在 Python 中会得到负数，取决于位数表示，这里只看正数演示低位
    # print(f"~a (按位取反): {~a}") # e.g., -61

    print(f"a << 2 (左移 2 位): {a << 2} (Binary: {bin(a << 2)[2:]})") # 240 (1111 0000)
    print(f"a >> 2 (右移 2 位): {a >> 2} (Binary: {bin(a >> 2)[2:]})") # 15 (0000 1111)

    print("\n--- 基本位操作技巧示例 ---")
    num_check = 12
    print(f"数字 {num_check}:")
    print(f"是偶数吗? {is_even(num_check)}") # 期望: True
    print(f"是奇数吗? {not is_even(num_check)}") # 期望: False

    num_check = 13
    print(f"\n数字 {num_check}:")
    print(f"是偶数吗? {is_even(num_check)}") # 期望: False
    print(f"是奇数吗? {not is_even(num_check)}") # 期望: True

    num_bit = 20 # Binary: 010100
    k_bit = 2
    print(f"\n数字 {num_bit} (Binary: {bin(num_bit)[2:]}):")
    print(f"第 {k_bit} 位是: {get_bit(num_bit, k_bit)}") # 期望: 1 (因为 20 的二进制是 10100, 第2位是1)
    k_bit = 3
    print(f"第 {k_bit} 位是: {get_bit(num_bit, k_bit)}") # 期望: 0

    num_set = 20 # Binary: 010100
    k_set = 1
    print(f"\n设置数字 {num_set} 的第 {k_set} 位为 1:")
    result_set = set_bit(num_set, k_set)
    print(f"结果: {result_set} (Binary: {bin(result_set)[2:]})") # 期望: 22 (010110)

    num_clear = 20 # Binary: 010100
    k_clear = 2
    print(f"\n清除数字 {num_clear} 的第 {k_clear} 位 (设为 0):")
    result_clear = clear_bit(num_clear, k_clear)
    print(f"结果: {result_clear} (Binary: {bin(result_clear)[2:]})") # 期望: 16 (010000)

    num_toggle = 20 # Binary: 010100
    k_toggle = 3
    print(f"\n翻转数字 {num_toggle} 的第 {k_toggle} 位:")
    result_toggle = toggle_bit(num_toggle, k_toggle)
    print(f"结果: {result_toggle} (Binary: {bin(result_toggle)[2:]})") # 期望: 28 (011100)

    num_lowest = 12 # Binary: 1100
    print(f"\n数字 {num_lowest} (Binary: {bin(num_lowest)[2:]}):")
    print(f"清除最低设置位: {clear_lowest_set_bit(num_lowest)} (Binary: {bin(clear_lowest_set_bit(num_lowest))[2:]})") # 期望: 8 (1000)
    print(f"获取最低设置位的值: {get_lowest_set_bit(num_lowest)} (Binary: {bin(get_lowest_set_bit(num_lowest))[2:]})") # 期望: 4 (0100)

    num_lowest = 8 # Binary: 1000
    print(f"\n数字 {num_lowest} (Binary: {bin(num_lowest)[2:]}):")
    print(f"清除最低设置位: {clear_lowest_set_bit(num_lowest)} (Binary: {bin(clear_lowest_set_bit(num_lowest))[2:]})") # 期望: 0 (0000)
    print(f"获取最低设置位的值: {get_lowest_set_bit(num_lowest)} (Binary: {bin(get_lowest_set_bit(num_lowest))[2:]})") # 期望: 8 (1000)


    print("\n--- 经典问题示例 (找出只出现一次的数字) ---")
    nums_single = [2, 2, 1]
    print(f"数组: {nums_single}")
    print(f"只出现一次的数字是: {single_number(nums_single)}") # 期望: 1

    nums_single_2 = [4, 1, 2, 1, 2]
    print(f"\n数组: {nums_single_2}")
    print(f"只出现一次的数字是: {single_number(nums_single_2)}") # 期望: 4

    nums_single_3 = [1]
    print(f"\n数组: {nums_single_3}")
    print(f"只出现一次的数字是: {single_number(nums_single_3)}") # 期望: 1
"""