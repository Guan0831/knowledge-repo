# algorithms/common_algorithms/leetcode_patterns/two_pointers_sliding_window/two_pointers_sliding_window_code.py

"""
双指针与滑动窗口算法示例 (Python)

本文件包含了双指针和滑动窗口两种常用技巧的经典问题实现。
这些技巧常用于数组和字符串问题，能有效降低时间复杂度。
"""

# --- 双指针示例 (Two Pointers Examples) ---

def two_sum_sorted(arr, target):
    """
    在 **有序** 数组中查找和为目标值的两个数 (双指针)

    原理：使用两个指针，一个从数组头部开始 (left)，一个从数组尾部开始 (right)。
         如果 arr[left] + arr[right] 等于目标值，则找到。
         如果和小于目标值，说明 left 指向的数太小，需要增大和，将 left 右移。
         如果和大于目标值，说明 right 指向的数太大，需要减小和，将 right 左移。
    时间复杂度：O(n) - 只需要一次遍历。
    空间复杂度：O(1) - 原地操作。
    先决条件：输入数组必须是已排序的。
    返回值：如果找到，返回两个数的索引 (left, right) 的元组；如果找不到，返回 (-1, -1)。
    """
    left, right = 0, len(arr) - 1 # 初始化左右指针

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return (left, right) # 找到，返回索引
        elif current_sum < target:
            left += 1 # 和太小，left 右移增大和
        else: # current_sum > target
            right -= 1 # 和太大，right 左移减小和

    return (-1, -1) # 遍历结束，没有找到

def reverse_list_inplace(arr):
    """
    原地反转列表 (双指针)

    原理：使用两个指针，一个从列表头部开始 (left)，一个从列表尾部开始 (right)。
         交换 left 和 right 指针指向的元素，然后 left 右移，right 左移，直到 left >= right。
    时间复杂度：O(n) - 只需遍历列表的一半。
    空间复杂度：O(1) - 原地操作。
    注意：直接修改输入的列表。
    """
    left, right = 0, len(arr) - 1 # 初始化左右指针

    while left < right:
        # 交换 left 和 right 指针指向的元素
        arr[left], arr[right] = arr[right], arr[left]
        left += 1  # left 右移
        right -= 1 # right 左移

    return arr # 返回修改后的列表

def remove_duplicates_sorted(arr):
    """
    移除 **有序** 数组中的重复项 (双指针)

    原理：使用两个指针，一个慢指针 (slow) 记录不重复元素的下一个位置，一个快指针 (fast) 遍历整个数组。
         如果 fast 指向的元素与 slow 指向的元素不同，说明 fast 指向的是一个新的不重复元素，将其放到 slow 指针的下一个位置，并移动 slow。
    时间复杂度：O(n) - 只需一次遍历。
    空间复杂度：O(1) - 原地修改。
    先决条件：输入数组必须是已排序的。
    返回值：不重复元素的个数 (也是修改后数组的有效长度)。原数组会被修改。
    """
    if not arr:
        return 0

    slow = 0 # 慢指针，指向当前不重复元素的最后一个位置

    # 快指针遍历整个数组
    for fast in range(1, len(arr)):
        # 如果快指针指向的元素与慢指针指向的元素不同
        if arr[fast] != arr[slow]:
            slow += 1 # 慢指针向前移动一位
            arr[slow] = arr[fast] # 将快指针指向的新不重复元素放到慢指针位置

    # slow 指针的索引 + 1 就是不重复元素的个数
    return slow + 1

def is_palindrome(s):
    """
    判断字符串是否为回文串 (忽略大小写和非字母数字字符) (双指针)

    原理：使用两个指针，一个从字符串头部开始 (left)，一个从尾部开始 (right)。
         在移动指针时，跳过非字母数字字符。
         比较 left 和 right 指针指向的字符 (忽略大小写)，如果不相等则不是回文串。
    时间复杂度：O(n) - 只需遍历字符串。
    空间复杂度：O(1) - 原地操作 (不考虑字符串本身存储)。
    """
    left, right = 0, len(s) - 1 # 初始化左右指针

    while left < right:
        # 移动 left 指针直到指向一个字母数字字符
        while left < right and not s[left].isalnum():
            left += 1
        # 移动 right 指针直到指向一个字母数字字符
        while left < right and not s[right].isalnum():
            right -= 1

        # 如果 left 仍然小于 right (说明还没相遇或越过)
        if left < right:
            # 比较左右指针指向的字符 (忽略大小写)
            if s[left].lower() != s[right].lower():
                return False # 不相等，不是回文串

            # 字符相等，继续向内移动指针
            left += 1
            right -= 1

    return True # 循环结束，说明是回文串


# --- 滑动窗口示例 (Sliding Window Examples) ---

def max_subarray_sum_fixed(arr, k):
    """
    查找固定大小 (k) 子数组的最大和 (滑动窗口 - 固定大小)

    原理：维护一个大小为 k 的窗口，在数组上滑动。
         计算第一个窗口的和。
         然后，每当窗口向右移动一步，就在当前窗口和中减去窗口最左边的元素，加上新进入窗口的最右边的元素。
         同时跟踪过程中遇到的最大窗口和。
    时间复杂度：O(n) - 一次遍历计算所有窗口的和。
    空间复杂度：O(1) - 不需要额外的空间存储窗口。
    """
    if k > len(arr) or k <= 0:
        return 0 # 无效输入

    # 计算第一个窗口 (前 k 个元素) 的和
    current_window_sum = sum(arr[:k])
    max_sum = current_window_sum # 初始化最大和为第一个窗口的和

    # 滑动窗口，从第二个窗口开始 (索引 k 到 len(arr)-1)
    # window_end 是窗口的最右边元素的索引
    for window_end in range(k, len(arr)):
        # 滑动一步：减去最左边的元素，加上新进入的元素
        # window_end - k 是当前窗口最左边元素的索引
        current_window_sum = current_window_sum - arr[window_end - k] + arr[window_end]

        # 更新最大和
        max_sum = max(max_sum, current_window_sum)

    return max_sum

def length_of_longest_substring(s):
    """
    查找最长无重复字符的子串的长度 (滑动窗口 - 可变大小)

    原理：使用一个可变大小的窗口 [window_start, window_end]。
         使用一个集合 (或者字典) 来记录当前窗口中出现的字符。
         window_end 不断向右移动，扩大窗口。
         如果遇到重复字符，说明当前窗口不再满足“无重复字符”的条件，需要收缩窗口：移动 window_start 向右，直到重复字符不再在窗口中。
         在窗口滑动过程中，不断更新最长无重复子串的长度 (当前窗口的大小)。
    时间复杂度：O(n) - 每个字符最多被 window_start 和 window_end 各访问一次。
    空间复杂度：O(min(n, alphabet_size)) - 集合或字典存储窗口中的字符，最大大小取决于字符串长度或字符集大小。
    """
    if not s:
        return 0

    window_start = 0 # 滑动窗口的起始索引
    max_length = 0   # 记录最长无重复子串的长度
    char_set = set() # 使用集合来记录当前窗口中出现的字符

    # window_end 是滑动窗口的结束索引，不断向右移动
    for window_end in range(len(s)):
        right_char = s[window_end] # 当前考察的字符 (窗口最右边的字符)

        # 如果当前字符已经在窗口的集合中 (说明重复了)
        while right_char in char_set:
            # 移除窗口最左边的字符，并收缩窗口 (window_start 右移)
            left_char = s[window_start]
            char_set.remove(left_char)
            window_start += 1

        # 当前字符不在窗口集合中，将其加入 (扩大窗口)
        char_set.add(right_char)

        # 更新最长无重复子串的长度 (当前窗口的大小是 window_end - window_start + 1)
        max_length = max(max_length, window_end - window_start + 1)

    return max_length

"""

# --- 示例用法 ---
if __name__ == "__main__":
    print("--- 双指针示例 ---")
    sorted_arr_sum = [1, 2, 3, 4, 5, 6, 7]
    target_sum = 10
    print(f"在有序数组 {sorted_arr_sum} 中查找和为 {target_sum} 的两个数:")
    indices = two_sum_sorted(sorted_arr_sum, target_sum)
    print(f"找到的索引: {indices}\n") # 期望: (3, 6) 或 (4, 5) 等一对

    arr_to_reverse = [1, 2, 3, 4, 5]
    print(f"原地反转列表 {arr_to_reverse}:")
    reversed_arr = reverse_list_inplace(arr_to_reverse)
    print(f"反转后列表: {reversed_arr}\n") # 期望: [5, 4, 3, 2, 1]

    sorted_arr_dup = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print(f"移除有序数组 {sorted_arr_dup} 中的重复项:")
    # 复制一份用于显示原始数组
    original_arr_dup = list(sorted_arr_dup)
    new_length = remove_duplicates_sorted(sorted_arr_dup)
    print(f"原数组: {original_arr_dup}")
    print(f"新长度: {new_length}")
    print(f"修改后数组 (前 {new_length} 个元素): {sorted_arr_dup[:new_length]}\n") # 期望新长度: 5, 数组前5个元素: [0, 1, 2, 3, 4]

    string_palindrome_1 = "A man, a plan, a canal: Panama"
    string_palindrome_2 = "race a car"
    print(f"判断 '{string_palindrome_1}' 是否回文: {is_palindrome(string_palindrome_1)}") # 期望: True
    print(f"判断 '{string_palindrome_2}' 是否回文: {is_palindrome(string_palindrome_2)}\n") # 期望: False


    print("--- 滑动窗口示例 ---")
    arr_window = [1, 4, 2, 10, 23, 3, 1, 0, 20]
    k_window = 4
    print(f"在列表 {arr_window} 中查找大小为 {k_window} 的子数组的最大和:")
    max_sum = max_subarray_sum_fixed(arr_window, k_window)
    print(f"最大和: {max_sum}\n") # 期望: 39 (子数组 [4, 2, 10, 23])

    string_window = "abcabcbb"
    print(f"字符串 '{string_window}' 的最长无重复字符子串长度:")
    length1 = length_of_longest_substring(string_window)
    print(f"长度: {length1}\n") # 期望: 3 ("abc")

    string_window_2 = "pwwkew"
    print(f"字符串 '{string_window_2}' 的最长无重复字符子串长度:")
    length2 = length_of_longest_substring(string_window_2)
    print(f"长度: {length2}\n") # 期望: 3 ("wke" 或 "kew")

    string_window_3 = "bbbbb"
    print(f"字符串 '{string_window_3}' 的最长无重复字符子串长度:")
    length3 = length_of_longest_substring(string_window_3)
    print(f"长度: {length3}\n") # 期望: 1 ("b")

"""