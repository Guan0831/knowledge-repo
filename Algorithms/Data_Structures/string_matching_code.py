# algorithms/common_algorithms/beyond_basics/string_matching/string_matching_code.py

"""
字符串匹配 (String Matching) 算法实现示例 (Python)

本文件包含了朴素匹配算法和 Knuth-Morris-Pratt (KMP) 算法的 Python 实现。
字符串匹配是在文本串中查找模式串出现位置的问题。

算法比较：
- 朴素匹配: 简单直观，但效率较低。最坏时间复杂度 O(n*m)。
- KMP: 利用模式串自身的特性 (部分匹配表 LPS)，避免不必要的比较，效率高。时间复杂度 O(n+m)。
"""

def naive_string_match(text, pattern):
    """
    朴素字符串匹配算法 (Naive String Matching)

    原理：逐一尝试文本串中所有可能的起始位置。对于每个起始位置，比较模式串和文本串对应子串的所有字符。
    text: 文本串 (string)
    pattern: 模式串 (string)
    返回值：模式串在文本串中所有出现位置的起始索引列表。
    时间复杂度：O((n-m+1) * m) = O(nm)，n 是文本串长度，m 是模式串长度。最坏情况下如在 "AAAAA" 中查找 "AAA"。
    空间复杂度：O(1) (不计结果列表空间)。
    """
    n = len(text)
    m = len(pattern)
    occurrences = [] # 存储匹配的起始索引

    # 遍历文本串中所有可能的起始位置
    # 只需要遍历到 n - m 的位置，因为模式串长度是 m
    for i in range(n - m + 1):
        # 检查从索引 i 开始的文本子串是否与模式串匹配
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False # 字符不匹配
                break # 停止当前起始位置的比较

        # 如果内层循环完整执行完且 match 仍为 True，说明找到了一个匹配
        if match:
            occurrences.append(i)

    return occurrences

# --- KMP 算法 (Knuth-Morris-Pratt) ---

def compute_lps(pattern):
    """
    计算模式串的部分匹配表 (LPS - Longest Proper Prefix which is also a Suffix)

    LPS 表 (或称失败函数) 的长度与模式串长度相同。
    lps[i] 存储的是模式串中索引 i (包括) 之前子串的最长的、同时是前缀也是后缀的子串的长度。
    例如: pattern = "ABABAA"
    lps[0] = 0 ("A")
    lps[1] = 0 ("AB")
    lps[2] = 1 ("ABA", 最长公共前后缀是 "A", 长度1)
    lps[3] = 2 ("ABAB", 最长公共前后缀是 "AB", 长度2)
    lps[4] = 3 ("ABABA", 最长公共前后缀是 "ABA", 长度3)
    lps[5] = 1 ("ABABAA", 最长公共前后缀是 "A", 长度1)
    LPS 表: [0, 0, 1, 2, 3, 1]

    原理：使用两个指针，length 表示当前计算的前一个位置的最长公共前后缀的长度，i 是当前正在计算 LPS 的位置。
    时间复杂度：O(m)，m 是模式串长度。
    空间复杂度：O(m)。
    """
    m = len(pattern)
    lps = [0] * m      # 初始化 LPS 表，大小为 m
    length = 0         # length 变量存储前一个位置 (i-1) 的最长公共前后缀的长度
    i = 1              # 从索引 1 开始计算 LPS (lps[0] 总是 0)

    # 循环计算 lps[i]，直到 i 到达模式串末尾
    while i < m:
        # 情况 1: pattern[i] 和 pattern[length] 匹配
        if pattern[i] == pattern[length]:
            length += 1      # 最长公共前后缀长度加 1
            lps[i] = length  # 将长度赋给当前位置 i 的 lps 值
            i += 1           # 移动到下一个位置
        else:
            # 情况 2: pattern[i] 和 pattern[length] 不匹配
            # 如果 length 不为 0，说明可以回溯到前一个最长公共前后缀的长度处继续比较
            # 即将 length 更新为 lps[length - 1]
            if length != 0:
                length = lps[length - 1]
                # 注意：这里不递增 i，因为还要用新的 length 值来和 pattern[i] 比较
            else:
                # 如果 length 为 0 (已经回溯到最前面，还是不匹配)
                lps[i] = 0 # 当前位置 i 的最长公共前后缀长度就是 0
                i += 1     # 移动到下一个位置

    return lps


def kmp_search(text, pattern):
    """
    KMP 字符串匹配算法

    原理：首先计算模式串的 LPS 表。然后在文本串中搜索时，如果发生不匹配，
         利用 LPS 表的信息，模式串不是简单地右移一位，而是根据已匹配的部分
         的最长公共前后缀长度进行跳跃，从而避免重复比较已经匹配过的字符。
    text: 文本串 (string)
    pattern: 模式串 (string)
    返回值：模式串在文本串中所有出现位置的起始索引列表。
    时间复杂度：O(n + m)，n 是文本串长度，m 是模式串长度。计算 LPS 是 O(m)，搜索是 O(n)。
    空间复杂度：O(m) (存储 LPS 表)。
    """
    n = len(text)
    m = len(pattern)

    # 如果模式串为空，则认为在文本串的每个位置都“匹配”
    if m == 0:
        return list(range(n + 1))
    # 如果文本串长度小于模式串长度，不可能匹配
    if n < m:
        return []

    # 1. 计算模式串的 LPS 表
    lps = compute_lps(pattern)
    # print(f"模式串 '{pattern}' 的 LPS 表: {lps}") # 辅助调试打印

    occurrences = [] # 存储匹配的起始索引
    i = 0 # 文本串当前字符的索引
    j = 0 # 模式串当前字符的索引

    # 循环比较，直到文本串遍历完
    while i < n:
        # 情况 1: 当前字符匹配 (text[i] == pattern[j])
        if text[i] == pattern[j]:
            i += 1 # 文本串指针右移
            j += 1 # 模式串指针右移

        # 情况 2: 模式串全部匹配完成 (j == m)
        if j == m:
            # 找到了一个匹配，匹配的起始索引是 i - m
            occurrences.append(i - m)

            # 继续搜索：利用 LPS 表进行跳跃
            # 下一个可能的匹配的起始位置是在当前匹配结束位置 (i) 之前
            # 模式串需要移动到使得 pattern[lps[j-1]] 对齐 text[i]
            # 也就是将 j 更新为 lps[j-1]
            j = lps[j - 1]

        # 情况 3: 当前字符不匹配 (text[i] != pattern[j]) 且 j > 0
        # (j > 0 说明模式串已经匹配了一部分)
        # 利用 LPS 表进行跳跃：将模式串指针 j 回退到 lps[j-1] 的位置
        # text[i] 不变，继续与 pattern[lps[j-1]] 比较
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            # 情况 4: 当前字符不匹配 (text[i] != pattern[j]) 且 j == 0
            # (j == 0 说明模式串的第一个字符就不匹配)
            else:
                # 模式串无法通过 LPS 表进行跳跃，文本串指针右移一位
                i += 1

    return occurrences

"""
# --- 示例用法 ---
if __name__ == "__main__":
    # 在 main 块中定义测试数据

    text1 = "ABABDABACDABABCABAB"
    pattern1 = "ABABCABAB" # 在 text1 中出现两次，起始索引 10 和 12

    text2 = "AAAAAA"
    pattern2 = "AAA" # 在 text2 中出现 4 次，起始索引 0, 1, 2, 3

    text3 = "this is a test text"
    pattern3 = "test" # 在 text3 中出现 1 次，起始索引 10

    text4 = "ABABAB"
    pattern4 = "ABA"

    text5 = "abc"
    pattern5 = "abcd" # 模式串长于文本串

    text6 = "aaaaa"
    pattern6 = "aa"

    print("--- 朴素匹配算法示例 ---")
    print(f"文本: '{text1}'")
    print(f"模式: '{pattern1}'")
    print(f"匹配起始索引: {naive_string_match(text1, pattern1)}\n") # 期望: [10, 12]

    print(f"文本: '{text2}'")
    print(f"模式: '{pattern2}'")
    print(f"匹配起始索引: {naive_string_match(text2, pattern2)}\n") # 期望: [0, 1, 2, 3]

    print(f"文本: '{text3}'")
    print(f"模式: '{pattern3}'")
    print(f"匹配起始索引: {naive_string_match(text3, pattern3)}\n") # 期望: [10]

    print(f"文本: '{text5}'")
    print(f"模式: '{pattern5}'")
    print(f"匹配起始索引: {naive_string_match(text5, pattern5)}\n") # 期望: []


    print("--- KMP 算法示例 ---")
    print(f"文本: '{text1}'")
    print(f"模式: '{pattern1}'")
    print(f"匹配起始索引: {kmp_search(text1, pattern1)}\n") # 期望: [10, 12]

    print(f"文本: '{text2}'")
    print(f"模式: '{pattern2}'")
    print(f"匹配起始索引: {kmp_search(text2, pattern2)}\n") # 期望: [0, 1, 2, 3]

    print(f"文本: '{text3}'")
    print(f"模式: '{pattern3}'")
    print(f"匹配起始索引: {kmp_search(text3, pattern3)}\n") # 期望: [10]

    print(f"文本: '{text4}'")
    print(f"模式: '{pattern4}'")
    print(f"匹配起始索引: {kmp_search(text4, pattern4)}\n") # 期望: [0, 2]

    print(f"文本: '{text5}'")
    print(f"模式: '{pattern5}'")
    print(f"匹配起始索引: {kmp_search(text5, pattern5)}\n") # 期望: []

    print(f"文本: '{text6}'")
    print(f"模式: '{pattern6}'")
    print(f"匹配起始索引: {kmp_search(text6, pattern6)}\n") # 期望: [0, 1, 2, 3]
"""