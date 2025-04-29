# algorithms/common_algorithms/leetcode_patterns/trie/trie_code.py

"""
字典树 / 前缀树 (Trie / Prefix Tree) 实现 (Python)

本文件包含了字典树 (Trie) 数据结构的实现，常用于高效地存储和检索字符串集合，
特别擅长处理与字符串前缀相关的操作。

核心应用：
- 自动补全
- 拼写检查
- 快速过滤或查找带有特定前缀的字符串
"""

class TrieNode:
    """
    字典树的节点

    每个节点存储：
    - children: 一个字典，键是字符，值是对应的子节点 TrieNode。
                表示从当前节点出发，通过该字符到达的下一个节点。
    - is_end_of_word: 布尔值，如果为 True，表示从根节点到当前节点形成的路径是一个完整的单词的结尾。
    """
    def __init__(self):
        self.children = {}  # 使用字典来存储子节点 {char: TrieNode}
        self.is_end_of_word = False # 标记是否是某个单词的结束


class Trie:
    """
    字典树 (Trie) 数据结构

    包含 Trie 的根节点和核心操作方法。
    """
    def __init__(self):
        """
        初始化 Trie，创建一个空的根节点。
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        向 Trie 中插入一个单词。

        遍历单词的每个字符，沿着对应的路径向下。如果路径上的节点不存在，则创建新节点。
        在单词的最后一个字符对应的节点上，将 is_end_of_word 标记设为 True。
        时间复杂度：O(L)，L 是单词的长度。
        空间复杂度：O(L * C)，L 是单词长度，C 是字符集大小（对于字典实现 C 是平均分支因子），最坏情况下每个字符都创建一个新节点。
        """
        node = self.root # 从根节点开始

        for char in word:
            # 如果当前节点的子节点中没有当前字符对应的节点
            if char not in node.children:
                # 创建一个新的 TrieNode 作为子节点
                node.children[char] = TrieNode()
            # 移动到下一个节点 (即当前字符对应的子节点)
            node = node.children[char]

        # 单词的所有字符遍历完毕，当前节点就是单词的结尾
        node.is_end_of_word = True

    def search(self, word):
        """
        在 Trie 中搜索一个完整的单词是否存在。

        遍历单词的每个字符，沿着对应的路径向下。
        如果任一字符对应的路径中断 (没有对应的子节点)，说明单词不存在。
        如果路径遍历完，检查最后一个字符对应的节点是否标记为单词结尾 (is_end_of_word)。
        时间复杂度：O(L)，L 是单词的长度。
        空间复杂度：O(1) (不考虑 Trie 本身的空间)。
        """
        node = self.root # 从根节点开始

        for char in word:
            # 如果当前字符对应的子节点不存在，单词不存在
            if char not in node.children:
                return False
            # 移动到下一个节点
            node = node.children[char]

        # 单词的所有字符路径都存在，最后检查该节点是否是某个单词的结尾
        return node.is_end_of_word

    def starts_with(self, prefix):
        """
        检查 Trie 中是否存在任何以给定前缀开头的单词。

        遍历前缀的每个字符，沿着对应的路径向下。
        如果任一字符对应的路径中断，说明没有单词以该前缀开头。
        如果前缀路径遍历完，说明存在至少一个节点代表该前缀的结尾，即存在以此为前缀的单词 (或前缀本身就是一个单词)。
        时间复杂度：O(L)，L 是前缀的长度。
        空间复杂度：O(1) (不考虑 Trie 本身的空间)。
        """
        node = self.root # 从根节点开始

        for char in prefix:
            # 如果当前字符对应的子节点不存在，说明没有单词以该前缀开头
            if char not in node.children:
                return False
            # 移动到下一个节点
            node = node.children[char]

        # 前缀的所有字符路径都存在
        return True

""""""
# --- 示例用法 ---
if __name__ == "__main__":
    # 创建一个 Trie 实例
    trie = Trie()

    # 插入单词
    words_to_insert = ["apple", "app", "apricot", "banana", "bandana"]
    print(f"插入单词: {words_to_insert}")
    for word in words_to_insert:
        trie.insert(word)

    print("\n--- 搜索单词 ---")
    print(f"搜索 'apple': {trie.search('apple')}")   # 期望: True
    print(f"搜索 'app': {trie.search('app')}")     # 期望: True
    print(f"搜索 'apricot': {trie.search('apricot')}") # 期望: True
    print(f"搜索 'ban': {trie.search('ban')}")     # 期望: False (只是前缀)
    print(f"搜索 'appl': {trie.search('appl')}")    # 期望: False (不是完整的单词)
    print(f"搜索 'grape': {trie.search('grape')}")   # 期望: False (不存在)

    print("\n--- 查找前缀 ---")
    print(f"是否存在以 'app' 开头的单词: {trie.starts_with('app')}")     # 期望: True ("apple", "app", "apricot")
    print(f"是否存在以 'ban' 开头的单词: {trie.starts_with('ban')}")     # 期望: True ("banana", "bandana")
    print(f"是否存在以 'appl' 开头的单词: {trie.starts_with('appl')}")    # 期望: True ("apple")
    print(f"是否存在以 'grape' 开头的单词: {trie.starts_with('grape')}")   # 期望: False
    print(f"是否存在以 'applepie' 开头的单词: {trie.starts_with('applepie')}") # 期望: False (路径不存在)