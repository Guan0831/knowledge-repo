# algorithms/common_algorithms/leetcode_patterns/trees/tree_algorithms_code.py

"""
树算法实现示例 (Python)

本文件包含了二叉树的基本结构定义以及常见的树遍历和算法实现。
特别关注在算法问题中常用的迭代遍历和关键算法如验证 BST、查找 LCA。
"""

import collections # 用于层序遍历的队列

# --- 节点定义 (Node Definition) ---

class TreeNode:
    """
    二叉树节点定义
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        # 方便打印节点值
        return f"TreeNode({self.val})"


# --- 树遍历 (Tree Traversals) ---

# 递归实现 (Recursion Implementations)
# 通常用于概念理解，代码简洁

def preorder_traversal_recursive(root):
    """前序遍历 (递归): 根 -> 左 -> 右"""
    order = []
    def traverse(node):
        if node is None:
            return
        order.append(node.val)      # 访问根
        traverse(node.left)         # 遍历左子树
        traverse(node.right)        # 遍历右子树
    traverse(root)
    return order

def inorder_traversal_recursive(root):
    """中序遍历 (递归): 左 -> 根 -> 右 (对于 BST 结果是有序的)"""
    order = []
    def traverse(node):
        if node is None:
            return
        traverse(node.left)         # 遍历左子树
        order.append(node.val)      # 访问根
        traverse(node.right)        # 遍历右子树
    traverse(root)
    return order

def postorder_traversal_recursive(root):
    """后序遍历 (递归): 左 -> 右 -> 根"""
    order = []
    def traverse(node):
        if node is None:
            return
        traverse(node.left)         # 遍历左子树
        traverse(node.right)        # 遍历右子树
        order.append(node.val)      # 访问根
    traverse(root)
    return order

# 迭代实现 (Iterative Implementations)
# 在 LeetCode 中更常用，避免递归深度限制

def preorder_traversal_iterative(root):
    """前序遍历 (迭代): 根 -> 左 -> 右 (使用栈)"""
    order = []
    if root is None:
        return order

    stack = [root] # 初始化栈，将根节点入栈

    while stack:
        node = stack.pop() # 弹出当前节点
        order.append(node.val) # 访问当前节点

        # 先压右子节点，再压左子节点，这样左子节点会先被弹出处理
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return order

def inorder_traversal_iterative(root):
    """中序遍历 (迭代): 左 -> 根 -> 右 (使用栈)"""
    order = []
    if root is None:
        return order

    stack = []
    current = root # 从根节点开始

    while current or stack:
        # 一直向左走，并将路径上的节点入栈
        while current:
            stack.append(current)
            current = current.left

        # 栈顶节点没有左子树了，弹出它 (访问根节点)
        current = stack.pop()
        order.append(current.val)

        # 转到当前节点的右子树
        current = current.right

    return order

def postorder_traversal_iterative(root):
    """后序遍历 (迭代): 左 -> 右 -> 根 (有多种实现方法，这里是一种使用两个栈的思路)"""
    order = []
    if root is None:
        return order

    stack1 = [root] # 第一个栈用于模拟前序遍历 (根右左)
    stack2 = []     # 第二个栈用于存储后序遍历的结果 (最后逆序得到左右根)

    while stack1:
        node = stack1.pop()
        stack2.append(node.val) # 将节点值压入第二个栈

        # 先压左子节点，再压右子节点 (这样在 stack2 中会是根左右的顺序)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)

    # stack2 中的元素顺序是 根 -> 右 -> 左，反转后就是 左 -> 右 -> 根 (后序)
    order = stack2[::-1]
    return order

def level_order_traversal(root):
    """层序遍历 (Level Order Traversal): 从上到下，从左到右 (使用队列)"""
    order = []
    if root is None:
        return order

    queue = collections.deque([root]) # 初始化队列，放入根节点

    while queue:
        node = queue.popleft() # 弹出队列头部节点
        order.append(node.val) # 访问当前节点

        # 将左右子节点（如果存在）依次加入队列尾部
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return order

# --- 树的属性与算法 (Tree Properties & Algorithms) ---

def max_depth(root):
    """
    计算二叉树的最大深度 (高度)

    原理：树的深度等于其左子树和右子树深度的最大值 + 1。
    Base Case: 空树的深度为 0。
    """
    if root is None:
        return 0 # Base Case

    # 递归计算左右子树的深度
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    # 返回左右子树最大深度 + 1 (当前节点所在的一层)
    return max(left_depth, right_depth) + 1

def is_valid_bst(root):
    """
    检查是否为有效的二叉搜索树 (Valid BST)

    原理：二叉搜索树 (BST) 的中序遍历结果是有序的。
          或者使用递归方法，在遍历过程中传递节点的有效取值范围 (min_val, max_val)。
          对于当前节点 node:
          - 其值必须在 (min_val, max_val) 范围内。
          - 其左子树必须是有效的 BST，且所有节点值都在 (min_val, node.val) 范围内。
          - 其右子树必须是有效的 BST，且所有节点值都在 (node.val, max_val) 范围内。
    Base Case: 空树是有效的 BST。
    """
    # 辅助函数，递归检查节点 node 的值是否在 (low, high) 范围内
    # 使用 float('-inf') 和 float('inf') 表示初始的无穷范围
    def validate(node, low=float('-inf'), high=float('inf')):
        # Base Case: 空节点是有效的
        if node is None:
            return True

        # 检查当前节点的值是否在有效范围内
        if not (low < node.val < high):
            return False

        # 递归检查左子树和右子树
        # 左子树的范围是 (low, node.val)
        # 右子树的范围是 (node.val, high)
        return (validate(node.left, low, node.val) and
                validate(node.right, node.val, high))

    return validate(root)

def lowest_common_ancestor(root, p, q):
    """
    查找二叉树中两个给定节点 p 和 q 的最低公共祖先 (LCA)

    原理：递归方法。
          - 如果当前节点是 p 或 q，那么它就是 p 和 q (之一) 的 LCA (或者 p 是 q 的祖先，或反之)。
          - 如果 p 和 q 分别位于当前节点的左子树和右子树中，那么当前节点就是 LCA。
          - 如果 p 和 q 都位于当前节点的左子树中，那么 LCA 在左子树中，继续在左子树查找。
          - 如果 p 和 q 都位于当前节点的右子树中，那么 LCA 在右子树中，继续在右子树查找。
    Base Case: 当前节点为空，或者当前节点就是 p 或 q，返回当前节点。
    假设 p 和 q 都在树中。
    """
    # Base Case: 节点为空，或者当前节点就是 p 或 q
    if root is None or root == p or root == q:
        return root

    # 递归查找 p 和 q 在左子树和右子树中的 LCA
    left_lca = lowest_common_ancestor(root.left, p, q)
    right_lca = lowest_common_ancestor(root.right, p, q)

    # 根据左右子树的查找结果判断当前节点的角色
    if left_lca and right_lca:
        # 如果左右子树都找到了结果 (说明 p 和 q 分别在左右子树)，当前节点就是 LCA
        return root
    elif left_lca:
        # 如果只有左子树找到了结果 (说明 p 和 q 都在左子树)，LCA 在左子树中
        return left_lca
    else: # right_lca is not None (implicitly, since left_lca is None)
        # 如果只有右子树找到了结果 (说明 p 和 q 都在右子树)，LCA 在右子树中
        return right_lca
    # 如果左右子树都没找到 (left_lca is None and right_lca is None)，返回 None，
    # 这通常发生在 p 和 q 都不在以 root 为根的子树中，或者 p 或 q 不在树中。
    # 但根据问题假设 p 和 q 在树中，所以至少会找到 p 或 q 中的一个。

"""

# --- 示例用法 ---
if __name__ == "__main__":
    # 构建一个示例二叉树
    #         3
    #        / \
    #       9   20
    #          /  \
    #         15   7
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)

    print("--- 树遍历示例 ---")
    print("示例树结构: [3, 9, 20, null, null, 15, 7] (按层序)")
    print(f"前序遍历 (递归): {preorder_traversal_recursive(root)}") # 期望: [3, 9, 20, 15, 7]
    print(f"前序遍历 (迭代): {preorder_traversal_iterative(root)}") # 期望: [3, 9, 20, 15, 7]

    print(f"中序遍历 (递归): {inorder_traversal_recursive(root)}") # 期望: [9, 3, 15, 20, 7]
    print(f"中序遍历 (迭代): {inorder_traversal_iterative(root)}") # 期望: [9, 3, 15, 20, 7]

    print(f"后序遍历 (递归): {postorder_traversal_recursive(root)}") # 期望: [9, 15, 7, 20, 3]
    print(f"后序遍历 (迭代): {postorder_traversal_iterative(root)}") # 期望: [9, 15, 7, 20, 3]

    print(f"层序遍历: {level_order_traversal(root)}") # 期望: [3, 9, 20, 15, 7]


    print("\n--- 树的属性与算法示例 ---")
    print(f"示例树的最大深度: {max_depth(root)}") # 期望: 3

    # 构建一个示例 BST
    #         5
    #        / \
    #       2   7
    #      / \ / \
    #     1  3 6  8
    root_bst = TreeNode(5)
    root_bst.left = TreeNode(2)
    root_bst.right = TreeNode(7)
    root_bst.left.left = TreeNode(1)
    root_bst.left.right = TreeNode(3)
    root_bst.right.left = TreeNode(6)
    root_bst.right.right = TreeNode(8)

    print("\n--- 验证 BST ---")
    print("示例 BST:")
    print(f"是否为有效的 BST: {is_valid_bst(root_bst)}") # 期望: True

    # 构建一个无效的 BST
    #         5
    #        / \
    #       2   7
    #      / \ / \
    #     1  6 6  8  (左子树的 6 不应大于根 5)
    root_invalid_bst = TreeNode(5)
    root_invalid_bst.left = TreeNode(2)
    root_invalid_bst.right = TreeNode(7)
    root_invalid_bst.left.left = TreeNode(1)
    root_invalid_bst.left.right = TreeNode(6) # 问题在这里
    root_invalid_bst.right.left = TreeNode(6)
    root_invalid_bst.right.right = TreeNode(8)

    print("\n示例无效 BST:")
    print(f"是否为有效的 BST: {is_valid_bst(root_invalid_bst)}") # 期望: False


    print("\n--- 查找 LCA ---")
    print("在示例 BST 中查找 LCA:")
    # 查找节点 1 和 3 的 LCA (期望是 2)
    p1 = root_bst.left.left # 节点 1
    q1 = root_bst.left.right # 节点 3
    lca1 = lowest_common_ancestor(root_bst, p1, q1)
    print(f"节点 {p1.val} 和 {q1.val} 的 LCA 是: {lca1.val}") # 期望: 2

    # 查找节点 2 和 8 的 LCA (期望是 5)
    p2 = root_bst.left # 节点 2
    q2 = root_bst.right.right # 节点 8
    lca2 = lowest_common_ancestor(root_bst, p2, q2)
    print(f"节点 {p2.val} 和 {q2.val} 的 LCA 是: {lca2.val}") # 期望: 5

    # 查找节点 6 和 7 的 LCA (期望是 7)
    p3 = root_bst.right.left # 节点 6
    q3 = root_bst.right # 节点 7
    lca3 = lowest_common_ancestor(root_bst, p3, q3)
    print(f"节点 {p3.val} 和 {q3.val} 的 LCA 是: {lca3.val}") # 期望: 7 (因为 7 是 6 的祖先)
"""