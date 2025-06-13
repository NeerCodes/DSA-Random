
# 🧠 3-7-15 Rule DSA Practice Sheet

> ✅ First **solve** the question. Then **note down** the answer. After **3 days**, recall the question from the answer and **solve again**.  
> Follow **3-7-15 spaced repetition** for mastery.

---


# 📘 Arrays and Strings – Java Cheat Sheet

This cheat sheet covers essential problems involving arrays and strings in Java. Each problem includes both **brute-force** and **optimized** solutions along with explanations.

---

## 🔹 1. Maximum Sum Subarray

### Problem:
Find the contiguous subarray with the maximum sum.

### 🧪 Brute Force (O(n²)):
```java
public static int maxSubArrayBrute(int[] nums) {
    int maxSum = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
        int sum = 0;
        for (int j = i; j < nums.length; j++) {
            sum += nums[j];
            maxSum = Math.max(maxSum, sum);
        }
    }
    return maxSum;
}
```

### ⚡ Optimized (Kadane’s Algorithm, O(n)):
```java
public static int maxSubArrayOptimized(int[] nums) {
    int maxSum = nums[0], currentSum = nums[0];
    for (int i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }
    return maxSum;
}
```

---

## 🔹 2. Find All Palindromic Substrings

### Problem:
Find and return all substrings that are palindromes.

### 🧪 Brute Force (O(n³)):
```java
public static List<String> allPalindromesBrute(String s) {
    List<String> result = new ArrayList<>();
    for (int i = 0; i < s.length(); i++) {
        for (int j = i; j < s.length(); j++) {
            String sub = s.substring(i, j + 1);
            if (isPalindrome(sub)) result.add(sub);
        }
    }
    return result;
}

private static boolean isPalindrome(String s) {
    int l = 0, r = s.length() - 1;
    while (l < r) if (s.charAt(l++) != s.charAt(r--)) return false;
    return true;
}
```

### ⚡ Optimized (Expand Around Center, O(n²)):
```java
public static List<String> allPalindromesOptimized(String s) {
    List<String> result = new ArrayList<>();
    for (int center = 0; center < 2 * s.length() - 1; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            result.add(s.substring(left, right + 1));
            left--; right++;
        }
    }
    return result;
}
```

---

## 🔹 3. Two Sum

### Problem:
Find indices of two numbers that add up to the target.

### 🧪 Brute Force (O(n²)):
```java
public static int[] twoSumBrute(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++) {
        for (int j = i + 1; j < nums.length; j++) {
            if (nums[i] + nums[j] == target)
                return new int[]{i, j};
        }
    }
    return new int[]{-1, -1};
}
```

### ⚡ Optimized (HashMap, O(n)):
```java
public static int[] twoSumOptimized(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int comp = target - nums[i];
        if (map.containsKey(comp))
            return new int[]{map.get(comp), i};
        map.put(nums[i], i);
    }
    return new int[]{-1, -1};
}
```

---

## 🔹 4. Kadane's Algorithm

Kadane’s algorithm is already covered in Problem 1 (Optimized).

---

## 🔹 5. Find the Missing Number

### Problem:
Array contains numbers from 0 to n with one missing.

### 🧪 Brute Force (O(n²)):
```java
public static int missingNumberBrute(int[] nums) {
    for (int i = 0; i <= nums.length; i++) {
        boolean found = false;
        for (int num : nums) {
            if (num == i) {
                found = true;
                break;
            }
        }
        if (!found) return i;
    }
    return -1;
}
```

### ⚡ Optimized (Sum Formula, O(n)):
```java
public static int missingNumberOptimized(int[] nums) {
    int n = nums.length, sum = n * (n + 1) / 2;
    for (int num : nums) sum -= num;
    return sum;
}
```

---

## 🔹 6. Merge Two Sorted Arrays

### 🧪 Brute Force (O(n log n)):
```java
public static int[] mergeSortedArraysBrute(int[] a, int[] b) {
    int[] merged = new int[a.length + b.length];
    System.arraycopy(a, 0, merged, 0, a.length);
    System.arraycopy(b, 0, merged, a.length, b.length);
    Arrays.sort(merged);
    return merged;
}
```

### ⚡ Optimized (Two Pointers, O(n)):
```java
public static int[] mergeSortedArraysOptimized(int[] a, int[] b) {
    int[] merged = new int[a.length + b.length];
    int i = 0, j = 0, k = 0;
    while (i < a.length && j < b.length)
        merged[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
    while (i < a.length) merged[k++] = a[i++];
    while (j < b.length) merged[k++] = b[j++];
    return merged;
}
```

---

## 🔹 7. Check If String is Palindrome

```java
public static boolean isPalindromeString(String s) {
    int l = 0, r = s.length() - 1;
    while (l < r) {
        if (s.charAt(l++) != s.charAt(r--)) return false;
    }
    return true;
}
```

---

## 🔹 8. First Non-Repeating Character

### 🧪 Brute Force (O(n²)):
```java
public static char firstNonRepeatingCharBrute(String s) {
    for (int i = 0; i < s.length(); i++) {
        boolean unique = true;
        for (int j = 0; j < s.length(); j++) {
            if (i != j && s.charAt(i) == s.charAt(j)) {
                unique = false;
                break;
            }
        }
        if (unique) return s.charAt(i);
    }
    return '\0';
}
```

### ⚡ Optimized (O(n)):
```java
public static char firstNonRepeatingCharOptimized(String s) {
    Map<Character, Integer> freq = new LinkedHashMap<>();
    for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
    for (Map.Entry<Character, Integer> e : freq.entrySet()) {
        if (e.getValue() == 1) return e.getKey();
    }
    return '\0';
}
```

---

## 🔹 9. Remove Duplicates from Sorted Array

```java
public static int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    int index = 1;
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] != nums[i - 1]) {
            nums[index++] = nums[i];
        }
    }
    return index;
}
```
🛠️ Modifies array in-place and returns new length.

---



# 🔗 Linked Lists – Java Cheat Sheet

This cheat sheet covers common linked list problems with both **brute-force** and **optimized** solutions in Java. Each solution is explained clearly for interview preparation.

---

## 🔹 10. Reverse a Linked List

### 🧪 Brute Force (Using Stack):
```java
public static ListNode reverseBrute(ListNode head) {
    Stack<ListNode> stack = new Stack<>();
    while (head != null) {
        stack.push(head);
        head = head.next;
    }
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    while (!stack.isEmpty()) {
        curr.next = stack.pop();
        curr = curr.next;
    }
    curr.next = null;
    return dummy.next;
}
```

### ⚡ Optimized (Iterative):
```java
public static ListNode reverseOptimized(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```

---

## 🔹 11. Detect a Cycle in a Linked List

### 🧪 Brute Force (Using Set):
```java
public static boolean hasCycleBrute(ListNode head) {
    Set<ListNode> visited = new HashSet<>();
    while (head != null) {
        if (!visited.add(head)) return true;
        head = head.next;
    }
    return false;
}
```

### ⚡ Optimized (Floyd’s Cycle Detection):
```java
public static boolean hasCycleOptimized(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```

---

## 🔹 12. Find the Middle of a Linked List

### 🧪 Brute Force (Count nodes then loop again):
```java
public static ListNode findMiddleBrute(ListNode head) {
    int count = 0;
    ListNode curr = head;
    while (curr != null) {
        count++;
        curr = curr.next;
    }
    curr = head;
    for (int i = 0; i < count / 2; i++) {
        curr = curr.next;
    }
    return curr;
}
```

### ⚡ Optimized (Two Pointers):
```java
public static ListNode findMiddleOptimized(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

---

## 🔹 13. Merge Two Sorted Linked Lists

### 🧪 Brute Force (Convert to array, sort, re-link):
```java
public static ListNode mergeSortedBrute(ListNode l1, ListNode l2) {
    List<Integer> list = new ArrayList<>();
    while (l1 != null) { list.add(l1.val); l1 = l1.next; }
    while (l2 != null) { list.add(l2.val); l2 = l2.next; }
    Collections.sort(list);
    ListNode dummy = new ListNode(0), curr = dummy;
    for (int val : list) {
        curr.next = new ListNode(val);
        curr = curr.next;
    }
    return dummy.next;
}
```

### ⚡ Optimized (Two Pointers):
```java
public static ListNode mergeSortedOptimized(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), curr = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            curr.next = l1;
            l1 = l1.next;
        } else {
            curr.next = l2;
            l2 = l2.next;
        }
        curr = curr.next;
    }
    curr.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```

---

## 🔹 14. Implement a Stack Using Linked List

```java
class StackNode {
    int data;
    StackNode next;
    StackNode(int data) { this.data = data; }
}

class Stack {
    StackNode top;

    public void push(int x) {
        StackNode node = new StackNode(x);
        node.next = top;
        top = node;
    }

    public int pop() {
        if (top == null) throw new EmptyStackException();
        int val = top.data;
        top = top.next;
        return val;
    }

    public int peek() {
        if (top == null) throw new EmptyStackException();
        return top.data;
    }

    public boolean isEmpty() {
        return top == null;
    }
}
```

---

## 🔹 15. Find the Intersection Point of Two Linked Lists

### 🧪 Brute Force (Compare every node with every other):
```java
public static ListNode getIntersectionBrute(ListNode headA, ListNode headB) {
    for (ListNode a = headA; a != null; a = a.next) {
        for (ListNode b = headB; b != null; b = b.next) {
            if (a == b) return a;
        }
    }
    return null;
}
```

### ⚡ Optimized (Length difference approach):
```java
public static ListNode getIntersectionOptimized(ListNode headA, ListNode headB) {
    int lenA = 0, lenB = 0;
    ListNode a = headA, b = headB;
    while (a != null) { lenA++; a = a.next; }
    while (b != null) { lenB++; b = b.next; }
    a = headA; b = headB;
    if (lenA > lenB) for (int i = 0; i < lenA - lenB; i++) a = a.next;
    else for (int i = 0; i < lenB - lenA; i++) b = b.next;
    while (a != null && b != null) {
        if (a == b) return a;
        a = a.next;
        b = b.next;
    }
    return null;
}
```

🧠 Tip: Always consider edge cases such as empty lists or cycles when working with linked list problems.

 ---

 ## 📚 Stacks and Queues

### 🔹 16. Implement a Stack Using an Array

#### 🧪 Brute Force
```java
class Stack {
    int[] arr = new int[1000];
    int top = -1;

    void push(int x) {
        if (top < arr.length - 1) arr[++top] = x;
    }

    int pop() {
        return top >= 0 ? arr[top--] : -1;
    }

    int peek() {
        return top >= 0 ? arr[top] : -1;
    }
}
```

#### ⚡ Optimized
```java
class Stack {
    ArrayList<Integer> list = new ArrayList<>();

    void push(int x) {
        list.add(x);
    }

    int pop() {
        return list.isEmpty() ? -1 : list.remove(list.size() - 1);
    }

    int peek() {
        return list.isEmpty() ? -1 : list.get(list.size() - 1);
    }
}
```

### 🔹 17. Stack with getMin

#### 🧪 Brute Force
```java
class MinStack {
    Stack<Integer> stack = new Stack<>();

    void push(int x) {
        stack.push(x);
    }

    void pop() {
        stack.pop();
    }

    int top() {
        return stack.peek();
    }

    int getMin() {
        int min = Integer.MAX_VALUE;
        for (int num : stack) min = Math.min(min, num);
        return min;
    }
}
```

#### ⚡ Optimized (With extra stack)
```java
class MinStack {
    Stack<Integer> stack = new Stack<>();
    Stack<Integer> minStack = new Stack<>();

    void push(int x) {
        stack.push(x);
        if (minStack.isEmpty() || x <= minStack.peek()) minStack.push(x);
    }

    void pop() {
        if (stack.pop().equals(minStack.peek())) minStack.pop();
    }

    int top() {
        return stack.peek();
    }

    int getMin() {
        return minStack.peek();
    }
}
```

### 🔹 18. Circular Queue

#### 🧪 Brute Force (Also optimal)
```java
class CircularQueue {
    int[] arr;
    int size, front = -1, rear = -1;

    CircularQueue(int k) {
        arr = new int[k];
        size = k;
    }

    boolean enqueue(int val) {
        if ((rear + 1) % size == front) return false;
        if (front == -1) front = 0;
        rear = (rear + 1) % size;
        arr[rear] = val;
        return true;
    }

    boolean dequeue() {
        if (front == -1) return false;
        if (front == rear) front = rear = -1;
        else front = (front + 1) % size;
        return true;
    }
}
```

### 🔹 19. Max Stack

#### 🧪 Brute Force
```java
class MaxStack {
    Stack<Integer> stack = new Stack<>();

    void push(int x) {
        stack.push(x);
    }

    int pop() {
        return stack.pop();
    }

    int top() {
        return stack.peek();
    }

    int getMax() {
        int max = Integer.MIN_VALUE;
        for (int num : stack) max = Math.max(max, num);
        return max;
    }
}
```

#### ⚡ Optimized
```java
class MaxStack {
    Stack<Integer> stack = new Stack<>();
    Stack<Integer> maxStack = new Stack<>();

    void push(int x) {
        stack.push(x);
        if (maxStack.isEmpty() || x >= maxStack.peek()) maxStack.push(x);
    }

    int pop() {
        if (stack.peek().equals(maxStack.peek())) maxStack.pop();
        return stack.pop();
    }

    int top() {
        return stack.peek();
    }

    int getMax() {
        return maxStack.peek();
    }
}
```

### 🔹 20. Queue Using Stacks

#### 🧪 Brute Force
```java
class MyQueue {
    Stack<Integer> input = new Stack<>();

    void push(int x) {
        input.push(x);
    }

    int pop() {
        Stack<Integer> temp = new Stack<>();
        while (!input.isEmpty()) temp.push(input.pop());
        int res = temp.pop();
        while (!temp.isEmpty()) input.push(temp.pop());
        return res;
    }
}
```

#### ⚡ Optimized (Two stacks)
```java
class MyQueue {
    Stack<Integer> input = new Stack<>();
    Stack<Integer> output = new Stack<>();

    void push(int x) {
        input.push(x);
    }

    int pop() {
        if (output.isEmpty()) while (!input.isEmpty()) output.push(input.pop());
        return output.pop();
    }
}
```


## 🌲 Trees and BSTs

### 21. Find the Height of a Binary Tree

#### 🧪 Brute Force
```java
public int height(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(height(root.left), height(root.right));
}
```
- Recursively finds the height of left and right subtrees.

#### ⚡ Optimized
(Same as brute force. This is the optimal recursive approach.)

---

### 22. Find the Lowest Common Ancestor (LCA) in Binary Tree

#### 🧪 Brute Force
```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if (left != null && right != null) return root;
    return left != null ? left : right;
}
```
- Recursive solution checking both left and right subtrees.

#### ⚡ Optimized
(Same as brute force. It is optimal for general binary trees.)

---

### 23. Validate Binary Search Tree

#### 🧪 Brute Force (In-order traversal, check sorted list)
```java
List<Integer> vals = new ArrayList<>();
public void inorder(TreeNode node) {
    if (node == null) return;
    inorder(node.left);
    vals.add(node.val);
    inorder(node.right);
}

public boolean isValidBST(TreeNode root) {
    inorder(root);
    for (int i = 1; i < vals.size(); i++) {
        if (vals.get(i) <= vals.get(i - 1)) return false;
    }
    return true;
}
```

#### ⚡ Optimized (Check valid range)
```java
public boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

public boolean validate(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return validate(node.left, min, node.val) && validate(node.right, node.val, max);
}
```

---

### 24. Serialize and Deserialize Binary Tree

#### ⚡ Optimal (Using Preorder and Null Marker)
```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "X,";
        return root.val + "," + serialize(root.left) + serialize(root.right);
    }

    public TreeNode deserialize(String data) {
        Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
        return build(nodes);
    }

    private TreeNode build(Queue<String> nodes) {
        String val = nodes.poll();
        if (val.equals("X")) return null;
        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = build(nodes);
        node.right = build(nodes);
        return node;
    }
}
```
- No brute force applicable.

---

### 25. Inorder Traversal of Binary Tree

#### 🧪 Brute Force (Recursive)
```java
public void inorder(TreeNode root) {
    if (root != null) {
        inorder(root.left);
        System.out.print(root.val + " ");
        inorder(root.right);
    }
}
```

#### ⚡ Optimized (Iterative using Stack)
```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Stack<TreeNode> stack = new Stack<>();
    TreeNode curr = root;
    while (curr != null || !stack.isEmpty()) {
        while (curr != null) {
            stack.push(curr);
            curr = curr.left;
        }
        curr = stack.pop();
        res.add(curr.val);
        curr = curr.right;
    }
    return res;
}
```

---

### 26. Diameter of Binary Tree

#### ⚡ Optimized
```java
int max = 0;
public int diameterOfBinaryTree(TreeNode root) {
    height(root);
    return max;
}

private int height(TreeNode node) {
    if (node == null) return 0;
    int left = height(node.left);
    int right = height(node.right);
    max = Math.max(max, left + right);
    return 1 + Math.max(left, right);
}
```

---

### 27. Mirror of Binary Tree

#### 🧪 Brute Force / ⚡ Optimal (Recursive)
```java
public TreeNode mirrorTree(TreeNode root) {
    if (root == null) return null;
    TreeNode left = mirrorTree(root.left);
    TreeNode right = mirrorTree(root.right);
    root.left = right;
    root.right = left;
    return root;
}
```

---

## 🔗 Graphs

### 28. Depth-First Search (DFS)

```java
public void dfs(int node, boolean[] visited, List<List<Integer>> adj) {
    visited[node] = true;
    for (int neighbor : adj.get(node)) {
        if (!visited[neighbor]) {
            dfs(neighbor, visited, adj);
        }
    }
}
```

---

### 29. Breadth-First Search (BFS)

```java
public void bfs(int start, List<List<Integer>> adj) {
    boolean[] visited = new boolean[adj.size()];
    Queue<Integer> queue = new LinkedList<>();
    queue.offer(start);
    visited[start] = true;

    while (!queue.isEmpty()) {
        int node = queue.poll();
        for (int neighbor : adj.get(node)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.offer(neighbor);
            }
        }
    }
}
```

---

### 30. Shortest Path (Unweighted Graph - BFS)

```java
public int[] shortestPath(int start, List<List<Integer>> adj) {
    int[] dist = new int[adj.size()];
    Arrays.fill(dist, -1);
    Queue<Integer> q = new LinkedList<>();
    q.offer(start);
    dist[start] = 0;

    while (!q.isEmpty()) {
        int node = q.poll();
        for (int neighbor : adj.get(node)) {
            if (dist[neighbor] == -1) {
                dist[neighbor] = dist[node] + 1;
                q.offer(neighbor);
            }
        }
    }
    return dist;
}
```

---

### 31. Cycle Detection in Undirected Graph (DFS)

```java
public boolean hasCycle(int node, int parent, boolean[] visited, List<List<Integer>> adj) {
    visited[node] = true;
    for (int neighbor : adj.get(node)) {
        if (!visited[neighbor]) {
            if (hasCycle(neighbor, node, visited, adj)) return true;
        } else if (neighbor != parent) return true;
    }
    return false;
}
```

---

### 32. Check if Graph is Bipartite

```java
public boolean isBipartite(int[][] graph) {
    int[] color = new int[graph.length];
    Arrays.fill(color, -1);
    for (int i = 0; i < graph.length; i++) {
        if (color[i] == -1 && !dfsColor(i, 0, color, graph)) return false;
    }
    return true;
}

private boolean dfsColor(int node, int c, int[] color, int[][] graph) {
    color[node] = c;
    for (int neighbor : graph[node]) {
        if (color[neighbor] == -1) {
            if (!dfsColor(neighbor, 1 - c, color, graph)) return false;
        } else if (color[neighbor] == c) return false;
    }
    return true;
}
```

---

### 33. Number of Connected Components (DFS)

```java
public int countComponents(int n, int[][] edges) {
    List<List<Integer>> adj = new ArrayList<>();
    for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
    for (int[] edge : edges) {
        adj.get(edge[0]).add(edge[1]);
        adj.get(edge[1]).add(edge[0]);
    }

    boolean[] visited = new boolean[n];
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfs(i, visited, adj);
            count++;
        }
    }
    return count;
}
```

---

### 34. Find Bridges in a Graph

```java
int time = 0;
List<List<Integer>> bridges = new ArrayList<>();

public void dfs(int u, int parent, int[] disc, int[] low, List<List<Integer>> adj) {
    disc[u] = low[u] = ++time;
    for (int v : adj.get(u)) {
        if (v == parent) continue;
        if (disc[v] == 0) {
            dfs(v, u, disc, low, adj);
            low[u] = Math.min(low[u], low[v]);
            if (low[v] > disc[u]) bridges.add(Arrays.asList(u, v));
        } else {
            low[u] = Math.min(low[u], disc[v]);
        }
    }
}
```


## 📚 Stacks and Queues
16. Implement a stack using an array.  
17. Implement a stack that supports push, pop, top, and retrieving the minimum element.  
18. Implement a circular queue.  
19. Design a max stack that supports push, pop, top, retrieve maximum element.  
20. Design a queue using stacks.  

## 🌲 Trees and BSTs
21. Find the height of a binary tree.  
22. Find the lowest common ancestor of two nodes in a binary tree.  
23. Validate if a binary tree is a valid binary search tree.  
24. Serialize and deserialize a binary tree.  
25. Implement an inorder traversal of a binary tree.  
26. Find the diameter of a binary tree.  
27. Convert a binary tree to its mirror tree.  

## 🔗 Graphs
28. Implement depth-first search (DFS).  
29. Implement breadth-first search (BFS).  
30. Find the shortest path between two nodes in an unweighted graph.  
31. Detect a cycle in an undirected graph using DFS.  
32. Check if a graph is bipartite.  
33. Find the number of connected components in an undirected graph.  
34. Find bridges in a graph.  

## 📊 Sorting and Searching
35. Implement (bubble, insertion, selection, merge) sort.  
36. Implement quicksort.  
37. Implement binary search.  
38. Implement interpolation search.  
39. Find the kth smallest element in an array.  
40. Count number of inversions in an array.  

## 💡 Dynamic Programming (DP)
41. Find the nth Fibonacci number using dynamic programming.  
42. Solve the 0/1 knapsack problem using dynamic programming.  
43. Use memoization to optimize recursive solutions.  
44. Find the longest common subsequence of two strings.  
45. Solve the coin change problem.  
46. Use tabulation in dynamic programming.  

## 🌀 Backtracking
47. Solve the N-Queens problem using backtracking.  
48. Generate all permutations of a set using backtracking.  
49. Solve Sudoku using backtracking.  
50. Subset sum problem using backtracking.  
51. Graph coloring problem using backtracking.  
52. Hamiltonian cycle using backtracking.  

## #️⃣ Hashing
53. Implement a hash table using separate chaining.  
54. Find first non-repeating character in a string using hashing.  
55. Explain collision resolution techniques.  
56. Solve the two-sum problem using hashing.  
57. Implement a hash set.  
58. Count frequency of elements in an array using hashing.  

## 🔺 Heaps
59. Implement a priority queue using a min-heap.  
60. Merge K sorted arrays using a min-heap.  
61. Perform heap sort.  
62. Find the kth largest element using a min-heap.  
63. Implement a priority queue.  
64. Build a max heap from an array.  

## 🌐 Tries
65. Implement a trie.  
66. Search for a word in a trie.  
67. Implement autocomplete using a trie.  
68. Delete a word from a trie.  
69. Find all words matching a pattern in a trie.  

## 💰 Greedy Algorithms
70. Solve the activity selection problem.  
71. Implement Huffman coding.  
72. Find MST using Prim's algorithm.  
73. Solve coin change using greedy approach.  
74. Dijkstra's algorithm.  
75. Job sequencing problem.  

## 📌 Miscellaneous
76. Stack vs Queue.  
77. Difference between BFS and DFS traversal.  
78. Concept of Big O notation.  
79. What is an AVL tree and how does it maintain balance?  
80. Difference between BFS and DFS in terms of recursion and queue usage.  
