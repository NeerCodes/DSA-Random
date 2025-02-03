# DSA-Random
Random DSA Patterns for revision.

## ARRAYS
#### Kadane's Algorithm
> Used to find the maximum subarray sum.

```java
  public int maxSubArray(int[] nums) {
    int maxSum = nums[0], currentSum = nums[0];  
    for (int i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }
    return maxSum;
 }
```

### Two Pointers Technique
> Used for problems involving arrays or strings, such as finding pairs or reversing.
> Example: Two Sum Problem.
```java
public int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return new int[]{left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return new int[]{-1, -1};
}
```

### Sliding Window
- Used for subarray or substring problems.
- Example: Maximum Sum Subarray of Size K.
```java
public int maxSumSubarray(int[] nums, int k) {
    int maxSum = 0, windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += nums[i];
    }
    maxSum = windowSum;
    for (int i = k; i < nums.length; i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = Math.max(maxSum, windowSum);
    }
    return maxSum;
}
```

### Merge Two Sorted Arrays
```java
public int[] mergeSortedArrays(int[] arr1, int[] arr2) {
    int i = 0, j = 0, k = 0;
    int[] result = new int[arr1.length + arr2.length];
    while (i < arr1.length && j < arr2.length) {
        if (arr1[i] < arr2[j]) result[k++] = arr1[i++];
        else result[k++] = arr2[j++];
    }
    while (i < arr1.length) result[k++] = arr1[i++];
    while (j < arr2.length) result[k++] = arr2[j++];
    return result;
}
```










### General 2-pointer pseudo code
```java
public boolean twoSumProblem(int[] a, int N, int X){
   //first pointer
   int left = 0;

   //second pointer
   int right = N-1;

   while(left <= right){
     //question condition match
     if(){
       //do something
       return true;
     }
     //first wrong condition
     else if(){
       //close in the array from the left
       left+=1;
     }
     //second wrong condtion
     else{
       //close in the array from right
       right-=1;
     }
   }
   return false;
}
```

### Two Sum + Sorted
```java
boolean pairSum(int[] a, int N, int X){
  int i = 0, j = N-1;
  while(i < j){
    if(a[i]+a[j]== X) 
      return true;
    else if(a[i]+a[j] < X)
      i++;
    else 
      j--;
  }
  return false;
}
```

## LINKED LIST
### Linked List Cycle
//Using Set
```java
boolean hasCycle(ListNode head){
  Set<ListNode> set = new HashSet<>();
  while(head != null){
    if(set.contains(head)){
      return true;
    }
    set.add(head);
    head = head.next;
  }
  return false;
}
```

//Using slow and fast pointers
```java
boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }

        ListNode slow = head;  // Moves one step at a time
        ListNode fast = head;  // Moves two steps at a time

        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast) { // Cycle detected
                return true;
            }
        }

        return false; // No cycle found
    }
```

### Reverse a Linked List
```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode nextNode = head.next;
        head.next = prev;
        prev = head;
        head = nextNode;
    }
    return prev;
}
```

### Merge 2 Sorted Lists
```java
   if(l1.val < l2.val){
     cur.next = l1;
     l1 = l1.next;
   }
   else{
     cur.next = l2;
     l2 = l2.next;
   }
```

### Merge K Sorted Lists
- compare every 2 lists, call 'merge2sortedLists' function, and keep doing until we have a bigger list.
- Add all the lists into one big array or list -> sort the array -> then put the elements back into a new LL
- Priority Queue is another option


### Sliding Window General Pseudo Code
```java
int max_sum = 0, window_sum = 0; 
/* calculate sum of 1st window */
for (int i = 0; i < k; i++)  window_sum += arr[i]; 

/* slide window from start to end in array. */
for (int i = k; i < n; i++){ 
    window_sum += arr[i] - arr[i-k];    // saving re-computation
    max_sum = max(max_sum, window_sum);
}
```

## STACKS and QUEUES
### Implement Queue using Stack
```java
class MyQueue {
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();
    
    public void push(int x) {
        stack1.push(x);
    }
    
    public int pop() {
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}
```

### Next Greater Element
```java
public int[] nextGreaterElement(int[] nums) {
    Stack<Integer> stack = new Stack<>();
    int[] result = new int[nums.length];
    Arrays.fill(result, -1);
    for (int i = 0; i < nums.length; i++) {
        while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
            result[stack.pop()] = nums[i];
        }
        stack.push(i);
    }
    return result;
}
```

### Implementing a Stack
```java
class MyStack {
    private List<Integer> data;
    public MyStack() {
        data = new ArrayList<>();
    }
    public void push(int x) {
        data.add(x);
    }
    public int pop() {
        return data.remove(data.size() - 1);
    }
    public int top() {
        return data.get(data.size() - 1);
    }
    public boolean isEmpty() {
        return data.isEmpty();
    }
}
```

## RECURSION and BACKTRACKING
### Generate All Subsets (Power Set)
```java
public void subsets(int[] nums, int index, List<Integer> current, List<List<Integer>> result) {
    if (index == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }
    subsets(nums, index + 1, current, result);
    current.add(nums[index]);
    subsets(nums, index + 1, current, result);
    current.remove(current.size() - 1);
}
```

### N-Queens Problem
```java
public boolean solveNQueens(int board[][], int row) {
    if (row >= board.length) return true;
    for (int i = 0; i < board.length; i++) {
        if (isSafe(board, row, i)) {
            board[row][i] = 1;
            if (solveNQueens(board, row + 1)) return true;
            board[row][i] = 0;
        }
    }
    return false;
}
```

## SEARCHING and SORTING
### Binary Search
```java
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

### Quick Sort
```java
public void quickSort(int[] arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

### Merge Sort
```java
public void mergeSort(int[] arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
private void merge(int[] arr, int left, int mid, int right) {
    // Merge logic
}
```


## Graph Algorithms
### Breadth-First Search (BFS)
```java
public void bfs(int start, List<List<Integer>> graph) {
    Queue<Integer> queue = new LinkedList<>();
    boolean[] visited = new boolean[graph.size()];
    queue.add(start);
    visited[start] = true;
    while (!queue.isEmpty()) {
        int node = queue.poll();
        System.out.print(node + " ");
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                queue.add(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}
```


### Depth-First Search (DFS)
```java
public void dfs(int node, List<List<Integer>> graph, boolean[] visited) {
    visited[node] = true;
    System.out.print(node + " ");
    for (int neighbor : graph.get(node)) {
        if (!visited[neighbor]) {
            dfs(neighbor, graph, visited);
        }
    }
}
```


### Dijkstra’s Algorithm
```java
public int[] dijkstra(int V, List<List<int[]>> adj, int src) {
    int[] dist = new int[V];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
    pq.add(new int[]{src, 0});
    while (!pq.isEmpty()) {
        int[] node = pq.poll();
        int u = node[0], d = node[1];
        for (int[] edge : adj.get(u)) {
            int v = edge[0], weight = edge[1];
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.add(new int[]{v, dist[v]});
            }
        }
    }
    return dist;
}
```

### Floyd-Warshall Algorithm
```java
public void floydWarshall(int[][] graph) {
    int V = graph.length;
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                graph[i][j] = Math.min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
}
```


## Dynamic Programming
### Fibonacci Sequence
```java
public int fib(int n) {
    if (n <= 1) return n;
    int[] dp = new int[n + 1];
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

### Longest Common Subsequence (LCS)
```java
public int lcs(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}
```

### Longest Increasing Subsequence (LIS)
> Problem Statement: Find the length of the longest subsequence of a given sequence such that all elements are in increasing order.
```java
public int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length];
    Arrays.fill(dp, 1);
    int maxLen = 1;
    for (int i = 1; i < nums.length; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        maxLen = Math.max(maxLen, dp[i]);
    }
    return maxLen;
}
```


### Coin Change Problem
> Problem Statement: Find the minimum number of coins required to make a given amount.
```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

### 0-1 Knapsack Problem
```java
public int knapsack(int[] weights, int[] values, int W) {
    int n = weights.length;
    int[][] dp = new int[n + 1][W + 1];
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= W; j++) {
            if (weights[i - 1] <= j) {
                dp[i][j] = Math.max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]]);
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[n][W];
}
```

### Unbounded Knapsack
```java
public int unboundedKnapsack(int[] weights, int[] values, int W) {
    int n = weights.length;
    int[] dp = new int[W + 1];
    for (int j = 0; j <= W; j++) {
        for (int i = 0; i < n; i++) {
            if (weights[i] <= j) {
                dp[j] = Math.max(dp[j], values[i] + dp[j - weights[i]]);
            }
        }
    }
    return dp[W];
}
```


## Greedy Algorithms
### Activity Selection Problem
```java
public int maxActivities(int[][] intervals) {
    Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
    int count = 1, end = intervals[0][1];
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] >= end) {
            count++;
            end = intervals[i][1];
        }
    }
    return count;
}
```

## Bit Manipulation
- AND (&): Sets a bit to 1 if both bits are 1.

- OR (|): Sets a bit to 1 if at least one bit is 1.

- XOR (^): Sets a bit to 1 if the bits are different.

- NOT (~): Flips all bits.

- Left Shift (<<): Shifts bits to the left, filling with 0.

- Right Shift (>>): Shifts bits to the right, filling with the sign bit.

### Check if a Number is a Power of Two
```java
public boolean isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

### Count the Number of Set Bits (Hamming Weight)
```java
public int countSetBits(int n) {
    int count = 0;
    while (n > 0) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
```

### Find the Missing Number
> Problem Statement: Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the missing number.
```java
public int missingNumber(int[] nums) {
    int xor = 0;
    for (int i = 0; i < nums.length; i++) {
        xor ^= nums[i] ^ (i + 1);
    }
    return xor;
}
```

### Single Number
> Problem Statement: Given a non-empty array of integers where every element appears twice except for one, find that single one.
```java
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```


