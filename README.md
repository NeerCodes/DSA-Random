# DSA-Random
Random DSA Patterns for revision.

## ARRAYS
#### Maximum Subarray (Kadane's Algorithm)
> Problem: Find the contiguous subarray with the largest sum.
> Approach: Use Kadane's algorithm to track the maximum sum ending at each index.
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

#### Two Sum
> Problem: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

> Approach: Use a hash map to store the difference between target and the current element.
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[]{map.get(complement), i};
        }
        map.put(nums[i], i);
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

#### Merge Intervals
> Problem: Merge overlapping intervals.

> Approach: Sort the intervals and merge overlapping ones.
```java
public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1) return intervals;
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    List<int[]> result = new ArrayList<>();
    int[] newInterval = intervals[0];
    result.add(newInterval);
    for (int[] interval : intervals) {
        if (interval[0] <= newInterval[1]) {
            newInterval[1] = Math.max(newInterval[1], interval[1]);
        } else {
            newInterval = interval;
            result.add(newInterval);
        }
    }
    return result.toArray(new int[result.size()][]);
}
```

#### Rotate Array
> Problem: Rotate an array to the right by k steps.

> Approach: Reverse the entire array, then reverse the first k elements and the remaining elements.
```java
public void rotate(int[] nums, int k) {
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
}
private void reverse(int[] nums, int start, int end) {
    while (start < end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}
```

#### Product of Array Except Self
> Problem: Given an array nums, return an array output such that output[i] is equal to the product of all elements of nums except nums[i].

> Approach: Use prefix and suffix products.
```java
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    result[0] = 1;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] * nums[i - 1];
    }
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
        result[i] *= right;
        right *= nums[i];
    }
    return result;
}
```

#### Find All Duplicates in an Array
> Problem: Given an array of integers where each integer is between 1 and n (inclusive), find all duplicates.

> Approach: Use the array indices to mark visited numbers.
```java
public List<Integer> findDuplicates(int[] nums) {
    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
        int index = Math.abs(nums[i]) - 1;
        if (nums[index] < 0) {
            result.add(index + 1);
        } else {
            nums[index] = -nums[index];
        }
    }
    return result;
}
```

#### Longest Consecutive Sequence
> Problem: Given an unsorted array of integers, find the length of the longest consecutive sequence.

> Approach: Use a hash set to store all elements and check for sequences.
```java
public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) set.add(num);
    int longestStreak = 0;
    for (int num : set) {
        if (!set.contains(num - 1)) {
            int currentNum = num;
            int currentStreak = 1;
            while (set.contains(currentNum + 1)) {
                currentNum++;
                currentStreak++;
            }
            longestStreak = Math.max(longestStreak, currentStreak);
        }
    }
    return longestStreak;
}
```


#### Search in Rotated Sorted Array
> Problem: Given a rotated sorted array, find the index of a target value.

> Approach: Use binary search with additional checks for rotation.
```java
public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}
```


#### Two Pointers Technique
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

#### Sliding Window
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

- Example: Given an array and a window size k, find the maximum in each sliding window.
- Approach: Use a deque to store indices of useful elements.
```java
public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || k <= 0) return new int[0];
    int n = nums.length;
    int[] result = new int[n - k + 1];
    Deque<Integer> deque = new ArrayDeque<>();
    int index = 0;
    for (int i = 0; i < n; i++) {
        while (!deque.isEmpty() && deque.peek() < i - k + 1) {
            deque.poll();
        }
        while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
            deque.pollLast();
        }
        deque.offer(i);
        if (i >= k - 1) {
            result[index++] = nums[deque.peek()];
        }
    }
    return result;
}
```

#### Missing Number
> Problem: Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the missing number.

> Approach: Use XOR or the sum of numbers.
```java
public int missingNumber(int[] nums) {
    int xor = 0;
    for (int i = 0; i < nums.length; i++) {
        xor ^= nums[i] ^ (i + 1);
    }
    return xor;
}
```

#### Merge Two Sorted Arrays
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

#### General 2-pointer pseudo code
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

#### Two Sum + Sorted
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
#### Linked List Cycle
> Using Set
> T = O(N)
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

> Using slow and fast pointers (Floyd’s Cycle Detection)
- Problem: Check if a linked list has a cycle.
- Approach: Use two pointers (slow, fast); if they meet, there’s a cycle.
- T = O(N)
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

#### Reverse a Linked List
> Problem: Reverse a given singly linked list.
> Approach: Use three pointers (prev, curr, next) to reverse links iteratively.
> T = O(N)
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

#### Find Start of Cycle in Linked List
> Problem: Find the node where the cycle begins.
> Approach: After cycle detection, move one pointer to head, move both one step at a time.
> T = O(N)
```java
public ListNode detectCycle(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow.next;
                fast = fast.next;
            }
            return slow;
        }
    }
    return null;
}
```

### Merge Two Sorted Linked Lists
> Problem: Merge two sorted linked lists into one sorted list.
> Approach: Use a dummy node and iterate through both lists, connecting nodes in sorted order.
> T = O(N)
```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(-1), cur = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            cur.next = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    cur.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```

### Merge K Sorted Lists
> compare every 2 lists, call 'merge2sortedLists' function, and keep doing until we have a bigger list.
> 
> Merge Two Lists at a Time (Divide & Conquer) - O(N log K)
```java
public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return mergeLists(lists, 0, lists.length - 1);
    }

    private ListNode mergeLists(ListNode[] lists, int left, int right) {
        if (left == right) return lists[left]; // Only one list left
        int mid = left + (right - left) / 2;
        ListNode l1 = mergeLists(lists, left, mid);
        ListNode l2 = mergeLists(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = (l1 != null) ? l1 : l2;
        return dummy.next;
    }
```
- Recursively divide the lists into two halves.
- Merge each half recursively using mergeTwoLists.
- Similar to Merge Sort, resulting in O(N log K) time complexity.

> Add all the lists into one big array or list -> sort the array -> then put the elements back into a new LL
> Using a Priority Queue (Min-Heap) - O(N log K)
```java
public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode node : lists) {
            if (node != null) pq.offer(node);
        }

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;

        while (!pq.isEmpty()) {
            ListNode minNode = pq.poll();
            cur.next = minNode;
            cur = cur.next;
            if (minNode.next != null) pq.offer(minNode.next);
        }

        return dummy.next;
    }
```

#### Remove N-th Node from End
> Problem: Remove the N-th node from the end of the linked list.
> Approach: Use two pointers with a gap of N, move both till end, delete required node.
> T = O(N)
```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode first = dummy, second = dummy;
    for (int i = 0; i <= n; i++) first = first.next;
    while (first != null) {
        first = first.next;
        second = second.next;
    }
    second.next = second.next.next;
    return dummy.next;
}
```

#### Find Middle of Linked List
> Problem: Find the middle node of a linked list.
> Approach: Use slow and fast pointers; fast moves twice as fast as slow.
> T = O(N)
```java
public ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

#### Check if Linked List is Palindrome
> Problem: Check if a linked list reads the same forward and backward.
> Approach: Find the middle, reverse second half, compare both halves.
> T = O(N)
```java
public boolean isPalindrome(ListNode head) {
    if (head == null || head.next == null) return true;
    ListNode slow = head, fast = head, prev = null;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    ListNode rev = reverse(slow);
    ListNode first = head, second = rev;
    while (second != null) {
        if (first.val != second.val) return false;
        first = first.next;
        second = second.next;
    }
    return true;
}

private ListNode reverse(ListNode head) {
    ListNode prev = null, curr = head;
    while (curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

#### Reverse in Groups of K
> Problem: Reverse the linked list in groups of K.
> Approach: Reverse first K nodes recursively, then connect.
> T = O(N)
```java
public ListNode reverseKGroup(ListNode head, int k) {
    ListNode curr = head;
    int count = 0;
    while (curr != null && count < k) {
        curr = curr.next;
        count++;
    }
    if (count == k) {
        curr = reverseKGroup(curr, k);
        while (count-- > 0) {
            ListNode temp = head.next;
            head.next = curr;
            curr = head;
            head = temp;
        }
        head = curr;
    }
    return head;
}
```

#### Intersection of Two Linked Lists
> Problem: Find the node where two linked lists intersect.
> Approach: Use two pointers, move them to opposite lists when reaching the end.
> T = O(N)

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode a = headA, b = headB;
    while (a != b) {
        a = (a == null) ? headB : a.next;
        b = (b == null) ? headA : b.next;
    }
    return a;
}
```

#### Flatten a Multilevel Linked List
> Problem: Flatten a linked list where nodes have an extra child pointer.
> Approach: Use DFS or an iterative approach with a stack. T = O(N)
```java
public Node flatten(Node head) {
    if (head == null) return head;
    Stack<Node> stack = new Stack<>();
    Node curr = head;
    while (curr != null) {
        if (curr.child != null) {
            if (curr.next != null) stack.push(curr.next);
            curr.next = curr.child;
            curr.child = null;
        }
        if (curr.next == null && !stack.isEmpty()) curr.next = stack.pop();
        curr = curr.next;
    }
    return head;
}
```



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
## STACK
#### Valid Parentheses
> Problem: Check if a given string of parentheses is valid.
> Approach: Use a stack to track opening brackets and match them with closing ones.
```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') stack.push(c);
        else {
            if (stack.isEmpty()) return false;
            char top = stack.pop();
            if ((c == ')' && top != '(') || (c == '}' && top != '{') || (c == ']' && top != '[')) return false;
        }
    }
    return stack.isEmpty();
}
```

#### Next Greater Element
> Problem: Find the next greater element for each element in an array.
> Approach: Use a stack to store indices and find the next greater element efficiently.
```java
public int[] nextGreaterElement(int[] nums) {
    int[] res = new int[nums.length];
    Stack<Integer> stack = new Stack<>();
    for (int i = nums.length - 1; i >= 0; i--) {
        while (!stack.isEmpty() && stack.peek() <= nums[i]) stack.pop();
        res[i] = stack.isEmpty() ? -1 : stack.peek();
        stack.push(nums[i]);
    }
    return res;
}
```

#### Daily Temperatures (Next Greater Element Variant)
> Problem: Find the number of days until a warmer temperature appears.
> Approach: Use a monotonic decreasing stack to store indices.
```java
public int[] dailyTemperatures(int[] T) {
    int[] res = new int[T.length];
    Stack<Integer> stack = new Stack<>();
    for (int i = 0; i < T.length; i++) {
        while (!stack.isEmpty() && T[i] > T[stack.peek()]) {
            int idx = stack.pop();
            res[idx] = i - idx;
        }
        stack.push(i);
    }
    return res;
}
```

#### Next Smaller Element
> Problem: For each element, find the next smaller element to the right.
> Approach: Similar to Next Greater Element, but using a monotonic increasing stack.
```java
public int[] nextSmallerElement(int[] nums) {
    int[] res = new int[nums.length];
    Stack<Integer> stack = new Stack<>();
    for (int i = nums.length - 1; i >= 0; i--) {
        while (!stack.isEmpty() && stack.peek() >= nums[i]) stack.pop();
        res[i] = stack.isEmpty() ? -1 : stack.peek();
        stack.push(nums[i]);
    }
    return res;
}
```

#### Largest Rectangle in Histogram
> Problem: Find the largest rectangular area in a histogram.
> Approach: Use a stack to maintain indices of increasing heights.
```java
public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0, n = heights.length;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (!stack.isEmpty() && h < heights[stack.peek()]) {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        stack.push(i);
    }
    return maxArea;
}
```

#### Max Area of Binary Matrix (Largest Rectangle in Binary Matrix)
> Problem: Find the largest rectangle of 1s in a binary matrix.
> Approach: Convert each row into histogram heights, then use Largest Rectangle in Histogram.
```java
public int maximalRectangle(char[][] matrix) {
    if (matrix.length == 0) return 0;
    int maxArea = 0, cols = matrix[0].length;
    int[] heights = new int[cols];
    for (char[] row : matrix) {
        for (int j = 0; j < cols; j++) heights[j] = (row[j] == '1') ? heights[j] + 1 : 0;
        maxArea = Math.max(maxArea, largestRectangleArea(heights));
    }
    return maxArea;
}
```

#### Simplify Path (Unix Path)
> Problem: Simplify a Unix file path like /home/../usr//bin/.
> Approach: Use a stack to track valid directory names.
```java
public String simplifyPath(String path) {
    Stack<String> stack = new Stack<>();
    for (String dir : path.split("/")) {
        if (dir.equals("..")) { if (!stack.isEmpty()) stack.pop(); }
        else if (!dir.equals("") && !dir.equals(".")) stack.push(dir);
    }
    return "/" + String.join("/", stack);
}
```

#### Min Stack (Stack with Get Minimum in O(1))
> Problem: Implement a stack that supports push, pop, top, and retrieving the minimum element in O(1).
> Approach: Use two stacks, one for values and one for minimums
```java
class MinStack {
    Stack<Integer> stack = new Stack<>();
    Stack<Integer> minStack = new Stack<>();

    public void push(int val) {
        stack.push(val);
        if (minStack.isEmpty() || val <= minStack.peek()) minStack.push(val);
    }

    public void pop() {
        if (stack.pop().equals(minStack.peek())) minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
```

#### Implement Stack using Queues
> Problem: Implement a stack using two queues.
> Approach: Use a single queue and push elements in reverse order.
```java
class MyStack {
    Queue<Integer> q = new LinkedList<>();

    public void push(int x) {
        q.add(x);
        for (int i = 0; i < q.size() - 1; i++) q.add(q.poll());
    }

    public int pop() {
        return q.poll();
    }

    public int top() {
        return q.peek();
    }

    public boolean empty() {
        return q.isEmpty();
    }
}
```

#### Implementing a Stack (using List)
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

#### Evaluate Reverse Polish Notation (Postfix Expression)
> Problem: Evaluate an arithmetic expression given in postfix notation.
> Approach: Use a stack, push numbers, pop and compute when encountering an operator.
```java
public int evalRPN(String[] tokens) {
    Stack<Integer> stack = new Stack<>();
    for (String token : tokens) {
        if ("+-*/".contains(token)) {
            int b = stack.pop(), a = stack.pop();
            switch (token) {
                case "+": stack.push(a + b); break;
                case "-": stack.push(a - b); break;
                case "*": stack.push(a * b); break;
                case "/": stack.push(a / b); break;
            }
        } else stack.push(Integer.parseInt(token));
    }
    return stack.pop();
}
```

#### Stock Span Problem
> Problem: Find the span of stock prices for each day.
> Approach: Use a monotonic decreasing stack to track indices.
```java
public int[] stockSpan(int[] prices) {
    int[] span = new int[prices.length];
    Stack<Integer> stack = new Stack<>();
    for (int i = 0; i < prices.length; i++) {
        while (!stack.isEmpty() && prices[stack.peek()] <= prices[i]) stack.pop();
        span[i] = stack.isEmpty() ? i + 1 : i - stack.peek();
        stack.push(i);
    }
    return span;
}
```

## QUEUE
#### Implement Queue using Stacks
> Problem: Implement a queue using two stacks.
> Approach: Use two stacks (one for enqueue, one for dequeue).
```java
class MyQueue {
    Stack<Integer> inStack = new Stack<>();
    Stack<Integer> outStack = new Stack<>();

    public void push(int x) {
        inStack.push(x);
    }

    public int pop() {
        if (outStack.isEmpty()) while (!inStack.isEmpty()) outStack.push(inStack.pop());
        return outStack.pop();
    }

    public int peek() {
        if (outStack.isEmpty()) while (!inStack.isEmpty()) outStack.push(inStack.pop());
        return outStack.peek();
    }

    public boolean empty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }
}
```

#### Implement Stack using Queue
> Problem: Implement a stack using two queues with push() and pop().
> Approach: Push elements into the queue and rotate it for LIFO order.
```java
class MyStack {
    Queue<Integer> q = new LinkedList<>();

    public void push(int x) {
        q.add(x);
        for (int i = 0; i < q.size() - 1; i++) q.add(q.poll());
    }

    public int pop() {
        return q.poll();
    }

    public int top() {
        return q.peek();
    }

    public boolean empty() {
        return q.isEmpty();
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
   public boolean solveNQueens(int[][] board, int row) {
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

    private boolean isSafe(int[][] board, int row, int col) {
        int n = board.length;
        
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 1) return false;
        }

        // Check upper-left diagonal
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 1) return false;
        }

        // Check upper-right diagonal
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 1) return false;
        }

        return true;
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

public int partition(int[] arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

```

### Merge Sort
```java
public void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2; // Avoids overflow for large left and right
            mergeSort(arr, left, mid);          // Sort the left half
            mergeSort(arr, mid + 1, right);     // Sort the right half
            merge(arr, left, mid, right);       // Merge the sorted halves
        }
}

private void merge(int[] arr, int left, int mid, int right) {
        // Sizes of the two subarrays to be merged
        int n1 = mid - left + 1;
        int n2 = right - mid;

        // Temporary arrays to hold the two halves
        int[] leftArray = new int[n1];
        int[] rightArray = new int[n2];

        // Copy data to temporary arrays
        for (int i = 0; i < n1; i++) {
            leftArray[i] = arr[left + i];
        }
        for (int j = 0; j < n2; j++) {
            rightArray[j] = arr[mid + 1 + j];
        }

        // Merge the two temporary arrays back into the original array
        int i = 0, j = 0; // Initial indices of the two subarrays
        int k = left;     // Initial index of the merged subarray

        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                arr[k] = leftArray[i];
                i++;
            } else {
                arr[k] = rightArray[j];
                j++;
            }
            k++;
        }

        // Copy remaining elements of leftArray (if any)
        while (i < n1) {
            arr[k] = leftArray[i];
            i++;
            k++;
        }

        // Copy remaining elements of rightArray (if any)
        while (j < n2) {
            arr[k] = rightArray[j];
            j++;
            k++;
        }
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
> **Problem Statement:** Find the length of the longest subsequence of a given sequence such that all elements are in increasing order.
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
> **Problem Statement:** Find the minimum number of coins required to make a given amount.
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
- **AND (&):** Sets a bit to 1 if both bits are 1.

- **OR (|):** Sets a bit to 1 if at least one bit is 1.

- **XOR (^):** Sets a bit to 1 if the bits are different.

- **NOT (~):** Flips all bits.

- **Left Shift (<<):** Shifts bits to the left, filling with 0.

- **Right Shift (>>):** Shifts bits to the right, filling with the sign bit.

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
> **Problem Statement:** Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the missing number.
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
> **Problem Statement:** Given a non-empty array of integers where every element appears twice except for one, find that single one.
```java
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```


## ADVANCE TOPICS
## Trie (Prefix Tree)
> **Used for:** Efficient string search and prefix matching.
```java
class TrieNode {
    TrieNode[] children = new TrieNode[26];
    boolean isEndOfWord;
}

class Trie {
    private TrieNode root;
    public Trie() {
        root = new TrieNode();
    }
    public void insert(String word) {
        TrieNode curr = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (curr.children[index] == null) {
                curr.children[index] = new TrieNode();
            }
            curr = curr.children[index];
        }
        curr.isEndOfWord = true;
    }
    public boolean search(String word) {
        TrieNode curr = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (curr.children[index] == null) return false;
            curr = curr.children[index];
        }
        return curr.isEndOfWord;
    }
    public boolean startsWith(String prefix) {
        TrieNode curr = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (curr.children[index] == null) return false;
            curr = curr.children[index];
        }
        return true;
    }
}
```


## Segment Tree
> **Used for:** Range queries and updates.
```java
class SegmentTree {
    int[] tree;
    int n;
    public SegmentTree(int[] nums) {
        n = nums.length;
        tree = new int[4 * n];
        build(nums, 0, 0, n - 1);
    }
    private void build(int[] nums, int idx, int left, int right) {
        if (left == right) {
            tree[idx] = nums[left];
            return;
        }
        int mid = left + (right - left) / 2;
        build(nums, 2 * idx + 1, left, mid);
        build(nums, 2 * idx + 2, mid + 1, right);
        tree[idx] = tree[2 * idx + 1] + tree[2 * idx + 2];
    }
    public int query(int ql, int qr) {
        return query(0, 0, n - 1, ql, qr);
    }
    private int query(int idx, int left, int right, int ql, int qr) {
        if (ql > right || qr < left) return 0;
        if (ql <= left && qr >= right) return tree[idx];
        int mid = left + (right - left) / 2;
        return query(2 * idx + 1, left, mid, ql, qr) + query(2 * idx + 2, mid + 1, right, ql, qr);
    }
}
```
