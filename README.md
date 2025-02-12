# DSA-Random
Random DSA Patterns (Algorithms and important problems) for revision.

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

#### Merge Two Sorted Linked Lists
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

#### Merge K Sorted Lists
> compare every 2 lists, call 'merge2sortedLists' function, and keep doing until we have a bigger list.

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

#### Sliding Window Maximum
> Problem: Find the maximum in every sliding window of size k.

> Approach: Use a Deque (monotonic decreasing) to keep track of max elements.
```java
public int[] maxSlidingWindow(int[] nums, int k) {
    int[] res = new int[nums.length - k + 1];
    Deque<Integer> dq = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
        if (!dq.isEmpty() && dq.peek() < i - k + 1) dq.poll();
        while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i]) dq.pollLast();
        dq.offer(i);
        if (i >= k - 1) res[i - k + 1] = nums[dq.peek()];
    }
    return res;
}
```

#### First Unique Character in a String
> Problem: Find the first non-repeating character in a string.

> Approach: Use a queue to track characters and remove repeated ones.
```java
public char firstUniqChar(String s) {
    int[] count = new int[26];
    Queue<Character> q = new LinkedList<>();
    for (char c : s.toCharArray()) {
        count[c - 'a']++;
        q.offer(c);
        while (!q.isEmpty() && count[q.peek() - 'a'] > 1) q.poll();
    }
    return q.isEmpty() ? '#' : q.peek();
}
```

#### Rotten Oranges (Shortest Path in Grid)
> Problem: Find the minimum time to rot all oranges in a grid.

> Approach: Use BFS (multi-source) starting from rotten oranges.
```java
public int orangesRotting(int[][] grid) {
    int fresh = 0, time = 0;
    Queue<int[]> q = new LinkedList<>();
    int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == 2) q.offer(new int[]{i, j});
            else if (grid[i][j] == 1) fresh++;
        }
    }

    while (!q.isEmpty() && fresh > 0) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            int[] pos = q.poll();
            for (int[] d : dirs) {
                int x = pos[0] + d[0], y = pos[1] + d[1];
                if (x >= 0 && y >= 0 && x < grid.length && y < grid[0].length && grid[x][y] == 1) {
                    grid[x][y] = 2;
                    q.offer(new int[]{x, y});
                    fresh--;
                }
            }
        }
        time++;
    }

    return fresh == 0 ? time : -1;
}
```

#### Number of Islands
> Problem: Count the number of islands in a grid (1s are land).

> Approach: Use BFS (or DFS) to traverse and mark connected components.
```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == '1') {
                bfs(grid, i, j);
                count++;
            }
        }
    }
    return count;
}

private void bfs(char[][] grid, int i, int j) {
    Queue<int[]> q = new LinkedList<>();
    q.offer(new int[]{i, j});
    grid[i][j] = '0';
    int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    while (!q.isEmpty()) {
        int[] pos = q.poll();
        for (int[] d : dirs) {
            int x = pos[0] + d[0], y = pos[1] + d[1];
            if (x >= 0 && y >= 0 && x < grid.length && y < grid[0].length && grid[x][y] == '1') {
                grid[x][y] = '0';
                q.offer(new int[]{x, y});
            }
        }
    }
}
```

#### LRU Cache (Least Recently Used Cache)
> Problem: Design an LRU cache that evicts the least recently used item.

> Approach: Use HashMap + Doubly LinkedList for O(1) operations.
```java
class LRUCache {
    class Node {
        int key, value;
        Node prev, next;
        Node(int k, int v) { key = k; value = v; }
    }

    private final int capacity;
    private final Map<Integer, Node> map;
    private final Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node);
        insert(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) remove(map.get(key));
        if (map.size() == capacity) remove(tail.prev);
        insert(new Node(key, value));
    }

    private void remove(Node node) {
        map.remove(node.key);
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void insert(Node node) {
        map.put(node.key, node);
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
}
```




## RECURSION 
#### Factorial of a Number
> Problem: Compute the factorial of n.

> Approach: Use recursion, fact(n) = n * fact(n - 1).
```java
public int factorial(int n) {
    return (n == 0 || n == 1) ? 1 : n * factorial(n - 1);
}
```

#### Fibonacci Series
> Problem: Find the nth Fibonacci number.

> Approach: Use recursion, fib(n) = fib(n-1) + fib(n-2), optimized with memoization.
```java
public int fibonacci(int n, Map<Integer, Integer> memo) {
    if (n <= 1) return n;
    if (!memo.containsKey(n)) memo.put(n, fibonacci(n - 1, memo) + fibonacci(n - 2, memo));
    return memo.get(n);
}
```

#### Print All Subsequences of an Array
> Problem: Print all subsequences (power set).

> Approach: Recursively include/exclude each element.
```java
public void printSubsequences(int[] arr, int index, List<Integer> path) {
    if (index == arr.length) {
        System.out.println(path);
        return;
    }
    path.add(arr[index]);
    printSubsequences(arr, index + 1, path);
    path.remove(path.size() - 1);
    printSubsequences(arr, index + 1, path);
}
```

#### Subset Sum
> Problem: Find all subsets that sum to a target value.

> Approach: Recursively include/exclude numbers and track the sum.
```java
public void subsetSum(int[] nums, int index, int sum, int target, List<Integer> path) {
    if (sum == target) System.out.println(path);
    if (index == nums.length) return;
    path.add(nums[index]);
    subsetSum(nums, index + 1, sum + nums[index], target, path);
    path.remove(path.size() - 1);
    subsetSum(nums, index + 1, sum, target, path);
}
```

#### Tower of Hanoi
> Problem: Move n disks from source to destination using an auxiliary peg.
> Approach:
  - 1. Move n-1 disks from source to auxiliary.
  - 2. Move the nth disk from source to destination.
  - 3. Move n-1 disks from auxiliary to destination.
  - 📈 Time Complexity: O(2^N)
```java
public class TowerOfHanoi {
    public static void solveHanoi(int n, char source, char auxiliary, char destination) {
        if (n == 1) {
            System.out.println("Move disk 1 from " + source + " to " + destination);
            return;
        }
        solveHanoi(n - 1, source, destination, auxiliary);
        System.out.println("Move disk " + n + " from " + source + " to " + destination);
        solveHanoi(n - 1, auxiliary, source, destination);
    }

    public static void main(String[] args) {
        solveHanoi(3, 'A', 'B', 'C');
    }
}
```

#### Reverse a String using Recursion
```java
public String reverseString(String s) {
    if (s.isEmpty()) return s;
    return reverseString(s.substring(1)) + s.charAt(0);
}
```

#### Generate All Permutations of a String
> Problem: Print all permutations of a given string.

> Approach: Swap characters and recurse.
```java
public void permute(char[] str, int left) {
    if (left == str.length) System.out.println(new String(str));
    for (int i = left; i < str.length; i++) {
        swap(str, left, i);
        permute(str, left + 1);
        swap(str, left, i);
    }
}
private void swap(char[] arr, int i, int j) {
   char temp = arr[i];
   arr[i] = arr[j];
   arr[j] = temp;
}
```

#### N-Queens Problem
> Problem: Place N queens on an N×N chessboard such that no two queens attack each other.

> Approach: Use backtracking and check row, column, diagonal conflicts.
```java
public void solveNQueens(int n) {
    char[][] board = new char[n][n];
    for (char[] row : board) Arrays.fill(row, '.');
    placeQueens(board, 0);
}
private void placeQueens(char[][] board, int row) {
    if (row == board.length) { printBoard(board); return; }
    for (int col = 0; col < board.length; col++) {
        if (isValid(board, row, col)) {
            board[row][col] = 'Q';
            placeQueens(board, row + 1);
            board[row][col] = '.';
        }
    }
}
private boolean isValid(char[][] board, int r, int c) {
    for (int i = 0; i < r; i++) if (board[i][c] == 'Q') return false;
    for (int i = r, j = c; i >= 0 && j >= 0; i--, j--) if (board[i][j] == 'Q') return false;
    for (int i = r, j = c; i >= 0 && j < board.length; i--, j++) if (board[i][j] == 'Q') return false;
    return true;
}
private void printBoard(char[][] board) {
    for (char[] row : board) System.out.println(new String(row));
    System.out.println();
}
```

#### Word Search (Backtracking)
> Problem: Check if a word exists in a grid.

> Approach: Use DFS + Backtracking.
```java
public boolean exist(char[][] board, String word) {
    for (int i = 0; i < board.length; i++)
        for (int j = 0; j < board[0].length; j++)
            if (dfs(board, i, j, word, 0)) return true;
    return false;
}
private boolean dfs(char[][] board, int i, int j, String word, int index) {
    if (index == word.length()) return true;
    if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] != word.charAt(index)) return false;
    char temp = board[i][j];
    board[i][j] = '#';
    boolean found = dfs(board, i + 1, j, word, index + 1) ||
                    dfs(board, i - 1, j, word, index + 1) ||
                    dfs(board, i, j + 1, word, index + 1) ||
                    dfs(board, i, j - 1, word, index + 1);
    board[i][j] = temp;
    return found;
}
```

#### Sudoku Solver
> Problem: Solve a given Sudoku puzzle.

> Approach: Use backtracking to try placing numbers 1-9.
```java
public boolean solveSudoku(char[][] board) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == '.') {
                for (char num = '1'; num <= '9'; num++) {
                    if (isValid(board, i, j, num)) {
                        board[i][j] = num;
                        if (solveSudoku(board)) return true;
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
    }
    return true;
}
private boolean isValid(char[][] board, int r, int c, char num) {
    for (int i = 0; i < 9; i++) 
        if (board[i][c] == num || board[r][i] == num || board[3 * (r / 3) + i / 3][3 * (c / 3) + i % 3] == num) return false;
    return true;
}
```

#### Generate All Subsets (Power Set)
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

#### N-Queens Problem
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


## BACKTRACKING
#### Permutations of a String/Array
> Approach: Swap elements, recurse, then backtrack
```java
public void permute(char[] str, int left) {
    if (left == str.length) System.out.println(new String(str));
    for (int i = left; i < str.length; i++) {
        swap(str, left, i);
        permute(str, left + 1);
        swap(str, left, i);
    }
}
private void swap(char[] arr, int i, int j) {
   char temp = arr[i];
   arr[i] = arr[j];
   arr[j] = temp;
}
```

#### Combination Sum
> Problem: Find all unique combinations that sum up to target.

> Approach: Include the same element multiple times.
```java
public void combinationSum(int[] candidates, int index, int target, List<Integer> path) {
    if (target == 0) { System.out.println(path); return; }
    if (target < 0 || index == candidates.length) return;
    path.add(candidates[index]);
    combinationSum(candidates, index, target - candidates[index], path);
    path.remove(path.size() - 1);
    combinationSum(candidates, index + 1, target, path);
}
```

#### N-Queens Problem
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

#### Rat in a Maze
> Problem: Find a path for the rat from the top-left to the bottom-right of a maze, moving only right or down through open cells (1s).

> Approach: Recursively explore right and down moves, marking the path in the solution matrix; if a move leads to a dead end, backtrack by resetting the cell.
```java
public boolean solveMaze(int[][] maze, int x, int y, int[][] solution) {
    if (x == maze.length - 1 && y == maze.length - 1) { solution[x][y] = 1; return true; }
    if (x >= 0 && y >= 0 && x < maze.length && y < maze.length && maze[x][y] == 1) {
        solution[x][y] = 1;
        if (solveMaze(maze, x + 1, y, solution) || solveMaze(maze, x, y + 1, solution)) return true;
        solution[x][y] = 0;
    }
    return false;
}
```

#### Generate Balanced Parentheses
> Problem: Generate all valid combinations of n pairs of parentheses.

> Approach: Use recursion to add ( if open brackets remain and ) if it keeps the sequence valid, backtracking as needed.
```java
public void generateParentheses(int open, int close, String path) {
    if (open == 0 && close == 0) { System.out.println(path); return; }
    if (open > 0) generateParentheses(open - 1, close, path + "(");
    if (close > open) generateParentheses(open, close - 1, path + ")");
}
```

## SEARCHING and SORTING
## Searching
#### Binary Search
> 💡 Problem: Find an element in a sorted array in O(log N) time.

> 🛠️ Approach: Repeatedly divide the search range in half until the target is found.
```java
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

#### Lower Bound (First Position of Target or Greater)
> 💡 Problem: Find the first index where arr[i] >= target in a sorted array.

> 🛠️ Approach: Use binary search, updating right when condition is met.
```java
public int lowerBound(int[] arr, int target) {
    int left = 0, right = arr.length;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] >= target) right = mid;
        else left = mid + 1;
    }
    return left;
}
```

#### Upper Bound (First Position Greater than Target)
> 💡 Problem: Find the first index where arr[i] > target in a sorted array.

> 🛠️ Approach: Similar to lower bound but finds the first greater element.
```java
public int upperBound(int[] arr, int target) {
    int left = 0, right = arr.length;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > target) right = mid;
        else left = mid + 1;
    }
    return left;
}
```

#### Search in Rotated Sorted Array
> 💡 Problem: Find an element in a rotated sorted array in O(log N).

> 🛠️ Approach: Use binary search and determine which half is sorted.
```java
public int searchRotated(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) { 
            if (nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```


## Sorting
#### Bubble Sort
> 💡 Problem: Sort an array by repeatedly swapping adjacent elements.

> 🛠️ Approach: Compare and swap adjacent elements if they are out of order.
```java
public void bubbleSort(int[] arr) {
    int n = arr.length;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
```

#### Selection Sort
> 💡 Problem: Sort an array by repeatedly finding the smallest element.

> 🛠️ Approach: Select the minimum element and swap it with the first unsorted element.
```java
public void selectionSort(int[] arr) {
    int n = arr.length;
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) minIdx = j;
        }
        int temp = arr[minIdx];
        arr[minIdx] = arr[i];
        arr[i] = temp;
    }
}
```

#### Insertion Sort
> 💡 Problem: Sort an array by inserting elements at their correct position.

> 🛠️ Approach: Insert each element in its correct position in the sorted part.
```java
public void insertionSort(int[] arr) {
    int n = arr.length;
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
```

#### Quick Sort (O(n log n) Average, O(n²) Worst)
> 💡 Problem: Sort an array using a pivot-based partitioning method.

> 🛠️ Approach: Select a pivot, partition elements, and recursively sort partitions.
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

#### Merge Sort (O(n log n))
> 💡 Problem: Sort an array using the divide-and-conquer technique.

> 🛠️ Approach: Recursively divide the array and merge sorted subarrays.
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
### Traversal Algorithms
#### Breadth-First Search (BFS)
> 💡 Problem: Traverse all nodes in a graph using BFS.

> 🛠️ Approach: Use a queue to explore nodes level by level.
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

#### Depth-First Search (DFS)
> 💡 Problem: Traverse all nodes in a graph using DFS.
> 🛠️ Approach: Recursively visit nodes, marking them as visited.
```java
import java.util.*;

public class GraphDFS {
    public void dfs(int node, List<List<Integer>> adj, boolean[] visited) {
        visited[node] = true;
        System.out.print(node + " ");
        for (int neighbor : adj.get(node)) {
            if (!visited[neighbor]) dfs(neighbor, adj, visited);
        }
    }
}
```
### Shortest Path Algorithms
#### Dijkstra’s Algorithm (Single Source Shortest Path for Weighted Graph)
> 💡 Problem: Find the shortest path from a single source to all nodes.

> 🛠️ Approach: Use a priority queue (min-heap) to process nodes with the smallest distance.
```java
import java.util.*;

public class Dijkstra {
    static class Pair {
        int node, weight;
        Pair(int node, int weight) { this.node = node; this.weight = weight; }
    }

    public int[] dijkstra(int V, List<List<Pair>> adj, int src) {
        PriorityQueue<Pair> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.weight));
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        pq.add(new Pair(src, 0));

        while (!pq.isEmpty()) {
            Pair current = pq.poll();
            int u = current.node;
            int d = current.weight;

            if (d > dist[u]) continue;

            for (Pair neighbor : adj.get(u)) {
                int v = neighbor.node, weight = neighbor.weight;
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.add(new Pair(v, dist[v]));
                }
            }
        }
        return dist;
    }
}
```

#### Floyd-Warshall Algorithm (All-Pairs Shortest Path)
> 💡 Problem: Find the shortest paths between all pairs of vertices in a weighted graph.

> 🛠️ Approach: Use dynamic programming to update shortest distances iteratively for all pairs of nodes.
```java
public class FloydWarshall {
    static final int INF = 1000000; // Representing Infinity

    public void floydWarshall(int[][] graph) {
        int V = graph.length;
        int[][] dist = new int[V][V];

        // Initialize distance matrix with given graph
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }

        // Updating distances considering each node as an intermediate
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        // Print shortest distances matrix
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                System.out.print((dist[i][j] == INF ? "INF" : dist[i][j]) + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 3, INF, 5},
            {2, 0, INF, 4},
            {INF, 1, 0, INF},
            {INF, INF, 2, 0}
        };

        FloydWarshall fw = new FloydWarshall();
        fw.floydWarshall(graph);
    }
}
```

#### Bellman-Ford Algorithm (Handles Negative Weights)
> 💡 Problem: Find the shortest path from a single source, even with negative weights.

> 🛠️ Approach: Relax all edges V-1 times and check for negative cycles.
```java
import java.util.*;

public class BellmanFord {
    static class Edge {
        int src, dest, weight;
        Edge(int src, int dest, int weight) { this.src = src; this.dest = dest; this.weight = weight; }
    }

    public int[] bellmanFord(int V, List<Edge> edges, int src) {
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;

        for (int i = 0; i < V - 1; i++) {
            for (Edge edge : edges) {
                if (dist[edge.src] != Integer.MAX_VALUE && dist[edge.src] + edge.weight < dist[edge.dest]) {
                    dist[edge.dest] = dist[edge.src] + edge.weight;
                }
            }
        }
        return dist;
    }
}
```

### Cycle Detection Algorithms
#### Detect Cycle in a Directed Graph (Using DFS)
> 💡 Problem: Check if a directed graph contains a cycle.

> 🛠️ Approach: Use DFS with a recursion stack to detect back edges.
```java
public class CycleDetectionDFS {
    public boolean dfs(int node, List<List<Integer>> adj, boolean[] visited, boolean[] recStack) {
        visited[node] = true;
        recStack[node] = true;

        for (int neighbor : adj.get(node)) {
            if (!visited[neighbor] && dfs(neighbor, adj, visited, recStack)) return true;
            else if (recStack[neighbor]) return true;
        }

        recStack[node] = false;
        return false;
    }

    public boolean hasCycle(int V, List<List<Integer>> adj) {
        boolean[] visited = new boolean[V];
        boolean[] recStack = new boolean[V];

        for (int i = 0; i < V; i++) {
            if (!visited[i] && dfs(i, adj, visited, recStack)) return true;
        }
        return false;
    }
}
```

#### Detect Cycle in an Undirected Graph (Using BFS)
> 💡 Problem: Check if an undirected graph contains a cycle.

> 🛠️ Approach: Use BFS with a parent-tracking method.
```java
import java.util.*;

public class CycleDetectionBFS {
    public boolean hasCycle(int V, List<List<Integer>> adj) {
        boolean[] visited = new boolean[V];
        Queue<int[]> queue = new LinkedList<>();

        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                queue.add(new int[]{i, -1});
                visited[i] = true;

                while (!queue.isEmpty()) {
                    int[] node = queue.poll();
                    int curr = node[0], parent = node[1];

                    for (int neighbor : adj.get(curr)) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            queue.add(new int[]{neighbor, curr});
                        } else if (neighbor != parent) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}
```

### Minimum Spanning Tree (MST) Algorithms
#### Kruskal’s Algorithm (MST Using Union-Find)
> 💡 Problem: Find the minimum spanning tree of a graph.

> 🛠️ Approach: Sort edges by weight and use Union-Find to add edges to MST.
```java
import java.util.*;

public class KruskalMST {
    static class Edge implements Comparable<Edge> {
        int src, dest, weight;
        Edge(int src, int dest, int weight) { this.src = src; this.dest = dest; this.weight = weight; }
        public int compareTo(Edge e) { return this.weight - e.weight; }
    }

    static class UnionFind {
        int[] parent, rank;
        UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
        }
        int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }
        boolean union(int x, int y) {
            int rootX = find(x), rootY = find(y);
            if (rootX == rootY) return false;
            if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
            else if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
            else { parent[rootY] = rootX; rank[rootX]++; }
            return true;
        }
    }

    public int kruskal(int V, List<Edge> edges) {
        Collections.sort(edges);
        UnionFind uf = new UnionFind(V);
        int mstWeight = 0, edgeCount = 0;

        for (Edge edge : edges) {
            if (uf.union(edge.src, edge.dest)) {
                mstWeight += edge.weight;
                edgeCount++;
                if (edgeCount == V - 1) break;
            }
        }
        return mstWeight;
    }
}
```


## Dynamic Programming
#### Fibonacci Number (Basic DP)
> 💡 Problem: Find the nth Fibonacci number.

> 🛠️ Approach: Use memoization (top-down) or tabulation (bottom-up) to avoid recomputation.
```java
public class Fibonacci {
    public static int fib(int n, int[] dp) {
        if (n <= 1) return n;
        if (dp[n] != -1) return dp[n];
        return dp[n] = fib(n - 1, dp) + fib(n - 2, dp);
    }

    public static void main(String[] args) {
        int n = 10;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, -1);
        System.out.println(fib(n, dp)); // Output: 55
    }
}
```

####  Climbing Stairs
> 💡 Problem: Given n stairs, find ways to reach the top taking 1 or 2 steps at a time.

> 🛠️ Approach: Similar to Fibonacci, use DP to store solutions to subproblems.
```java
public class ClimbStairs {
    public static int climbStairs(int n) {
        if (n <= 2) return n;
        int[] dp = new int[n + 1];
        dp[1] = 1; dp[2] = 2;
        for (int i = 3; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n];
    }

    public static void main(String[] args) {
        System.out.println(climbStairs(5)); // Output: 8
    }
}
```

#### House Robber Problem
> 💡 Problem: Given an array of houses with values, find the max sum by robbing non-adjacent houses.

> 🛠️ Approach: At each house, choose to rob it (skip the next) or skip it.
```java
public class HouseRobber {
    public static int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];

        int[] dp = new int[nums.length];
        dp[0] = nums[0]; dp[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < nums.length; i++)
            dp[i] = Math.max(nums[i] + dp[i - 2], dp[i - 1]);

        return dp[nums.length - 1];
    }

    public static void main(String[] args) {
        System.out.println(rob(new int[]{2, 7, 9, 3, 1})); // Output: 12
    }
}
```

#### Longest Common Subsequence (LCS)
> 💡 Problem: Given two strings, find the longest subsequence common to both.

> 🛠️ Approach: Use a 2D DP table to store results for each substring pair.
```java
public class LCS {
    public static int lcs(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1))
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                else
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    public static void main(String[] args) {
        System.out.println(lcs("abcde", "ace")); // Output: 3
    }
}
```

#### Longest Increasing Subsequence (LIS)
> 💡 Problem: Given an array, find the length of the longest subsequence where elements are in increasing order.

> 🛠️ Approach: Use DP where dp[i] stores the LIS ending at index i. Update dp[i] by checking previous elements.
```java
public class LIS {
    public static int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int maxLIS = 1;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j])
                    dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            maxLIS = Math.max(maxLIS, dp[i]);
        }
        return maxLIS;
    }

    public static void main(String[] args) {
        int[] nums = {10, 9, 2, 5, 3, 7, 101, 18};
        System.out.println(lengthOfLIS(nums)); // Output: 4 (2,3,7,101)
    }
}
```

#### Coin Change Problem (Minimum Coins)
> 💡 Problem: Given a set of coin denominations and a target sum, find the minimum number of coins required to make that sum.

> 🛠️ Approach: Use a 1D DP array where dp[i] represents the minimum number of coins needed to form amount i.
```java
public class CoinChangeMin {
    public static int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++)
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    public static void main(String[] args) {
        int[] coins = {1, 2, 5};
        int amount = 11;
        System.out.println(coinChange(coins, amount)); // Output: 3 (5+5+1)
    }
}
```

#### Coin Change Problem (Total Ways)
> 💡 Problem: Given a set of coin denominations and a target sum, find the number of ways to form that sum.

> 🛠️ Approach: Use a 1D DP array where dp[i] represents the number of ways to form amount i.
```java
public class CoinChangeWays {
    public static int coinChangeWays(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++)
                dp[i] += dp[i - coin];
        }
        return dp[amount];
    }

    public static void main(String[] args) {
        int[] coins = {1, 2, 5};
        int amount = 5;
        System.out.println(coinChangeWays(coins, amount)); // Output: 4
    }
}
```

#### 0/1 Knapsack Problem
> 💡 Problem: Given weights and values of items, find the max value we can carry within a weight limit.

> 🛠️ Approach: Use a DP table to store the best value for each weight limit.
```java
public class Knapsack {
    public static int knapsack(int W, int[] wt, int[] val, int n) {
        int[][] dp = new int[n + 1][W + 1];

        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (wt[i - 1] <= w)
                    dp[i][w] = Math.max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
                else
                    dp[i][w] = dp[i - 1][w];
            }
        }
        return dp[n][W];
    }

    public static void main(String[] args) {
        int[] val = {60, 100, 120};
        int[] wt = {10, 20, 30};
        int W = 50;
        System.out.println(knapsack(W, wt, val, val.length)); // Output: 220
    }
}
```

### Unbounded Knapsack
> 💡 Problem: Given weights and values of items, find the maximum value we can achieve within a weight limit, but unlike 0/1 Knapsack, we can take unlimited instances of an item.
> 🛠️ Approach: Use a 1D DP array where dp[i] represents the maximum value that can be achieved with weight i.
```java
public class UnboundedKnapsack {
    public static int unboundedKnapsack(int W, int[] wt, int[] val) {
        int[] dp = new int[W + 1];

        for (int i = 0; i <= W; i++) {
            for (int j = 0; j < wt.length; j++) {
                if (wt[j] <= i)
                    dp[i] = Math.max(dp[i], dp[i - wt[j]] + val[j]);
            }
        }
        return dp[W];
    }

    public static void main(String[] args) {
        int[] val = {10, 40, 50, 70};
        int[] wt = {1, 3, 4, 5};
        int W = 8;
        System.out.println(unboundedKnapsack(W, wt, val)); // Output: 110
    }
}
```

#### Edit Distance (String Transformation)
> 💡 Problem: Given two strings, find the minimum operations to convert one into the other.

> 🛠️ Approach: Use a DP table to track insertions, deletions, and replacements.
```java
public class EditDistance {
    public static int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0) dp[i][j] = j;
                else if (j == 0) dp[i][j] = i;
                else if (word1.charAt(i - 1) == word2.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1];
                else
                    dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
        return dp[m][n];
    }

    public static void main(String[] args) {
        System.out.println(minDistance("horse", "ros")); // Output: 3
    }
}
```

## Greedy Algorithms
#### Activity Selection Problem
> 💡 Problem: Given n activities with start and end times, select the maximum number of activities that don’t overlap.

> 🛠️ Approach: Sort activities by end time, pick the first one, then keep selecting the next non-overlapping activity.
```java
import java.util.*;

public class ActivitySelection {
    static int maxActivities(int[] start, int[] end) {
        int n = start.length;
        int[][] activities = new int[n][2];
        
        for (int i = 0; i < n; i++) {
            activities[i][0] = start[i];
            activities[i][1] = end[i];
        }
        
        Arrays.sort(activities, Comparator.comparingInt(a -> a[1]));
        
        int count = 1, lastEnd = activities[0][1];
        for (int i = 1; i < n; i++) {
            if (activities[i][0] >= lastEnd) {
                count++;
                lastEnd = activities[i][1];
            }
        }
        return count;
    }

    public static void main(String[] args) {
        int[] start = {1, 3, 2, 5};
        int[] end = {2, 4, 3, 6};
        System.out.println(maxActivities(start, end)); // Output: 2
    }
}
```

#### Huffman Coding (Data Compression)
> 💡 Problem: Given character frequencies, build an optimal prefix-free binary encoding tree.

> 🛠️ Approach: Use a Min-Heap to iteratively merge two lowest-frequency nodes into one.
```java
import java.util.*;

class HuffmanNode {
    char ch;
    int freq;
    HuffmanNode left, right;
    
    HuffmanNode(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
    }
}

class HuffmanComparator implements Comparator<HuffmanNode> {
    public int compare(HuffmanNode a, HuffmanNode b) {
        return a.freq - b.freq;
    }
}

public class HuffmanCoding {
    static void printCodes(HuffmanNode root, String code) {
        if (root == null) return;
        if (root.ch != '-') System.out.println(root.ch + ": " + code);
        printCodes(root.left, code + "0");
        printCodes(root.right, code + "1");
    }

    public static void main(String[] args) {
        char[] chars = {'a', 'b', 'c', 'd'};
        int[] freq = {5, 9, 12, 13};

        PriorityQueue<HuffmanNode> pq = new PriorityQueue<>(new HuffmanComparator());
        for (int i = 0; i < chars.length; i++) {
            pq.add(new HuffmanNode(chars[i], freq[i]));
        }

        while (pq.size() > 1) {
            HuffmanNode left = pq.poll();
            HuffmanNode right = pq.poll();
            HuffmanNode newNode = new HuffmanNode('-', left.freq + right.freq);
            newNode.left = left;
            newNode.right = right;
            pq.add(newNode);
        }
        
        printCodes(pq.poll(), "");
    }
}
```

#### Fractional Knapsack Problem
> 💡 Problem: Given n items with weights and values, maximize profit by selecting fractions of items.

> 🛠️ Approach: Sort items by value/weight, pick as much as possible greedily.
```java
import java.util.*;

class Item {
    int weight, value;
    Item(int v, int w) { value = v; weight = w; }
}

public class FractionalKnapsack {
    static double getMaxValue(Item[] items, int capacity) {
        Arrays.sort(items, (a, b) -> Double.compare((double)b.value/b.weight, (double)a.value/a.weight));
        double totalValue = 0.0;

        for (Item item : items) {
            if (capacity >= item.weight) {
                totalValue += item.value;
                capacity -= item.weight;
            } else {
                totalValue += (double)item.value / item.weight * capacity;
                break;
            }
        }
        return totalValue;
    }

    public static void main(String[] args) {
        Item[] items = {new Item(60, 10), new Item(100, 20), new Item(120, 30)};
        System.out.println(getMaxValue(items, 50)); // Output: 240.0
    }
}
```

#### Job Sequencing Problem
> 💡 Problem: Given jobs with deadlines and profits, schedule them to maximize total profit.

> 🛠️ Approach: Sort jobs by profit, use a greedy strategy to find the best available slot.
```java
import java.util.*;

class Job {
    int id, deadline, profit;
    Job(int i, int d, int p) { id = i; deadline = d; profit = p; }
}

public class JobSequencing {
    static int maxProfit(Job[] jobs) {
        Arrays.sort(jobs, (a, b) -> b.profit - a.profit);
        int n = jobs.length, totalProfit = 0;
        boolean[] slots = new boolean[n];

        for (Job job : jobs) {
            for (int j = Math.min(n, job.deadline) - 1; j >= 0; j--) {
                if (!slots[j]) {
                    slots[j] = true;
                    totalProfit += job.profit;
                    break;
                }
            }
        }
        return totalProfit;
    }

    public static void main(String[] args) {
        Job[] jobs = {new Job(1, 2, 100), new Job(2, 1, 50), new Job(3, 2, 10)};
        System.out.println(maxProfit(jobs)); // Output: 150
    }
}
```

#### Minimum Spanning Tree (Prim’s Algorithm)
> 💡 Problem: Find the minimum spanning tree (MST) for a given weighted graph.

> 🛠️ Approach: Use a Priority Queue (Min-Heap) to greedily select edges with the least weight.
```java
import java.util.*;

class Edge implements Comparable<Edge> {
    int dest, weight;
    Edge(int d, int w) { dest = d; weight = w; }
    public int compareTo(Edge e) { return this.weight - e.weight; }
}

public class PrimsAlgorithm {
    static int primMST(int[][] graph) {
        int V = graph.length, cost = 0;
        boolean[] visited = new boolean[V];
        PriorityQueue<Edge> pq = new PriorityQueue<>();

        pq.add(new Edge(0, 0));
        while (!pq.isEmpty()) {
            Edge edge = pq.poll();
            if (visited[edge.dest]) continue;
            visited[edge.dest] = true;
            cost += edge.weight;

            for (int i = 0; i < V; i++) {
                if (!visited[i] && graph[edge.dest][i] != 0) {
                    pq.add(new Edge(i, graph[edge.dest][i]));
                }
            }
        }
        return cost;
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 2, 0, 6, 0},
            {2, 0, 3, 8, 5},
            {0, 3, 0, 0, 7},
            {6, 8, 0, 0, 9},
            {0, 5, 7, 9, 0}
        };
        System.out.println(primMST(graph)); // Output: Minimum cost of MST
    }
}
```

## Bit Manipulation
- **AND (&):** Sets a bit to 1 if both bits are 1.

- **OR (|):** Sets a bit to 1 if at least one bit is 1.

- **XOR (^):** Sets a bit to 1 if the bits are different.

- **NOT (~):** Flips all bits.

- **Left Shift (<<):** Shifts bits to the left, filling with 0.

- **Right Shift (>>):** Shifts bits to the right, filling with the sign bit.

#### Check if a Number is a Power of Two
```java
public boolean isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

#### Count the Number of Set Bits (Hamming Weight)
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

#### Find the Missing Number
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

#### Single Number
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
