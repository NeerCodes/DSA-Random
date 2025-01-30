# DSA-Random
Random DSA Patterns for revision.

## ARRAYS
#### Kadane's Algorithm
- Used to find the maximum subarray sum.

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
- Used for problems involving arrays or strings, such as finding pairs or reversing.
- Example: Two Sum Problem.
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


### Slidind Window General Pseudo Code
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

:+1:  ↖️↗️
      ↙️↘️
