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

### General 2-pointer pseudo code 
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

### Two Sum + Sorted
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


### Linked List Cycle
//Using Set
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

//Using slow and fast pointers
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


### Merge 2 Sorted Lists
   if(l1.val < l2.val){
     cur.next = l1;
     l1 = l1.next;
   }
   else{
     cur.next = l2;
     l2 = l2.next;
   }

### Merge K Sorted Lists
- compare every 2 lists, call 'merge2sortedLists' function, and keep doing until we have a bigger list.
- Add all the lists into one big array or list -> sort the array -> then put the elements back into a new LL
- Priority Queue is another option


### Slidind Window General Pseudo Code
int max_sum = 0, window_sum = 0; 
/* calculate sum of 1st window */
for (int i = 0; i < k; i++)  window_sum += arr[i]; 

/* slide window from start to end in array. */
for (int i = k; i < n; i++){ 
    window_sum += arr[i] - arr[i-k];    // saving re-computation
    max_sum = max(max_sum, window_sum);
}

:+1:  ↖️↗️
      ↙️↘️
