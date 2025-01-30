# DSA-Random
Random DSA Patterns for revision.

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
   
