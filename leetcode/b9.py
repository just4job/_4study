class Solution:
    def isPalindrome(self, x: int) -> bool:
        test = str(x)
        if test[0] in ['+','-']: 
            return False
        else: 
            i = 0
            j = len(test) - 1
            while i<j:
                if test[i] != test[j]:
                    return False
                i +=1
                j -=1
        return True        