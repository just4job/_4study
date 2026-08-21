class Solution:
    def convert(self, s: str, numRows: int) -> str:
        mt = ['' * len(s) for _ in range(numRows)]
        if numRows == 1 or numRows >= len(s):
            return s
        i,j, idx =0
        while (idx <len(s)):
            while i< numRows:
                mt[i][j]= s[idx]
                i+=1
                idx+=1
            i-=2
            j+=1
            while i>0:
                mt[i][j]= s[idx]
                
