class Solution:
    def convert(self, s: str, numRows: int) -> str:
        mt = [[''] * len(s) for _ in range(numRows)]
        if numRows == 1 or numRows >= len(s):
            return s
        i,j, idx =0,0,0
        while (idx <len(s)):
            while i< numRows and idx< len(s):
                mt[i][j]= s[idx]
                i+=1
                idx+=1
            i-=2
            j+=1
            while i>0 and idx < len(s):
                mt[i][j]= s[idx]
                i-=1
                j+=1
                idx +=1
        res = []
        for r in range(numRows):
            for c in range(len(s)):
                if mt[r][c] != '':
                    res.append(mt[r][c])

        return "".join(res)

