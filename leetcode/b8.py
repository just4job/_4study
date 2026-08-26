class Solution:
    def myAtoi(self, s: str) -> int:
        i=0
        kq = 0
        
        while i < len(s) and s[i] == ' ':
            i += 1
        if i == len(s):
            return 0
        sign = 1
        if s[i] == '-':
            sign = -1
            i += 1
        elif s[i] == '+':
            i += 1
        while i<len(s) and s[i].isdigit():
            du = int(s[i])
            kq = kq*10 +du
            i+=1
        kq *= sign
        INT_MIN, INT_MAX = -2**31, 2**31 - 1
        if kq < INT_MIN:
            return INT_MIN
        if kq > INT_MAX:
            return INT_MAX
        return kq
            
        