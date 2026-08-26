class Solution:
    def reverse(self, x: int) -> int:
        sign = 1
        if x < 0: 
            sign = -1
        x = abs(x)
        kq = 0
        while x > 0:
            du = x % 10
            x = x // 10
            kq = kq*10 + du
        kq *=sign
        if kq < -2**31 or kq > 2**31 - 1:
            return 0
        return kq