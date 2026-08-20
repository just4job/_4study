class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        kq= ''
        max_len = 0
        for i in s:
            if i in kq:
                kq=kq[kq.index(i)+1:]
            kq += i
            max_len = max(max_len, len(kq))
        return max_len