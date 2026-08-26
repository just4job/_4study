class Solution:
    def longestPalindrome(self, s: str) -> str:
        ke =""
        def expan(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1:r]
        for i in range(len(s)):
            le = expan(i,i)
            chan = expan(i,i+1)
            if len(le) > len(ke):
                ke = le
            if len(chan) > len(ke):
                ke = chan
        return ke