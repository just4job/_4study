class Solution:
    def maxArea(self, height: List[int]) -> int:
        l= 0
        r= len(height)-1
        kq =0
        while l<r:
            s= (r-l)*min(height[l],height[r])
            if s> kq:
                kq = s
            if height[l]< height[r]:
                l+=1
            else:
                r-=1
        return kq