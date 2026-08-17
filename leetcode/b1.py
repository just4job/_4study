class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ds = [(nums[i],i) for i in range(len(nums))]
        ds.sort(key = lambda x : x[0])
        i = 0
        j = len(ds) -1
        while i <  j:
            if ds[i][0] + ds[j][0] == target:
                return [ds[i][1], ds[j][1]]
            elif ds[i][0] + ds[j][0] > target:
                j -=1  
            else: i +=1
        