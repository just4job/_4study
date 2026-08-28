class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi',
            '5': 'jkl', '6': 'mno', '7': 'pqrs',
            '8': 'tuv', '9': 'wxyz'
        }
        kq = [""]
        for digit in digits:
            tmp=[]
            for so in phone_map[digit]:
                for pre in kq:
                    tmp.append(pre+so)
            kq= tmp
        return kq