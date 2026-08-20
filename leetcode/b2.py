# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        du = 0
        kq = ListNode()
        cur = kq
        while l1 or l2 or du:
            if l1: 
                du += l1.val
                l1= l1.next
            if l2:
                du += l2.val
                l2 = l2.next
            cur.next = ListNode(du%10)
            cur = cur.next
            du //= 10
        return kq.next