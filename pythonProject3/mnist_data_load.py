
def jump(nums):
    i = 0
    minstep=9999
    def soor(i,minstep,iter):
        j=i
        step=i
        while j<len(nums):
            j+=nums[j]
            step+=1
        i+=1
        if step<minstep:
            minstep=step
        if i<len(nums)-1:
            soor(i,minstep,iter)
        else:
            return minstep
    a=soor(i,minstep,iter)
    return a

print(jump([2,1,4,1,3]))