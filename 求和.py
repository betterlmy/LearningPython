def get_index(nums, target):
    while True:
        if isinstance(nums, list):
            if isinstance(target, int):
                for i in range(len(nums) - 1):
                    a = nums[i]
                    for j in range(i + 1, len(nums)):
                        if a + nums[j] == target:
                            return [i, j]
                return None
            else:
                target = int(input('target 输入错误,请重新输入:'))
        else:
            nums = list(input('nums 输入错误,请重新输入:'))


print(get_index([2, 7, 11, 15], 9))
