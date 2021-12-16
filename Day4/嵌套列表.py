# author:TYUT-Lmy
# date:2021/12/16
# description:
from random import uniform

# 录入五个学生的三门成绩
names = ['关羽', '张飞', '赵云', '马超', '黄忠']
courses = ['语文', '数学', '英语']
# scores1 = [[None] * len(courses)] * len(names)
# 不能使用上述的方法,否则对每一行的操作会影响到其他行的操作,因为 每一行的内存地址都相等,需要通过for循环依次创建不同的内存地址
scores = [[None] * len(courses) for _ in range(len(names))]

# 两个for循环写入成绩
for row, name in enumerate(names):
    for col, course in enumerate(courses):
        scores[row][col] = round(uniform(1.0, 100.0), 2)

# 读取成绩
name = "张飞"
course = "英语"
name_index = names.index(name)
course_index = courses.index(course)
print("     语文    数学   英语")

for i, score in enumerate(scores):
    print(names[i], score)

print(f"{name}的{course}成绩:{scores[name_index][course_index]}")
