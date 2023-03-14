# A로 B 만들기
# 문자열 before와 after가 매개변수로 주어질 때,
# before의 순서를 바꾸어 after를 만들 수 있으면 1을,
# 만들 수 없으면 0을 return 하도록 solution 함수를 완성해보세요.
def solution(before, after):
    before = sorted(before)
    after = sorted(after)
    if before == after:
        return 1
    else:
        return 0

# 팩토리얼
# i팩토리얼 (i!)은 1부터 i까지 정수의 곱을 의미합니다.
# 예를들어 5! = 5 * 4 * 3 * 2 * 1 = 120 입니다.
# 정수 n이 주어질 때 다음 조건을 만족하는 가장 큰 정수 i를 return 하도록 solution 함수를 완성해주세요.
from math import factorial
def solution(n):
    answer = 10
    while n < factorial(answer):
        answer -= 1
    return answer

# k의 개수
# 1부터 13까지의 수에서, 1은 1, 10, 11, 12, 13 이렇게 총 6번 등장합니다.
# 정수 i, j, k가 매개변수로 주어질 때, i부터 j까지 k가 몇 번 등장하는지 return 하도록 solution 함수를 완성해주세요.
from collections import Counter 
def solution(i, j, k):
    answer = 0
    for a in range(i,j+1):
        answer += Counter(str(a))[str(k)]
    return answer

# 7의 개수
# 머쓱이는 행운의 숫자 7을 가장 좋아합니다.
# 정수 배열 array가 매개변수로 주어질 때, 7이 총 몇 개 있는지 return 하도록 solution 함수를 완성해보세요.
def solution(array):
    answer = ""
    for i in array:
        answer+=str(i)
    return answer.count("7")
# 다른사람 풀이
def solution(array):
    return str(array).count('7')

# 공 던지기
# 머쓱이는 친구들과 동그랗게 서서 공 던지기 게임을 하고 있습니다.
# 공은 1번부터 던지며 오른쪽으로 한 명을 건너뛰고 그다음 사람에게만 던질 수 있습니다.
# 친구들의 번호가 들어있는 정수 배열 numbers와 정수 K가 주어질 때,
# k번째로 공을 던지는 사람의 번호는 무엇인지 return 하도록 solution 함수를 완성해보세요.
def solution(numbers, k):
    answer = 0
    while k > 1:
        answer += 2
        answer %= len(numbers)
        k -= 1
    return numbers[answer]
# 다른사람 풀이
def solution(numbers, k):
    return numbers[2 * (k - 1) % len(numbers)]

# 소인수분해
# 소인수분해란 어떤 수를 소수들의 곱으로 표현하는 것입니다.
# 예를 들어 12를 소인수 분해하면 2 * 2 * 3 으로 나타낼 수 있습니다.
# 따라서 12의 소인수는 2와 3입니다. 자연수 n이 매개변수로 주어질 때
# n의 소인수를 오름차순으로 담은 배열을 return하도록 solution 함수를 완성해주세요.
def solution(n):
    answer = []
    a = 2
    while a <= n:
        if n%a==0:
            n=n/a
            answer.append(a)
        else:
            a += 1
    answer = list(set(answer)) # set()으로 중복을 제거해주고
    answer.sort()  # 오름차순으로 정렬해준다.
    return answer

# 컨트롤 제트
# 숫자와 "Z"가 공백으로 구분되어 담긴 문자열이 주어집니다.
# 문자열에 있는 숫자를 차례대로 더하려고 합니다.
# 이 때 "Z"가 나오면 바로 전에 더했던 숫자를 뺀다는 뜻입니다.
# 숫자와 "Z"로 이루어진 문자열 s가 주어질 때, 머쓱이가 구한 값을 return 하도록 solution 함수를 완성해보세요.
def solution(s):
    answer = []
    for i in s.split(' '):
        try:
            answer.append(int(i))
        except:
            answer.pop()      
    return sum(answer)
