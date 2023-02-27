# 제곱수 판별하기
# 어떤 자연수를 제곱했을 때 나오는 정수를 제곱수라고 합니다.
# 정수 n이 매개변수로 주어질 때, n이 제곱수라면 1을 아니라면 2를 return하도록 solution 함수를 완성해주세요.
import math
def solution(n):
    x = math.sqrt(n)
    if x % 1 == 0:
        answer = 1
    else:
        answer = 2
    return answer
# 다른사람 풀이
def solution(n):
    return 1 if (n ** 0.5).is_integer() else 2

# 가위바위보
def solution(rsp):
    result = {'2':'0','0':'5','5':'2'}
    answer = ''
    for i in rsp:
        answer += result.get(i)
    return answer