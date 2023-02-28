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
# 가위는 2 바위는 0 보는 5로 표현합니다.
# 가위 바위 보를 내는 순서대로 나타낸 문자열 rsp가 매개변수로 주어질 때,
# rsp에 저장된 가위 바위 보를 모두 이기는 경우를 순서대로 나타낸 문자열을 return하도록 solution 함수를 완성해보세요.
def solution(rsp):
    result = {'2':'0','0':'5','5':'2'}
    answer = ''
    for i in rsp:  # 딕셔너리를 만들어서 get()을 사용하여 value 값을 찾아낸다.
        answer += result.get(i)
    return answer
# 다른사람 풀이
def solution(rsp):
    d = {'0':'5','2':'0','5':'2'}
    return ''.join(d[i] for i in rsp) # '',join을 이용하여 하나로 합쳐, dict[key]를 사용하여 value 값을 가져온다.