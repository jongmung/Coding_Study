# 나머지가 1이 되는 수 찾기
# 자연수 n이 매개변수로 주어집니다.
# n을 x로 나눈 나머지가 1이 되도록 하는 가장 작은 자연수 x를 return 하도록 solution 함수를 완성해주세요. 답이 항상 존재함은 증명될 수 있습니다.
def solution(n):
    for i in range(1,n):
        if n%i == 1:
            answer =i
            break
    return answer
# 다른사람 풀이
def solution(n):
    return [x for x in range(1,n+1) if n%x==1][0]

# 음양 더하기
# 어떤 정수들이 있습니다.
# 이 정수들의 절댓값을 차례대로 담은 정수 배열 absolutes와
# 이 정수들의 부호를 차례대로 담은 불리언 배열 signs가 매개변수로 주어집니다.
# 실제 정수들의 합을 구하여 return 하도록 solution 함수를 완성해주세요.
def solution(absolutes, signs):
    answer = 0
    for i in range(len(absolutes)):
        if signs[i]:
            answer += absolutes[i]
        else:
            answer -= absolutes[i]
    return answer
# 다른사람 풀이
def solution(absolutes, signs):
    return sum(absolutes if sign else -absolutes for absolutes, sign in zip(absolutes, signs))

# 없는 숫자 더하기
def solution(numbers):
    answer = 0
    for i in range(10):
        if i not in numbers:
            answer += i
    return answer