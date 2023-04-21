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
# 0부터 9까지의 숫자 중 일부가 들어있는 정수 배열 numbers가 매개변수로 주어집니다.
# numbers에서 찾을 수 없는 0부터 9까지의 숫자를 모두 찾아 더한 수를 return 하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = 0
    for i in range(10):
        if i not in numbers:
            answer += i
    return answer
# 다른사람 풀이
def solution(numbers):
    return 45 - sum(numbers)

# 가운데 글자 가져오기
# 단어 s의 가운데 글자를 반환하는 함수, solution을 만들어 보세요.
# 단어의 길이가 짝수라면 가운데 두글자를 반환하면 됩니다.
def solution(s):
    center = int(len(s)/2)
    if len(s) % 2 != 0:
        return s[center]
    else:
        return s[center-1:center+1]
# 다른사람 풀이
def string_middle(str):
    return str[(len(str)-1)//2 : len(str)//2 + 1]

# 내적
# 길이가 같은 두 1차원 정수 배열 a, b가 매개변수로 주어집니다.
# a와 b의 내적을 return 하도록 solution 함수를 완성해주세요.
# 이때, a와 b의 내적은 a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1] 입니다. (n은 a, b의 길이)
def solution(a, b):
    answer = 0
    for i in range(len(a)):
        answer += a[i]*b[i]
    return answer
# 다른사람 풀이
def solution(a, b):
    return sum([x*y for x, y in zip(a,b)])

# 약수의 개수와 덧셈
# 두 정수 left와 right가 매개변수로 주어집니다.
# left부터 right까지의 모든 수들 중에서, 약수의 개수가 짝수인 수는 더하고,
# 약수의 개수가 홀수인 수는 뺀 수를 return 하도록 solution 함수를 완성해주세요.
def solution(left, right):
    answer = 0
    for i in range(left, right + 1):
        cnt = 0
        for j in range(1, i + 1):
            if i % j == 0:
                cnt += 1
        if cnt % 2 == 0:
            answer += i
        else:
            answer -= i
    return answer
# 다른사람 풀이
def solution(left, right):
    answer = 0
    for i in range(left,right+1):
        if int(i**0.5)==i**0.5:
            answer -= i
        else:
            answer += i
    return answer