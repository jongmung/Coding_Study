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

# 부족한 금액 계산하기
# 이 놀이기구의 원래 이용료는 price원 인데,
# 놀이기구를 N 번 째 이용한다면 원래 이용료의 N배를 받기로 하였습니다.
# 즉, 처음 이용료가 100이었다면 2번째에는 200, 3번째에는 300으로 요금이 인상됩니다.
# 놀이기구를 count번 타게 되면 현재 자신이 가지고 있는 금액에서 얼마가 모자라는지를 return 하도록 solution 함수를 완성하세요.
# 단, 금액이 부족하지 않으면 0을 return 하세요.
def solution(price, money, count):
    for i in range(1,count+1):
        money -= price*i
    if money < 0:
        money = abs(money)
    else:
        money = 0 
    return money
# 다른사람 풀이
def solution(price, money, count):
    return max(0,price*(count+1)*count//2-money)

# 3진법 뒤집기
# 자연수 n이 매개변수로 주어집니다.
# n을 3진법 상에서 앞뒤로 뒤집은 후,
# 이를 다시 10진법으로 표현한 수를 return 하도록 solution 함수를 완성해주세요.
def solution(n):
    answer = ''
    while n > 0:			
        n, re = divmod(n,3)	# n을 3으로 나눈 몫과 나머지
        answer += str(re)
    return int(answer, 3)
# divmod() : 몫과 나머지를 리턴합니다. 리턴 값이 2개이므로 튜플을 사용합니다.
# int(x, base) : base 진법으로 구성된 str 형식의 수를 10진법으로 변환해 줌
# 다른사람 풀이
def solution(n):
    tmp = ''
    while n:
        tmp += str(n % 3)
        n = n // 3
    answer = int(tmp, 3)
    return answer

# 문자열 다루기 기본
# 문자열 s의 길이가 4 혹은 6이고, 숫자로만 구성돼있는지 확인해주는 함수,
# solution을 완성하세요. 예를 들어 s가 "a234"이면 False를 리턴하고 "1234"라면 True를 리턴하면 됩니다.
def solution(s):
    if len(s) == 4 and s.isdigit() == 1:
        return True
    elif len(s) == 6 and s.isdigit() == 1:
        return True
    else:
        return False
    
# 삼총사
# 학교 학생 3명의 정수 번호를 더했을 때 0이 되면 3명의 학생은 삼총사라고 합니다.
# 예를 들어, 5명의 학생이 있고, 각각의 정수 번호가 순서대로 -2, 3, 0, 2, -5일 때,
# 첫 번째, 세 번째, 네 번째 학생의 정수 번호를 더하면 0이므로 세 학생은 삼총사입니다.
# 또한, 두 번째, 네 번째, 다섯 번째 학생의 정수 번호를 더해도 0이므로 세 학생도 삼총사입니다.
# 따라서 이 경우 한국중학교에서는 두 가지 방법으로 삼총사를 만들 수 있습니다.
# 한국중학교 학생들의 번호를 나타내는 정수 배열 number가 매개변수로 주어질 때,
# 학생들 중 삼총사를 만들 수 있는 방법의 수를 return 하도록 solution 함수를 완성하세요.
def solution(number):
    answer = 0
    for i in range(len(number)-2):
        for j in range(i + 1, len(number)-1):
            for k in range(j + 1, len(number)):
                if number[i] + number[j] + number[k] == 0:
                    answer += 1
    return answer
