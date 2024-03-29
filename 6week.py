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
# 다른사람 풀이
def solution (number) :
    from itertools import combinations
    cnt = 0
    for i in combinations(number,3) :
        if sum(i) == 0 :
            cnt += 1
    return cnt

# 숫자 문자열과 영단어
# 다음은 숫자의 일부 자릿수를 영단어로 바꾸는 예시입니다.
#   1478 → "one4seveneight"
#   234567 → "23four5six7"
#   10203 → "1zerotwozero3"
# 이렇게 숫자의 일부 자릿수가 영단어로 바뀌어졌거나,
# 혹은 바뀌지 않고 그대로인 문자열 s가 매개변수로 주어집니다.
# s가 의미하는 원래 숫자를 return 하도록 solution 함수를 완성해주세요.
def solution(s):
    answer = s 
    answer = answer.replace('zero', '0')
    answer = answer.replace('one', '1')
    answer = answer.replace('two', '2')
    answer = answer.replace('three', '3')
    answer = answer.replace('four', '4')
    answer = answer.replace('five', '5')
    answer = answer.replace('six', '6')
    answer = answer.replace('seven', '7')
    answer = answer.replace('eight', '8')
    answer = answer.replace('nine', '9')
    return int(answer)
# 다른사람 풀이
num_dic = {"zero":"0", "one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9"}
def solution(s):
    answer = s
    for key, value in num_dic.items():
        answer = answer.replace(key, value)
    return int(answer)

# 크기가 작은 부분문자열
# 숫자로 이루어진 문자열 t와 p가 주어질 때, t에서 p와 길이가 같은 부분문자열 중에서,
# 이 부분문자열이 나타내는 수가 p가 나타내는 수보다 작거나 같은 것이 나오는 횟수를 return하는 함수 solution을 완성하세요.
# 예를 들어, t="3141592"이고 p="271" 인 경우,
# t의 길이가 3인 부분 문자열은 314, 141, 415, 159, 592입니다.
# 이 문자열이 나타내는 수 중 271보다 작거나 같은 수는 141, 159 2개 입니다.
def solution(t, p):
  answer = 0
  p_len = len(p)
  t_len = len(t)
  p = int(p)
  for i in range(0,t_len+1-p_len):
      if int(t[i:i+p_len]) <= p:
          answer+=1
  return answer
# 다른사람 풀이
def solution(t, p):
    answer = 0
    for i in range(len(t) - len(p) + 1):
        if int(p) >= int(t[i:i+len(p)]):
            answer += 1
    return answer

# k번째수
# 배열 array의 i번째 숫자부터 j번째 숫자까지 자르고 정렬했을 때, k번째에 있는 수를 구하려 합니다.
# 예를 들어 array가 [1, 5, 2, 6, 3, 7, 4], i = 2, j = 5, k = 3이라면
# array의 2번째부터 5번째까지 자르면 [5, 2, 6, 3]입니다.
# 1에서 나온 배열을 정렬하면 [2, 3, 5, 6]입니다.
# 2에서 나온 배열의 3번째 숫자는 5입니다.
# 배열 array, [i, j, k]를 원소로 가진 2차원 배열 commands가 매개변수로 주어질 때,
# commands의 모든 원소에 대해 앞서 설명한 연산을 적용했을 때 나온 결과를 배열에 담아 return 하도록 solution 함수를 작성해주세요.
def solution(array, commands):
    answer = []    
    for i in commands:
        ary = array[i[0]-1: i[1]]    # 문제에서 주어진 크기만큼 자르기
        ary.sort()    # sort 함수로 정렬
        answer.append(ary[i[2]-1])    # k 번째 값 집어넣기
    return answer
# 다른사람 풀이
def solution(array, commands):
    return list(map(lambda x:sorted(array[x[0]-1:x[1]])[x[2]-1], commands))