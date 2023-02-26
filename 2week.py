# 특정 문자 제거하기
# 문자열 my_string과 문자 letter이 매개변수로 주어집니다.
# my_string에서 letter를 제거한 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(my_string, letter):
    answer = ''
    answer = my_string.replace(letter,'') # .replace로 제거할 수 있다.
    return answer

# 피자 나눠 먹기 (1)
# 머쓱이네 피자가게는 피자를 일곱 조각으로 잘라 줍니다.
# 피자를 나눠먹을 사람의 수 n이 주어질 때,
# 모든 사람이 피자를 한 조각 이상 먹기 위해 필요한 피자의 수를 return 하는 solution 함수를 완성해보세요.
def solution(n):
    answer = 0
    if n%7 == 0:
        answer = n//7
    else:
        answer = (n//7)+1 
    return answer
# 다른사람 풀이
def solution(n):
    return (n - 1) // 7 + 1

# 배열 자르기
# 정수 배열 numbers와 정수 num1, num2가 매개변수로 주어질 때,
# numbers의 num1번 째 인덱스부터 num2번째 인덱스까지 자른 정수 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(numbers, num1, num2):
    answer = []
    for i in range(num1, num2+1):
        answer.append(numbers[i])
    return answer
# 다른사람 풀이
def solution(numbers, num1, num2):
    return numbers[num1:num2+1]

# 옷가게 할인 받기
# 머쓱이네 옷가게는 10만 원 이상 사면 5%, 30만 원 이상 사면 10%,
# 50만 원 이상 사면 20%를 할인해줍니다.
# 구매한 옷의 가격 price가 주어질 때,
# 지불해야 할 금액을 return 하도록 solution 함수를 완성해보세요.
def solution(price):
    answer = 0
    if price >= 100000 and price < 300000:
        answer = price * 0.95
    elif price >= 300000 and price < 500000:
        answer = price * 0.9
    elif price >= 500000:
        answer = price * 0.8
    else:
        answer = price
    return int(answer)
# 다른사람 풀이
def solution(price):
    discount_rates = {500000: 0.8, 300000: 0.9, 100000: 0.95, 0: 1}
    for discount_price, discount_rate in discount_rates.items():
        if price >= discount_price:
            return int(price * discount_rate)
        
# 피자 나눠 먹기 (3)
# 머쓱이네 피자가게는 피자를 두 조각에서 열 조각까지 원하는 조각 수로 잘라줍니다.
# 피자 조각 수 slice와 피자를 먹는 사람의 수 n이 매개변수로 주어질 때,
# n명의 사람이 최소 한 조각 이상 피자를 먹으려면
# 최소 몇 판의 피자를 시켜야 하는지를 return 하도록 solution 함수를 완성해보세요.
def solution(slice, n):
    return (n-1) // slice +1

# 삼각형의 완성조건 (1)
# 선분 세 개로 삼각형을 만들기 위해서는 다음과 같은 조건을 만족해야 합니다.
#  가장 긴 변의 길이는 다른 두 변의 길이의 합보다 작아야 합니다.
# 삼각형의 세 변의 길이가 담긴 배열 sides이 매개변수로 주어집니다.
# 세 변으로 삼각형을 만들 수 있다면 1, 만들 수 없다면 2를 return하도록 solution 함수를 완성해주세요.
def solution(sides):
    answer = 1
    a = 0
    b = []
    if sides[0] > sides[1]:
        a = sides[0]
        b.append(sides[1])
    else:
        a = sides[1]
        b.append(sides[0])
    if a > sides[2]:
        b.append(sides[2])
    else:
        b.append(a)
        a = sides[2]
    if a < b[0]+b[1]:
        return answer
    else:
        answer = 2
    return answer
# 다른사람 풀이
def solution(sides):
    return 1 if max(sides) < (sum(sides) - max(sides)) else 2

# 최댓값 만들기 (1)
# 정수 배열 numbers가 매개변수로 주어집니다.
# numbers의 원소 중 두 개를 곱해 만들 수 있는 최댓값을 return하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = max(numbers)
    numbers.remove(answer)
    answer *= max(numbers)
    return answer
# 다른사람 풀이
def solution(numbers):
    numbers.sort()
    return numbers[-2] * numbers[-1]

# 아이스 아메리카노
# 머쓱이는 추운 날에도 아이스 아메리카노만 마십니다.
# 아이스 아메리카노는 한잔에 5,500원입니다.
# 머쓱이가 가지고 있는 돈 money가 매개변수로 주어질 때,
# 머쓱이가 최대로 마실 수 있는 아메리카노의 잔 수와 남는 돈을 순서대로 담은 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(money):
    answer = []
    answer.extend([money//5500,money%5500])
    return answer
def solution(money):
    return [money//5500,money%5500]

# 중복된 숫자 개수
# 정수가 담긴 배열 array와 정수 n이 매개변수로 주어질 때,
# array에 n이 몇 개 있는 지를 return 하도록 solution 함수를 완성해보세요.
def solution(array, n):
    return array.count(n)

# 피자 나눠 먹기 (2)
# 머쓱이네 피자가게는 피자를 여섯 조각으로 잘라 줍니다.
# 피자를 나눠먹을 사람의 수 n이 매개변수로 주어질 때,
# n명이 주문한 피자를 남기지 않고 모두 같은 수의 피자 조각을 먹어야 한다면
# 최소 몇 판을 시켜야 하는지를 return 하도록 solution 함수를 완성해보세요.
def solution(n):
    answer = 0
    for i in range (0,(6*n)+1):
        if i%n == 0 and i%6 == 0:  # 최소공배수를 구하였다.
            answer = i//6
            if answer != 0: # 최소공배수를 구하기 위해 값이 담겨지면 종료
                break
    return answer

# 순서쌍의 개수
# 순서쌍이란 두 개의 숫자를 순서를 정하여 짝지어 나타낸 쌍으로 (a, b)로 표기합니다.
# 자연수 n이 매개변수로 주어질 때 두 숫자의 곱이 n인 자연수 순서쌍의 개수를 return하도록 solution 함수를 완성해주세요.
def solution(n):
    answer = []
    for i in range(1, n+1):
        if n%i == 0:
            answer.extend([(i,n//i)])
    return len(answer)

# 편지
# 머쓱이는 할머니께 생신 축하 편지를 쓰려고 합니다.
# 할머니가 보시기 편하도록 글자 한 자 한 자를 가로 2cm 크기로 적으려고 하며,
# 편지를 가로로만 적을 때, 축하 문구 message를 적기 위해
# 필요한 편지지의 최소 가로길이를 return 하도록 solution 함수를 완성해주세요.
# 제한사항
#  공백도 하나의 문자로 취급합니다.
#  1 ≤ message의 길이 ≤ 50
#  편지지의 여백은 생각하지 않습니다.
#  message는 영문 알파벳 대소문자, ‘!’, ‘~’ 또는 공백으로만 이루어져 있습니다.
def solution(message):
    return 2*len(message)

# 배열의 유사도
# 두 배열이 얼마나 유사한지 확인해보려고 합니다.
# 문자열 배열 s1과 s2가 주어질 때 같은 원소의 개수를 return하도록 solution 함수를 완성해주세요.
def solution(s1, s2):  
    answer = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                answer +=1
    return answer
# 다른사람 풀이
def solution(s1, s2):
    return len(set(s1).intersection(s2))

# 자릿수 더하기
# 정수 n이 매개변수로 주어질 때 n의 각 자리 숫자의 합을 return하도록 solution 함수를 완성해주세요
def solution(n):
    answer=0
    while n>0 :
        answer+=n%10
        n//=10
    return answer
# 다른사람 풀이
def solution(n):
    N=[int(i) for i in str(n)]
    return sum(N)

# 숨어있는 숫자의 덧셈 (1)
# 문자열 my_string이 매개변수로 주어집니다.
# my_string안의 모든 자연수들의 합을 return하도록 solution 함수를 완성해주세요.
import re
def solution(my_string):
    answer = re.findall(r'\d',my_string)
    answer = [int(i) for i in answer]
    return sum(answer)
# 다른사람 풀이
def solution(my_string):
    return sum(int(i) for i in my_string if i.isdigit())
# 2
def solution(my_string):
    answer = 0
    for i in my_string:
        try:
            answer = answer + int(i)
        except:
            pass
    return answer

# 문자열안에 문자열
# 문자열 str1, str2가 매개변수로 주어집니다.
# str1 안에 str2가 있다면 1을 없다면 2를 return하도록 solution 함수를 완성해주세요.
def solution(str1, str2):   
    if str2 in str1:   # in 연산자 이용
        answer = 1
    else:
        answer = 2
    return answer   

# 모음 제거
# 영어에선 a, e, i, o, u 다섯 가지 알파벳을 모음으로 분류합니다.
# 문자열 my_string이 매개변수로 주어질 때 모음을 제거한 문자열을 return하도록 solution 함수를 완성해주세요.
import re
def solution(my_string):
    answer = re.sub("a|e|i|u|o",'',my_string)
    return answer   # sub(regex, replacement, str) 문자열 str에서 regex 패턴을 찾고 해당하는 부분을 replacement로 변경합니다.
                    # 이것을 이용하여 특정 문자를 삭제할 수 있습니다.
