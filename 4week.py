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

# 이진수 더하기
# 이진수를 의미하는 두 개의 문자열 bin1과 bin2가 매개변수로 주어질 때,
# 두 이진수의 합을 return하도록 solution 함수를 완성해주세요.
def solution(bin1, bin2):
    answer = bin(int(bin1,2) + int(bin2,2))
    return answer[2:]

# 문자열 계산하기
# my_string은 "3 + 5"처럼 문자열로 된 수식입니다.
# 문자열 my_string이 매개변수로 주어질 때, 수식을 계산한 값을 return 하는 solution 함수를 완성해주세요.
def solution(my_string):
    return eval(my_string)
# 다른사람 풀이
def solution(my_string):
    return sum(int(i) for i in my_string.replace(' - ', ' + -').split(' + '))

# 잘라서 배열로 저장하기
# 문자열 my_str과 n이 매개변수로 주어질 때,
# my_str을 길이 n씩 잘라서 저장한 배열을 return하도록 solution 함수를 완성해주세요.
def solution(my_str, n):
    return [my_str[i:i+n] for i in range(0, len(my_str),n)]

# 치킨쿠폰
# 프로그래머스 치킨은 치킨을 시켜먹으면 한 마리당 쿠폰을 한 장 발급합니다.
# 쿠폰을 열 장 모으면 치킨을 한 마리 서비스로 받을 수 있고,
# 서비스 치킨에도 쿠폰이 발급됩니다. 시켜먹은 치킨의 수 chicken이 매개변수로 주어질 때 
# 받을 수 있는 최대 서비스 치킨의 수를 return하도록 solution 함수를 완성해주세요.
def solution(chicken):
    answer = 0
    while chicken >= 10:
        div = chicken // 10
        mod = chicken % 10
        answer += div
        chicken = div+mod
    return answer
# 다른사람 풀이
def solution(chicken):
    return int(chicken*0.11111111111) 

def solution(chicken):
    answer = (max(chicken,1)-1)//9
    return answer

# 등수 매기기
# 영어 점수와 수학 점수의 평균 점수를 기준으로 학생들의 등수를 매기려고 합니다.
# 영어 점수와 수학 점수를 담은 2차원 정수 배열 score가 주어질 때,
# 영어 점수와 수학 점수의 평균을 기준으로 매긴 등수를 담은 배열을 return하도록 solution 함수를 완성해주세요.
def solution(score):
    answer = []
    li = []
    for i in score:
        li.append(sum(i)/len(i))
    sort_arr = sorted(li,reverse = True)
    for i in li:
        answer.append(sort_arr.index(i)+1)
    return answer

# 종이 자르기
# 정수 M, N이 매개변수로 주어질 때,
# M x N 크기의 종이를 최소로 가위질 해야하는 횟수를 return 하도록 solution 함수를 완성해보세요.
def solution(M, N):
    answer = (M*N)-1
    return answer

# 직사각형 넓이 구하기
# 2차원 좌표 평면에 변이 축과 평행한 직사각형이 있습니다.
# 직사각형 네 꼭짓점의 좌표
# [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]가
# 담겨있는 배열 dots가 매개변수로 주어질 때, 직사각형의 넓이를 return 하도록 solution 함수를 완성해보세요.
def solution(dots):
    w = max(dots)[0] - min(dots)[0]
    h = max(dots)[1] - min(dots)[1]
    area = w*h
    return area

