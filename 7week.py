# 두 개 뽑아서 더하기
# 정수 배열 numbers가 주어집니다.
# numbers에서 서로 다른 인덱스에 있는 두 개의 수를 뽑아 더해서
# 만들 수 있는 모든 수를 배열에 오름차순으로 담아 return 하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = set()
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            answer.add(numbers[i] + numbers[j])
    answer = list(answer)
    answer.sort()
    return answer
# 다른사람 풀이
def solution(numbers):
    answer = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            answer.append(numbers[i] + numbers[j])
    return sorted(list(set(answer)))

from itertools import combinations
def solution(numbers):
    return sorted(set(sum(i) for i in list(combinations(numbers, 2))))

# 2016년
# 2016년 1월 1일은 금요일입니다. 2016년 a월 b일은 무슨 요일일까요?
# 두 수 a ,b를 입력받아 2016년 a월 b일이 무슨 요일인지 리턴하는 함수,
# solution을 완성하세요.
# 요일의 이름은 일요일부터 토요일까지 각각 SUN,MON,TUE,WED,THU,FRI,SAT
def solution(a, b):
    answer = 0
    days = ['FRI','SAT','SUN','MON','TUE','WED','THU']
    months = [31, 29, 31, 30, 31, 30,31, 31, 30, 31, 30, 31]    
    for i in range(a-1):
        answer += months[i]
    answer += b-1
    answer = answer%7    
    return days[answer]
# 다른사람 풀이
import datetime
def getDayName(a,b):
    t = 'MON TUE WED THU FRI SAT SUN'.split()
    return t[datetime.datetime(2016, a, b).weekday()]

# 모의고사
# 수포자는 수학을 포기한 사람의 준말입니다.
# 수포자 삼인방은 모의고사에 수학 문제를 전부 찍으려 합니다.
# 수포자는 1번 문제부터 마지막 문제까지 다음과 같이 찍습니다.
#  1번 수포자가 찍는 방식: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
#  2번 수포자가 찍는 방식: 2, 1, 2, 3, 2, 4, 2, 5, 2, 1, 2, 3, 2, 4, 2, 5, ...
#  3번 수포자가 찍는 방식: 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, ...
# 1번 문제부터 마지막 문제까지의 정답이 순서대로 들은 배열 answers가 주어졌을 때,
# 가장 많은 문제를 맞힌 사람이 누구인지 배열에 담아 return 하도록 solution 함수를 작성해주세요.
def solution(answers):   
    answer = []
    score = [0,0,0]   
    student1 = [1,2,3,4,5]
    student2 = [2,1,2,3,2,4,2,5]
    student3 = [3,3,1,1,2,2,4,4,5,5]
    for i in range(len(answers)) :
        if answers[i] == student1[i%5] :
            score[0] += 1
        if answers[i] == student2[i%8] :
            score[1] += 1
        if answers[i] == student3[i%10] :
            score[2] += 1       
    for idx, num in enumerate(score) :
        if num == max(score) :
            answer.append(idx +1)
    return answer
# 다른사람 풀이
def solution(answers):
    pattern1 = [1,2,3,4,5]
    pattern2 = [2,1,2,3,2,4,2,5]
    pattern3 = [3,3,1,1,2,2,4,4,5,5]
    score = [0, 0, 0]
    result = []
    for idx, answer in enumerate(answers):
        if answer == pattern1[idx%len(pattern1)]:
            score[0] += 1
        if answer == pattern2[idx%len(pattern2)]:
            score[1] += 1
        if answer == pattern3[idx%len(pattern3)]:
            score[2] += 1
    for idx, s in enumerate(score):
        if s == max(score):
            result.append(idx+1)
    return result

# 완주하지 못한 선수
# 수많은 마라톤 선수들이 마라톤에 참여하였습니다.
# 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.
# 마라톤에 참여한 선수들의 이름이 담긴 배열 participant와
# 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때,
# 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.
def solution(participant, completion):
    completion.sort()
    participant.sort()    
    for i in range(len(completion)):
        if participant[i] != completion[i]:
            return participant[i]    
    return  participant[-1]
# 다른사람 풀이
import collections
def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]

# 숫자 짝꿍
# 두 정수 X, Y의 임의의 자리에서 공통으로 나타나는 정수 k(0 ≤ k ≤ 9)들을 이용하여
# 만들 수 있는 가장 큰 정수를 두 수의 짝꿍이라 합니다(단, 공통으로 나타나는 정수 중 서로 짝지을 수 있는 숫자만 사용합니다).
# X, Y의 짝꿍이 존재하지 않으면, 짝꿍은 -1입니다.
# X, Y의 짝꿍이 0으로만 구성되어 있다면, 짝꿍은 0입니다.
#  예를 들어, X = 3403이고 Y = 13203이라면,
#  X와 Y의 짝꿍은 X와 Y에서 공통으로 나타나는 3, 0, 3으로 만들 수 있는 가장 큰 정수인 330입니다.
#  다른 예시로 X = 5525이고 Y = 1255이면
#  X와 Y의 짝꿍은 X와 Y에서 공통으로 나타나는 2, 5, 5로 만들 수 있는 가장 큰 정수인 552입니다
#  (X에는 5가 3개, Y에는 5가 2개 나타나므로 남는 5 한 개는 짝 지을 수 없습니다.)
# 두 정수 X, Y가 주어졌을 때, X, Y의 짝꿍을 return하는 solution 함수를 완성해주세요.
def solution(X, Y):
    answer = ''
    for i in range(9,-1,-1) :
        answer += (str(i) * min(X.count(str(i)), Y.count(str(i))))
    if answer == '' :
        return '-1'
    elif len(answer) == answer.count('0'):
        return '0'
    else :
        return answer
    
# 정수 부분
# 실수 flo가 매개 변수로 주어질 때,
# flo의 정수 부분을 return하도록 solution 함수를 완성해주세요.
def solution(flo):
    return int(flo)

# flag에 따라 다른 값 반환하기
# 두 정수 a, b와 boolean 변수 flag가 매개변수로 주어질 때,
# flag가 true면 a + b를 false면 a - b를 return 하는 solution 함수를 작성해 주세요.
def solution(a, b, flag):
    answer = 0
    if flag == True:
        answer = a+b
    else:
        answer = a-b
    return answer
# 다른사람 풀이
def solution(a, b, flag):
    return a + b if flag else a - b

# 공배수
# 정수 number와 n, m이 주어집니다.
# number가 n의 배수이면서 m의 배수이면 1을 아니라면 0을 return하도록 solution 함수를 완성해주세요.
def solution(number, n, m):
    answer = 1
    if number%n == 0 and number%m == 0:
        return answer
    else:
        answer = 0
    return answer
# 다른사람 풀이
def solution(number, n, m):
    return 1 if number%n==0 and number%m==0 else 0

# n의 배수
# 정수 num과 n이 매개 변수로 주어질 때,
# num이 n의 배수이면 1을 return n의 배수가 아니라면 0을 return하도록 solution 함수를 완성해주세요.
def solution(num, n):
    return 1 if num%n==0 else 0
# 다른사람 풀이
def solution(num, n):
    return int(not(num % n))

# 문자열의 뒤의 n글자
# 문자열 my_string과 정수 n이 매개변수로 주어질 때,
# my_string의 뒤의 n글자로 이루어진 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, n):
    answer = my_string[-n:]
    return answer

# 문자열 곱하기
# 문자열 my_string과 정수 k가 주어질 때,
# my_string을 k번 반복한 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, k):
    answer = my_string * (k)
    return answer

# 문자열 출력하기
# 문자열 str이 주어질 때, str을 출력하는 코드를 작성해 보세요.
str = input()
print(str)

# a와 b 출력하기
# 정수 a와 b가 주어집니다.
# 각 수를 입력받아 입출력 예와 같은 형식으로 출력하는 코드를 작성해 보세요.
a, b = map(int, input().strip().split(' '))
print("a =",a)
print("b =",b)

# 문자열 반복해서 출력하기
# 문자열 str과 정수 n이 주어집니다.
# str이 n번 반복된 문자열을 만들어 출력하는 코드를 작성해 보세요.
a, b = input().strip().split(' ')
b = int(b)
for i in range(0,b):
    print(a, end='')

# 덧셈식 출력하기
# 두 정수 a, b가 주어질 때 다음과 같은 형태의 계산식을 출력하는 코드를 작성해 보세요.
# a + b = c
a, b = map(int, input().strip().split(' '))
print(a,"+",b,"=",a + b)

# 특수문자 출력하기
# 다음과 같이 출력하도록 코드를 작성해 주세요.
#  !@#$%^&*(\'"<>?:;
print(r'!@#$%^&*(\'"<>?:;')

# 홀짝 구분하기
# 자연수 n이 입력으로 주어졌을 때 만약 n이 짝수이면 "n is even"을,
# 홀수이면 "n is odd"를 출력하는 코드를 작성해 보세요.
a = int(input())
if a%2 == 0:
    print(a, "is even")
else:
    print(a, "is odd")

# 문자열 붙여서 출력하기
# 두 개의 문자열 str1, str2가 공백으로 구분되어 입력으로 주어집니다.
# 입출력 예와 같이 str1과 str2을 이어서 출력하는 코드를 작성해 보세요.
str1, str2 = input().strip().split(' ')
print(str1, end="")
print(str2)
# 다른사람 풀이
print(input().strip().replace(' ', ''))

# 소문자로 바꾸기
# 알파벳으로 이루어진 문자열 myString이 주어집니다.
# 모든 알파벳을 소문자로 변환하여 return 하는 solution 함수를 완성해 주세요.
def solution(myString):
    return myString.lower()
