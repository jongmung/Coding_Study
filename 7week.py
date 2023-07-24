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

# 문자열 돌리기
# 문자열 str이 주어집니다.
# 문자열을 시계방향으로 90도 돌려서 아래 입출력 예와 같이 출력하는 코드를 작성해 보세요.
str = input()
for i in range(len(str)):
    print(str[i])

# 대소문자 바꿔서 출력하기
# 영어 알파벳으로 이루어진 문자열 str이 주어집니다.
# 각 알파벳을 대문자는 소문자로 소문자는 대문자로 변환해서 출력하는 코드를 작성해 보세요.
str = input()
print(str.swapcase())

# 두 수의 연산값 비교하기
# 연산 ⊕는 두 정수에 대한 연산으로 두 정수를 붙여서 쓴 값을 반환합니다.
# 예를 들면 다음과 같습니다.
#   12 ⊕ 3 = 123
#   3 ⊕ 12 = 312
#양의 정수 a와 b가 주어졌을 때, a ⊕ b와 2 * a * b 중 더 큰 값을 return하는 solution 함수를 완성해 주세요.
#단, a ⊕ b와 2 * a * b가 같으면 a ⊕ b를 return 합니다.
def solution(a, b):
    answer1 = int(str(a) + str(b))
    answer2 = 2 * a * b
    return answer1 if answer1 >= answer2 else answer2

# 문자 리스트를 문자열로 변환하기
# 문자들이 담겨있는 배열 arr가 주어집니다.
# arr의 원소들을 순서대로 이어 붙인 문자열을 return 하는 solution함수를 작성해 주세요.
def solution(arr):
    answer = ''
    for i in range(len(arr)):
        answer += arr[i]
    return answer

# 더 크게 합치기
# 연산 ⊕는 두 정수에 대한 연산으로 두 정수를 붙여서 쓴 값을 반환합니다. 예를 들면 다음과 같습니다.
#   12 ⊕ 3 = 123
#   3 ⊕ 12 = 312
# 양의 정수 a와 b가 주어졌을 때, a ⊕ b와 b ⊕ a 중 더 큰 값을 return 하는 solution 함수를 완성해 주세요.
# 단, a ⊕ b와 b ⊕ a가 같다면 a ⊕ b를 return 합니다.
def solution(a, b):
    answer1 = int(str(a) + str(b))
    answer2 = int(str(b) + str(a))
    return answer1 if answer1 >= answer2 else answer2
# 다른사람 풀이
def solution(a, b):
    return int(max(f"{a}{b}", f"{b}{a}"))

# 문자열 섞기
# 길이가 같은 두 문자열 str1과 str2가 주어집니다.
# 두 문자열의 각 문자가 앞에서부터 서로 번갈아가면서 한 번씩 등장하는 문자열을 만들어 return 하는 solution 함수를 완성해 주세요.
def solution(str1, str2):
    answer = ''
    for i in range(len(str1)):
        answer += str1[i] + str2[i]
    return answer

# 첫 번째로 나오는 음수
# 정수 리스트 num_list가 주어질 때,
# 첫 번째로 나오는 음수의 인덱스를 return하도록 solution 함수를 완성해주세요.
# 음수가 없다면 -1을 return합니다.
def solution(num_list):
    answer = -1
    for i, num in enumerate(num_list):
        if num < 0 :
            return i
    return answer
# 다른사람 풀이
def solution(num_list):
    for i in range(len(num_list)):
        if num_list[i]<0:return i
    return -1

# n 번째 원소까지
# 정수 리스트 num_list와 정수 n이 주어질 때,
# num_list의 첫 번째 원소부터 n 번째 원소까지의 모든 원소를 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(num_list, n):
    return num_list[:n]

# 문자열로 변환
# 정수 n이 주어질 때, n을 문자열로 변환하여 return하도록 solution 함수를 완성해주세요.
def solution(n):
    return str(n)

# 정수 찾기
# 정수 리스트 num_list와 찾으려는 정수 n이 주어질 때,
# num_list안에 n이 있으면 1을 없으면 0을 return하도록 solution 함수를 완성해주세요.
def solution(num_list, n):
    answer = 0
    for i in range(len(num_list)):
        if num_list[i]==n:
            answer = 1
    return answer

# 대문자로 바꾸기
# 알파벳으로 이루어진 문자열 myString이 주어집니다.
# 모든 알파벳을 대문자로 변환하여 return 하는 solution 함수를 완성해 주세요.
def solution(myString):
    return myString.upper()

# 문자열의 앞의 n글자
def solution(my_string, n):
    return my_string[:n]

# 원소들의 곱과 합
# 정수가 담긴 리스트 num_list가 주어질 때,
# 모든 원소들의 곱이 모든 원소들의 합의 제곱보다 작으면 1을 크면 0을 return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    answer = 0
    a = 0
    b = 1
    for i in range(len(num_list)):
        a += num_list[i]
        b *= num_list[i]
    if a*a > b:
        answer = 1
    else:
        answer = 0
    return answer

# n 번째 원소부터
# 정수 리스트 num_list와 정수 n이 주어질 때,
# n 번째 원소부터 마지막 원소까지의 모든 원소를 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(num_list, n):
    answer = []
    for i in range(n-1,len(num_list)):
        answer.append(num_list[i])
    return answer

# 문자열 겹쳐쓰기
# 문자열 my_string, overwrite_string과 정수 s가 주어집니다.
# 문자열 my_string의 인덱스 s부터 overwrite_string의 길이만큼을
# 문자열 overwrite_string으로 바꾼 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, overwrite_string, s):
    answer = ''
    answer = my_string[:s]
    answer += overwrite_string
    answer += my_string[len(answer):]
    return answer
# 다른사람 풀이
def solution(my_string, overwrite_string, s):
    return my_string[:s] + overwrite_string + my_string[s + len(overwrite_string):]

# 홀짝에 따라 다른 값 반환하기
# 양의 정수 n이 매개변수로 주어질 때,
# n이 홀수라면 n 이하의 홀수인 모든 양의 정수의 합을 return 하고
# n이 짝수라면 n 이하의 짝수인 모든 양의 정수의 제곱의 합을 return 하는 solution 함수를 작성해 주세요.
def solution(n):
    answer = 0
    if n%2 != 0:
        for i in range(1, n+1):
            if i%2 != 0:
                answer += i
    else:
        for j in range(1, n+1):
            if j%2 == 0:
                answer += j*j           
    return answer
# 다른사람 풀이
def solution(n):
    return sum(x ** (2 - x % 2) for x in range(n + 1) if n % 2 == x % 2)

# n보다 커질 때까지 더하기
# 정수 배열 numbers와 정수 n이 매개변수로 주어집니다.
# numbers의 원소를 앞에서부터 하나씩 더하다가
# 그 합이 n보다 커지는 순간 이때까지 더했던 원소들의 합을 return 하는 solution 함수를 작성해 주세요.
def solution(numbers, n):
    answer = 0
    for i in numbers:
        if answer <= n:
            answer += i
    return answer

# 문자열을 정수로 변환하기
# 숫자로만 이루어진 문자열 n_str이 주어질 때,
# n_str을 정수로 변환하여 return하도록 solution 함수를 완성해주세요.
def solution(n_str):
    return int(n_str)

# 가장 큰 수
# 0 또는 양의 정수가 주어졌을 때,
# 정수를 이어 붙여 만들 수 있는 가장 큰 수를 알아내 주세요.
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x: x * 3, reverse=True)
    return str(int(''.join(numbers)))

# 주사위 게임1
# 1부터 6까지 숫자가 적힌 주사위가 두 개 있습니다. 두 주사위를 굴렸을 때 나온 
# 숫자를 각각 a, b라고 했을 때 얻는 점수는 다음과 같습니다.
#   a와 b가 모두 홀수라면 a2 + b2 점을 얻습니다.
#   a와 b 중 하나만 홀수라면 2 × (a + b) 점을 얻습니다.
#   a와 b 모두 홀수가 아니라면 |a - b| 점을 얻습니다.
#두 정수 a와 b가 매개변수로 주어질 때, 얻는 점수를 return 하는 solution 함수를 작성해 주세요.
def solution(a, b):
    answer = 0
    if a%2 != 0 and b%2 != 0:
        answer = a*a + b*b
    elif a%2 != 0 or b%2 != 0:
        answer = 2*(a+b)
    else:
        answer = abs(a-b)
    return answer
# 다른사람 풀이
def solution(a, b):
    if a%2 and b%2: return a*a+b*b
    elif a%2 or b%2: return 2*(a+b)
    return abs(a-b)

# 뒤에서 5등까지
# 정수로 이루어진 리스트 num_list가 주어집니다.
# num_list에서 가장 작은 5개의 수를 오름차순으로 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    num_list.sort()
    return num_list[:5]

# 뒤에서 5등 위로
# 정수로 이루어진 리스트 num_list가 주어집니다.
# num_list에서 가장 작은 5개의 수를 제외한 수들을 오름차순으로 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    num_list.sort()
    return num_list[5:]

# 문자열 정수의 합
# 한 자리 정수로 이루어진 문자열 num_str이 주어질 때, 각 자리수의 합을 return하도록 solution 함수를 완성해주세요.
def solution(num_str):
    answer = 0
    num_str = int(num_str)
    while num_str>0:
        answer += num_str%10
        num_str //= 10
    return answer
# 다른사람 풀이
def solution(num_str):
    return sum(map(int, list(num_str)))

# 카운트 업
# 정수 start와 end가 주어질 때, start부터 end까지의 숫자를 차례로 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(start, end):
    answer = []
    for i in range(start, end+1):
        answer.append(i)
    return answer

# 카운트 다운
# 정수 start와 end가 주어질 때,
# start에서 end까지 1씩 감소하는 수들을 차례로 담은 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(start, end):
    answer = []
    for i in range(start, end-1, -1):
        answer.append(i)
    return answer

# 배열 만들기1
# 정수 n과 k가 주어졌을 때,
# 1 이상 n이하의 정수 중에서 k의 배수를 오름차순으로 저장한 배열을 return 하는 solution 함수를 완성해 주세요.
def solution(n, k):
    answer = [i for i in range(1, n+1) if i % k == 0]
    return answer

# rny_string
# 'm'과 "rn"이 모양이 비슷하게 생긴 점을 활용해 문자열에 장난을 하려고 합니다.
# 문자열 rny_string이 주어질 때,
# rny_string의 모든 'm'을 "rn"으로 바꾼 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(rny_string):
    return rny_string.replace('m','rn')

# 공백으로 구분하기1
# 단어가 공백 한 개로 구분되어 있는 문자열 my_string이 매개변수로 주어질 때,
# my_string에 나온 단어를 앞에서부터 순서대로 담은 문자열 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string):
    answer = []
    answer = my_string.split()
    return answer

# 특정한 문자를 대문자로 바꾸기
# 영소문자로 이루어진 문자열 my_string과
# 영소문자 1글자로 이루어진 문자열 alp가 매개변수로 주어질 때,
# my_string에서 alp에 해당하는 모든 글자를 대문자로 바꾼 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, alp):
    return my_string.replace(alp, alp.upper())

# 꼬리 문자열
# 문자열들이 담긴 리스트가 주어졌을 때, 모든 문자열들을 순서대로 합친 문자열을 꼬리 문자열이라고 합니다.
# 꼬리 문자열을 만들 때 특정 문자열을 포함한 문자열은 제외시키려고 합니다.
# 예를 들어 문자열 리스트 ["abc", "def", "ghi"]가 있고 문자열 "ef"를 포함한 문자열은 제외하고 꼬리 문자열을 만들면 "abcghi"가 됩니다.
# 문자열 리스트 str_list와 제외하려는 문자열 ex가 주어질 때,
# str_list에서 ex를 포함한 문자열을 제외하고 만든 꼬리 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(str_list, ex):
    answer = ''
    for some_string in str_list:
        if not ex in some_string:
            answer += some_string
    return answer

# A 강조하기
# 문자열 myString이 주어집니다.
# myString에서 알파벳 "a"가 등장하면 전부 "A"로 변환하고,
# "A"가 아닌 모든 대문자 알파벳은 소문자 알파벳으로 변환하여 return 하는 solution 함수를 완성하세요.
def solution(myString):
    return (myString.lower()).replace('a', 'A')

# 배열 비교하기
# 이 문제에서 두 정수 배열의 대소관계를 다음과 같이 정의합니다.
#   두 배열의 길이가 다르다면, 배열의 길이가 긴 쪽이 더 큽니다.
#   배열의 길이가 같다면 각 배열에 있는 모든 원소의 합을 비교하여 다르다면 더 큰 쪽이 크고, 같다면 같습니다.
# 두 정수 배열 arr1과 arr2가 주어질 때, 위에서 정의한 배열의 대소관계에 대하여 arr2가 크다면 -1,
# arr1이 크다면 1, 두 배열이 같다면 0을 return 하는 solution 함수를 작성해 주세요.
def solution(arr1, arr2):
    answer = 0
    if len(arr1) != len(arr2):
        answer = -1 if len(arr2) > len(arr1) else 1
    elif len(arr1) == len(arr2):
        if sum(arr1) == sum(arr2):
            answer = 0
        else:
            if sum(arr1) > sum(arr2):
                answer = 1
            else:
                answer = -1
    return answer

# 글자 이어 붙여 문자열 만들기
# 문자열 my_string과 정수 배열 index_list가 매개변수로 주어집니다.
# my_string의 index_list의 원소들에 해당하는 인덱스의 글자들을
# 순서대로 이어 붙인 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, index_list):
    answer = ''
    for i in index_list:
        answer += my_string[i]
    return answer

# 공백으로 구분하기2
# 단어가 공백 한 개 이상으로 구분되어 있는 문자열 my_string이 매개변수로 주어질 때,
# my_string에 나온 단어를 앞에서부터 순서대로 담은 문자열 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string):
    return my_string.split()

# 간단한 식 계산하기
# 문자열 binomial이 매개변수로 주어집니다.
# binomial은 "a op b" 형태의 이항식이고 a와 b는 음이 아닌 정수,
# op는 '+', '-', '*' 중 하나입니다. 주어진 식을 계산한 정수를 return 하는 solution 함수를 작성해 주세요.
def solution(binomial):
    return eval(binomial)

# 순서 바꾸기
# 정수 리스트 num_list와 정수 n이 주어질 때,
# num_list를 n 번째 원소 이후의 원소들과 n 번째까지의 원소들로 나눠
# n 번째 원소 이후의 원소들을 n 번째까지의 원소들 앞에 붙인 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(num_list, n):
    answer = []
    answer = num_list[n:]
    answer += num_list[:n]
    return answer

# 배열의 원소 삭제하기
# 정수 배열 arr과 delete_list가 있습니다.
# arr의 원소 중 delete_list의 원소를 모두 삭제하고
# 남은 원소들은 기존의 arr에 있던 순서를 유지한 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(arr, delete_list):
    answer = []
    rm_arr = []
    for i in arr:        
        for j in delete_list:
            if i==j:
                rm_arr.append(i)    
                arr = [x for x in arr if x not in rm_arr]       
    answer = arr
    return answer

# 날짜 비교하기
# 정수 배열 date1과 date2가 주어집니다.
# 두 배열은 각각 날짜를 나타내며 [year, month, day] 꼴로 주어집니다.
# 각 배열에서 year는 연도를, month는 월을, day는 날짜를 나타냅니다.
# 만약 date1이 date2보다 앞서는 날짜라면 1을, 아니면 0을 return 하는 solution 함수를 완성해 주세요.
def solution(date1, date2):
    return 1 if int("".join(map(str, date1))) < int("".join(map(str, date2))) else 0

# 9로 나눈 나머지
# 음이 아닌 정수를 9로 나눈 나머지는 그 정수의 각 자리 숫자의 합을 9로 나눈 나머지와 같은 것이 알려져 있습니다.
# 이 사실을 이용하여 음이 아닌 정수가 문자열 number로 주어질 때,
# 이 정수를 9로 나눈 나머지를 return 하는 solution 함수를 작성해주세요.
def solution(number):
    return int(number)%9

# 문자열 잘라서 정렬하기
# 문자열 myString이 주어집니다.
# "x"를 기준으로 해당 문자열을 잘라내 배열을 만든 후 사전순으로 정렬한 배열을 return 하는 solution 함수를 완성해 주세요.
def solution(myString):
    answer = list(filter(None, myString.split("x")))
    return sorted(answer)

# 가까운 1찾기
# 정수 배열 arr가 주어집니다. 이때 arr의 원소는 1 또는 0입니다.
# 정수 idx가 주어졌을 때, idx보다 크면서 배열의 값이 1인 가장 작은 인덱스를 찾아서 반환하는 solution 함수를 완성해 주세요.
# 단, 만약 그러한 인덱스가 없다면 -1을 반환합니다.
def solution(arr, idx):
    answer = -1
    for i, num in enumerate(arr):
        if idx <= i and num == 1:
            return i
    return answer

# 세로 읽기
# 문자열 my_string과 두 정수 m, c가 주어집니다.
# my_string을 한 줄에 m 글자씩 가로로 적었을 때
# 왼쪽부터 세로로 c번째 열에 적힌 글자들을 문자열로 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, m, c):
    return my_string[c-1::m]

# 홀수 vs 짝수
# 정수 리스트 num_list가 주어집니다. 가장 첫 번째 원소를 1번 원소라고 할 때,
# 홀수 번째 원소들의 합과 짝수 번째 원소들의 합 중 큰 값을 return 하도록 solution 함수를 완성해주세요.
# 두 값이 같을 경우 그 값을 return합니다.
def solution(num_list):
    odd_sum = sum(num_list[1::2])
    even_sum = sum(num_list[0::2])
    if odd_sum<even_sum:
        return even_sum
    else:
        return odd_sum

# 마지막 두 원소
# 정수 리스트 num_list가 주어질 때,
# 마지막 원소가 그전 원소보다 크면 마지막 원소에서 그전 원소를 뺀 값을
# 마지막 원소가 그전 원소보다 크지 않다면 마지막 원소를 두 배한 값을 추가하여
# return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    if num_list[-1] > num_list[-2]:
        num_list.append(num_list[-1]-num_list[-2])
    else:
        num_list.append(2*num_list[-1])
    return num_list

# 0 떼기
# 정수로 이루어진 문자열 n_str이 주어질 때,
# n_str의 가장 왼쪽에 처음으로 등장하는 0들을 뗀 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(n_str):
    return n_str.lstrip('0')

# 주사위 게임 2
# 1부터 6까지 숫자가 적힌 주사위가 세 개 있습니다.
# 세 주사위를 굴렸을 때 나온 숫자를 각각 a, b, c라고 했을 때 얻는 점수는 다음과 같습니다.
#   세 숫자가 모두 다르다면 a + b + c 점을 얻습니다.
#   세 숫자 중 어느 두 숫자는 같고 나머지 다른 숫자는 다르다면 (a + b + c) × (a2 + b2 + c2 )점을 얻습니다.
#   세 숫자가 모두 같다면 (a + b + c) × (a2 + b2 + c2 ) × (a3 + b3 + c3 )점을 얻습니다.
# 세 정수 a, b, c가 매개변수로 주어질 때, 얻는 점수를 return 하는 solution 함수를 작성해 주세요.
def solution(a, b, c):
    answer = 0
    one, two, three = (a + b + c), (a**2 + b**2 + c**2), (a**3 + b**3 + c**3)
    if a == b and b == c :
        answer = one * two * three
    elif a == b or b == c or a == c:
        answer = one * two
    else:
        answer = one
    return answer

# 조건 문자열
# 문자열에 따라 다음과 같이 두 수의 크기를 비교하려고 합니다.
# 두 수가 n과 m이라면
#   ">", "=" : n >= m
#   "<", "=" : n <= m
#   ">", "!" : n > m
#   "<", "!" : n < m
# 두 문자열 ineq와 eq가 주어집니다.
# ineq는 "<"와 ">"중 하나고, eq는 "="와 "!"중 하나입니다.
# 그리고 두 정수 n과 m이 주어질 때, n과 m이 ineq와 eq의 조건에 맞으면 1을 아니면 0을 return하도록 solution 함수를 완성해주세요.
def solution(ineq, eq, n, m):
    answer = 0
    if (ineq == '>' and eq == '='):
        answer = 1 if n >= m else 0
    elif (ineq == '<' and eq == '='):
        answer = 1 if n <= m else 0
    elif (ineq == '>' and eq == '!'):
        answer = 1 if n > m else 0
    elif (ineq == '<' and eq == '!'):
        answer = 1 if n < m else 0
    return answer

# 배열 만들기 5
# 배열 intStrs의 각 원소마다 s번 인덱스에서 시작하는 길이 l짜리 부분 문자열을 잘라내 정수로 변환합니다.
# 이때 변환한 정수값이 k보다 큰 값들을 담은 배열을 return 하는 solution 함수를 완성해 주세요.
def solution(intStrs, k, s, l):
    answer = []
    
    for i in intStrs:
        if int(i[s:s+l]) > k:
            answer.append(int(i[s:s+l]))
    
    return answer

# 1로 만들기
# 정수가 있을 때, 짝수라면 반으로 나누고, 홀수라면 1을 뺀 뒤 반으로 나누면, 마지막엔 1이 됩니다.
# 정수들이 담긴 리스트 num_list가 주어질 때, num_list의 모든 원소를 1로 만들기 위해서
# 필요한 나누기 연산의 횟수를 return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    answer = 0
    for i in num_list:
        count = 0
        while i != 1:
            count += 1
            if i % 2 == 0:
                i = i / 2
            else:
                i = (i - 1) / 2
        answer += count
    return answer

# 2의 영역
# 정수 배열 arr가 주어집니다.
# 배열 안의 2가 모두 포함된 가장 작은 연속된 부분 배열을 return 하는 solution 함수를 완성해 주세요.
# 단, arr에 2가 없는 경우 [-1]을 return 합니다.
def solution(arr):
    answer = []
    if 2 in arr:
        if arr.count(2) > 1:
            start = arr.index(2)
            end = len(arr) - arr[::-1].index(2)
            return arr[start:end]
        else:
            idx = arr.index(2)
            return [arr[idx]]
    else:
        return [-1]
    return answer

# 왼쪽 오른쪽
# 문자열 리스트 str_list에는 "u", "d", "l", "r" 네 개의 문자열이 여러 개 저장되어 있습니다.
# str_list에서 "l"과 "r" 중 먼저 나오는 문자열이 "l"이라면 해당 문자열을 기준으로 왼쪽에 있는 문자열들을 순서대로 담은 리스트를,
# 먼저 나오는 문자열이 "r"이라면 해당 문자열을 기준으로 오른쪽에 있는 문자열들을 순서대로 담은 리스트를 return하도록 solution 함수를 완성해주세요.
# "l"이나 "r"이 없다면 빈 리스트를 return합니다.
def solution(str_list):
    for i in range(len(str_list)):
        if str_list[i] == "l":
            return str_list[:i]
        elif str_list[i] == "r":
            return str_list[i+1:]
    return []

# 배열 만들기 2
# 정수 l과 r이 주어졌을 때, l 이상 r이하의 정수 중에서 숫자 "0"과 "5"로만
# 이루어진 모든 정수를 오름차순으로 저장한 배열을 return 하는 solution 함수를 완성해 주세요.
# 만약 그러한 정수가 없다면, -1이 담긴 배열을 return 합니다.
def solution(l, r):
    answer = []
    for i in range(l, r+1):
        if all(num in ['0', '5'] for num in str(i)):
            answer.append(i)
    if len(answer) == 0:
        answer.append(-1)
    return answer

# 주사위 게임3
# 1부터 6까지 숫자가 적힌 주사위가 네 개 있습니다. 네 주사위를 굴렸을 때 나온 숫자에 따라 다음과 같은 점수를 얻습니다.
#   네 주사위에서 나온 숫자가 모두 p로 같다면 1111 × p점을 얻습니다.
#   세 주사위에서 나온 숫자가 p로 같고 나머지 다른 주사위에서 나온 숫자가 q(p ≠ q)라면 (10 × p + q)2 점을 얻습니다.
#   주사위가 두 개씩 같은 값이 나오고, 나온 숫자를 각각 p, q(p ≠ q)라고 한다면 (p + q) × |p - q|점을 얻습니다.
#   어느 두 주사위에서 나온 숫자가 p로 같고 나머지 두 주사위에서 나온 숫자가 각각 p와 다른 q, r(q ≠ r)이라면 q × r점을 얻습니다.
#   네 주사위에 적힌 숫자가 모두 다르다면 나온 숫자 중 가장 작은 숫자 만큼의 점수를 얻습니다.
# 네 주사위를 굴렸을 때 나온 숫자가 정수 매개변수 a, b, c, d로 주어질 때, 얻는 점수를 return 하는 solution 함수를 작성해 주세요.
def solution(a, b, c, d):
    answer = 0
    origin = [a, b, c, d]
    arr = list(set(origin))
    if len(arr) == 4:
        answer = min(arr)
    elif len(arr) == 3:
        p = max(origin, key=origin.count)
        tmp = [num for num in arr if num != p]
        answer = tmp[0] * tmp[1]
    elif len(arr) == 2:
        if max([origin.count(num) for num in arr]) > 2:
            p = max(origin, key=origin.count)
            q = min(origin, key=origin.count)
            answer = pow(((10 * p) + q), 2)
        else:
            answer = ((arr[0] + arr[1]) * abs(arr[0] - arr[1]))  
    elif len(arr) == 1:
        answer = int(str(arr[0]) * 4)
    return answer

# 세 개의 구분자
# 임의의 문자열이 주어졌을 때 문자 "a", "b", "c"를 구분자로 사용해 문자열을 나누고자 합니다.
# 예를 들어 주어진 문자열이 "baconlettucetomato"라면 나눠진 문자열 목록은 ["onlettu", "etom", "to"] 가 됩니다.
# 문자열 myStr이 주어졌을 때 위 예시와 같이 "a", "b", "c"를 사용해 나눠진 문자열을 순서대로 저장한 배열을 return 하는 solution 함수를 완성해 주세요.
# 단, 두 구분자 사이에 다른 문자가 없을 경우에는 아무것도 저장하지 않으며, return할 배열이 빈 배열이라면 ["EMPTY"]를 return 합니다.
import re
def solution(myStr):
    answer = re.sub("[a-c]", " ", myStr).split()
    return answer if len(answer) > 0 else ["EMPTY"]

# 그림 확대
# 직사각형 형태의 그림 파일이 있고, 이 그림 파일은 1 × 1 크기의 정사각형 크기의 픽셀로 이루어져 있습니다.
# 이 그림 파일을 나타낸 문자열 배열 picture과 정수 k가 매개변수로 주어질 때,
# 이 그림 파일을 가로 세로로 k배 늘린 그림 파일을 나타내도록 문자열 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(picture, k):
    answer = []
    for row in picture:
        resized = ''
        for pixel in row:
            resized += pixel * k
        for _ in range(k):
            answer.append(resized)
    return answer

# 두 수의 합
# 0 이상의 두 정수가 문자열 a, b로 주어질 때,
# a + b의 값을 문자열로 return 하는 solution 함수를 작성해 주세요.
def solution(a, b):
    return str(int(a)+int(b))

# 원하는 문자열 찾기
# 알파벳으로 이루어진 문자열 myString과 pat이 주어집니다.
# myString의 연속된 부분 문자열 중 pat이 존재하면 1을 그렇지 않으면 0을 return 하는 solution 함수를 완성해 주세요.
# 단, 알파벳 대문자와 소문자는 구분하지 않습니다.
def solution(myString, pat):
    return 1 if pat.lower() in myString.lower() else 0

# 5명씩
# 최대 5명씩 탑승가능한 놀이기구를 타기 위해 줄을 서있는 사람들의 이름이 담긴 문자열 리스트 names가 주어질 때,
# 앞에서 부터 5명씩 묶은 그룹의 가장 앞에 서있는 사람들의 이름을 담은 리스트를 return하도록 solution 함수를 완성해주세요. 마지막 그룹이 5명이 되지 않더라도 가장 앞에 있는 사람의 이름을 포함합니다.
def solution(names):
    return [names[i] for i in range(0, len(names), 5)]

# ad 제거하기
# 문자열 배열 strArr가 주어집니다.
# 배열 내의 문자열 중 "ad"라는 부분 문자열을 포함하고 있는 모든 문자열을 제거하고
# 남은 문자열을 순서를 유지하여 배열로 return 하는 solution 함수를 완성해 주세요.
def solution(strArr):
    return [char for char in strArr if "ad" not in char]

# 간단한 논리 연산
# boolean 변수 x1, x2, x3, x4가 매개변수로 주어질 때,
# 다음의 식의 true/false를 return 하는 solution 함수를 작성해 주세요.
def solution(x1, x2, x3, x4):
    return ((x1 or x2) and (x3 or x4))

# 문자열 묶기
# 문자열 배열 strArr이 주어집니다. strArr의 원소들을 길이가 같은 문자열들끼리 그룹으로 묶었을 때
# 가장 개수가 많은 그룹의 크기를 return 하는 solution 함수를 완성해 주세요.
def solution(strArr):
    answer = [len(i) for i in strArr]
    tmp = []
    for i in set(answer):
        tmp.append(answer.count(i))
    return max(tmp)