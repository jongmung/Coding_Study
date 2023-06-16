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