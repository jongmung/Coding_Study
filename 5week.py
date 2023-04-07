# 유한소수 판별하기
# 소수점 아래 숫자가 계속되지 않고 유한개인 소수를 유한소수라고 합니다.
# 분수를 소수로 고칠 때 유한소수로 나타낼 수 있는 분수인지 판별하려고 합니다.
# 유한소수가 되기 위한 분수의 조건은 다음과 같습니다.
#  기약분수로 나타내었을 때, 분모의 소인수가 2와 5만 존재해야 합니다.
# 두 정수 a와 b가 매개변수로 주어질 때, a/b가 유한소수이면 1을,
# 무한소수라면 2를 return하도록 solution 함수를 완성해주세요.
from math import gcd
def solution(a, b):
    b //= gcd(a,b)
    while b%2==0:
        b//=2
    while b%5==0:
        b//=5
    return 1 if b==1 else 2

# 다른사람 풀이
def solution(a, b):
    return 1 if a/b * 1000 % 1 == 0 else 2

# 연속된 수의 합
# 연속된 세 개의 정수를 더해 12가 되는 경우는 3, 4, 5입니다.
# 두 정수 num과 total이 주어집니다. 연속된 수 num개를 더한 값이 total이 될 때,
# 정수 배열을 오름차순으로 담아 return하도록 solution함수를 완성해보세요.
def solution(num, total):
    average = total // num
    return [i for i in range(average - (num-1)//2, average + (num + 2)//2)]

# 평행
# 점 네 개의 좌표를 담은 이차원 배열  dots가 다음과 같이 매개변수로 주어집니다.
#  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# 주어진 네 개의 점을 두 개씩 이었을 때, 두 직선이 평행이 되는 경우가 있으면 1을 없으면 0을 return 하도록 solution 함수를 완성해보세요.
def solution(dots):
    answer = 0
    if slope(dots[0],dots[1]) == slope(dots[2],dots[3]):
        answer = 1
    if slope(dots[0],dots[2]) == slope(dots[1],dots[3]):
        answer = 1
    if slope(dots[0],dots[3]) == slope(dots[1],dots[2]):
        answer = 1
    return answer
# 다른사람 풀이
def slope(dot1,dot2):
    return (dot2[1] - dot1[1] ) / (dot2[0] - dot1[0])

# 개미 군단
# 개미 군단이 사냥을 나가려고 합니다.
# 개미군단은 사냥감의 체력에 딱 맞는 병력을 데리고 나가려고 합니다.
# 장군개미는 5의 공격력을, 병정개미는 3의 공격력을 일개미는 1의 공격력을 가지고 있습니다.
# 예를 들어 체력 23의 여치를 사냥하려고 할 때, 일개미 23마리를 데리고 가도 되지만,
# 장군개미 네 마리와 병정개미 한 마리를 데리고 간다면 더 적은 병력으로 사냥할 수 있습니다.
# 사냥감의 체력 hp가 매개변수로 주어질 때,
# 사냥감의 체력에 딱 맞게 최소한의 병력을 구성하려면 몇 마리의 개미가 필요한지를 return하도록 solution 함수를 완성해주세요.
def solution(hp):
    answer = 0
    if hp//5>0:
        answer += (hp//5)
        hp = hp%5
    if hp//3>0:
        answer += (hp//3)
        hp= hp%3
    if hp//1>0:
        answer+=(hp//1)
    return answer
# 다른사람 풀이
def solution(hp):    
    return hp // 5 + (hp % 5 // 3) + ((hp % 5) % 3)

# 진료 순서 정하기
# 외과의사 머쓱이는 응급실에 온 환자의 응급도를 기준으로 진료 순서를 정하려고 합니다.
# 정수 배열 emergency가 매개변수로 주어질 때 응급도가 높은 순서대로
# 진료 순서를 정한 배열을 return하도록 solution 함수를 완성해주세요.
def solution(emergency):
    answer = []
    a = sorted(emergency, reverse = True)
    for i in emergency:
        answer.append(a.index(i)+1)
    return answer
# 다른사람 풀이
def solution(emergency):
    return [sorted(emergency, reverse=True).index(e) + 1 for e in emergency]

# 영어가 싫어요
# 영어가 싫은 머쓱이는 영어로 표기되어있는 숫자를 수로 바꾸려고 합니다.
# 문자열 numbers가 매개변수로 주어질 때, numbers를 정수로 바꿔 return 하도록 solution 함수를 완성해 주세요.
def solution(numbers):
    answer = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for index,num in enumerate(answer):            # enumerate()함수는 기본적으로 인덱스와 원소로 이루어진 튜플을 만들어준다.
        numbers = numbers.replace(num,str(index))  # for i,letter in enumerate(['A','B','C']):
    return int(numbers)                            # print(i,letter) -> 0 A , 1 B , 2 C
# 다른사람 풀이
def solution(numbers):
    for num, eng in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]):
        numbers = numbers.replace(eng, str(num))
    return int(numbers)

# 구슬을 나누는 경우의 수
# 머쓱이는 구슬을 친구들에게 나누어주려고 합니다. 구슬은 모두 다르게 생겼습니다.
# 머쓱이가 갖고 있는 구슬의 개수 balls와 친구들에게 나누어 줄 구슬 개수 share이 매개변수로 주어질 때,
# balls개의 구슬 중 share개의 구슬을 고르는 가능한 모든 경우의 수를 return 하는 solution 함수를 완성해주세요.
import math
def solution(balls, share):
    return math.comb(balls, share)

# 숨어있는 숫자의 덧셈 (2)
# 문자열 my_string이 매개변수로 주어집니다. my_string은 소문자, 대문자, 자연수로만 구성되어있습니다.
# my_string안의 자연수들의 합을 return하도록 solution 함수를 완성해주세요.
def solution(my_string):
    s = ''.join(i if i.isdigit() else ' ' for i in my_string)
    return sum(int(i) for i in s.split())

# 모스부호 (1)
# 머쓱이는 친구에게 모스부호를 이용한 편지를 받았습니다. 
# 그냥은 읽을 수 없어 이를 해독하는 프로그램을 만들려고 합니다.
# 문자열 letter가 매개변수로 주어질 때, letter를 영어 소문자로 바꾼 문자열을 return 하도록 solution 함수를 완성해보세요.
# morse = { 
#    '.-':'a','-...':'b','-.-.':'c','-..':'d','.':'e','..-.':'f',
#    '--.':'g','....':'h','..':'i','.---':'j','-.-':'k','.-..':'l',
#    '--':'m','-.':'n','---':'o','.--.':'p','--.-':'q','.-.':'r',
#    '...':'s','-':'t','..-':'u','...-':'v','.--':'w','-..-':'x',
#    '-.--':'y','--..':'z'
# } 
def solution(letter):
    morse = {
        '.-':'a','-...':'b','-.-.':'c','-..':'d','.':'e','..-.':'f',
        '--.':'g','....':'h','..':'i','.---':'j','-.-':'k','.-..':'l',
        '--':'m','-.':'n','---':'o','.--.':'p','--.-':'q','.-.':'r',
        '...':'s','-':'t','..-':'u','...-':'v','.--':'w','-..-':'x',
        '-.--':'y','--..':'z'
    }
    return ''.join([morse[i] for i in letter.split(' ')])

# 최빈값 구하기
# 최빈값은 주어진 값 중에서 가장 자주 나오는 값을 의미합니다.
# 정수 배열 array가 매개변수로 주어질 때, 최빈값을 return 하도록 solution 함수를 완성해보세요.
# 최빈값이 여러 개면 -1을 return 합니다.
def solution(array):
    while len(array) != 0:
        for i, a in enumerate(set(array)):
            array.remove(a)
        if i == 0: return a
    return -1

# 삼각형의 완성조건 (2)
# 선분 세 개로 삼각형을 만들기 위해서는 다음과 같은 조건을 만족해야 합니다.
# 가장 긴 변의 길이는 다른 두 변의 길이의 합보다 작아야 합니다.
# 삼각형의 두 변의 길이가 담긴 배열 sides이 매개변수로 주어집니다.
# 나머지 한 변이 될 수 있는 정수의 개수를 return하도록 solution 함수를 완성해주세요.
def solution(sides):
    return (sorted(sides)[0] * 2) - 1

# 외계어 사전
# 알파벳이 담긴 배열 spell과 외계어 사전 dic이 매개변수로 주어집니다.
# spell에 담긴 알파벳을 한번씩만 모두 사용한 단어가 dic에 존재한다면 1,
# 존재하지 않는다면 2를 return하도록 solution 함수를 완성해주세요.
def solution(spell, dic):
    spell = set(spell) 
    for i in dic:
        if spell.issubset(set(i)):
            return 1 
    return 2

# 한 번만 등장한 문자
# 문자열 s가 매개변수로 주어집니다.
# s에서 한 번만 등장하는 문자를 사전 순으로 정렬한 문자열을 return 하도록 solution 함수를 완성해보세요.
# 한 번만 등장하는 문자가 없을 경우 빈 문자열을 return 합니다.
def solution(s):
    return ''.join(sorted(i for i in s if s.count(i) == 1))

# 정수 배열 array와 정수 n이 매개변수로 주어질 때,
# array에 들어있는 정수 중 n과 가장 가까운 수를 return 하도록 solution 함수를 완성해주세요.
def solution(array, n):
    array.sort()
    answer = 0
    com = n+100
    for i in array:
        if abs(i-n) < com:
            com = abs(i-n)
            answer = i
    return answer
# 다른사람 풀이
solution=lambda a,n:sorted(a,key=lambda x:(abs(x-n),x))[0]

# 2차원으로 만들기
# 정수 배열 num_list와 정수 n이 매개변수로 주어집니다.
# num_list를 다음 설명과 같이 2차원 배열로 바꿔 return하도록 solution 함수를 완성해주세요.
# num_list가 [1, 2, 3, 4, 5, 6, 7, 8] 로 길이가 8이고
# n이 2이므로 num_list를 2 * 4 배열로 다음과 같이 변경합니다.
# 2차원으로 바꿀 때에는 num_list의 원소들을 앞에서부터 n개씩 나눠 2차원 배열로 변경합니다.
import numpy as np
def solution(num_list, n):
    return (np.array(num_list).reshape((len(num_list) // n ,n))).tolist()
# 다른사람 풀이
def solution(num_list, n):
    answer = []
    for i in range(0, len(num_list), n):
        answer.append(num_list[i:i+n])
    return answer

# 저주의 숫자 3
# 3x 마을 사람들은 3을 저주의 숫자라고 생각하기 때문에 3의 배수와 숫자 3을 사용하지 않습니다.
# 3x 마을 사람들의 숫자는 다음과 같습니다.
# 10진법  3x마을숫자  10진법  3x마을숫자
#   1       1 	     6       8
#   2	    2	     7	    10
#   3	    4	     8	    11
#   4	    5	     9	    14
#   5	    7	     10	    16
def solution(n):
    answer = 0
    for i in range(n):
        answer += 1
        while answer%3 == 0 or '3' in str(answer):
            answer += 1
    return answer

# 문자열 밀기
# 문자열 "hello"에서 각 문자를 오른쪽으로 한 칸씩 밀고 마지막 문자는 맨 앞으로 이동시키면 "ohell"이 됩니다.
# 이것을 문자열을 민다고 정의한다면 문자열 A와 B가 매개변수로 주어질 때,
# A를 밀어서 B가 될 수 있다면 밀어야 하는 최소 횟수를 return하고 밀어서 B가 될 수 없으면 -1을 return 하도록 solution 함수를 완성해보세요.
from collections import deque
def solution(A, B):
    Alist = deque(A)
    Blist = deque(B)
    for i in range(len(Alist)):
        if Alist == Blist:
            return i
        Alist.rotate(1)
    return -1
# 다른사람 풀이
solution=lambda a,b:(b*2).find(a)

# 특이한 정렬
# 정수 n을 기준으로 n과 가까운 수부터 정렬하려고 합니다.
# 이때 n으로부터의 거리가 같다면 더 큰 수를 앞에 오도록 배치합니다.
# 정수가 담긴 배열 numlist와 정수 n이 주어질 때
# numlist의 원소를 n으로부터 가까운 순서대로 정렬한 배열을 return하도록 solution 함수를 완성해주세요.
def solution(numlist, n):
    answer = sorted(numlist, key = lambda x : (abs(x-n), -x))
    return answer

# 옹알이 (1)
# 머쓱이는 태어난 지 6개월 된 조카를 돌보고 있습니다.
# 조카는 아직 "aya", "ye", "woo", "ma" 네 가지 발음을
# 최대 한 번씩 사용해 조합한(이어 붙인) 발음밖에 하지 못합니다.
# 문자열 배열 babbling이 매개변수로 주어질 때,
# 머쓱이의 조카가 발음할 수 있는 단어의 개수를 return하도록 solution 함수를 완성해주세요.
def solution(babbling):
    answer = 0
    for i in babbling:
        cnt = 0
        word = ''
        for j in i:
            word += j
            if word in ['aya','ye','woo','ma']:
                word=''
                cnt +=1
        if len(word) ==0 and cnt>0:
                    answer +=1
    return answer
# 다른사람 풀이
def solution(babbling):
    c = 0
    for b in babbling:
        for w in [ "aya", "ye", "woo", "ma" ]:
            if w * 2 not in b:
                b = b.replace(w, ' ')
        if len(b.strip()) == 0:
            c += 1
    return c

# 겹치는 선분의 길이
# 선분 3개가 평행하게 놓여 있습니다.
# 세 선분의 시작과 끝 좌표가 [[start, end], [start, end], [start, end]]
# 형태로 들어있는 2차원 배열 lines가 매개변수로 주어질 때,
# 두 개 이상의 선분이 겹치는 부분의 길이를 return 하도록 solution 함수를 완성해보세요.
def solution(lines):
    answer = 0
    count = [0 for _ in range(200)]
    for line in lines:
        for i in range(line[0], line[1]): 
            count[i + 100] += 1
    answer += count.count(2)
    answer += count.count(3)
    return answer
# 다른사람 풀이
def solution(lines):
    sets = [set(range(min(l), max(l))) for l in lines]
    return len(sets[0] & sets[1] | sets[0] & sets[2] | sets[1] & sets[2])