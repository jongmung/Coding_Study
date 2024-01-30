# [PCCE 기출문제] 1번 / 출력
# 주어진 초기 코드는 변수에 데이터를 저장하고 출력하는 코드입니다.
# 아래와 같이 출력되도록 빈칸을 채워 코드를 완성해 주세요.
# Spring is beginning
# 13
# 310
string_msg = 'Spring is beginning'
int_val = int(3)
string_val = "3"
print(string_msg)
print(int_val + 10)
print(string_val + "10")

# [PCCE 기출문제] 2번 / 피타고라스의 정리
# 직각삼각형이 주어졌을 때 빗변의 제곱은 다른 두 변을 각각 제곱한 것의 합과 같습니다.
# 직각삼각형의 한 변의 길이를 나타내는 정수 a와 빗변의 길이를 나타내는 정수 c가 주어질 때,
# 다른 한 변의 길이의 제곱, b_square 을 출력하도록 한 줄을 수정해 코드를 완성해 주세요.
a = int(input())
c = int(input())
b_square = c*c - a*a
print(b_square)

# 
# 나이를 세는 방법은 여러 가지가 있습니다.
# 그중 한국식 나이는 태어난 순간 1살이 되며 해가 바뀔 때마다 1살씩 더 먹게 됩니다.
# 연 나이는 태어난 순간 0살이며 해가 바뀔 때마다 1살씩 더 먹게 됩니다.
# 각각 나이의 계산법은 다음과 같습니다.
#   한국식 나이 : 현재 연도 - 출생 연도 + 1
#   연 나이 : 현재 연도 - 출생 연도
# 출생 연도를 나타내는 정수 year와 구하려는 나이의 종류를 나타내는 문자열 age_type이 주어질 때
# 2030년에 몇 살인지 출력하도록 빈칸을 채워 코드를 완성해 주세요.
# age_type이 "Korea"라면 한국식 나이를, "Year"라면 연 나이를 출력합니다.
year = int(input())
age_type = input()
if age_type == "Korea":
    answer = 2030 - year + 1
elif age_type == "Year":
    answer = 2030 - year
print(answer)

# [PCCE 기출문제] 4번 / 저축
# 첫 달에 저축하는 금액을 나타내는 정수 start,
# 두 번째 달 부터 70만 원 이상 모일 때까지 매월 저축하는 금액을 나타내는 정수 before,
# 100만 원 이상 모일 때 까지 매월 저축하는 금액을 나타내는 정수 after가 주어질 때,
# 100만 원 이상을 모을 때까지 걸리는 개월 수를 출력하도록 빈칸을 채워 코드를 완성해 주세요.
start = int(input())
before = int(input())
after = int(input())
money = start
month = 1
while money < 70:
    money += before
    month += 1
while money < 100:
    money += after
    month += 1
print(month)

# PCCE 기출문제] 5번 / 산책
# 여름이는 강아지를 산책시키려고 합니다.
# 여름이는 2차원 좌표평면에서 동/서/남/북 방향으로 1m 단위로 이동하면서 강아지를 산책시킵니다.
# 산책루트가 담긴 문자열 route가 주어질 때,
# 도착점의 위치를 return하도록 빈칸을 채워 solution함수를 완성해 주세요.
#   route는 "N", "S", "E", "W"로 이루어져 있습니다.
#   "N"은 북쪽으로 1만큼 움직입니다.
#   "S"는 남쪽으로 1만큼 움직입니다.
#       북쪽으로 -1만큼 움직인 것과 같습니다.
#   "E"는 동쪽으로 1만큼 움직입니다.
#   "W"는 서쪽으로 1만큼 움직입니다.
#       동쪽으로 -1만큼 움직인 것과 같습니다.
# 출발점으로부터 [동쪽으로 떨어진 거리, 북쪽으로 떨어진 거리]형태로 강아지의 최종 위치를 구해서 return해야 합니다.
# 출발점을 기준으로 서쪽, 남쪽에 있는 경우는 동쪽, 북쪽으로 음수만큼 떨어진 것으로 표현합니다.
# 출발점으로부터 동쪽으로 2, 북쪽으로 3만큼 떨어졌다면 [2, 3]을 return 합니다.
# 출발점으로부터 서쪽으로 1, 남쪽으로 4만큼 떨어졌다면 [-1, -4]를 return 합니다.
def solution(route):
    east = 0
    north = 0
    for i in route:
        if i == "N":
            north += 1
        elif i == "S":      
            north -= 1
        elif i == "E":
            east += 1
        elif i == "W":
            east -= 1
    return [east, north]

# [PCCE 기출문제] 6번 / 가채점
# 성적을 문의하려는 학생들의 번호가 담긴 정수 리스트 numbers와
# 가채점한 점수가 성적을 문의하려는 학생 순서대로 담긴 정수 리스트 our_score,
# 실제 성적이 번호 순서대로 담긴 정수 리스트 score_list가 주어집니다.
# 주어진 solution 함수는 가채점한 점수가 실제 성적과 동일하다면 "Same"을,
# 다르다면 "Different"를 순서대로 리스트에 담아 return하는 함수입니다.
# solution 함수가 올바르게 작동하도록 한 줄을 수정해 주세요.
def solution(numbers, our_score, score_list):
    answer = []
    for i in range(len(numbers)):
        if our_score[i] == score_list[numbers[i]-1]:
            answer.append("Same")
        else:
            answer.append("Different")
    return answer

# [PCCE 기출문제] 7번 / 가습기
# 상우가 사용하는 가습기에는 "auto", "target", "minimum"의 세 가지 모드가 있습니다.
# 가습기의 가습량은 0~5단계로 구분되며 각 모드 별 동작 방식은 다음과 같습니다.
#   "auto" 모드
#       습도가 0 이상 10 미만인 경우 : 5단계
#       습도가 10 이상 20 미만인 경우 : 4단계
#       습도가 20 이상 30 미만인 경우 : 3단계
#       습도가 30 이상 40 미만인 경우 : 2단계
#       습도가 40 이상 50 미만인 경우 : 1단계
#       습도가 50 이상인 경우 : 0단계
#   "target" 모드
#       습도가 설정값 미만일 경우 : 3단계
#       습도가 설정값 이상일 경우 : 1단계
#   "minimum"모드
#       습도가 설정값 미만일 경우 : 1단계
#       습도가 설정값 이상일 경우 : 0단계
# 상우가 설정한 가습기의 모드를 나타낸 문자열 mode_type,
# 현재 공기 중 습도를 나타낸 정수 humidity,
# 설정값을 나타낸 정수 val_set이 주어질 때
# 현재 가습기가 몇 단계로 작동 중인지 return하도록 빈칸을 채워 solution 함수를 완성해 주세요.
def func1(humidity, val_set):
    if humidity < val_set:
        return 3
    return 1

def func2(humidity):
    if humidity >= 50:
        return 0
    elif humidity >= 40:
        return 1
    elif humidity >= 30:
        return 2
    elif humidity >= 20:
        return 3
    elif humidity >= 10:
        return 4
    else:
        return 5

def func3(humidity, val_set):
    if humidity < val_set:
        return 1
    return 0

def solution(mode_type, humidity, val_set):
    answer = 0
    if mode_type == "auto":
        answer = func2(humidity)
    elif mode_type == "target":
        answer = func1(humidity, val_set)
    elif mode_type == "minimum":
        answer = func3(humidity, val_set)
    return answer

# [PCCE 기출문제] 8번 / 창고 정리
# 예를 들어 창고의 각 칸에 담겨있는 물건의 이름이
# storage = ["pencil", "pencil", "pencil", "book"],
# 각 물건의 개수가 num = [2, 4, 3, 1]이라면
# 연필과 책을 한 칸에 각각 겹쳐 쌓아 간단하게
# clean_storage = ["pencil", "book"],
# clean_num = [9, 1]로 만들 수 있습니다.
# 주어진 solution 함수는 정리되기 전 창고의 물건 이름이 담긴 문자열 리스트 storage와
# 각 물건의 개수가 담긴 정수 리스트 num이 주어질 때,
# 정리된 창고에서 개수가 가장 많은 물건의 이름을 return 하는 함수입니다.
# solution 함수가 올바르게 작동하도록 한 줄을 수정해 주세요.
def solution(storage, num):
    clean_storage = []
    clean_num = []
    for i in range(len(storage)):
        if storage[i] in clean_storage:
            pos = clean_storage.index(storage[i])
            clean_num[pos] += num[i]
        else:
            clean_storage.append(storage[i])
            clean_num.append(num[i])
    # 아래 코드에는 틀린 부분이 없습니다.     
    max_num = max(clean_num)
    answer = clean_storage[clean_num.index(max_num)]
    return answer

# [1차] 캐시
# 어피치는 제이지에게 해당 로직을 개선하라고 닦달하기 시작하였고,
# 제이지는 DB 캐시를 적용하여 성능 개선을 시도하고 있지만 캐시 크기를 얼마로 해야 효율적인지 몰라 난감한 상황이다.
# 어피치에게 시달리는 제이지를 도와, DB 캐시를 적용할 때 캐시 크기에 따른 실행시간 측정 프로그램을 작성하시오.
def solution(cacheSize, cities):
    answer = 0
    memolist=[]
    
    if cacheSize==0:
        return len(cities)*5

    else:
        cities[0]=cities[0].upper()
        memolist.append(cities[0])
        answer+=5
        
        for i in range(1,len(cities)):
            cities[i]=cities[i].upper()
            check=True
            
            for j in range(len(memolist)):
                if memolist[j]==cities[i]:
                    memolist.remove(cities[i])
                    memolist.append(cities[i])
                    answer+=1
                    check=False # 중복된거 있음!
                    break
            if  check:
                answer+=5
                if len(memolist)==cacheSize:
                    memolist.pop(0)
                memolist.append(cities[i])    
    return answer

# 튜플
# 셀수있는 수량의 순서있는 열거 또는 어떤 순서를 따르는 요소들의 모음을 튜플(tuple)이라고 합니다.
# n개의 요소를 가진 튜플을 n-튜플(n-tuple)이라고 하며, 다음과 같이 표현할 수 있습니다.
# 원소의 개수가 n개이고,
# 중복되는 원소가 없는 튜플 (a1, a2, a3, ..., an)이 주어질 때(단, a1, a2, ..., an은 자연수),
# 이는 다음과 같이 집합 기호 '{', '}'를 이용해 표현할 수 있습니다.
# 특정 튜플을 표현하는 집합이 담긴 문자열 s가 매개변수로 주어질 때,
# s가 표현하는 튜플을 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def solution(s):
    li=[]
    for i in s.split("},"):
        li.append(i.replace("{","").replace("}","").split(","))
    li.sort(key=len)
    answer=[]
    for i in li:
        for j in i:
            if j not in answer:
                answer.append(j)
    return list(map(int,answer))

# k진수에서 소수 개수 구하기
# 양의 정수 n이 주어집니다. 이 숫자를 k진수로 바꿨을 때,
# 변환된 수 안에 아래 조건에 맞는 소수(Prime number)가 몇 개인지 알아보려 합니다.
#   0P0처럼 소수 양쪽에 0이 있는 경우
#   P0처럼 소수 오른쪽에만 0이 있고 왼쪽에는 아무것도 없는 경우
#   0P처럼 소수 왼쪽에만 0이 있고 오른쪽에는 아무것도 없는 경우
#   P처럼 소수 양쪽에 아무것도 없는 경우
# 정수 n과 k가 매개변수로 주어집니다.
# n을 k진수로 바꿨을 때, 변환된 수 안에서 찾을 수 있는
# 위 조건에 맞는 소수의 개수를 return 하도록 solution 함수를 완성해 주세요.
def solution(n, k):
    word=""
    while n:            # 숫자를 k진법으로 변환
        word = str(n%k)+word
        n=n//k 
    word=word.split("0")  # 변환된 숫자를 0을 기준으로 나눈다.
    count=0
    for w in word:
        if len(w)==0:    # 만약 0또는 1이거나 빈공간이라면 continue를 통해 건너뛴다.
            continue
        if int(w)<2:
            continue
        sosu=True
        for i in range(2,int(int(w)**0.5)+1): # 소수찾기
            if int(w)%i==0:
                sosu=False
                break
        if sosu:
            count+=1      
    return count

# 게임 맵 최단거리
# ROR 게임은 두 팀으로 나누어서 진행하며, 상대 팀 진영을 먼저 파괴하면 이기는 게임입니다.
# 따라서, 각 팀은 상대 팀 진영에 최대한 빨리 도착하는 것이 유리합니다.
# 게임 맵의 상태 maps가 매개변수로 주어질 때,
# 캐릭터가 상대 팀 진영에 도착하기 위해서 지나가야 하는 칸의 개수의 최솟값을 return 하도록 solution 함수를 완성해주세요.
# 단, 상대 팀 진영에 도착할 수 없을 때는 -1을 return 해주세요.
# 검은색 부분은 벽으로 막혀있어 갈 수 없는 길이며, 흰색 부분은 갈 수 있는 길입니다.
# 캐릭터가 움직일 때는 동, 서, 남, 북 방향으로 한 칸씩 이동하며, 게임 맵을 벗어난 길은 갈 수 없습니다.
from collections import deque
def solution(maps):
    answer = 0
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]

    queue = deque()
    queue.append((0,0))		# 0,0에서 시작

    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            N = len(maps)
            M = len(maps[0])
            if 0<=nx<N and 0<=ny<M and maps[nx][ny]==1:
                maps[nx][ny] = maps[x][y]+1
                queue.append((nx,ny))
    answer = maps[N-1][M-1]
    
    if answer ==1:
        answer = -1
    return answer

# 주차 요금 계산
# 주차장의 요금표와 차량이 들어오고(입차) 나간(출차) 기록이 주어졌을 때,
# 차량별로 주차 요금을 계산하려고 합니다.
# 아래는 하나의 예시를 나타냅니다.
# 주차 요금을 나타내는 정수 배열 fees, 자동차의 입/출차 내역을 나타내는 문자열 배열 records가 매개변수로 주어집니다.
# 차량 번호가 작은 자동차부터 청구할 주차 요금을 차례대로 정수 배열에 담아서 return 하도록 solution 함수를 완성해주세요.
import math
from collections import defaultdict

def solution(fees, records):
    answer = []
    table = defaultdict(list)
    
    for i in range(len(records)):
        time,number,state = records[i].split()
        minutes = int(time[:2]) * 60 + int(time[3:])
        table[number].append(minutes)
    
    for i in table :
        if ( len(table[i] ) % 2 == 1 ) :
            table[i].append(23*60+59)
    
    cars = sorted(table.keys())
    
    for c in cars :
        money = 0
        time = sum(table[c][1::2]) - sum(table[c][::2])
        if time > fees[0] :
            money += fees[1] + math.ceil((time - fees[0]) / fees[2]) * fees[3]
        else :
            money += fees[1]
        answer.append(money)

    return answer

# 땅따먹기
# 땅따먹기 게임을 하려고 합니다. 땅따먹기 게임의 땅(land)은 총 N행 4열로 이루어져 있고,
# 모든 칸에는 점수가 쓰여 있습니다. 1행부터 땅을 밟으며 한 행씩 내려올 때,
# 각 행의 4칸 중 한 칸만 밟으면서 내려와야 합니다.
# 단, 땅따먹기 게임에는 한 행씩 내려올 때, 같은 열을 연속해서 밟을 수 없는 특수 규칙이 있습니다.
# 마지막 행까지 모두 내려왔을 때, 얻을 수 있는 점수의 최대값을 return하는 solution 함수를 완성해 주세요.
# 위 예의 경우,
# 1행의 네번째 칸 (5), 2행의 세번째 칸 (7), 3행의 첫번째 칸 (4) 땅을 밟아
# 16점이 최고점이 되므로 16을 return 하면 됩니다.
def solution(land):

    for i in range(1,len(land)):
        for j in range(len(land[0])):
            land[i][j] += max(land[i-1][:j] + land[i-1][j+1:])

    return max(land[len(land)-1])

# 다리를 지나는 트럭
# 트럭 여러 대가 강을 가로지르는 일차선 다리를 정해진 순으로 건너려 합니다.
# 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 알아내야 합니다.
# 다리에는 트럭이 최대 bridge_length대 올라갈 수 있으며,
# 다리는 weight 이하까지의 무게를 견딜 수 있습니다.
# 단, 다리에 완전히 오르지 않은 트럭의 무게는 무시합니다.
# solution 함수의 매개변수로 다리에 올라갈 수 있는 트럭 수 bridge_length,
# 다리가 견딜 수 있는 무게 weight, 트럭 별 무게 truck_weights가 주어집니다.
# 이때 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 return 하도록 solution 함수를 완성하세요.
def solution(length, threshold, trucks):
    answer = 0
    bridge = [0]*length
    cur_weight = 0
    trucks = trucks[::-1]
    while trucks:
        answer += 1
        cur_weight -= bridge.pop(0)
        w = trucks.pop() if cur_weight + trucks[-1] <= threshold else 0
        cur_weight += w
        bridge.append(w)

    return answer + len(bridge)

# 큰 수 만들기
# 어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 합니다.
# 예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있습니다.
# 이 중 가장 큰 숫자는 94 입니다.
# 문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어집니다.
# number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 return 하도록 solution 함수를 완성하세요.
def solution(number, k):
    answer = [] # Stack
    for num in number:
        while k > 0 and answer and answer[-1] < num:
            answer.pop()
            k -= 1
        answer.append(num)
        
    return ''.join(answer[:len(answer) - k])

# 연속된 부분 수열의 합
# 비내림차순으로 정렬된 수열이 주어질 때, 다음 조건을 만족하는 부분 수열을 찾으려고 합니다.
#   기존 수열에서 임의의 두 인덱스의 원소와 그 사이의 원소를 모두 포함하는 부분 수열이어야 합니다.
#   부분 수열의 합은 k입니다.
#   합이 k인 부분 수열이 여러 개인 경우 길이가 짧은 수열을 찾습니다.
#   길이가 짧은 수열이 여러 개인 경우 앞쪽(시작 인덱스가 작은)에 나오는 수열을 찾습니다.
# 수열을 나타내는 정수 배열 sequence와 부분 수열의 합을 나타내는 정수 k가 매개변수로 주어질 때,
# 위 조건을 만족하는 부분 수열의 시작 인덱스와 마지막 인덱스를 배열에 담아 return 하는 solution 함수를 완성해주세요.
# 이때 수열의 인덱스는 0부터 시작합니다.
def solution(sequence, k):
    l = r = 0
    answer = [0, len(sequence)]
    sum = sequence[0]
    while True:
        if sum < k:
            r += 1
            if r == len(sequence): break
            sum += sequence[r]
        else:
            if sum == k:
                if r-l < answer[1]-answer[0]:
                    answer = [l, r]
            sum -= sequence[l]
            l += 1
    return answer

# 호텔 대실
# 호텔을 운영 중인 코니는 최소한의 객실만을 사용하여 예약 손님들을 받으려고 합니다.
# 한 번 사용한 객실은 퇴실 시간을 기준으로 10분간 청소를 하고 다음 손님들이 사용할 수 있습니다.
# 예약 시각이 문자열 형태로 담긴 2차원 배열 book_time이 매개변수로 주어질 때,
# 코니에게 필요한 최소 객실의 수를 return 하는 solution 함수를 완성해주세요.
def solution(book_time):
    # 풀이설명1 : 함수 만들기
    def change_min(str_time: str) -> int:
        return int(str_time[0:2]) * 60 + int(str_time[3:])
    #풀이 설명2 : 예약 시간이 빠른 순으로 정렬하기
    book_times = sorted([[change_min(i[0]), change_min(i[1]) + 10] for i in book_time])
    #풀이 설명3 : 방 배정하기
    rooms = []
    for book_time in book_times:
        if not rooms:
            rooms.append(book_time)
            continue
        for index, room in enumerate(rooms):
            if book_time[0] >= room[-1]:
                rooms[index] = room + book_time
                break
        else:
            rooms.append(book_time)
    return len(rooms)

# 줄 서는 방법
# n명의 사람이 일렬로 줄을 서고 있습니다. n명의 사람들에게는 각각 1번부터 n번까지 번호가 매겨져 있습니다.
# n명이 사람을 줄을 서는 방법은 여러가지 방법이 있습니다.
# 예를 들어서 3명의 사람이 있다면 다음과 같이 6개의 방법이 있습니다.
#   [1, 2, 3]
#   [1, 3, 2]
#   [2, 1, 3]
#   [2, 3, 1]
#   [3, 1, 2]
#   [3, 2, 1]
# 사람의 수 n과, 자연수 k가 주어질 때, 사람을 나열 하는 방법을 사전 순으로 나열 했을 때,
# k번째 방법을 return하는 solution 함수를 완성해주세요.
import math

def solution(n, k):
    arr = [i for i in range(1, n + 1)]
    answer = []
    
    while arr:
        a = (k - 1) // math.factorial(n - 1)
        answer.append(arr.pop(a))
        k = k % math.factorial(n - 1)
        n -= 1

    return answer

# 하이노이의 탑
# 하노이 탑(Tower of Hanoi)은 퍼즐의 일종입니다. 세 개의 기둥과 이 기동에 꽂을 수 있는 크기가 다양한 원판들이 있고,
# 퍼즐을 시작하기 전에는 한 기둥에 원판들이 작은 것이 위에 있도록 순서대로 쌓여 있습니다.
# 게임의 목적은 다음 두 가지 조건을 만족시키면서, 한 기둥에 꽂힌 원판들을 그 순서 그대로 다른 기둥으로 옮겨서 다시 쌓는 것입니다.
#   한 번에 하나의 원판만 옮길 수 있습니다.
#   큰 원판이 작은 원판 위에 있어서는 안됩니다.
# 하노이 탑의 세 개의 기둥을 왼쪽 부터 1번, 2번, 3번이라고 하겠습니다.
# 1번에는 n개의 원판이 있고 이 n개의 원판을 3번 원판으로 최소 횟수로 옮기려고 합니다.
# 1번 기둥에 있는 원판의 개수 n이 매개변수로 주어질 때,
# n개의 원판을 3번 원판으로 최소로 옮기는 방법을 return하는 solution를 완성해주세요.
def solution(n):
    answer = []
    def hanoi(src, tgt, inter, n): # 인자 순서 넣어주는 게 좀 헷갈렸음.
        if n == 1:
            answer.append([src, tgt])
        else:
            hanoi(src,inter,tgt,n-1)
            hanoi(src,tgt,inter,1)
            hanoi(inter,tgt,src,n-1)
            
    hanoi(1,3,2,n)
    
    return answer

# 시소 짝궁
# 어느 공원 놀이터에는 시소가 하나 설치되어 있습니다.
# 이 시소는 중심으로부터 2(m), 3(m), 4(m) 거리의 지점에 좌석이 하나씩 있습니다.
# 이 시소를 두 명이 마주 보고 탄다고 할 때,
# 시소가 평형인 상태에서 각각에 의해 시소에 걸리는 토크의 크기가 서로 상쇄되어 완전한 균형을 이룰 수 있다면
# 그 두 사람을 시소 짝꿍이라고 합니다.
# 즉, 탑승한 사람의 무게와 시소 축과 좌석 간의 거리의 곱이 양쪽 다 같다면 시소 짝꿍이라고 할 수 있습니다.
# 사람들의 몸무게 목록 weights이 주어질 때,
# 시소 짝꿍이 몇 쌍 존재하는지 구하여 return 하도록 solution 함수를 완성해주세요.
def solution(weights):
    answer = 0
    dic = {}
    for i in weights:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1    
    for i in dic:
        if dic[i] > 1:
            answer += (dic[i] * (dic[i]-1)) / 2
        if i*2 in dic:
            answer += dic[i] * dic[2*i]
        if i*2/3 in dic:
            answer += dic[i] * dic[i*2/3]
        if i*3/4 in dic:
            answer += dic[i] * dic[i*3/4]
    return answer

# 개인정보 수집 유효기간
# 고객의 약관 동의를 얻어서 수집된 1~n번으로 분류되는 개인정보 n개가 있습니다.
# 약관 종류는 여러 가지 있으며 각 약관마다 개인정보 보관 유효기간이 정해져 있습니다.
# 당신은 각 개인정보가 어떤 약관으로 수집됐는지 알고 있습니다.
# 수집된 개인정보는 유효기간 전까지만 보관 가능하며, 유효기간이 지났다면 반드시 파기해야 합니다.
# 오늘 날짜를 의미하는 문자열 today, 약관의 유효기간을 담은 1차원 문자열 배열 terms와
# 수집된 개인정보의 정보를 담은 1차원 문자열 배열 privacies가 매개변수로 주어집니다.
# 이때 파기해야 할 개인정보의 번호를 오름차순으로 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
def solution(today, terms, privacies):
    d ={}
    answer = []
    today_lst = list(map(int,today.split('.'))) # 오늘 날짜 리스트로 변환
    
    for term in terms: # 약관종류와 개월수 딕셔너리로 저장
        n, m = term.split()
        d[n] = int(m)*28 # 한 달 일 수 곱해줌
    
    for i in range(len(privacies)):
        date, s = privacies[i].split()
        date_lst = list(map(int, date.split('.'))) # 수집일자 리스트로 변환
        year = (today_lst[0] - date_lst[0])*336 # 연도 차이에 일 수 곱해주기
        month = (today_lst[1] - date_lst[1])*28 # 달 수 차이에 일 수 곱해주기
        day = today_lst[2] - date_lst[2]
        total = year+month+day
        if d[s] <= total:
            answer.append(i+1)
    return answer

# 2개 이하로 다른 비트
# 양의 정수 x에 대한 함수 f(x)를 다음과 같이 정의합니다.

# x보다 크고 x와 비트가 1~2개 다른 수들 중에서 제일 작은 수
#   예를 들어,
#   f(2) = 3 입니다. 다음 표와 같이 2보다 큰 수들 중에서 비트가 다른 지점이 2개 이하이면서 제일 작은 수가 3이기 때문입니다.
# 정수들이 담긴 배열 numbers가 매개변수로 주어집니다. 
# numbers의 모든 수들에 대하여 각 수의 f 값을 배열에 차례대로 담아 return 하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = []
    for number in numbers:
        bin_number = list('0' + bin(number)[2:])
        idx = ''.join(bin_number).rfind('0')
        bin_number[idx] = '1'
        
        if number % 2 == 1:
            bin_number[idx+1] = '0'
        
        answer.append(int(''.join(bin_number), 2))
    return answer

# 124 나라의 숫자
# 124 나라가 있습니다.
# 124 나라에서는 10진법이 아닌 다음과 같은 자신들만의 규칙으로 수를 표현합니다.
#   124 나라에는 자연수만 존재합니다.
#   124 나라에는 모든 수를 표현할 때 1, 2, 4만 사용합니다.
# 자연수 n이 매개변수로 주어질 때,
# n을 124 나라에서 사용하는 숫자로 바꾼 값을 return 하도록 solution 함수를 완성해 주세요.
def solution(n):
    answer = ''
    while n:
        if n % 3:
            answer += str(n % 3)
            n //= 3
        else:
            answer += "4"
            n = n//3 - 1
    return answer[::-1]

# 가장 큰 정사각형 찾기
# 1와 0로 채워진 표(board)가 있습니다.
# 표 1칸은 1 x 1 의 정사각형으로 이루어져 있습니다.
# 표에서 1로 이루어진 가장 큰 정사각형을 찾아 넓이를 return 하는 solution 함수를 완성해 주세요.
# (단, 정사각형이란 축에 평행한 정사각형을 말합니다.)
def solution(board):
    answer = board[0][0]
    for i in range(1, len(board)):
        for j in range(1, len(board[i])):
            if board[i][j] == 1:
                board[i][j] = 1 + min(board[i-1][j-1], board[i-1][j], board[i][j-1])
                answer = max(answer, board[i][j])
    return answer**2

# 멀쩡한 사각형
# 가로 길이가 Wcm, 세로 길이가 Hcm인 직사각형 종이가 있습니다.
# 종이에는 가로, 세로 방향과 평행하게 격자 형태로 선이 그어져 있으며,
# 모든 격자칸은 1cm x 1cm 크기입니다. 이 종이를 격자 선을 따라 1cm × 1cm의 정사각형으로 잘라 사용할 예정이었는데,
# 누군가가 이 종이를 대각선 꼭지점 2개를 잇는 방향으로 잘라 놓았습니다.
# 그러므로 현재 직사각형 종이는 크기가 같은 직각삼각형 2개로 나누어진 상태입니다.
# 새로운 종이를 구할 수 없는 상태이기 때문에,
# 이 종이에서 원래 종이의 가로, 세로 방향과 평행하게 1cm × 1cm로 잘라 사용할 수 있는 만큼만 사용하기로 하였습니다.
# 가로의 길이 W와 세로의 길이 H가 주어질 때, 사용할 수 있는 정사각형의 개수를 구하는 solution 함수를 완성해 주세요.
import math
def solution(w,h):
    return w*h - (w+h-math.gcd(w,h))

# 과제 진행하기
# 과제를 받은 루는 다음과 같은 순서대로 과제를 하려고 계획을 세웠습니다.
#   과제는 시작하기로 한 시각이 되면 시작합니다.
#   새로운 과제를 시작할 시각이 되었을 때, 기존에 진행 중이던 과제가 있다면 진행 중이던 과제를 멈추고 새로운 과제를 시작합니다.
#   진행중이던 과제를 끝냈을 때, 잠시 멈춘 과제가 있다면, 멈춰둔 과제를 이어서 진행합니다.
#   만약, 과제를 끝낸 시각에 새로 시작해야 되는 과제와 잠시 멈춰둔 과제가 모두 있다면, 새로 시작해야 하는 과제부터 진행합니다.
#   멈춰둔 과제가 여러 개일 경우, 가장 최근에 멈춘 과제부터 시작합니다.
# 과제 계획을 담은 이차원 문자열 배열 plans가 매개변수로 주어질 때, 과제를 끝낸 순서대로 이름을 배열에 담아 return 하는 solution 함수를 완성해주세요.
def solution(plans):
    answer = []
    for i in range(len(plans)):
        h, m = map(int, plans[i][1].split(':'))
        st = h*60+m
        plans[i][1] = st
        plans[i][2] = int(plans[i][2])
        
    plans.sort(key=lambda x:x[1])
    stack = []
    for i in range(len(plans)):
        if i == len(plans)-1:
            stack.append(plans[i])
            break
        
        sub, st, t = plans[i]
        nsub, nst, nt = plans[i+1]
        if st + t <= nst:
            answer.append(sub)
            temp_time = nst - (st+t)
            
            while temp_time != 0 and stack:
                tsub, tst, tt = stack.pop()
                if temp_time >= tt:
                    answer.append(tsub)
                    temp_time -= tt
                else:
                    stack.append([tsub, tst, tt - temp_time])
                    temp_time = 0
            
        else:
            plans[i][2] = t - (nst - st)
            stack.append(plans[i])
        
    while stack:
        sub, st, tt = stack.pop()
        answer.append(sub)

    return answer

# 두 원 사이의 정수 쌍
# x축과 y축으로 이루어진 2차원 직교 좌표계에 중심이 원점인 서로 다른 크기의 원이 두 개 주어집니다.
# 반지름을 나타내는 두 정수 r1, r2가 매개변수로 주어질 때,
# 두 원 사이의 공간에 x좌표와 y좌표가 모두 정수인 점의 개수를 return하도록 solution 함수를 완성해주세요.
# ※ 각 원 위의 점도 포함하여 셉니다.
import math
def solution(r1, r2):
    answer = 0
    for i in range(1, r2+1):
        if i < r1 :
            s = math.ceil(math.sqrt((r1**2 - i**2)))
        else : 
            s = 0

        e = int(math.sqrt((r2**2 - i**2)))
        answer = answer + e - s + 1

    return answer*4

# 마법의 엘리베이터
# 마법의 엘리베이터에는 -1, +1, -10, +10, -100, +100 등과 같이
# 절댓값이 10c (c ≥ 0 인 정수) 형태인 정수들이 적힌 버튼이 있습니다.
# 마법의 엘리베이터의 버튼을 누르면 현재 층 수에 버튼에 적혀 있는 값을 더한 층으로 이동하게 됩니다.
# 단, 엘리베이터가 위치해 있는 층과 버튼의 값을 더한 결과가 0보다 작으면 엘리베이터는 움직이지 않습니다.
# 민수의 세계에서는 0층이 가장 아래층이며 엘리베이터는 현재 민수가 있는 층에 있습니다.
# 마법의 돌을 아끼기 위해 민수는 항상 최소한의 버튼을 눌러서 이동하려고 합니다.
# 민수가 어떤 층에서 엘리베이터를 타고 0층으로 내려가는데 필요한 마법의 돌의 최소 개수를 알고 싶습니다.
# 민수와 마법의 엘리베이터가 있는 층을 나타내는 정수 storey 가 주어졌을 때,
# 0층으로 가기 위해 필요한 마법의 돌의 최소값을 return 하도록 solution 함수를 완성하세요.
def solution(storey):
    answer = 0
    while storey:
        remainder = storey % 10
        # 6 ~ 9
        if remainder > 5:
            answer += (10 - remainder)
            storey += 10
        # 0 ~ 4
        elif remainder < 5:
            answer += remainder
        # 5
        else:
            if (storey // 10) % 10 > 4:
                storey += 10
            answer += remainder
        storey //= 10
    return answer

# 무인도 여행
# 메리는 여름을 맞아 무인도로 여행을 가기 위해 지도를 보고 있습니다.
# 지도에는 바다와 무인도들에 대한 정보가 표시돼 있습니다.
# 지도는 1 x 1크기의 사각형들로 이루어진 직사각형 격자 형태이며,
# 격자의 각 칸에는 'X' 또는 1에서 9 사이의 자연수가 적혀있습니다.
# 지도의 'X'는 바다를 나타내며, 숫자는 무인도를 나타냅니다.
# 이때, 상, 하, 좌, 우로 연결되는 땅들은 하나의 무인도를 이룹니다.
# 지도의 각 칸에 적힌 숫자는 식량을 나타내는데, 상, 하, 좌, 우로 연결되는 칸에
# 적힌 숫자를 모두 합한 값은 해당 무인도에서 최대 며칠동안 머물 수 있는지를 나타냅니다.
# 어떤 섬으로 놀러 갈지 못 정한 메리는 우선 각 섬에서 최대 며칠씩 머물 수 있는지 알아본 후
# 놀러갈 섬을 결정하려 합니다.
# 지도를 나타내는 문자열 배열 maps가 매개변수로 주어질 때,
# 각 섬에서 최대 며칠씩 머무를 수 있는지 배열에 오름차순으로 담아 return 하는 solution 함수를 완성해주세요.
# 만약 지낼 수 있는 무인도가 없다면 -1을 배열에 담아 return 해주세요.
import collections
def bfs(x,y,maps,visited):
    moves = {(-1,0),(0,1),(1,0),(0,-1)}
    
    q = collections.deque()
    q.append((x,y))
    
    visited[x][y] = 1 #visited
    
    days = int(maps[x][y])
    while q: 
        x,y = q.popleft()
        
        for tx, ty in moves:
            nx = x + tx
            ny = y + ty
            
            if 0<=nx<len(maps) and 0<=ny<len(maps[0]):
                if maps[nx][ny] != 'X' and visited[nx][ny] != 1:
                    q.append((nx,ny))
                    visited[nx][ny] = 1
                    days += int(maps[nx][ny]) 
    return days
def solution(maps):
    total = []
    visited = [[0 for _ in range(len(maps[0]))] for _ in range(len(maps))]
    for i in range(len(maps)):
        for j in range(len(maps[0])):
            if maps[i][j] != 'X' and visited[i][j] == 0:
                total.append(bfs(i,j,maps,visited))

    if total:
        return sorted(total)
    else: return [-1]

# 전력망을 둘로 나누기
# n개의 송전탑이 전선을 통해 하나의 트리 형태로 연결되어 있습니다.
# 당신은 이 전선들 중 하나를 끊어서 현재의 전력망 네트워크를 2개로 분할하려고 합니다.
# 이때, 두 전력망이 갖게 되는 송전탑의 개수를 최대한 비슷하게 맞추고자 합니다.
# 송전탑의 개수 n, 그리고 전선 정보 wires가 매개변수로 주어집니다.
# 전선들 중 하나를 끊어서 송전탑 개수가 가능한 비슷하도록 두 전력망으로 나누었을 때,
# 두 전력망이 가지고 있는 송전탑 개수의 차이(절대값)를 return 하도록 solution 함수를 완성해주세요.
def solution(n, wires):
    ans = n
    for sub in (wires[i+1:] + wires[:i] for i in range(len(wires))):
        s = set(sub[0])
        [s.update(v) for _ in sub for v in sub if set(v) & s]  # 집합연산자 & : 교집합 연산,   집합연산자 update : 여러데이터를 한번에 추가
        ans = min(ans, abs(2 * len(s) - n))
    return ans

# 쿼드압축 후 개수 세기
# 0과 1로 이루어진 2^ x 2^ 크기의 2차원 정수 배열 arr이 있습니다.
# 당신은 이 arr을 쿼드 트리와 같은 방식으로 압축하고자 합니다.
# 구체적인 방식은 다음과 같습니다.
#   당신이 압축하고자 하는 특정 영역을 S라고 정의합니다.
#   만약 S 내부에 있는 모든 수가 같은 값이라면, S를 해당 수 하나로 압축시킵니다.
#   그렇지 않다면, S를 정확히 4개의 균일한 정사각형 영역(입출력 예를 참고해주시기 바랍니다.)으로 쪼갠 뒤,
# 각 정사각형 영역에 대해 같은 방식의 압축을 시도합니다.
# arr이 매개변수로 주어집니다. 위와 같은 방식으로 arr을 압축했을 때,
# 배열에 최종적으로 남는 0의 개수와 1의 개수를 배열에 담아서 return 하도록 solution 함수를 완성해주세요.
def solution(arr):
    result=[0,0]
    length=len(arr)
    def compression(a,b,l):
        start=arr[a][b]
        for i in range(a,a+l):
            for j in range(b,b+l):
                if arr[i][j]!=start:
                    l=l//2
                    compression(a,b,l)
                    compression(a,b+l,l)
                    compression(a+l,b,l)
                    compression(a+l,b+l,l)
                    return
                
        result[start]+=1   
    compression(0,0,length)    
    return (result)

# 테이블 해시 함수
# 완호가 관리하는 어떤 데이터베이스의 한 테이블은 모두 정수 타입인 컬럼들로 이루어져 있습니다.
# 테이블은 2차원 행렬로 표현할 수 있으며 열은 컬럼을 나타내고, 행은 튜플을 나타냅니다.
# 첫 번째 컬럼은 기본키로서 모든 튜플에 대해 그 값이 중복되지 않도록 보장됩니다.
# 완호는 이 테이블에 대한 해시 함수를 다음과 같이 정의하였습니다.
#   해시 함수는 col, row_begin, row_end을 입력으로 받습니다.
#   테이블의 튜플을 col번째 컬럼의 값을 기준으로 오름차순 정렬을 하되,
#   만약 그 값이 동일하면 기본키인 첫 번째 컬럼의 값을 기준으로 내림차순 정렬합니다.
#   정렬된 데이터에서 S_i를 i 번째 행의 튜플에 대해 각 컬럼의 값을 i 로 나눈 나머지들의 합으로 정의합니다.
#   row_begin ≤ i ≤ row_end 인 모든 S_i를 누적하여 bitwise XOR 한 값을 해시 값으로서 반환합니다.
# 테이블의 데이터 data와 해시 함수에 대한 입력 col, row_begin, row_end이 주어졌을 때 
# 테이블의 해시 값을 return 하도록 solution 함수를 완성해주세요.
def solution(data, col, row_begin, row_end):
    data.sort(reverse=True)
    data.sort(key=lambda tup: tup[col - 1])
    mods = [sum(x % (r+1) for x in data[r]) for r in range(row_begin-1, row_end)]
    answer = mods[0]
    for x in mods[1:]:
        answer ^= x 
    return answer

# 점찍기
# 좌표평면을 좋아하는 진수는 x축과 y축이 직교하는 2차원 좌표평면에 점을 찍으면서 놀고 있습니다.
# 진수는 두 양의 정수 k, d가 주어질 때 다음과 같이 점을 찍으려 합니다.
#   원점(0, 0)으로부터 x축 방향으로 a*k(a = 0, 1, 2, 3 ...),
#   y축 방향으로 b*k(b = 0, 1, 2, 3 ...)만큼 떨어진 위치에 점을 찍습니다.
#   원점과 거리가 d를 넘는 위치에는 점을 찍지 않습니다.
# 예를 들어, k가 2, d가 4인 경우에는 (0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (4, 0) 위치에
# 점을 찍어 총 6개의 점을 찍습니다.
# 정수 k와 원점과의 거리를 나타내는 정수 d가 주어졌을 때,
# 점이 총 몇 개 찍히는지 return 하는 solution 함수를 완성하세요.
import math
def solution(k, d):
    answer = 0
    # x 기준으로 세기
    for x in range(0, d + 1, k):
        res_d = math.floor(math.sqrt(d*d - x*x))
        answer += (res_d // k) + 1
    return answer 

# N-Queen
# 가로, 세로 길이가 n인 정사각형으로된 체스판이 있습니다.
# 체스판 위의 n개의 퀸이 서로를 공격할 수 없도록 배치하고 싶습니다.
# 예를 들어서 n이 4인경우 다음과 같이 퀸을 배치하면 n개의 퀸은 서로를 한번에 공격 할 수 없습니다.
# 체스판의 가로 세로의 세로의 길이 n이 매개변수로 주어질 때,
# n개의 퀸이 조건에 만족 하도록 배치할 수 있는 방법의 수를 return하는 solution함수를 완성해주세요.
def solution(n):
	# 어태까지의 queen 위치 ls, 내가 두려는 위치 new
    def check(ls, new):
        for i in range(len(ls)):
        	# 같은 열에 퀸을 둔 적이 있거나, 대각 위치에 둔 적이 있다면, return False
            if new == ls[i] or (len(ls)-i) == abs(ls[i]-new):
                return False
        return True
    def dfs(n, ls):
    	# 끝 행까지 도달! return 1
        if len(ls) == n:
            return 1
        # 끝 행이 아니라면, 다음 줄을 다시 탐색
        cnt = 0
        for i in range(n):
            if check(ls, i):
                cnt += dfs(n, ls+[i])
        # 탐색 결과를 return
        return cnt
        
    return dfs(n, [])

# 조이스틱 
# 조이스틱으로 알파벳 이름을 완성하세요. 맨 처음엔 A로만 이루어져 있습니다.
# ex) 완성해야 하는 이름이 세 글자면 AAA, 네 글자면 AAAA
# 조이스틱을 각 방향으로 움직이면 아래와 같습니다.
#   ▲ - 다음 알파벳
#   ▼ - 이전 알파벳 (A에서 아래쪽으로 이동하면 Z로)
#   ◀ - 커서를 왼쪽으로 이동 (첫 번째 위치에서 왼쪽으로 이동하면 마지막 문자에 커서)
#   ▶ - 커서를 오른쪽으로 이동 (마지막 위치에서 오른쪽으로 이동하면 첫 번째 문자에 커서)
# 예를 들어 아래의 방법으로 "JAZ"를 만들 수 있습니다.
#   - 첫 번째 위치에서 조이스틱을 위로 9번 조작하여 J를 완성합니다.
#   - 조이스틱을 왼쪽으로 1번 조작하여 커서를 마지막 문자 위치로 이동시킵니다.
#   - 마지막 위치에서 조이스틱을 아래로 1번 조작하여 Z를 완성합니다.
# 따라서 11번 이동시켜 "JAZ"를 만들 수 있고, 이때가 최소 이동입니다.
# 만들고자 하는 이름 name이 매개변수로 주어질 때,
# 이름에 대해 조이스틱 조작 횟수의 최솟값을 return 하도록 solution 함수를 만드세요.
def solution(name):

	# 조이스틱 조작 횟수 
    answer = 0
    
    # 기본 최소 좌우이동 횟수는 길이 - 1
    min_move = len(name) - 1
    
    for i, char in enumerate(name):
    	# 해당 알파벳 변경 최솟값 추가
        answer += min(ord(char) - ord('A'), ord('Z') - ord(char) + 1)
        
        # 해당 알파벳 다음부터 연속된 A 문자열 찾기
        next = i + 1
        while next < len(name) and name[next] == 'A':
            next += 1
            
        # 기존, 연속된 A의 왼쪽시작 방식, 연속된 A의 오른쪽시작 방식 비교 및 갱신
        min_move = min([min_move, 2 *i + len(name) - next, i + 2 * (len(name) -next)])
        
    # 알파벳 변경(상하이동) 횟수에 좌우이동 횟수 추가
    answer += min_move
    return answer

# 신규 아이디 추천
# 카카오에 입사한 신입 개발자 네오는 "카카오계정개발팀"에 배치되어,
# 카카오 서비스에 가입하는 유저들의 아이디를 생성하는 업무를 담당하게 되었습니다.
# "네오"에게 주어진 첫 업무는 새로 가입하는 유저들이 카카오 아이디 규칙에 맞지 않는 아이디를 입력했을 때,
# 입력된 아이디와 유사하면서 규칙에 맞는 아이디를 추천해주는 프로그램을 개발하는 것입니다.
# 다음은 카카오 아이디의 규칙입니다.
#   아이디의 길이는 3자 이상 15자 이하여야 합니다.
#   아이디는 알파벳 소문자, 숫자, 빼기(-), 밑줄(_), 마침표(.) 문자만 사용할 수 있습니다.
#   단, 마침표(.)는 처음과 끝에 사용할 수 없으며 또한 연속으로 사용할 수 없습니다.
# 신규 유저가 입력한 아이디를 나타내는 new_id가 매개변수로 주어질 때,
# "네오"가 설계한 7단계의 처리 과정을 거친 후의 추천 아이디를 return 하도록 solution 함수를 완성해 주세요.
def solution(new_id):
    # 1단계
    new_id = new_id.lower()
    # 2단계
    answer = ''
    for word in new_id:
        if word.isalnum() or word in '-_.':
            answer += word
    # 3단계
    while '..' in answer:
        answer = answer.replace('..', '.')
    # 4단계
    answer = answer[1:] if answer[0] == '.' and len(answer) > 1 else answer
    answer = answer[:-1] if answer[-1] == '.' else answer
    # 5단계
    answer = 'a' if answer == '' else answer
    # 6단계
    if len(answer) >= 16:
        answer = answer[:15]
        if answer[-1] == '.':
            answer = answer[:-1]
    # 7단계
    if len(answer) <= 3:
        answer = answer + answer[-1] * (3-len(answer))
    return answer

# 삼각 달팽이
# 정수 n이 매개변수로 주어집니다.
# 다음 그림과 같이 밑변의 길이와 높이가 n인 삼각형에서 맨 위 꼭짓점부터 반시계 방향으로
# 달팽이 채우기를 진행한 후, 첫 행부터 마지막 행까지
# 모두 순서대로 합친 새로운 배열을 return 하도록 solution 함수를 완성해주세요.
def solution(n):
    triangle = [ [0] * n for _ in range(n) ]
    answer = []
    x, y = -1, 0
    num = 1
    for i in range(n):
        for j in range(i, n):
            # Down
            if i % 3 == 0:
                x += 1
            # Right
            elif i % 3 == 1:
                y += 1
            # Up
            elif i % 3 == 2:
                x -= 1
                y -= 1
            triangle[x][y] = num
            num += 1
    for i in range(n):
        for j in range(i+1):
            answer.append(triangle[i][j])
    return answer
    
# 메뉴 리뉴얼
# 레스토랑을 운영하던 스카피는 코로나19로 인한 불경기를 극복하고자 메뉴를 새로 구성하려고 고민하고 있습니다.
# 기존에는 단품으로만 제공하던 메뉴를 조합해서 코스요리 형태로 재구성해서 새로운 메뉴를 제공하기로 결정했습니다.
# 어떤 단품메뉴들을 조합해서 코스요리 메뉴로 구성하면 좋을 지 고민하던 "스카피"는 이전에 각 손님들이 주문할 때
# 가장 많이 함께 주문한 단품메뉴들을 코스요리 메뉴로 구성하기로 했습니다.
# 단, 코스요리 메뉴는 최소 2가지 이상의 단품메뉴로 구성하려고 합니다.
# 또한, 최소 2명 이상의 손님으로부터 주문된 단품메뉴 조합에 대해서만 코스요리 메뉴 후보에 포함하기로 했습니다.
# 각 손님들이 주문한 단품메뉴들이 문자열 형식으로 담긴 배열 orders,
# "스카피"가 추가하고 싶어하는 코스요리를 구성하는 단품메뉴들의 갯수가 담긴 배열 course가
# 매개변수로 주어질 때, "스카피"가 새로 추가하게 될 코스요리의 메뉴 구성을 문자열 형태로 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
from itertools import combinations
from collections import Counter
def solution(orders, course):
    answer = []
    for k in course:
        candidates = []
        for menu_li in orders:
            for li in combinations(menu_li, k):
                res = ''.join(sorted(li))
                candidates.append(res)
        sorted_candidates = Counter(candidates).most_common()
        answer += [menu for menu, cnt in sorted_candidates if cnt > 1 and cnt == sorted_candidates[0][1]]
    return sorted(answer)

# [3차] 파일명 정렬
# 저장소 서버에는 프로그램의 과거 버전을 모두 담고 있어, 이름 순으로 정렬된 파일 목록은 보기가 불편했다.
# 파일을 이름 순으로 정렬하면 나중에 만들어진 ver-10.zip이 ver-9.zip보다 먼저 표시되기 때문이다.
# 버전 번호 외에도 숫자가 포함된 파일 목록은 여러 면에서 관리하기 불편했다.
# 예컨대 파일 목록이 ["img12.png", "img10.png", "img2.png", "img1.png"]일 경우,
# 일반적인 정렬은 ["img1.png", "img10.png", "img12.png", "img2.png"] 순이 되지만,
# 숫자 순으로 정렬된 ["img1.png", "img2.png", "img10.png", img12.png"] 순이 훨씬 자연스럽다.
# 무지는 단순한 문자 코드 순이 아닌, 파일명에 포함된 숫자를 반영한 정렬 기능을 저장소 관리 프로그램에 구현하기로 했다.
# 소스 파일 저장소에 저장된 파일명은 100 글자 이내로, 영문 대소문자, 숫자, 공백(" "), 마침표("."), 빼기 부호("-")만으로 이루어져 있다.
# 파일명은 영문자로 시작하며, 숫자를 하나 이상 포함하고 있다.
# 파일명은 크게 HEAD, NUMBER, TAIL의 세 부분으로 구성된다.
#   HEAD는 숫자가 아닌 문자로 이루어져 있으며, 최소한 한 글자 이상이다.
#   NUMBER는 한 글자에서 최대 다섯 글자 사이의 연속된 숫자로 이루어져 있으며,
#       앞쪽에 0이 올 수 있다. 0부터 99999 사이의 숫자로, 00000이나 0101 등도 가능하다.
#   TAIL은 그 나머지 부분으로, 여기에는 숫자가 다시 나타날 수도 있으며, 아무 글자도 없을 수 있다.
# 파일명을 세 부분으로 나눈 후, 다음 기준에 따라 파일명을 정렬한다.
#   파일명은 우선 HEAD 부분을 기준으로 사전 순으로 정렬한다.
#       이때, 문자열 비교 시 대소문자 구분을 하지 않는다. MUZI와 muzi, MuZi는 정렬 시에 같은 순서로 취급된다.
#   파일명의 HEAD 부분이 대소문자 차이 외에는 같을 경우, NUMBER의 숫자 순으로 정렬한다.
#       9 < 10 < 0011 < 012 < 13 < 014 순으로 정렬된다. 숫자 앞의 0은 무시되며,
#       012와 12는 정렬 시에 같은 같은 값으로 처리된다.
#   두 파일의 HEAD 부분과, NUMBER의 숫자도 같을 경우,
#       원래 입력에 주어진 순서를 유지한다. MUZI01.zip과 muzi1.png가 입력으로 들어오면,
#       정렬 후에도 입력 시 주어진 두 파일의 순서가 바뀌어서는 안 된다.
import re
def solution(files):
    temp = [re.split(r"([0-9]+)", s) for s in files]
    
    sort = sorted(temp, key = lambda x: (x[0].lower(), int(x[1])))
    
    return [''.join(s) for s in sort]

# 배달
# N개의 마을로 이루어진 나라가 있습니다.
# 이 나라의 각 마을에는 1부터 N까지의 번호가 각각 하나씩 부여되어 있습니다.
# 각 마을은 양방향으로 통행할 수 있는 도로로 연결되어 있는데,
# 서로 다른 마을 간에 이동할 때는 이 도로를 지나야 합니다.
# 도로를 지날 때 걸리는 시간은 도로별로 다릅니다.
# 현재 1번 마을에 있는 음식점에서 각 마을로 음식 배달을 하려고 합니다.
# 각 마을로부터 음식 주문을 받으려고 하는데,
# N개의 마을 중에서 K 시간 이하로 배달이 가능한 마을에서만 주문을 받으려고 합니다.
# 마을의 개수 N, 각 마을을 연결하는 도로의 정보 road,
# 음식 배달이 가능한 시간 K가 매개변수로 주어질 때,
# 음식 주문을 받을 수 있는 마을의 개수를 return 하도록 solution 함수를 완성해주세요.
import heapq
def dijkstra(dist,adj):
    # 출발노드를 기준으로 각 노드들의 최소비용 탐색
    heap = []
    heapq.heappush(heap, [0,1])  # 거리,노드
    while heap:
        cost, node = heapq.heappop(heap)
        for c,n in adj[node]:
            if cost+c < dist[n]:
                dist[n] = cost+c
                heapq.heappush(heap, [cost+c,n])
def solution(N, road, K):
    dist = [float('inf')]*(N+1)  # dist 배열 만들고 최소거리 갱신
    dist[1] = 0  # 1번은 자기자신이니까 거리 0
    adj = [[] for _ in range(N+1)]  # 인접노드&거리 기록할 배열
    for r in road:
        adj[r[0]].append([r[2],r[1]])
        adj[r[1]].append([r[2],r[0]])
    dijkstra(dist,adj)
    return len([i for i in dist if i <=K])

# 미로 탈출
# 1 x 1 크기의 칸들로 이루어진 직사각형 격자 형태의 미로에서 탈출하려고 합니다.
# 각 칸은 통로 또는 벽으로 구성되어 있으며,
# 벽으로 된 칸은 지나갈 수 없고 통로로 된 칸으로만 이동할 수 있습니다.
# 통로들 중 한 칸에는 미로를 빠져나가는 문이 있는데, 이 문은 레버를 당겨서만 열 수 있습니다.
# 레버 또한 통로들 중 한 칸에 있습니다.
# 따라서, 출발 지점에서 먼저 레버가 있는 칸으로 이동하여 레버를 당긴 후 미로를 빠져나가는 문이 있는 칸으로 이동하면 됩니다.
# 이때 아직 레버를 당기지 않았더라도 출구가 있는 칸을 지나갈 수 있습니다.
# 미로에서 한 칸을 이동하는데 1초가 걸린다고 할 때, 최대한 빠르게 미로를 빠져나가는데 걸리는 시간을 구하려 합니다.
# 미로를 나타낸 문자열 배열 maps가 매개변수로 주어질 때,
# 미로를 탈출하는데 필요한 최소 시간을 return 하는 solution 함수를 완성해주세요.
# 만약, 탈출할 수 없다면 -1을 return 해주세요.
from collections import deque
def bfs(start, end, maps):
	# 탐색할 방향
    dy = [0, 1, -1, 0]
    dx = [1, 0, 0, -1]
    
    n = len(maps)       # 세로
    m = len(maps[0])    # 가로
    visited = [[False]*m for _ in range(n)]
    que = deque()
    flag = False
    
    # 초깃값 설정
    for i in range(n):
        for j in range(m):
        	# 출발하고자 하는 지점이라면 시작점의 좌표를 기록함
            if maps[i][j] == start:      
                que.append((i, j, 0))    
                # 별도의 cost 리스트를 만들지 않고 que에 바로 기록(0)
                visited[i][j] = True
                flag = True; break 
                # 시작 지점은 한 개만 존재하기 때문에 찾으면 바로 나옴
        if flag: break
                
    # BFS 알고리즘 수행 (핵심)
    while que:
        y, x, cost = que.popleft()
        
        if maps[y][x] == end:
            return cost
        
        for i in range(4):
            ny = y + dy[i]
            nx = x + dx[i]
            
            # maps 범위내에서 벽이 아니라면 지나갈 수 있음
            if 0<= ny <n and 0<= nx <m and maps[ny][nx] !='X':
                if not visited[ny][nx]:	# 아직 방문하지 않는 통로라면
                    que.append((ny, nx, cost+1))
                    visited[ny][nx] = True
                    
    return -1	# 탈출할 수 없다면
def solution(maps):
    path1 = bfs('S', 'L', maps)	# 시작 지점 --> 레버
    path2 = bfs('L', 'E', maps) # 레버 --> 출구
    
    # 둘다 -1 이 아니라면 탈출할 수 있음
    if path1 != -1 and path2 != -1:
        return path1 + path2
        
   	# 둘중 하나라도 -1 이면 탈출할 수 없음
    return -1

# 디펜스 게임
# 준호는 요즘 디펜스 게임에 푹 빠져 있습니다. 디펜스 게임은 준호가 보유한 병사 n명으로 연속되는 적의 공격을 순서대로 막는 게임입니다.
# 디펜스 게임은 다음과 같은 규칙으로 진행됩니다.
#   준호는 처음에 병사 n명을 가지고 있습니다.
#   매 라운드마다 enemy[i]마리의 적이 등장합니다.
#   남은 병사 중 enemy[i]명 만큼 소모하여 enemy[i]마리의 적을 막을 수 있습니다.
#       예를 들어 남은 병사가 7명이고, 적의 수가 2마리인 경우, 현재 라운드를 막으면 7 - 2 = 5명의 병사가 남습니다.
#       남은 병사의 수보다 현재 라운드의 적의 수가 더 많으면 게임이 종료됩니다.
#   게임에는 무적권이라는 스킬이 있으며, 무적권을 사용하면 병사의 소모없이 한 라운드의 공격을 막을 수 있습니다.
#   무적권은 최대 k번 사용할 수 있습니다.
# 준호는 무적권을 적절한 시기에 사용하여 최대한 많은 라운드를 진행하고 싶습니다.
# 준호가 처음 가지고 있는 병사의 수 n, 사용 가능한 무적권의 횟수 k, 매 라운드마다 공격해오는 적의 수가 순서대로 담긴 정수 배열 enemy가 매개변수로 주어집니다. 준호가 몇 라운드까지 막을 수 있는지 return 하도록 solution 함수를 완성해주세요.
from heapq import heappop, heappush
def solution(n, k, enemy):
    answer, sumEnemy = 0, 0
    heap = []
    
    for e in enemy:
        heappush(heap, -e)
        sumEnemy += e
        if sumEnemy > n:
            if k == 0: break
            sumEnemy += heappop(heap) 
            k -= 1
        answer += 1
    return answer

# 숫자 카드 나누기
# 철수와 영희는 선생님으로부터 숫자가 하나씩 적힌 카드들을 절반씩 나눠서 가진 후, 다음 두 조건 중 하나를 만족하는 가장 큰 양의 정수 a의 값을 구하려고 합니다.

# 철수가 가진 카드들에 적힌 모든 숫자를 나눌 수 있고 영희가 가진 카드들에 적힌 모든 숫자들 중 하나도 나눌 수 없는 양의 정수 a
# 영희가 가진 카드들에 적힌 모든 숫자를 나눌 수 있고, 철수가 가진 카드들에 적힌 모든 숫자들 중 하나도 나눌 수 없는 양의 정수 a
# 예를 들어, 카드들에 10, 5, 20, 17이 적혀 있는 경우에 대해 생각해 봅시다. 만약, 철수가 [10, 17]이 적힌 카드를 갖고, 영희가 [5, 20]이 적힌 카드를 갖는다면 두 조건 중 하나를 만족하는 양의 정수 a는 존재하지 않습니다. 하지만, 철수가 [10, 20]이 적힌 카드를 갖고, 영희가 [5, 17]이 적힌 카드를 갖는다면, 철수가 가진 카드들의 숫자는 모두 10으로 나눌 수 있고, 영희가 가진 카드들의 숫자는 모두 10으로 나눌 수 없습니다. 따라서 철수와 영희는 각각 [10, 20]이 적힌 카드, [5, 17]이 적힌 카드로 나눠 가졌다면 조건에 해당하는 양의 정수 a는 10이 됩니다.

# 철수가 가진 카드에 적힌 숫자들을 나타내는 정수 배열 arrayA와 영희가 가진 카드에 적힌 숫자들을 나타내는 정수 배열 arrayB가 주어졌을 때, 주어진 조건을 만족하는 가장 큰 양의 정수 a를 return하도록 solution 함수를 완성해 주세요. 만약, 조건을 만족하는 a가 없다면, 0을 return 해 주세요.
def solution(arrayA, arrayB):
    answer = 0
    
    # array의 첫 번째 값이 최대공약수로 가정 해 모든 요소와 비교하기 위해 아래처럼 초기화
    gcdA = arrayA[0]
    gcdB = arrayB[0]
    
    for n in arrayA[1:]:
        gcdA = gcd(n, gcdA)
        
    for n in arrayB[1:]:
        gcdB = gcd(n, gcdB)
        
    # 첫 번째 조건
    if notDiv(arrayA, gcdB):
        answer = max(answer, gcdB)
    
    # 두 번째 조건
    if notDiv(arrayB, gcdA):
        answer = max(answer, gcdA)
        
    return answer
 
# 최대공약수
def gcd(a, b):
    if a % b == 0:
        return b
    return gcd(b, a % b)
 
# 나누어떨어지는지
def notDiv(array, gcd):
    for n in array:
        if n % gcd == 0:
            return False
    return True

# 광물캐기
# 마인은 곡괭이로 광산에서 광석을 캐려고 합니다.
# 마인은 다이아몬드 곡괭이, 철 곡괭이, 돌 곡괭이를 각각 0개에서 5개까지 가지고 있으며,
# 곡괭이로 광물을 캘 때는 피로도가 소모됩니다. 각 곡괭이로 광물을 캘 때의 피로도는 아래 표와 같습니다.
# 예를 들어, 철 곡괭이는 다이아몬드를 캘 때 피로도 5가 소모되며,
# 철과 돌을 캘때는 피로도가 1씩 소모됩니다.
# 각 곡괭이는 종류에 상관없이 광물 5개를 캔 후에는 더 이상 사용할 수 없습니다.
# 마인은 다음과 같은 규칙을 지키면서 최소한의 피로도로 광물을 캐려고 합니다.
#   사용할 수 있는 곡괭이중 아무거나 하나를 선택해 광물을 캡니다.
#   한 번 사용하기 시작한 곡괭이는 사용할 수 없을 때까지 사용합니다.
#   광물은 주어진 순서대로만 캘 수 있습니다.
#   광산에 있는 모든 광물을 캐거나, 더 사용할 곡괭이가 없을 때까지 광물을 캡니다.
# 즉, 곡괭이를 하나 선택해서 광물 5개를 연속으로 캐고,
# 다음 곡괭이를 선택해서 광물 5개를 연속으로 캐는 과정을 반복하며,
# 더 사용할 곡괭이가 없거나 광산에 있는 모든 광물을 캘 때까지 과정을 반복하면 됩니다.
# 마인이 갖고 있는 곡괭이의 개수를 나타내는 정수 배열 picks와
# 광물들의 순서를 나타내는 문자열 배열 minerals가 매개변수로 주어질 때,
# 마인이 작업을 끝내기까지 필요한 최소한의 피로도를 return 하는 solution 함수를 완성해주세요.
def solution(picks, minerals):
    # 광물의 수가 (곡괭이의 수) x 5 보다 많다면,
    # 채굴 가능한 총 광물의 개수를 자원 (곡괭이의 수) x 5로 제한한다.
    if sum(picks) * 5 < len(minerals):
        minerals = minerals[:sum(picks) * 5]

    # 광물을 크기가 5인 청크로 분할하고 각 청크에 포함된 종류별 광물의 개수를 센다.
    # 광물의 개수를 기준으로 내림차순으로 정렬한다.
    counted = scan_minerals(minerals)

    # 정렬 방법에 따라 곡괭이의 개수를 줄여가며 피로도를 계산한다.
    answer = calculate_fatigue(counted, picks)
    return answer

def scan_minerals(minerals):
    i = 0
    counted = []
    flag = True
    while flag:
        target = []
        if i + 5 < len(minerals):
            target = minerals[i:i + 5]
        else:
            target = minerals[i:]
            flag = False
        dias, irons, stones = target.count('diamond'), target.count('iron'), target.count('stone')
        counted.append([dias, irons, stones])
        i += 5
    counted.sort(key=lambda _: (-_[0], -_[1]))
    return counted

def calculate_fatigue(counted, picks):
    result = 0
    for target in counted:
        if picks[0] > 0:
            picks[0] -= 1
            result += sum(target)
        elif picks[1] > 0:
            picks[1] -= 1
            result += target[0] * 5 + target[1] + target[2]
        elif picks[2] > 0:
            picks[2] -= 1
            result += target[0] * 25 + target[1] * 5 + target[2]
        else:
            break
    return result

# 요격 시스템
# A 나라가 B 나라를 침공하였습니다.
# B 나라의 대부분의 전략 자원은 아이기스 군사 기지에 집중되어 있기 때문에
# A 나라는 B 나라의 아이기스 군사 기지에 융단폭격을 가했습니다.
# A 나라의 공격에 대항하여 아이기스 군사 기지에서는 무수히 쏟아지는 폭격 미사일들을 요격하려고 합니다.
# 이곳에는 백발백중을 자랑하는 요격 시스템이 있지만 운용 비용이 상당하기 때문에 미사일을 최소로 사용해서 모든 폭격 미사일을 요격하려 합니다.
# A 나라와 B 나라가 싸우고 있는 이 세계는 2 차원 공간으로 이루어져 있습니다.
# A 나라가 발사한 폭격 미사일은 x 축에 평행한 직선 형태의 모양이며 개구간을 나타내는 정수 쌍 (s, e) 형태로 표현됩니다.
# B 나라는 특정 x 좌표에서 y 축에 수평이 되도록 미사일을 발사하며,
# 발사된 미사일은 해당 x 좌표에 걸쳐있는 모든 폭격 미사일을 관통하여 한 번에 요격할 수 있습니다.
# 단, 개구간 (s, e)로 표현되는 폭격 미사일은 s와 e에서 발사하는 요격 미사일로는 요격할 수 없습니다.
# 요격 미사일은 실수인 x 좌표에서도 발사할 수 있습니다.
# 각 폭격 미사일의 x 좌표 범위 목록 targets이 매개변수로 주어질 때,
# 모든 폭격 미사일을 요격하기 위해 필요한 요격 미사일 수의 최솟값을 return 하도록 solution 함수를 완성해 주세요.
def solution(targets):
    answer = 0
    targets.sort(key = lambda x: [x[1], x[0]])
    s = e = 0
    for target in targets:
        if target[0] >= e:
            answer += 1
            e = target[1]

    return answer 

# 두 큐 합 같게 만들기
# 길이가 같은 두 개의 큐가 주어집니다. 하나의 큐를 골라 원소를 추출(pop)하고,
# 추출된 원소를 다른 큐에 집어넣는(insert) 작업을 통해 각 큐의 원소 합이 같도록 만들려고 합니다.
# 이때 필요한 작업의 최소 횟수를 구하고자 합니다.
# 한 번의 pop과 한 번의 insert를 합쳐서 작업을 1회 수행한 것으로 간주합니다.
# 큐는 먼저 집어넣은 원소가 먼저 나오는 구조입니다.
# 이 문제에서는 큐를 배열로 표현하며,
# 원소가 배열 앞쪽에 있을수록 먼저 집어넣은 원소임을 의미합니다.
# 즉, pop을 하면 배열의 첫 번째 원소가 추출되며, insert를 하면 배열의 끝에 원소가 추가됩니다.
# 예를 들어 큐 [1, 2, 3, 4]가 주어졌을 때, pop을 하면 맨 앞에 있는 원소 1이 추출되어 [2, 3, 4]가 되며,
# 이어서 5를 insert하면 [2, 3, 4, 5]가 됩니다.
# 길이가 같은 두 개의 큐를 나타내는 정수 배열 queue1, queue2가 매개변수로 주어집니다.
# 각 큐의 원소 합을 같게 만들기 위해 필요한 작업의 최소 횟수를 return 하도록 solution 함수를 완성해주세요.
# 단, 어떤 방법으로도 각 큐의 원소 합을 같게 만들 수 없는 경우, -1을 return 해주세요.
from collections import deque
def solution(queue1, queue2):
    answer = 0
    
    q1 = deque(queue1)
    q2 = deque(queue2)
    
    sum1 = sum(queue1)
    sum2 = sum(queue2)
    
    # 홀수인 경우
    if (sum1 + sum2) % 2 != 0:
        return -1
    
    while True:
        if answer == 4 * len(queue1):
            return -1
        
        if sum1 > sum2:
            value = q1.popleft()
            q2.append(value)
            sum1 -= value
            sum2 += value
        elif sum1 < sum2:
            value = q2.popleft()
            q1.append(value)
            sum1 += value
            sum2 -= value
        else:
            return answer
        answer += 1

# 택배상자
# 영재는 택배상자를 트럭에 싣는 일을 합니다. 영재가 실어야 하는 택배상자는 크기가 모두 같으며
# 1번 상자부터 n번 상자까지 번호가 증가하는 순서대로 컨테이너 벨트에 일렬로 놓여 영재에게 전달됩니다.
# 컨테이너 벨트는 한 방향으로만 진행이 가능해서 벨트에 놓인 순서대로(1번 상자부터) 상자를 내릴 수 있습니다.
# 하지만 컨테이너 벨트에 놓인 순서대로 택배상자를 내려 바로 트럭에 싣게 되면 택배 기사님이 배달하는 순서와
# 택배상자가 실려 있는 순서가 맞지 않아 배달에 차질이 생깁니다.
# 따라서 택배 기사님이 미리 알려준 순서에 맞게 영재가 택배상자를 실어야 합니다.
# 만약 컨테이너 벨트의 맨 앞에 놓인 상자가 현재 트럭에 실어야 하는 순서가 아니라면 그 상자를 트럭에 실을 순서가 될 때까지 잠시 다른 곳에 보관해야 합니다.
# 하지만 고객의 물건을 함부로 땅에 둘 수 없어 보조 컨테이너 벨트를 추가로 설치하였습니다. 보조 컨테이너 벨트는 앞 뒤로 이동이 가능하지만 
# 입구 외에 다른 면이 막혀 있어서 맨 앞의 상자만 뺄 수 있습니다(즉, 가장 마지막에 보조 컨테이너 벨트에 보관한 상자부터 꺼내게 됩니다).
# 보조 컨테이너 벨트를 이용해도 기사님이 원하는 순서대로 상자를 싣지 못 한다면, 더 이상 상자를 싣지 않습니다.
#   예를 들어, 영재가 5개의 상자를 실어야 하며, 택배 기사님이 알려준 순서가
#   기존의 컨테이너 벨트에 네 번째, 세 번째, 첫 번째, 두 번째, 다섯 번째 놓인 택배상자 순서인 경우,
#   영재는 우선 첫 번째, 두 번째, 세 번째 상자를 보조 컨테이너 벨트에 보관합니다.
#   그 후 네 번째 상자를 트럭에 싣고 보조 컨테이너 벨트에서 세 번째 상자 빼서 트럭에싣습니다. 
#   다음으로 첫 번째 상자를 실어야 하지만 보조 컨테이너 벨트에서는 두 번째 상자를,
#   기존의 컨테이너 벨트에는 다섯 번째 상자를 꺼낼 수 있기 때문에 더이상의 상자는 실을 수 없습니다.
#   따라서 트럭에는 2개의 상자만 실리게 됩니다.
# 택배 기사님이 원하는 상자 순서를 나타내는 정수 배열 order가 주어졌을 때,
# 영재가 몇 개의 상자를 실을 수 있는지 return 하는 solution 함수를 완성하세요.
def solution(order):
    answer = 0
    tmp = [] # 임시 컨테이너 
    i=1 # [1,2,3,4,5] 기존 순서 유지 
    
    while i != len(order)+1:
        tmp.append(i)
        while tmp[-1] == order[answer]:
            answer += 1
            tmp.pop()
            if len(tmp) == 0:
                break
        i += 1
    return answer

# [1차] 뉴스 클러스터링
# 자카드 유사도는 집합 간의 유사도를 검사하는 여러 방법 중의 하나로 알려져 있다.
# 두 집합 A, B 사이의 자카드 유사도 J(A, B)는 두 집합의 교집합 크기를 두 집합의 합집합 크기로 나눈 값으로 정의된다.
#   예를 들어 집합 A = {1, 2, 3}, 집합 B = {2, 3, 4}라고 할 때,
#   교집합 A ∩ B = {2, 3}, 합집합 A ∪ B = {1, 2, 3, 4}이 되므로,
#   집합 A, B 사이의 자카드 유사도 J(A, B) = 2/4 = 0.5가 된다.
#   집합 A와 집합 B가 모두 공집합일 경우에는 나눗셈이 정의되지 않으니 따로 J(A, B) = 1로 정의한다.
#   자카드 유사도는 원소의 중복을 허용하는 다중집합에 대해서 확장할 수 있다. 다중집합 A는 원소 "1"을 3개 가지고 있고,
#   다중집합 B는 원소 "1"을 5개 가지고 있다고 하자.
#   이 다중집합의 교집합 A ∩ B는 원소 "1"을 min(3, 5)인 3개, 합집합 A ∪ B는 원소 "1"을 max(3, 5)인 5개 가지게 된다.
#   다중집합 A = {1, 1, 2, 2, 3}, 다중집합 B = {1, 2, 2, 4, 5}라고 하면,
#   교집합 A ∩ B = {1, 2, 2}, 합집합 A ∪ B = {1, 1, 2, 2, 3, 4, 5}가 되므로,
#   자카드 유사도 J(A, B) = 3/7, 약 0.42가 된다.
# 입력으로 들어온 두 문자열의 자카드 유사도를 출력한다.
# 유사도 값은 0에서 1 사이의 실수이므로,
# 이를 다루기 쉽도록 65536을 곱한 후에 소수점 아래를 버리고 정수부만 출력한다.
from collections import Counter

def solution(str1, str2):
    str1_low = str1.lower()
    str2_low = str2.lower()
    
    str1_lst = []
    str2_lst = []
    
    for i in range(len(str1_low)-1):
        if str1_low[i].isalpha() and str1_low[i+1].isalpha():
            str1_lst.append(str1_low[i] + str1_low[i+1])
    for j in range(len(str2_low)-1):
        if str2_low[j].isalpha() and str2_low[j+1].isalpha():
            str2_lst.append(str2_low[j] + str2_low[j+1])
            
    Counter1 = Counter(str1_lst)
    Counter2 = Counter(str2_lst)
    
    inter = list((Counter1 & Counter2).elements())
    union = list((Counter1 | Counter2).elements())
    
    if len(union) == 0 and len(inter) == 0:
        return 65536
    else:
        return int(len(inter) / len(union) * 65536)
    
# [3차] 압축
# 어피치는 여러 압축 알고리즘 중에서 성능이 좋고 구현이 간단한 LZW(Lempel–Ziv–Welch) 압축을 구현하기로 했다.
# LZW 압축은 1983년 발표된 알고리즘으로, 이미지 파일 포맷인 GIF 등 다양한 응용에서 사용되었다.
# LZW 압축은 다음 과정을 거친다.
#   길이가 1인 모든 단어를 포함하도록 사전을 초기화한다.
#   사전에서 현재 입력과 일치하는 가장 긴 문자열 w를 찾는다.
#   w에 해당하는 사전의 색인 번호를 출력하고, 입력에서 w를 제거한다.
#   입력에서 처리되지 않은 다음 글자가 남아있다면(c),
#       w+c에 해당하는 단어를 사전에 등록한다.
#   단계 2로 돌아간다.
# 압축 알고리즘이 영문 대문자만 처리한다고 할 때, 사전은 다음과 같이 초기화된다.
# 사전의 색인 번호는 정수값으로 주어지며, 1부터 시작한다고 하자.
def solution(msg):
    answer = []
    d = dict()
    for c in range(ord('A'), ord('Z') + 1):
        d[chr(c)] = c - ord('A') + 1
    idx = 27
    start, end = 0, 1
 
    while end < len(msg) + 1:
        s = msg[start:end]
        if s in d:
            end += 1
        else:
            answer.append(d[s[:-1]])
            d[s] = idx
            idx += 1
            start = end - 1
    answer.append(d[s])
    return answer

# [3차]n진수 게임
#튜브가 활동하는 코딩 동아리에서는 전통적으로 해오는 게임이 있다.
# 이 게임은 여러 사람이 둥글게 앉아서 숫자를 하나씩 차례대로 말하는 게임인데, 규칙은 다음과 같다.
#   숫자를 0부터 시작해서 차례대로 말한다.
#       첫 번째 사람은 0, 두 번째 사람은 1, … 열 번째 사람은 9를 말한다.
#   10 이상의 숫자부터는 한 자리씩 끊어서 말한다.
#       즉 열한 번째 사람은 10의 첫 자리인 1, 열두 번째 사람은 둘째 자리인 0을 말한다.
# 이렇게 게임을 진행할 경우,
#   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, …
# 순으로 숫자를 말하면 된다.
# 한편 코딩 동아리 일원들은 컴퓨터를 다루는 사람답게 이진수로 이 게임을 진행하기도 하는데, 이 경우에는
#   0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, …
# 순으로 숫자를 말하면 된다.
# 이진수로 진행하는 게임에 익숙해져 질려가던 사람들은 좀 더 난이도를 높이기 위해 이진법에서 십육진법까지 모든 진법으로 게임을 진행해보기로 했다.
# 숫자 게임이 익숙하지 않은 튜브는 게임에 져서 벌칙을 받는 굴욕을 피하기 위해,
# 자신이 말해야 하는 숫자를 스마트폰에 미리 출력해주는 프로그램을 만들려고 한다. 튜브의 프로그램을 구현하라.
def solution(n, t, m, p):
    #재귀함수 이용 - 10진수를 n진수로
    def convert(n, base):
        arr = "0123456789ABCDEF"
        q, r = divmod(n, base)
        if q == 0:
            return arr[r]
        else:
            return convert(q, base) + arr[r]
    answer = ''
    candidate = []
  # 모든 턴의 답
    for i in range(t*m):
        conv = convert(i, n)
        for c in conv:
            candidate.append(c)

    # 튜브의 답만 추출
    for i in range(p-1, t*m, m):
        answer += candidate[i]

    return answer

# 방문 길이
# 게임 캐릭터를 4가지 명령어를 통해 움직이려 합니다. 명령어는 다음과 같습니다.
#   U: 위쪽으로 한 칸 가기
#   D: 아래쪽으로 한 칸 가기
#   R: 오른쪽으로 한 칸 가기
#   L: 왼쪽으로 한 칸 가기
# 캐릭터는 좌표평면의 (0, 0) 위치에서 시작합니다.
# 좌표평면의 경계는 왼쪽 위(-5, 5), 왼쪽 아래(-5, -5), 오른쪽 위(5, 5), 오른쪽 아래(5, -5)로 이루어져 있습니다.
# 이때, 우리는 게임 캐릭터가 지나간 길 중 캐릭터가 처음 걸어본 길의 길이를 구하려고 합니다.
# 명령어가 매개변수 dirs로 주어질 때, 게임 캐릭터가 처음 걸어본 길의 길이를 구하여 return 하는 solution 함수를 완성해 주세요.
def solution(dirs):
    sets = set()
    y, x = 0, 0
    udrl = {'U': (1, 0), 'D': (-1, 0), 'R': (0, 1), 'L': (0, -1)}
    
    for d in dirs:
        dy, dx = udrl[d]
        ny = y + dy
        nx = x + dx
        if -5 <= ny <= 5 and -5 <= nx <= 5:
            sets.add(((y, x), (ny, nx)))
            sets.add(((ny, nx), (y, x)))
            y = ny
            x = nx
    return len(sets) // 2

# 오픈채팅방
# 신입사원인 김크루는 카카오톡 오픈 채팅방을 개설한 사람을 위해, 다양한 사람들이 들어오고,
# 나가는 것을 지켜볼 수 있는 관리자창을 만들기로 했다. 채팅방에 누군가 들어오면 다음 메시지가 출력된다.
#   "[닉네임]님이 들어왔습니다."
# 채팅방에서 누군가 나가면 다음 메시지가 출력된다.
#   "[닉네임]님이 나갔습니다."
# 채팅방에서 닉네임을 변경하는 방법은 다음과 같이 두 가지이다.
#   채팅방을 나간 후, 새로운 닉네임으로 다시 들어간다.
#   채팅방에서 닉네임을 변경한다.
# 닉네임을 변경할 때는 기존에 채팅방에 출력되어 있던 메시지의 닉네임도 전부 변경된다.
# 채팅방에 들어오고 나가거나, 닉네임을 변경한 기록이 담긴 문자열 배열 record가 매개변수로 주어질 때, 모든 기록이 처리된 후,
# 최종적으로 방을 개설한 사람이 보게 되는 메시지를 문자열 배열 형태로 return 하도록 solution 함수를 완성하라.
def solution(record):
    answer = []
    dic = {}
    
    for sentence in record:
        sentence_split = sentence.split()
        if len(sentence_split) == 3:
            dic[sentence_split[1]] = sentence_split[2]
            
    for sentence in record:
        sentence_split = sentence.split()
        if sentence_split[0] == 'Enter':
            answer.append('%s님이 들어왔습니다.' %dic[sentence_split[1]])
        elif sentence_split[0] == 'Leave':
            answer.append('%s님이 나갔습니다.' %dic[sentence_split[1]])
            
    return(answer)

# 롤케이크 자르기
# 철수는 롤케이크를 두 조각으로 잘라서 동생과 한 조각씩 나눠 먹으려고 합니다.
# 이 롤케이크에는 여러가지 토핑들이 일렬로 올려져 있습니다. 철수와 동생은 롤케이크를 공평하게 나눠먹으려 하는데,
# 그들은 롤케이크의 크기보다 롤케이크 위에 올려진 토핑들의 종류에 더 관심이 많습니다.
# 그래서 잘린 조각들의 크기와 올려진 토핑의 개수에 상관없이 각 조각에 동일한 가짓수의 토핑이 
# 올라가면 공평하게 롤케이크가 나누어진 것으로 생각합니다.
# 롤케이크에 올려진 토핑들의 번호를 저장한 정수 배열 topping이 매개변수로 주어질 때,
# 롤케이크를 공평하게 자르는 방법의 수를 return 하도록 solution 함수를 완성해주세요.
from collections import Counter
def solution(topping):
    dic = Counter(topping)
    set_dic = set()
    res = 0
    for i in topping:
        dic[i] -= 1
        set_dic.add(i)
        if dic[i] == 0:
            dic.pop(i)
        if len(dic) == len(set_dic):
            res += 1
    return res

# 프렌즈4블록
# 블라인드 공채를 통과한 신입 사원 라이언은 신규 게임 개발 업무를 맡게 되었다. 이번에 출시할 게임 제목은 "프렌즈4블록".
# 같은 모양의 카카오프렌즈 블록이 2×2 형태로 4개가 붙어있을 경우 사라지면서 점수를 얻는 게임이다.
# 만약 판이 위와 같이 주어질 경우, 라이언이 2×2로 배치된 7개 블록과 콘이 2×2로 배치된 4개 블록이 지워진다.
# 같은 블록은 여러 2×2에 포함될 수 있으며, 지워지는 조건에 만족하는 2×2 모양이 여러 개 있다면 한꺼번에 지워진다.
# TTTANT
# RRFACC
# RRRFCC
# TRRRAA
# TTMMMF
# TMMTTJ
# 각 문자는 라이언(R), 무지(M), 어피치(A), 프로도(F), 네오(N), 튜브(T), 제이지(J), 콘(C)을 의미한다
# 입력으로 블록의 첫 배치가 주어졌을 때, 지워지는 블록은 모두 몇 개인지 판단하는 프로그램을 제작하라.
def solution(m, n, board):
    for i in range(m):
        board[i] = list(board[i])
    
    cnt = 0
    rm = set()
    while True:
        # 보드를 순회하며 4블록이 된 곳의 좌표를 집합에 기록
        for i in range(m-1):
            for j in range(n-1):
                t = board[i][j]
                if t == []:
                    continue
                if board[i+1][j] == t and board[i][j+1] == t and board[i+1][j+1] == t:
                    rm.add((i,j));rm.add((i+1,j))
                    rm.add((i,j+1));rm.add((i+1,j+1))
        
        # 좌표가 존재한다면 집합의 길이만큼 세주고 블록을 지움 
        if rm:
            cnt += len(rm)
            for i,j in rm:
                board[i][j] = []
            rm = set()
        # 없다면 함수 종료
        else:
            return cnt
        
        # 블록을 위에서 아래로 당겨줌
        while True:
            moved = 0
            for i in range(m-1):
                for j in range(n):
                    if board[i][j] and board[i+1][j]==[]:
                        board[i+1][j] = board[i][j]
                        board[i][j] = []
                        moved = 1
            if moved == 0:
                break
            
# [3차] 방금그곡
# 네오는 자신이 기억한 멜로디를 가지고 방금그곡을 이용해 음악을 찾는다.
# 그런데 라디오 방송에서는 한 음악을 반복해서 재생할 때도 있어서 네오가 기억하고 있는 멜로디는 음악 끝부분과 처음 부분이 이어서 재생된 멜로디일 수도 있다.
# 반대로, 한 음악을 중간에 끊을 경우 원본 음악에는 네오가 기억한 멜로디가 들어있다 해도 그 곡이 네오가 들은 곡이 아닐 수도 있다.
# 그렇기 때문에 네오는 기억한 멜로디를 재생 시간과 제공된 악보를 직접 보면서 비교하려고 한다.
# 다음과 같은 가정을 할 때 네오가 찾으려는 음악의 제목을 구하여라.
#   방금그곡 서비스에서는 음악 제목, 재생이 시작되고 끝난 시각, 악보를 제공한다.
#   네오가 기억한 멜로디와 악보에 사용되는 음은 C, C#, D, D#, E, F, F#, G, G#, A, A#, B 12개이다.
#   각 음은 1분에 1개씩 재생된다. 음악은 반드시 처음부터 재생되며 음악 길이보다 재생된 시간이 길 때는 음악이 끊김 없이 처음부터 반복해서 재생된다.
#       음악 길이보다 재생된 시간이 짧을 때는 처음부터 재생 시간만큼만 재생된다.
#   음악이 00:00를 넘겨서까지 재생되는 일은 없다.
#   조건이 일치하는 음악이 여러 개일 때에는 라디오에서 재생된 시간이 제일 긴 음악 제목을 반환한다.
#       재생된 시간도 같을 경우 먼저 입력된 음악 제목을 반환한다.
#   조건이 일치하는 음악이 없을 때에는 “(None)”을 반환한다.
def change(music):
    if 'A#' in music:
        music = music.replace('A#', 'a')
    if 'F#' in music:
        music = music.replace('F#', 'f')
    if 'C#' in music:
        music = music.replace('C#', 'c')
    if 'G#' in music:
        music = music.replace('G#', 'g')
    if 'D#' in music:
        music = music.replace('D#', 'd')
    return music

def solution(m, musicinfos):
    answer = []
    index = 0  # 먼저 입력된 음악을 판단하기 위해 index 추가
    for info in musicinfos:
        index += 1
        music = info.split(',')
        start = music[0].split(':') # 시작 시간
        end = music[1].split(':')  # 종료 시간
        # 재생시간 계산
        time = (int(end[0])*60 + int(end[1])) - (int(start[0])*60 + int(start[1]))
        
        # 악보에 #이 붙은 음을 소문자로 변환
        changed = change(music[3])
        
        # 음악 길이
        a = len(changed)
        
        # 재생시간에 재생된 음
        b = changed * (time // a) + changed[:time%a]
        
        # 기억한 멜로디도 #을 제거
        m = change(m)
        
        # 기억한 멜로디가 재생된 음에 있다면 결과배열에 [시간, index, 제목]을 추가
        if m in b:
            answer.append([time, index, music[2]])
    
    # 결과배열이 비어있다면 "None" 리턴
    if not answer:
        return "(None)"
    # 결과배열의 크기가 1이라면 제목 리턴
    elif len(answer) == 1:
        return answer[0][2]
    # 결과 배열의 크기가 2보다 크다면 재생된 시간이 긴 음악, 먼저 입력된 음악순으로 정렬
    else:
        answer = sorted(answer, key=lambda x: (-x[0], x[1]))
        return answer[0][2] # 첫번째 제목을 리턴

# 괄호변환
# 문자열 p를 u, v로 분리하는 함수
def divide(p):
    open_p = 0
    close_p = 0

    for i in range(len(p)):
        if p[i] == '(':
            open_p += 1
        else:
            close_p += 1
        if open_p == close_p:
            return p[:i + 1], p[i + 1:]


# 문자열 u가 올바른 괄호 문자열인지 확인하는 함수
# '(' 와 ')' 로만 이루어진 문자열이 있을 경우, '(' 의 개수와 ')' 의 개수가 같다면 이를 균형잡힌 괄호 문자열이라고 부릅니다.
# 그리고 여기에 '('와 ')'의 괄호의 짝도 모두 맞을 경우에는 이를 올바른 괄호 문자열이라고 부릅니다.
#   예를 들어, "(()))("와 같은 문자열은 "균형잡힌 괄호 문자열" 이지만 "올바른 괄호 문자열"은 아닙니다.
#   반면에 "(())()"와 같은 문자열은 "균형잡힌 괄호 문자열" 이면서 동시에 "올바른 괄호 문자열" 입니다.
# '(' 와 ')' 로만 이루어진 문자열 w가 "균형잡힌 괄호 문자열" 이라면 다음과 같은 과정을 통해 "올바른 괄호 문자열"로 변환할 수 있습니다.
#   1. 입력이 빈 문자열인 경우, 빈 문자열을 반환합니다. 
#   2. 문자열 w를 두 "균형잡힌 괄호 문자열" u, v로 분리합니다. 단, u는 "균형잡힌 괄호 문자열"로 더 이상 분리할 수 없어야 하며, v는 빈 문자열이 될 수 있습니다. 
#   3. 문자열 u가 "올바른 괄호 문자열" 이라면 문자열 v에 대해 1단계부터 다시 수행합니다. 
#       3-1. 수행한 결과 문자열을 u에 이어 붙인 후 반환합니다. 
#   4. 문자열 u가 "올바른 괄호 문자열"이 아니라면 아래 과정을 수행합니다. 
#       4-1. 빈 문자열에 첫 번째 문자로 '('를 붙입니다. 
#       4-2. 문자열 v에 대해 1단계부터 재귀적으로 수행한 결과 문자열을 이어 붙입니다. 
#       4-3. ')'를 다시 붙입니다. 
#       4-4. u의 첫 번째와 마지막 문자를 제거하고, 나머지 문자열의 괄호 방향을 뒤집어서 뒤에 붙입니다. 
#       4-5. 생성된 문자열을 반환합니다.
# "균형잡힌 괄호 문자열" p가 매개변수로 주어질 때,
# 주어진 알고리즘을 수행해 "올바른 괄호 문자열"로 변환한 결과를 return 하도록 solution 함수를 완성해 주세요.
def check(u):
    stack = []
    for p in u:
        if p == '(':
            stack.append(p)
        else:
            if not stack:
                return False
            stack.pop()

    return True
def solution(p):
    # 과정 1
    if not p:
        return ""
    # 과정 2
    u, v = divide(p)
    # 과정 3
    if check(u):
        # 과정 3-1
        return u + solution(v)
    # 과정 4
    else:
        # 과정 4-1
        answer = '('
        # 과정 4-2
        answer += solution(v)
        # 과정 4-3
        answer += ')'
        # 과정 4-4
        for p in u[1:len(u) - 1]:
            if p == '(':
                answer += ')'
            else:
                answer += '('
        # 과정 4-5
        return answer
    
# 거리두기 확인하기
# 코로나 바이러스 감염 예방을 위해 응시자들은 거리를 둬서 대기를 해야하는데 개발 직군 면접인 만큼
# 아래와 같은 규칙으로 대기실에 거리를 두고 앉도록 안내하고 있습니다.
#   대기실은 5개이며, 각 대기실은 5x5 크기입니다.
#   거리두기를 위하여 응시자들 끼리는 맨해튼 거리1가 2 이하로 앉지 말아 주세요.
#   단 응시자가 앉아있는 자리 사이가 파티션으로 막혀 있을 경우에는 허용합니다.
# 5개의 대기실을 본 죠르디는 각 대기실에서 응시자들이 거리두기를 잘 기키고 있는지 알고 싶어졌습니다.
# 자리에 앉아있는 응시자들의 정보와 대기실 구조를 대기실별로 담은 2차원 문자열 배열 places가 매개변수로 주어집니다.
# 각 대기실별로 거리두기를 지키고 있으면 1을, 한 명이라도 지키지 않고 있으면 0을 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
def solution(places):
    answer = []
    #place를 하나씩 확인
    for p in places:
        #거리두기가 지켜지지 않음을 확인하면 바로 반복을 멈추기 위한 key
        key = False
        nowArr = []
        #이번 place를 nowArr에 담아줍니다.
        for n in p:
            nowArr.append(list(n))
        #이중 for문을 이용해 하나씩 확인합니다.
        for i in range(5):
            if key:
                break
            for j in range(5):
                if key:
                    break
                #사람을 찾아내면 판단을 시작합니다.
                if nowArr[i][j] == "P":  
                    if i+1<5:
                    	#만약 바로 아랫부분에 사람이 존재하면 break
                        if nowArr[i+1][j] == "P":
                            key = True
                            break
                        #만약 아랫부분이 빈테이블이고 그 다음에 바로 사람이 있다면 한칸 떨어져 있더라도 맨허튼 거리는 2이므로 break
                        if nowArr[i+1][j] == "O":
                            if i+2<5:
                                if nowArr[i+2][j] == "P":
                                    key = True
                                    break
                    #바로 오른쪽 부분에 사람이 존재하면 break    
                    if j+1<5:
                        if nowArr[i][j+1] == "P":
                            key = True
                            break
                         #만약 오른쪽 부분이 빈테이블이고 그 다음에 바로 사람이 있다면 한칸 떨어져 있더라도 맨허튼 거리는 2이므로 break   
                        if nowArr[i][j+1] == "O":
                            if j+2<5:
                                if nowArr[i][j+2] == "P":
                                    key = True
                                    break
                    #우측 아래 부분입니다.
                    if i+1<5 and j+1<5:
                    	#만약 우측 아래가 사람이고, 오른 쪽 또는 아랫부분중 하나라도 빈테이블이면 break
                        if nowArr[i+1][j+1] == "P" and (nowArr[i+1][j] == "O" or nowArr[i][j+1] == "O"):
                            key = True
                            break
                    
                    #좌측 아래부분입니다.
                    if i+1<5 and j-1>=0:
                    	#만약 좌측 아래가 사람이고, 왼쪽 또는 아랫부분중 하나라도 빈테이블이면 break
                        if nowArr[i+1][j-1] == "P" and (nowArr[i+1][j] == "O" or nowArr[i][j-1] == "O"):
                            key = True
                            break
        #위의 for문동안 key가 True로 변경되었다면 거리두가기 지켜지지 않은것 이므로 0을 answer에 추가
        if key:
            answer.append(0)
        else:
        #key가 false로 끝났다면 거리두기가 지켜졌으므로 1 추가
            answer.append(1)
    #끝!
    return answer

# 리코쳇 로봇
# 이 보드게임은 격자모양 게임판 위에서 말을 움직이는 게임으로,
# 시작 위치에서 목표 위치까지 최소 몇 번만에 도달할 수 있는지 말하는 게임입니다.
# 이 게임에서 말의 움직임은 상, 하, 좌, 우 4방향 중 하나를 선택해서
# 게임판 위의 장애물이나 맨 끝에 부딪힐 때까지 미끄러져 이동하는 것을 한 번의 이동으로 칩니다.
# 게임판의 상태를 나타내는 문자열 배열 board가 주어졌을 때,
# 말이 목표위치에 도달하는데 최소 몇 번 이동해야 하는지 return 하는 solution함수를 완성하세요.
# 만약 목표위치에 도달할 수 없다면 -1을 return 해주세요.
from collections import deque

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

N, M = 0, 0

def init(board) :
    global N, M
    N, M = len(board), len(board[0])
    for i in range(N) :
        for j in range(M) :
            if board[i][j] == 'R' :
                start = (j, i)
            if board[i][j] == 'G' :
                end = (j, i)
                
    return start, end

def move(board, x, y, k) :
    while -1 < x < M and -1 < y < N and board[y][x] != 'D':
        x, y = x + dx[k], y + dy[k]
    
    return x - dx[k], y - dy[k]

def bfs(board, start, end) :
    x, y = start
    q = deque([(x, y, 0)])
    visited = [[float('inf')]*M for _ in range(N)]
    visited[y][x] = 0
    
    while q :
        x, y, dist = q.popleft()
        if (x, y) == end :
            return dist
        
        for k in range(4) :
            ax, ay = move(board, x, y, k)
            if visited[ay][ax] > dist + 1 :
                visited[ay][ax] = dist + 1
                q.append((ax, ay, dist+1))

    return -1
    
def solution(board):
    start, end = init(board)
    result = bfs(board, start, end)

    return result

# 문자열 압축
# 
def solution(s):
    result=[]
    if len(s)==1:
        return 1
    for i in range(1, len(s)+1):
        b = ''
        cnt = 1
        tmp=s[:i]

        for j in range(i, len(s)+i, i):
            if tmp==s[j:i+j]:
                cnt+=1
            else:
                if cnt!=1:
                    b = b + str(cnt)+tmp
                else:
                    b = b + tmp
                    
                tmp=s[j:j+i]
                cnt = 1
        result.append(len(b))
    return min(result)