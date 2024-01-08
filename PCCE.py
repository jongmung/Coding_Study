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