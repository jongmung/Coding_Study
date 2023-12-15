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