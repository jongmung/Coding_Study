# 이중우선순위큐
# 이중 우선순위 큐는 다음 연산을 할 수 있는 자료구조를 말합니다.
#   명령어	  수신 탑(높이)
#   I 숫자	 큐에 주어진 숫자를 삽입합니다.
#   D 1	    큐에서 최댓값을 삭제합니다.
#   D -1	큐에서 최솟값을 삭제합니다.
# 이중 우선순위 큐가 할 연산 operations가 매개변수로 주어질 때,
# 모든 연산을 처리한 후 큐가 비어있으면 [0,0] 비어있지 않으면 [최댓값, 최솟값]을 return 하도록 solution 함수를 구현해주세요.
import heapq
def solution(operations):
    heap = []

    for i in operations:
        op1, op2 = i.split()
        if op1 == 'I':
            heapq.heappush(heap, int(op2))
        elif heap:
            if op2 == "1":
                heap = heapq.nlargest(len(heap), heap)[1:]
                heapq.heapify(heap)
            elif op2 == '-1':
                heapq.heappop(heap)
    if heap:
        min_value = heap[0]
        return [heapq.nlargest(1, heap)[0], min_value]
    else:
        return [0, 0]
    
# 정수 삼각형
# 위와 같은 삼각형의 꼭대기에서 바닥까지 이어지는 경로 중, 거쳐간 숫자의 합이 가장 큰 경우를 찾아보려고 합니다.
# 아래 칸으로 이동할 때는 대각선 방향으로 한 칸 오른쪽 또는 왼쪽으로만 이동 가능합니다.
#   예를 들어 3에서는 그 아래칸의 8 또는 1로만 이동이 가능합니다.
# 삼각형의 정보가 담긴 배열 triangle이 매개변수로 주어질 때, 거쳐간 숫자의 최댓값을 return 하도록 solution 함수를 완성하세요.
# 아래에서 위로 올라가기
def solution(triangle):
    floor = len(triangle) - 1  # N층의 인덱스
    while floor > 0:  # N, N-1,...2, 1
        for i in range(floor):  # N번째 인덱스엔 0~N-> N+1개의 숫자가 있음
            # 바로 위층의 칸에 아래칸의 두개중 큰값을 더해줌
            triangle[floor-1][i] += max(triangle[floor][i], triangle[floor][i+1])
        floor -= 1  # 층하나 올라가기

    return triangle[0][0]
# 위에서 아래로 내려가기
def solution(triangle):
    answer = 0
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            if j == 0:	# 왼쪽 끝이면
                triangle[i][j] += triangle[i-1][0]  # 이전 층의 0번째 값 더하기
            elif j == len(triangle[i])-1:	# 오른쪽 끝이면
                triangle[i][j] += triangle[i-1][-1]	# 이전 층의 -1번째 값 더하기
            else:
                triangle[i][j] += max(triangle[i-1][j], triangle[i-1][j-1])

    return max(triangle[-1])

# 네트워크
# 네트워크란 컴퓨터 상호 간에 정보를 교환할 수 있도록 연결된 형태를 의미합니다.
# 예를 들어, 컴퓨터 A와 컴퓨터 B가 직접적으로 연결되어있고,
# 컴퓨터 B와 컴퓨터 C가 직접적으로 연결되어 있을 때 컴퓨터 A와 컴퓨터 C도 간접적으로 연결되어 정보를 교환할 수 있습니다.
# 따라서 컴퓨터 A, B, C는 모두 같은 네트워크 상에 있다고 할 수 있습니다.
# 컴퓨터의 개수 n, 연결에 대한 정보가 담긴 2차원 배열 computers가 매개변수로 주어질 때,
# 네트워크의 개수를 return 하도록 solution 함수를 작성하시오.
def solution(n, computers):            
    def DFS(i):
        visited[i] = 1
        for a in range(n):
            if computers[i][a] and not visited[a]:
                DFS(a)            
    answer = 0
    visited = [0 for i in range(len(computers))]
    for i in range(n):
        if not visited[i]:
            DFS(i)
            answer += 1
        
    return answer

# 단어 변환
# 두 개의 단어 begin, target과 단어의 집합 words가 있습니다. 아래와 같은 규칙을 이용하여 begin에서 target으로 변환하는 가장 짧은 변환 과정을 찾으려고 합니다.
#   1. 한 번에 한 개의 알파벳만 바꿀 수 있습니다.
#   2. words에 있는 단어로만 변환할 수 있습니다.
#       예를 들어 begin이 "hit", target가 "cog", words가 ["hot","dot","dog","lot","log","cog"]라면
#       "hit" -> "hot" -> "dot" -> "dog" -> "cog"와 같이 4단계를 거쳐 변환할 수 있습니다.
# 두 개의 단어 begin, target과 단어의 집합 words가 매개변수로 주어질 때,
# 최소 몇 단계의 과정을 거쳐 begin을 target으로 변환할 수 있는지 return 하도록 solution 함수를 작성해주세요.
from collections import deque
def solution(begin, target, words):
    if target not in words : 
        return  0
    return bfs(begin, target, words)
#최소 단계를 찾아야 하므로 bfs
def bfs(begin, target, words):
    queue = deque()
    queue.append([begin, 0]) #시작 단어와 단계 0으로 초기화
    while queue:
        now, step = queue.popleft()
        if now == target:
            return step
        #단어를 모두 체크하면서, 해당 단어가 변경될 수 있는지 체크
        for word in words:
            count = 0
            for i in range(len(now)): #단어의 길이만큼 반복하여
                if now[i] != word[i]: #단어에 알파벳 한개씩 체크하기
                    count += 1
            if count == 1: 
                queue.append([word, step+1])
                
# 등굣길 
# 계속되는 폭우로 일부 지역이 물에 잠겼습니다. 물에 잠기지 않은 지역을 통해 학교를 가려고 합니다.
# 집에서 학교까지 가는 길은 m x n 크기의 격자모양으로 나타낼 수 있습니다.
# 가장 왼쪽 위, 즉 집이 있는 곳의 좌표는 (1, 1)로 나타내고 가장 오른쪽 아래, 즉 학교가 있는 곳의 좌표는 (m, n)으로 나타냅니다.
# 격자의 크기 m, n과 물이 잠긴 지역의 좌표를 담은 2차원 배열 puddles이 매개변수로 주어집니다.
# 오른쪽과 아래쪽으로만 움직여 집에서 학교까지 갈 수 있는 최단경로의 개수를 1,000,000,007로 나눈 나머지를 return 하도록 solution 함수를 작성해주세요.
def solution(m, n, puddles):
    puddles = [[q,p] for [p,q] in puddles]      # 미리 puddles 좌표 거꾸로
    dp = [[0] * (m + 1) for i in range(n + 1)]  # dp 초기화
    dp[1][1] = 1           # 집의 위치(시작위치)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if i == 1 and j == 1: continue 
            if [i, j] in puddles:    # 웅덩이 위치의 경우 값을 0으로
                dp[i][j] = 0
            else:                    # 현재 칸은 왼쪽 칸, 위 칸의 합산!
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % 1000000007
    return dp[n][m]

# 최고의 집합
# 자연수 n 개로 이루어진 중복 집합(multi set, 편의상 이후에는 "집합"으로 통칭) 중에 다음 두 조건을 만족하는 집합을 최고의 집합이라고 합니다.
# 각 원소의 합이 S가 되는 수의 집합
# 위 조건을 만족하면서 각 원소의 곱 이 최대가 되는 집합
#   예를 들어서 자연수 2개로 이루어진 집합 중 합이 9가 되는 집합은 다음과 같이 4개가 있습니다.
#       { 1, 8 }, { 2, 7 }, { 3, 6 }, { 4, 5 }
#       그중 각 원소의 곱이 최대인 { 4, 5 }가 최고의 집합입니다.
# 집합의 원소의 개수 n과 모든 원소들의 합 s가 매개변수로 주어질 때, 최고의 집합을 return 하는 solution 함수를 완성해주세요.
def solution(n, s):
    if n > s:
        return [-1]
    p, q = divmod(s, n)
    answer = [p] * n
    for i in range(q):
        answer[i] += 1
    return sorted(answer)

# 숫자게임
# xx 회사의 2xN명의 사원들은 N명씩 두 팀으로 나눠 숫자 게임을 하려고 합니다.
# 두 개의 팀을 각각 A팀과 B팀이라고 하겠습니다. 숫자 게임의 규칙은 다음과 같습니다.
#   먼저 모든 사원이 무작위로 자연수를 하나씩 부여받습니다.
#   각 사원은 딱 한 번씩 경기를 합니다.
#   각 경기당 A팀에서 한 사원이, B팀에서 한 사원이 나와 서로의 수를 공개합니다.
#       그때 숫자가 큰 쪽이 승리하게 되고, 승리한 사원이 속한 팀은 승점을 1점 얻게 됩니다.
#   만약 숫자가 같다면 누구도 승점을 얻지 않습니다.
# 전체 사원들은 우선 무작위로 자연수를 하나씩 부여받았습니다.
# 그다음 A팀은 빠르게 출전순서를 정했고 자신들의 출전 순서를 B팀에게 공개해버렸습니다.
# B팀은 그것을 보고 자신들의 최종 승점을 가장 높이는 방법으로 팀원들의 출전 순서를 정했습니다. 이때의 B팀이 얻는 승점을 구해주세요.
# A 팀원들이 부여받은 수가 출전 순서대로 나열되어있는 배열 A와
# i번째 원소가 B팀의 i번 팀원이 부여받은 수를 의미하는 배열 B가 주어질 때,
# B 팀원들이 얻을 수 있는 최대 승점을 return 하도록 solution 함수를 완성해주세요.
def solution(A, B):
    answer = 0
    A.sort(reverse = True)
    B.sort(reverse = True)
    for a in A:
        if a >= B[0]:
            continue
        else:
            answer += 1
            del B[0]
    return answer

# 단속카메라
# 고속도로를 이동하는 모든 차량이 고속도로를 이용하면서 단속용 카메라를 한 번은 만나도록 카메라를 설치하려고 합니다.
# 고속도로를 이동하는 차량의 경로 routes가 매개변수로 주어질 때,
# 모든 차량이 한 번은 단속용 카메라를 만나도록 하려면 최소 몇 대의 카메라를 설치해야 하는지를 return 하도록 solution 함수를 완성하세요.
def solution(routes):
	# 진출지점에 대해서 오름차순 정렬
    routes.sort(key=lambda x: x[1])
    # 기준은 제한사항 참조
    key = -30001
    # 필요한 카메라 수
    cnt = 0
    for route in routes:
    	# 기준(카메라)보다 진입지점이 뒤에 있으면
        if route[0] > key:
        	# 단속이 안되기에 카메라 하나 더 필요
            cnt += 1
            # 새로운 기준은 해당 경로의 진출지점(맨끝)
            key = route[1]
            
    return cnt

# 베스트 앨범
# 스트리밍 사이트에서 장르 별로 가장 많이 재생된 노래를 두 개씩 모아 베스트 앨범을 출시하려 합니다. 노래는 고유 번호로 구분하며, 노래를 수록하는 기준은 다음과 같습니다.
# 속한 노래가 많이 재생된 장르를 먼저 수록합니다.
# 장르 내에서 많이 재생된 노래를 먼저 수록합니다.
# 장르 내에서 재생 횟수가 같은 노래 중에서는 고유 번호가 낮은 노래를 먼저 수록합니다.
# 노래의 장르를 나타내는 문자열 배열 genres와 노래별 재생 횟수를 나타내는 정수 배열 plays가 주어질 때,
# 베스트 앨범에 들어갈 노래의 고유 번호를 순서대로 return 하도록 solution 함수를 완성하세요.
def solution(genres, plays):
    answer = []
    dic1 = {}
    dic2 = {}
    for i, (g, p) in enumerate(zip(genres, plays)):
        if g not in dic1:
            dic1[g] = [(i, p)]
        else:
            dic1[g].append((i, p))

        if g not in dic2:
            dic2[g] = p
        else:
            dic2[g] += p
    for (k, v) in sorted(dic2.items(), key=lambda x:x[1], reverse=True):
        for (i, p) in sorted(dic1[k], key=lambda x:x[1], reverse=True)[:2]:
            answer.append(i)
    return answer

# 야근지수
# 회사원 Demi는 가끔은 야근을 하는데요, 야근을 하면 야근 피로도가 쌓입니다.
# 야근 피로도는 야근을 시작한 시점에서 남은 일의 작업량을 제곱하여 더한 값입니다.
# Demi는 N시간 동안 야근 피로도를 최소화하도록 일할 겁니다.
# Demi가 1시간 동안 작업량 1만큼을 처리할 수 있다고 할 때,
# 퇴근까지 남은 N 시간과 각 일에 대한 작업량 works에 대해
# 야근 피로도를 최소화한 값을 리턴하는 함수 solution을 완성해주세요.
import heapq
def solution(n, works):
    if sum(works) <= n:
        return 0
    # 최대힙으로 만듬
    works = [-w for w in works]
    heapq.heapify(works)
    
    # n번 만큼만 works내 배열의 값을 하향평준화
    while n>0:
        val = heapq.heappop(works)
        val += 1
        heapq.heappush(works,val)
        n-=1
    
    answer = [ w**2 for w in works ]
    
    return sum(answer)