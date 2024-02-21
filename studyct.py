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
