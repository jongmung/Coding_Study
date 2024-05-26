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

# 수식 최대화
# 해커톤 대회에 참가하는 모든 참가자들에게는 숫자들과 3가지의 연산문자(+, -, *) 만으로 이루어진 연산 수식이 전달되며,
# 참가자의 미션은 전달받은 수식에 포함된 연산자의 우선순위를 자유롭게 재정의하여 만들 수 있는 가장 큰 숫자를 제출하는 것입니다.
# 단, 연산자의 우선순위를 새로 정의할 때, 같은 순위의 연산자는 없어야 합니다.
# 즉, + > - > * 또는 - > * > + 등과 같이 연산자 우선순위를 정의할 수 있으나
# +,* > - 또는 * > +,-처럼 2개 이상의 연산자가 동일한 순위를 가지도록 연산자 우선순위를 정의할 수는 없습니다.
# 수식에 포함된 연산자가 2개라면 정의할 수 있는 연산자 우선순위 조합은 2! = 2가지이며, 연산자가 3개라면 3! = 6가지 조합이 가능합니다.
# 참가자에게 주어진 연산 수식이 담긴 문자열 expression이 매개변수로 주어질 때,
# 우승 시 받을 수 있는 가장 큰 상금 금액을 return 하도록 solution 함수를 완성해주세요.
from re import split
from itertools import permutations

def solution(expression):
    # 연산자 우선 순위가 6종류밖에 안 되고, expression 길이도 짧습니다.
    # brute-force로 해결합니다.
    values = []
    
    for priority in permutations(['*', '+', '-'], 3):
        # 우선 연산자와 피연산자를 저장해두고,
        operands = list(map(int, split('[\*\+\-]', expression)))
        operators = [c for c in expression if c in '*+-']
        
        # 우선순위대로 연산을 수행합니다.
        for op in priority:
            
            while op in operators:
                # i번째 연산자는 i번째 피연산자와 i+1번째 피연산자에 대한 연산을 수행합니다.
                i = operators.index(op)
                
                if op == '*':
                    v = operands[i] * operands[i+1]
                elif op == '+':
                    v = operands[i] + operands[i+1]
                else:
                    v = operands[i] - operands[i+1]
                
                # 피연산자 리스트를 업데이트 합니다.
                operands[i] = v
                operands.pop(i+1)
                # 연산자 리스트를 업데이트 합니다.
                operators.pop(i)
        
        values.append(operands[0])
        
    return max(abs(v) for v in values)

# 가장 먼 노드
# n개의 노드가 있는 그래프가 있습니다. 각 노드는 1부터 n까지 번호가 적혀있습니다.
# 1번 노드에서 가장 멀리 떨어진 노드의 갯수를 구하려고 합니다.
# 가장 멀리 떨어진 노드란 최단경로로 이동했을 때 간선의 개수가 가장 많은 노드들을 의미합니다.
# 노드의 개수 n, 간선에 대한 정보가 담긴 2차원 배열 vertex가 매개변수로 주어질 때,
# 1번 노드로부터 가장 멀리 떨어진 노드가 몇 개인지를 return 하도록 solution 함수를 작성해주세요.
from collections import deque 
def solution(n, edge):
    graph = [[] for _ in range(n + 1)]
    visited = [0] * (n + 1)    
    
    for eg in edge:
        a, b = eg[0], eg[1]
        graph[a].append(b)
        graph[b].append(a)
    
    q = deque() 
    q.append(1)
    visited[1] = 1 
    while q:
        x = q.popleft() 
        for i in graph[x]:
            if not visited[i]:
                visited[i] = visited[x] + 1 
                q.append(i)
    
    max_value = max(visited)
    answer = visited.count(max_value)
    
    return answer

# 여행경로
# 주어진 항공권을 모두 이용하여 여행경로를 짜려고 합니다. 항상 "ICN" 공항에서 출발합니다.
# 항공권 정보가 담긴 2차원 배열 tickets가 매개변수로 주어질 때,
# 방문하는 공항 경로를 배열에 담아 return 하도록 solution 함수를 작성해주세요.
def solution(tickets):
    answer = []
    visited = [False]*len(tickets)
    def dfs(airport, path):
        if len(path) == len(tickets)+1:
            answer.append(path)
            return
        
        for idx, ticket in enumerate(tickets):
            if airport == ticket[0] and visited[idx] == False:
                visited[idx] = True
                dfs(ticket[1], path+[ticket[1]])
                visited[idx] = False
    
    dfs("ICN", ["ICN"])
    answer.sort()
    return answer[0]

# 입국심사
# n명이 입국심사를 위해 줄을 서서 기다리고 있습니다. 각 입국심사대에 있는 심사관마다 심사하는데 걸리는 시간은 다릅니다.
# 처음에 모든 심사대는 비어있습니다. 한 심사대에서는 동시에 한 명만 심사를 할 수 있습니다.
# 가장 앞에 서 있는 사람은 비어 있는 심사대로 가서 심사를 받을 수 있습니다. 하지만 더 빨리 끝나는 심사대가 있으면 기다렸다가 그곳으로 가서 심사를 받을 수도 있습니다.
# 모든 사람이 심사를 받는데 걸리는 시간을 최소로 하고 싶습니다.
# 입국심사를 기다리는 사람 수 n, 각 심사관이 한 명을 심사하는데 걸리는 시간이 담긴 배열 times가 매개변수로 주어질 때,
# 모든 사람이 심사를 받는데 걸리는 시간의 최솟값을 return 하도록 solution 함수를 작성해주세요.
def solution(n, times):
    answer = 0
    # right는 가장 비효율적으로 심사했을 때 걸리는 시간
    # 가장 긴 심사시간이 소요되는 심사관에게 n 명 모두 심사받는 경우이다.
    left, right = 1, max(times) * n
    while left <= right:
        mid = (left+ right) // 2
        people = 0
        for time in times:
            # people 은 모든 심사관들이 mid분 동안 심사한 사람의 수
            people += mid // time
            # 모든 심사관을 거치지 않아도 mid분 동안 n명 이상의 심사를 할 수 있다면 반복문을 나간다.
            if people >= n:
                break
        
        # 심사한 사람의 수가 심사 받아야할 사람의 수(n)보다 많거나 같은 경우
        if people >= n:
            answer = mid
            right = mid - 1
        # 심사한 사람의 수가 심사 받아야할 사람의 수(n)보다 적은 경우
        elif people < n:
            left = mid + 1
            
    return answer

# 섬 연결하기
# n개의 섬 사이에 다리를 건설하는 비용(costs)이 주어질 때,
# 최소의 비용으로 모든 섬이 서로 통행 가능하도록 만들 때 필요한 최소 비용을 return 하도록 solution을 완성하세요.
# 다리를 여러 번 건너더라도, 도달할 수만 있으면 통행 가능하다고 봅니다.
# 예를 들어 A 섬과 B 섬 사이에 다리가 있고, B 섬과 C 섬 사이에 다리가 있으면 A 섬과 C 섬은 서로 통행 가능합니다.
def solution(n, costs):
    answer = 0
    costs.sort(key = lambda x: x[2]) 
    link = set([costs[0][0]])

    # Kruskal 알고리즘으로 최소 비용 구하기
    while len(link) != n:
        for v in costs:
            if v[0] in link and v[1] in link:
                continue
            if v[0] in link or v[1] in link:
                link.update([v[0], v[1]])
                answer += v[2]
                break
                
    return answer

# 가장 긴 팰린드롬
# 앞뒤를 뒤집어도 똑같은 문자열을 팰린드롬(palindrome)이라고 합니다.
# 문자열 s가 주어질 때, s의 부분문자열(Substring)중 가장 긴 팰린드롬의 길이를 return 하는 solution 함수를 완성해 주세요.
# 예를들면, 문자열 s가 "abcdcba"이면 7을 return하고 "abacde"이면 3을 return합니다.
def isPalindrome(x):
    if x==x[::-1]:
        return True
def solution(s):
    MAX=0
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if isPalindrome(s[i:j]):
                if MAX<len(s[i:j]):
                    MAX=len(s[i:j])
    return MAX

# 징검다리 건너기
# 카카오 초등학교의 "니니즈 친구들"이 "라이언" 선생님과 함께 가을 소풍을 가는 중에 징검다리가 있는 개울을 만나서 건너편으로 건너려고 합니다.
# "라이언" 선생님은 "니니즈 친구들"이 무사히 징검다리를 건널 수 있도록 다음과 같이 규칙을 만들었습니다.
#   징검다리는 일렬로 놓여 있고 각 징검다리의 디딤돌에는 모두 숫자가 적혀 있으며 디딤돌의 숫자는 한 번 밟을 때마다 1씩 줄어듭니다.
#   디딤돌의 숫자가 0이 되면 더 이상 밟을 수 없으며 이때는 그 다음 디딤돌로 한번에 여러 칸을 건너 뛸 수 있습니다.
#   단, 다음으로 밟을 수 있는 디딤돌이 여러 개인 경우 무조건 가장 가까운 디딤돌로만 건너뛸 수 있습니다.
# "니니즈 친구들"은 개울의 왼쪽에 있으며, 개울의 오른쪽 건너편에 도착해야 징검다리를 건넌 것으로 인정합니다.
# "니니즈 친구들"은 한 번에 한 명씩 징검다리를 건너야 하며, 한 친구가 징검다리를 모두 건넌 후에 그 다음 친구가 건너기 시작합니다.
# 디딤돌에 적힌 숫자가 순서대로 담긴 배열 stones와 한 번에 건너뛸 수 있는 디딤돌의 최대 칸수 k가 매개변수로 주어질 때,
# 최대 몇 명까지 징검다리를 건널 수 있는지 return 하도록 solution 함수를 완성해주세요.
def solution(stones, k):
    answer = 0
    s = 1
    # 최대 밟을 수 있는 횟수
    e = max(stones)
    while s <= e:
        # mid명이 건넌 후 다음 사람이 건널 수 있는가?
        mid = (s+e)//2
        # 몇칸씩 건너는지
        l = []
        cnt = 1
        for i in range(len(stones)):
        	# mid명이 건넜으니 
            # 밟을 수 없는 돌 -> 건너뛰기
            if mid >= stones[i]:
                cnt += 1
            # 밟을 수 있는 돌 -> 몇개 건너뛰었는지 저장
            else:
                l.append(cnt)
                cnt = 1
        # 마지막 건너뛰기
        l.append(cnt)
        # 건너뛸 수 있는 최대보다 크면 다 못 건넌다... -> 사람 줄이자
        if max(l) > k:
            e = mid - 1
        # 다 건널 수 있다 -> 사람 늘려보자 + (mid+1)값 저장
        else:
            s = mid + 1
            answer = mid+1
    return answer

# 연속 펄스 부분 수열의 합
# 어떤 수열의 연속 부분 수열에 같은 길이의 펄스 수열을 각 원소끼리 곱하여 연속 펄스 부분 수열을 만들려 합니다.
# 펄스 수열이란 [1, -1, 1, -1 …] 또는 [-1, 1, -1, 1 …] 과 같이 1 또는 -1로 시작하면서 1과 -1이 번갈아 나오는 수열입니다.
#   예를 들어 수열 [2, 3, -6, 1, 3, -1, 2, 4]의 연속 부분 수열 [3, -6, 1]에 펄스 수열 [1, -1, 1]을 곱하면 연속 펄스 부분수열은 [3, 6, 1]이 됩니다.
#   또 다른 예시로 연속 부분 수열 [3, -1, 2, 4]에 펄스 수열 [-1, 1, -1, 1]을 곱하면 연속 펄스 부분수열은 [-3, -1, -2, 4]이 됩니다.
# 정수 수열 sequence가 매개변수로 주어질 때, 연속 펄스 부분 수열의 합 중 가장 큰 것을 return 하도록 solution 함수를 완성해주세요.
from sys import maxsize
INF = maxsize
def solution(sequence):
    answer = -INF
    size = len(sequence)
    s1 = s2 = 0				# 1
    s1_min = s2_min = 0		# 2
    pulse = 1
    
    for i in range(size):
        s1 += sequence[i] * pulse
        s2 += sequence[i] * (-pulse)
        
        # 3
        answer = max(answer, s1-s1_min, s2-s2_min)
        
        # 4
        s1_min = min(s1_min, s1)
        s2_min = min(s2_min, s2)
        
        # 5
        pulse *= -1
    return answer

# 거스름돈
# Finn은 편의점에서 야간 아르바이트를 하고 있습니다.
# 야간에 손님이 너무 없어 심심한 Finn은 손님들께 거스름돈을 n 원을 줄 때 방법의 경우의 수를 구하기로 하였습니다.
#   예를 들어서 손님께 5원을 거슬러 줘야 하고 1원, 2원, 5원이 있다면 다음과 같이 4가지 방법으로 5원을 거슬러 줄 수 있습니다.
#   1원을 5개 사용해서 거슬러 준다.
#   1원을 3개 사용하고, 2원을 1개 사용해서 거슬러 준다.
#   1원을 1개 사용하고, 2원을 2개 사용해서 거슬러 준다.
#   5원을 1개 사용해서 거슬러 준다.
# 거슬러 줘야 하는 금액 n과 Finn이 현재 보유하고 있는 돈의 종류 money가 매개변수로 주어질 때,
# Finn이 n 원을 거슬러 줄 방법의 수를 return 하도록 solution 함수를 완성해 주세요.
def solution(n, money):
    dp = [0]*(n+1)
    dp[0] = 1
    for typ in money :
        for price in range(typ, n+1) :
            dp[price] += dp[price - typ] 
    return dp[-1] % 1000000007

# 부대복귀
# 강철부대의 각 부대원이 여러 지역에 뿔뿔이 흩어져 특수 임무를 수행 중입니다.
# 지도에서 강철부대가 위치한 지역을 포함한 각 지역은 유일한 번호로 구분되며,
# 두 지역 간의 길을 통과하는 데 걸리는 시간은 모두 1로 동일합니다.
# 임무를 수행한 각 부대원은 지도 정보를 이용하여 최단시간에 부대로 복귀하고자 합니다.
# 다만 적군의 방해로 인해, 임무의 시작 때와 다르게 되돌아오는 경로가 없어져 복귀가 불가능한 부대원도 있을 수 있습니다.
# 강철부대가 위치한 지역을 포함한 총지역의 수 n, 두 지역을 왕복할 수 있는 길 정보를 담은 2차원 정수 배열 roads,
# 각 부대원이 위치한 서로 다른 지역들을 나타내는 정수 배열 sources, 강철부대의 지역 destination이 주어졌을 때,
# 주어진 sources의 원소 순서대로 강철부대로 복귀할 수 있는 최단시간을 담은 배열을 return하는 solution 함수를 완성해주세요.
# 복귀가 불가능한 경우 해당 부대원의 최단시간은 -1입니다.
from collections import deque, defaultdict
def solution(n, roads, sources, destination):
    answer = []
    
    # n = 총지역 수, roads = 길 정보
    # sources = 부대원 위치, destination = 강철부대 지역
    graph = [[] for _ in range(n+1)]
    for road in roads:
        a, b = road
        graph[a].append(b)
        graph[b].append(a)

    sub_answer = [-1]*(n+1)
    
    # bfs 수행
    q = deque()
    q.append((destination, 0)) 
    sub_answer[destination] = 0
    while q:
        now, level = q.popleft()
        
        for next_v in graph[now]:
            if sub_answer[next_v] == -1:
                q.append((next_v, level+1))
                sub_answer[next_v] = level+1

    for source in sources:
        answer.append(sub_answer[source])
    
    return answer

# 순위
# n명의 권투선수가 권투 대회에 참여했고 각각 1번부터 n번까지 번호를 받았습니다. 권투 경기는 1대1 방식으로 진행이 되고,
# 만약 A 선수가 B 선수보다 실력이 좋다면 A 선수는 B 선수를 항상 이깁니다.
# 심판은 주어진 경기 결과를 가지고 선수들의 순위를 매기려 합니다. 하지만 몇몇 경기 결과를 분실하여 정확하게 순위를 매길 수 없습니다.
# 선수의 수 n, 경기 결과를 담은 2차원 배열 results가 매개변수로 주어질 때
# 정확하게 순위를 매길 수 있는 선수의 수를 return 하도록 solution 함수를 작성해주세요.
def solution(n, results):
    answer = 0
    board = [[0]*n for _ in range(n)]
    
    for a,b in results:
        board[a-1][b-1] = 1
        board[b-1][a-1] = -1
        
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i == j or board[i][j] in [1,-1]:
                    continue
                if board[i][k] == board[k][j] == 1:
                    board[i][j] = 1
                    board[j][i] = board[k][i] = board[j][k] = -1
    for row in board:
        if row.count(0) == 1:
            answer += 1
    return answer

# 인사고과
# 완호네 회사는 연말마다 1년 간의 인사고과에 따라 인센티브를 지급합니다.
# 각 사원마다 근무 태도 점수와 동료 평가 점수가 기록되어 있는데 만약 어떤 사원이 다른 임의의 사원보다 두 점수가 모두 낮은 경우가 한 번이라도 있다면 그 사원은 인센티브를 받지 못합니다.
# 그렇지 않은 사원들에 대해서는 두 점수의 합이 높은 순으로 석차를 내어 석차에 따라 인센티브가 차등 지급됩니다.
# 이때, 두 점수의 합이 동일한 사원들은 동석차이며, 동석차의 수만큼 다음 석차는 건너 뜁니다.
# 예를 들어 점수의 합이 가장 큰 사원이 2명이라면 1등이 2명이고 2등 없이 다음 석차는 3등부터입니다.
# 각 사원의 근무 태도 점수와 동료 평가 점수 목록 scores이 주어졌을 때, 완호의 석차를 return 하도록 solution 함수를 완성해주세요.
def solution(scores):
    answer = 0
    target_a, target_b = scores[0]
    target_score = target_a + target_b
    # 첫번째 점수에 대해서 내림차순,
    # 첫 번째 점수가 같으면 두 번째 점수에 대해서 오름차순으로 정렬합니다.
    scores.sort(key=lambda x: (-x[0], x[1]))
    maxb = 0
    for a, b in scores:
        if target_a < a and target_b < b:
            return -1
        
        if b >= maxb:
            maxb = b
            if a + b > target_score:
                answer += 1
            
    return answer + 1

# 풍선 터트리기
# 일렬로 나열된 n개의 풍선이 있습니다. 모든 풍선에는 서로 다른 숫자가 써져 있습니다.
# 당신은 다음 과정을 반복하면서 풍선들을 단 1개만 남을 때까지 계속 터트리려고 합니다.
#   임의의 인접한 두 풍선을 고른 뒤, 두 풍선 중 하나를 터트립니다.
#   터진 풍선으로 인해 풍선들 사이에 빈 공간이 생겼다면, 빈 공간이 없도록 풍선들을 중앙으로 밀착시킵니다.
# 여기서 조건이 있습니다. 인접한 두 풍선 중에서 번호가 더 작은 풍선을 터트리는 행위는 최대 1번만 할 수 있습니다.
# 즉, 어떤 시점에서 인접한 두 풍선 중 번호가 더 작은 풍선을 터트렸다면,
# 그 이후에는 인접한 두 풍선을 고른 뒤 번호가 더 큰 풍선만을 터트릴 수 있습니다.
# 당신은 어떤 풍선이 최후까지 남을 수 있는지 알아보고 싶습니다. 위에 서술된 조건대로 풍선을 터트리다 보면,
# 어떤 풍선은 최후까지 남을 수도 있지만, 어떤 풍선은 무슨 수를 쓰더라도 마지막까지 남기는 것이 불가능할 수도 있습니다.
# 일렬로 나열된 풍선들의 번호가 담긴 배열 a가 주어집니다.
# 위에 서술된 규칙대로 풍선들을 1개만 남을 때까지 터트렸을 때 최후까지 남기는 것이 가능한 풍선들의 개수를 return 하도록 solution 함수를 완성해주세요.
def solution(a):
    if len(a) == 1:
        return 1

    answer = 2   # 기본적으로 양쪽 끝은 끝까지 살아남을 수 있음
    
    # 최솟값 쌓기 ----------------
    l_min = [a[0]]     
    r_min = [a[-1]]
    for i in range(1, len(a)):
        if a[i] < l_min[-1]:
            l_min.append(a[i])
        else:
            l_min.append(l_min[-1])
        if a[len(a) - 1 - i] < r_min[-1]:
            r_min.append(a[len(a) - 1 - i])
        else:
            r_min.append(r_min[-1])
    r_min.reverse()    
    # -----------------

	# 찾은 최솟값으로 비교 계산 -------------
    for i in range(1, len(a) - 1):
        if l_min[i - 1] > a[i] or r_min[i + 1] > a[i]:
            answer += 1
    # --------------------------------
            
    return answer

# 110 옮기기
# 0과 1로 이루어진 어떤 문자열 x에 대해서, 당신은 다음과 같은 행동을 통해 x를 최대한 사전 순으로 앞에 오도록 만들고자 합니다.
#   x에 있는 "110"을 뽑아서, 임의의 위치에 다시 삽입합니다.
#   예를 들어, x = "11100" 일 때, 여기서 중앙에 있는 "110"을 뽑으면 x = "10" 이 됩니다.
#   뽑았던 "110"을 x의 맨 앞에 다시 삽입하면 x = "11010" 이 됩니다.
# 변형시킬 문자열 x가 여러 개 들어있는 문자열 배열 s가 주어졌을 때,
# 각 문자열에 대해서 위의 행동으로 변형해서 만들 수 있는 문자열 중 사전 순으로 가장 앞에 오는 문자열을 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def solution(s):
    answer = []
    for string in s:
        count, idx, stack = 0, 0, ""
        while idx < len(string):            # 110 찾기
            if string[idx] == "0" and stack[-2:] == "11":
                stack = stack[:-2]
                count += 1
            else:
                stack += string[idx]
            idx += 1

        idx = stack.find("111")             # 110이 빠진 string에서 111 찾기
        if idx == -1:                       # 0뒤에 110 반복해 붙이기
            idx = stack.rfind('0')
            stack = stack[:idx+1]+"110"*count+stack[idx+1:]
        else:                               # 111앞에 110 반복해 붙이기
            stack = stack[:idx]+"110"*count+stack[idx:]
        answer.append(stack)
    return answer

# 외벽 점검
# 레스토랑을 운영하고 있는 "스카피"는 레스토랑 내부가 너무 낡아 친구들과 함께 직접 리모델링 하기로 했습니다.
# 레스토랑이 있는 곳은 스노우타운으로 매우 추운 지역이어서 내부 공사를 하는 도중에 주기적으로 외벽의 상태를 점검해야 할 필요가 있습니다.
# 레스토랑의 구조는 완전히 동그란 모양이고 외벽의 총 둘레는 n미터이며,
# 외벽의 몇몇 지점은 추위가 심할 경우 손상될 수도 있는 취약한 지점들이 있습니다.
# 따라서 내부 공사 도중에도 외벽의 취약 지점들이 손상되지 않았는 지, 주기적으로 친구들을 보내서 점검을 하기로 했습니다.
# 다만, 빠른 공사 진행을 위해 점검 시간을 1시간으로 제한했습니다. 친구들이 1시간 동안 이동할 수 있는 거리는 제각각이기 때문에,
# 최소한의 친구들을 투입해 취약 지점을 점검하고 나머지 친구들은 내부 공사를 돕도록 하려고 합니다.
# 편의 상 레스토랑의 정북 방향 지점을 0으로 나타내며, 취약 지점의 위치는 정북 방향 지점으로부터 시계 방향으로 떨어진 거리로 나타냅니다.
# 또, 친구들은 출발 지점부터 시계, 혹은 반시계 방향으로 외벽을 따라서만 이동합니다.
# 외벽의 길이 n, 취약 지점의 위치가 담긴 배열 weak, 각 친구가 1시간 동안 이동할 수 있는 거리가 담긴 배열 dist가 매개변수로 주어질 때,
# 취약 지점을 점검하기 위해 보내야 하는 친구 수의 최소값을 return 하도록 solution 함수를 완성해주세요.
def solution(n, weak, dist):

    W, F = len(weak), len(dist)
    repair_lst = [()]  # 현재까지 고칠 수 있는 취약점들 저장 (1,2,3)
    count = 0  # 투입친구 수
    dist.sort(reverse=True) # 움직일 수 있는 거리가 큰 친구 순서대로

    # 고칠 수 있는 것들 리스트 작성
    for can_move in dist:
        repairs = []  # 친구 별 고칠 수 있는 취약점들 저장
        count += 1

        # 수리 가능한 지점 찾기
        for i, wp in enumerate(weak):
            start = wp  # 각 위크포인트부터 시작
            ends = weak[i:] + [n+w for w in weak[:i]]  # 시작점 기준 끝 포인트 값 저장
            can = [end % n for end in ends if end -
                   start <= can_move]  # 가능한 지점 저장
            repairs.append(set(can))

        # 수리 가능한 경우 탐색
        cand = set()
        for r in repairs:  # 새친구의 수리가능 지점
            for x in repair_lst:  # 기존 수리가능 지점
                new = r | set(x)  # 새로운 수리가능 지점
                if len(new) == W:  # 모두 수리가능 한 경우 친구 수 리턴
                    return count
                cand.add(tuple(new))
        repair_lst = cand

    return -1

# 스타 수열
# 다음과 같은 것들을 정의합니다.
# 어떤 수열 x의 부분 수열(Subsequence)이란,
# x의 몇몇 원소들을 제거하거나 그러지 않고 남은 원소들이 원래 순서를 유지하여 얻을 수 있는 새로운 수열을 말합니다.
#   예를 들어, [1,3]은 [1,2,3,4,5]의 부분수열입니다. 원래 수열에서 2, 4, 5를 제거해서 얻을 수 있기 때문입니다.
# 다음과 같은 조건을 모두 만족하는 수열 x를 스타 수열이라고 정의합니다.
#   x의 길이가 2 이상의 짝수입니다. (빈 수열은 허용되지 않습니다.)
#   x의 길이를 2n이라 할 때, 다음과 같은 n개의 집합 {x[0], x[1]}, {x[2], x[3]}, ..., {x[2n-2], x[2n-1]} 의 교집합의 원소의 개수가 1 이상입니다.
#   x[0] != x[1], x[2] != x[3], ..., x[2n-2] != x[2n-1] 입니다.
#   예를 들어, [1,2,1,3,4,1,1,3]은 스타 수열입니다. {1,2}, {1,3}, {4,1}, {1,3} 의 교집합은 {1} 이고,
#       각 집합 내의 숫자들이 서로 다르기 때문입니다.
# 1차원 정수 배열 a가 매개변수로 주어집니다. a의 모든 부분 수열 중에서 가장 길이가 긴 스타 수열의 길이를 return 하도록 solution 함수를 완성해주세요.
# 이때, a의 모든 부분 수열 중에서 스타 수열이 없다면, 0을 return 해주세요.
from collections import Counter
def solution(a):
    answer = -1
    # 현재 배열에 숫자들이 나온 횟수들
    els = Counter(a)
    # a에 있는 각 원소 k를 기준으로 스타배열을 만들 수 있는지 검사
    for k in els.keys():
        # 현재 k의 등장횟수가 스타수열에 사용된 공통 인자 횟수 이하면 continue
        if els[k] <= answer:
            continue
        # k의 등장 횟수
        cnt = 0
        idx = 0
        while idx < len(a)-1:
            # 두 칸 모두 k가 포함 안되어있거나 두 칸이 같은 값이면 스타수열 안되니까 continue
            if (a[idx] != k and a[idx+1] != k) or (a[idx] == a[idx+1]):
                idx += 1
                continue

            # 스타수열의 원소로 추가할 수 있는 경우 k사용횟수 1 증가
            cnt += 1
            # 다음 배열 탐색을 위해 두칸 점프
            idx += 2
        # 스타 수열 완성에 쓰인 공통 원소 k가 사용된 최대 횟수 갱신
        answer = max(cnt, answer)
    return -1 if answer == -1 else answer*2

# 셔틀버스
# 카카오에서는 무료 셔틀버스를 운행하기 때문에 판교역에서 편하게 사무실로 올 수 있다.
# 카카오의 직원은 서로를 '크루'라고 부르는데, 아침마다 많은 크루들이 이 셔틀을 이용하여 출근한다.
# 이 문제에서는 편의를 위해 셔틀은 다음과 같은 규칙으로 운행한다고 가정하자.
#   셔틀은 09:00부터 총 n회 t분 간격으로 역에 도착하며, 하나의 셔틀에는 최대 m명의 승객이 탈 수 있다.
#   틀은 도착했을 때 도착한 순간에 대기열에 선 크루까지 포함해서 대기 순서대로 태우고 바로 출발한다.
#       예를 들어 09:00에 도착한 셔틀은 자리가 있다면 09:00에 줄을 선 크루도 탈 수 있다.
# 일찍 나와서 셔틀을 기다리는 것이 귀찮았던 콘은, 일주일간의 집요한 관찰 끝에 어떤 크루가 몇 시에 셔틀 대기열에 도착하는지 알아냈다.
# 콘이 셔틀을 타고 사무실로 갈 수 있는 도착 시각 중 제일 늦은 시각을 구하여라.
# 단, 콘은 게으르기 때문에 같은 시각에 도착한 크루 중 대기열에서 제일 뒤에 선다.
# 또한, 모든 크루는 잠을 자야 하므로 23:59에 집에 돌아간다. 따라서 어떤 크루도 다음날 셔틀을 타는 일은 없다.
# 셔틀 운행 횟수 n, 셔틀 운행 간격 t, 한 셔틀에 탈 수 있는 최대 크루 수 m, 크루가 대기열에 도착하는 시각을 모은 배열 timetable이 입력으로 주어진다.
def solution(n, t, m, timetable):
    answer = 0
    timetable = [int(time[:2])*60+int(time[3:]) for time in timetable]  # 시간 -> 분 change
    timetable.sort()
    busTime = [9*60+t*i for i in range(n)]  # 버스 시간
    
    i = 0  # 버스에 탈 크루의 인덱스
    for bt in busTime:  # 버스 도착 시간을 순회하면서
        cnt = 0  # 버스에 타는 크루 수
        while cnt<m and i<len(timetable) and timetable[i]<=bt:
            i += 1
            cnt += 1
        if cnt<m:  # 버스에 자리 남았으면 버스타임에 내가 타면 됨
            answer = bt
        else:  # 버스에 탈 자리 없으면 맨 마지막 크루보다 1분전에 도착
            answer = timetable[i-1]-1
            
    return str(answer//60).zfill(2)+":"+str(answer%60).zfill(2)

# 기지국 설치
# N개의 아파트가 일렬로 쭉 늘어서 있습니다.
# 이 중에서 일부 아파트 옥상에는 4g 기지국이 설치되어 있습니다.
# 기술이 발전해 5g 수요가 높아져 4g 기지국을 5g 기지국으로 바꾸려 합니다.
# 그런데 5g 기지국은 4g 기지국보다 전달 범위가 좁아,
# 4g 기지국을 5g 기지국으로 바꾸면 어떤 아파트에는 전파가 도달하지 않습니다.
# 아파트의 개수 N, 현재 기지국이 설치된 아파트의 번호가 담긴 1차원 배열 stations,
# 전파의 도달 거리 W가 매개변수로 주어질 때,
# 모든 아파트에 전파를 전달하기 위해 증설해야 할 기지국 개수의 최솟값을 리턴하는 solution 함수를 완성해주세요
import math
def solution(n, stations, w):
    answer = 0
    dist = []  # 전파 전달 안되는 구간 길이 저장할 리스트
    for i in range(1, len(stations)):
        dist.append((stations[i]-w-1)-(stations[i-1]+w))
    
    dist.append(stations[0]-w-1)  # 맨앞
    dist.append(n-(stations[-1]+w))  # 맨뒤
    
    for i in dist:
        if i <= 0:
            continue
        else:
            answer += math.ceil(i/(2*w+1))  # 올림
    return answer

# 선입 선출 스케줄링
# 처리해야 할 동일한 작업이 n 개가 있고, 이를 처리하기 위한 CPU가 있습니다.
#   이 CPU는 다음과 같은 특징이 있습니다.
#   CPU에는 여러 개의 코어가 있고, 코어별로 한 작업을 처리하는 시간이 다릅니다.
#   한 코어에서 작업이 끝나면 작업이 없는 코어가 바로 다음 작업을 수행합니다.
#   2개 이상의 코어가 남을 경우 앞의 코어부터 작업을 처리 합니다.
# 처리해야 될 작업의 개수 n과, 각 코어의 처리시간이 담긴 배열 cores 가 매개변수로 주어질 때,
# 마지막 작업을 처리하는 코어의 번호를 return 하는 solution 함수를 완성해주세요.
def solution(n, cores):
    if n <= len(cores):
        return n
    else:
        n -= len(cores)
        left = 1
        right = max(cores) * n
        while left < right:
            mid = (left + right) // 2
            capacity = 0
            for c in cores:
                capacity += mid // c
            if capacity >= n:
                right = mid
            else:
                left = mid + 1

        for c in cores:
            n -= (right-1) // c

        for i in range(len(cores)):
            if right % cores[i] == 0:
                n -= 1
                if n == 0:
                    return i + 1
                
# 억억단을 외우자
# 억억단은 1억 x 1억 크기의 행렬입니다. 억억단을 외우던 영우는 친구 수연에게 퀴즈를 내달라고 부탁하였습니다.
# 수연은 평범하게 문제를 내봐야 영우가 너무 쉽게 맞히기 때문에 좀 어렵게 퀴즈를 내보려고 합니다.
# 적당한 수 e를 먼저 정하여 알려주고 e 이하의 임의의 수 s를 여러 개 얘기합니다.
# 영우는 각 s에 대해서 s보다 크거나 같고 e 보다 작거나 같은 수 중에서 억억단에서 가장 많이 등장한 수를 답해야 합니다.
# 만약 가장 많이 등장한 수가 여러 개라면 그 중 가장 작은 수를 답해야 합니다.
# 수연은 영우가 정답을 말하는지 확인하기 위해 당신에게 프로그램 제작을 의뢰하였습니다.
# e와 s의 목록 starts가 매개변수로 주어질 때 각 퀴즈의 답 목록을 return 하도록 solution 함수를 완성해주세요.
def solution(e, starts):
    numbers = [0 for i in range(e+1)]
    numbers[1] = 1
    # 약수 개수 찾기
    for num in range(2, e+1):
        for i in range(1, int(num**0.5)+1):
            if num % i == 0:
                numbers[num] += 2
        
        if num**0.5 == int(num**0.5) :
            numbers[num] -= 1

    # 가장 큰 수 정렬하기
    big_numbers = [i for i in range(e+1)]
    for i in reversed(range(1, e)):
        if numbers[i] < numbers[big_numbers[i+1]]:
            big_numbers[i] = big_numbers[i+1]

    # 정답 찾기
    return [ big_numbers[s] for s in starts]

# 2차원 동전 뒤집기
# 한수는 직사각형 모양의 공간에 놓인 동전들을 뒤집는 놀이를 하고 있습니다.
# 모든 동전들은 앞과 뒤가 구분되어 있으며, 동전을 뒤집기 위해서는 같은 줄에 있는 모든 동전을 뒤집어야 합니다.
# 동전들의 초기 상태와 목표 상태가 주어졌을 때, 초기 상태에서 최소 몇 번의 동전을 뒤집어야 목표 상태가 되는지 알아봅시다.
# 직사각형 모양의 공간에 놓인 동전들의 초기 상태를 나타내는 2차원 정수 배열 beginning,
# 목표 상태를 나타내는 target이 주어졌을 때, 초기 상태에서 목표 상태로 만들기 위해
# 필요한 동전 뒤집기 횟수의 최솟값을 return 하는 solution 함수를 완성하세요.
# 단, 목표 상태를 만들지 못하는 경우에는 -1을 return 합니다.
# 행 뒤집기
def flip(ary, bits, row):
    row_cnt = 0
    for i in range(row):
        if bits & (1 << i): # i번째 행을 뒤집어야 하는 경우
            ary[i] = [1-e for e in ary[i]]
            row_cnt += 1
    return ary, row_cnt
       
# 열 확인하기
def check(ary, col):
    col_cnt = 0
    for j in range(col):
        tmp = set([row[j] for row in ary])  # j열
        if len(tmp) == 2:   # 목표 상태 도달 불가능
            return -1
        elif 1 in tmp :     # 뒤집어야 하는 경우
            col_cnt += 1
    return col_cnt
    
def solution(beginning, target):
    answer = float('inf')
    row, col = len(beginning), len(beginning[0])
    board = [[1 if beginning[i][j] != target[i][j] else 0 for j in range(col)] for i in range(row)]
    
    # 0...0 : 모든 행을 뒤집지 않는다
    # 1...1 : 모든 행을 뒤집는다
    for bits in range(2**row):
        flipped, row_cnt = flip(board[:], bits, row) # 뒤집어야 할 행만큼 행 뒤집기
        col_cnt = check(flipped, col)   # 열 확인하기
        if col_cnt == -1 :              # 목표 상태로 도달 불가능
            continue
    
        answer = min(row_cnt+col_cnt, answer)
    
    if answer == float('inf'):
        return -1
    return answer

# 카운트 다운
# "카운트 다운"은 게임이 시작되면 무작위로 점수가 정해지고, 다트를 던지면서 점수를 깎아서 정확히 0점으로 만드는 게임입니다.
# 단, 남은 점수보다 큰 점수로 득점하면 버스트가 되며 실격 합니다.
# 다트 과녁에는 1 부터 20 까지의 수가 하나씩 있고 각 수마다 "싱글", "더블", "트리플" 칸이 있습니다.
# "싱글"을 맞히면 해당 수만큼 점수를 얻고 "더블"을 맞히면 해당 수의 두 배만큼 점수를 얻고 "트리플"을 맞히면 해당 수의 세 배만큼 점수를 얻습니다.
# 가운데에는 "불"과 "아우터 불"이 있는데 "카운트 다운" 게임에서는 구분 없이 50점을 얻습니다.
# 대회는 토너먼트로 진행되며 한 게임에는 두 선수가 참가하게 됩니다. 게임은 두 선수가 교대로 한 번씩 던지는 라운드 방식으로 진행됩니다.
# 가장 먼저 0점을 만든 선수가 승리하는데 만약 두 선수가 같은 라운드에 0점을 만들면
# 두 선수 중 "싱글" 또는 "불"을 더 많이 던진 선수가 승리하며 그마저도 같다면 선공인 선수가 승리합니다.
# 다트 실력에 자신 있던 종호는 이 대회에 출전하기로 하였습니다. 최소한의 다트로 0점을 만드는 게 가장 중요하고,
# 그러한 방법이 여러가지가 있다면 "싱글" 또는 "불"을 최대한 많이 던지는 방법을 선택해야 합니다.
# 목표 점수 target이 매개변수로 주어졌을 때 최선의 경우 던질 다트 수와
# 그 때의 "싱글" 또는 "불"을 맞춘 횟수(합)를 순서대로 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
def solution(target):
    # target까지 도달하는 데 필요한 최소 다트 수와 "싱글" 또는 "불"을 맞춘 횟수를 기록하는 배열
    # 최댓값으로 초기화
    dp = [[float('inf'), 0] for _ in range(300000)]
    # 가능한 과녁 값들
    targetList = [i + 1 for i in range(20)]
    # 시작 지점의 다트 수는 0으로 초기화
    dp[0][0] = 0
    # target까지 탐색
    for i in range(target):
        def updateDart(addIdx, count):
            # 던진 다트 수를 갱신할 필요가 있는 경우
            if dp[i + addIdx][0] >= dp[i][0] + 1:
                if dp[i + addIdx][0] == dp[i][0] + 1:
                    # 기존 값과 비교하여 "싱글" 또는 "불"을 맞춘 횟수(합) 갱신
                    dp[i + addIdx][1] = max(dp[i + addIdx][1], dp[i][1] + count)
                else:
                    # 던진 다트 수와 "싱글" 또는 "불"을 맞춘 횟수(합)을 갱신
                    dp[i + addIdx] = [dp[i][0] + 1, dp[i][1] + count]

        # 모든 과녁 값에 대해 확인
        for j in targetList:
            # 싱글, 더블, 트리플을 순서대로 적용하여 갱신
            for multiplier, hitCount in [[1, 1], [2, 0], [3, 0]]:
                updateDart(j * multiplier, hitCount)

        # "불"에 대해서도 확인하여 갱신
        updateDart(50, 1)

    # target까지 도달하는 데 필요한 최소 다트 수와 "싱글" 또는 "불"을 맞춘 횟수 반환
    return dp[target]

# 모두 0으로 만들기
# 각 점에 가중치가 부여된 트리가 주어집니다. 당신은 다음 연산을 통하여, 이 트리의 모든 점들의 가중치를 0으로 만들고자 합니다.
#   임의의 연결된 두 점을 골라서 한쪽은 1 증가시키고, 다른 한쪽은 1 감소시킵니다.
# 하지만, 모든 트리가 위의 행동을 통하여 모든 점들의 가중치를 0으로 만들 수 있는 것은 아닙니다.
# 당신은 주어진 트리에 대해서 해당 사항이 가능한지 판별하고,
# 만약 가능하다면 최소한의 행동을 통하여 모든 점들의 가중치를 0으로 만들고자 합니다.
# 트리의 각 점의 가중치를 의미하는 1차원 정수 배열 a와 트리의 간선 정보를 의미하는 edges가 매개변수로 주어집니다.
# 주어진 행동을 통해 트리의 모든 점들의 가중치를 0으로 만드는 것이 불가능하다면 -1을,
# 가능하다면 최소 몇 번만에 가능한지를 찾아 return 하도록 solution 함수를 완성해주세요.
# (만약 처음부터 트리의 모든 정점의 가중치가 0이라면, 0을 return 해야 합니다.)
from collections import deque
def solution(a, edges):
    answer = 0
    n = len(a)
    graph = [[] for _ in range(n)]
    # 간성정보 입력
    for v1, v2 in edges:
        graph[v1].append(v2)
        graph[v2].append(v1)
    # 루트 노드부터 리프 노드까지 이동경로
    route = []
    visit = [0]*n
    visit[0] = 1
    Q = deque([0])
    # route 찾기
    while Q:
        now = Q.popleft()
        route.append(now)
        for j in graph[now]:
            if visit[j] == 0:
                visit[j] = 1
                Q.append(j)

    # 리프노드를 0으로 만들고 부모노드에 더해감
    # 최종적으로 부모노드에 도착
    visit = [0]*n
    for i in range(n-1, -1, -1):
        node = route[i]
        visit[node] = 1
        # 현재 노드가 0이 아니라면 탐색, 0이면 넘어감
        if a[node]:
            for v in graph[node]:
                if visit[v] == 0 and a[node]:
                    a[v] += a[node]
                    answer += abs(a[node])
                    a[node] = 0

    return answer if a[0] == 0 else -1

# 도넛과 막대 그래프
# 도넛 모양 그래프, 막대 모양 그래프, 8자 모양 그래프들이 있습니다. 이 그래프들은 1개 이상의 정점과, 정점들을 연결하는 단방향 간선으로 이루어져 있습니다.
#   크기가 n인 도넛 모양 그래프는 n개의 정점과 n개의 간선이 있습니다.
#       도넛 모양 그래프의 아무 한 정점에서 출발해 이용한 적 없는 간선을 계속 따라가면
#       나머지 n-1개의 정점들을 한 번씩 방문한 뒤 원래 출발했던 정점으로 돌아오게 됩니다. 도넛 모양 그래프의 형태는 다음과 같습니다.
#   크기가 n인 막대 모양 그래프는 n개의 정점과 n-1개의 간선이 있습니다.
#       막대 모양 그래프는 임의의 한 정점에서 출발해 간선을 계속 따라가면 나머지 n-1개의 정점을 한 번씩 방문하게 되는 정점이 단 하나 존재합니다.
#       막대 모양 그래프의 형태는 다음과 같습니다.
#   크기가 n인 8자 모양 그래프는 2n+1개의 정점과 2n+2개의 간선이 있습니다.
#       8자 모양 그래프는 크기가 동일한 2개의 도넛 모양 그래프에서 정점을 하나씩 골라 결합시킨 형태의 그래프입니다. 8자 모양 그래프의 형태는 다음과 같습니다.
# 도넛 모양 그래프, 막대 모양 그래프, 8자 모양 그래프가 여러 개 있습니다. 이 그래프들과 무관한 정점을 하나 생성한 뒤,
# 각 도넛 모양 그래프, 막대 모양 그래프, 8자 모양 그래프의 임의의 정점 하나로 향하는 간선들을 연결했습니다.
# 그 후 각 정점에 서로 다른 번호를 매겼습니다.
# 이때 당신은 그래프의 간선 정보가 주어지면 생성한 정점의 번호와 정점을 생성하기 전
# 도넛 모양 그래프의 수, 막대 모양 그래프의 수, 8자 모양 그래프의 수를 구해야 합니다.
# 그래프의 간선 정보를 담은 2차원 정수 배열 edges가 매개변수로 주어집니다.
# 이때, 생성한 정점의 번호, 도넛 모양 그래프의 수, 막대 모양 그래프의 수,
# 8자 모양 그래프의 수를 순서대로 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
def solution(edges):  
    def count_edges(edges):
        edge_counts = {}
        for a, b in edges:
            # 각 노드별로 간선의 수를 추가할 딕셔너리를 생성 - .get() 함수를 이용해 딕셔너리의 키 값 추가
            if not edge_counts.get(a):
                edge_counts[a] = [0, 0]
            if not edge_counts.get(b):
                edge_counts[b] = [0, 0]
            # output edge와 input edge의 개수를 추가
            edge_counts[a][0] += 1  # a는 n번 노드에서 나가는 간선
            edge_counts[b][1] += 1  # b는 n번 노드로 들어오는 간선
        return edge_counts

    def check_answer(egde_counts):
        answer = [0, 0, 0, 0]
        for key, counts in edge_counts.items():
            # 생성된 정점의 번호 확인
            if counts[0] >= 2 and counts[1] == 0:
                answer[0] = key
            # 막대 모양 그래프의 수 확인
            elif counts[0] == 0 and counts[1] > 0:
                answer[2] += 1
            # 8자 모양 그래프의 수 확인
            elif counts[0] >= 2 and counts[1] >= 2:
                answer[3] += 1
        # 도넛 모양 그래프의 수 확인
        answer[1] = (edge_counts[answer[0]][0] - answer[2] - answer[3])
        return answer
    edge_counts = count_edges(edges)
    answer = check_answer(edge_counts)
    return answer

# 양과 늑대
# 2진 트리 모양 초원의 각 노드에 늑대와 양이 한 마리씩 놓여 있습니다.
# 이 초원의 루트 노드에서 출발하여 각 노드를 돌아다니며 양을 모으려 합니다. 
# 각 노드를 방문할 때 마다 해당 노드에 있던 양과 늑대가 당신을 따라오게 됩니다. 
# 이때, 늑대는 양을 잡아먹을 기회를 노리고 있으며,
# 당신이 모은 양의 수보다 늑대의 수가 같거나 더 많아지면 바로 모든 양을 잡아먹어 버립니다. 
# 당신은 중간에 양이 늑대에게 잡아먹히지 않도록 하면서 최대한 많은 수의 양을 모아서 다시 루트 노드로 돌아오려 합니다.
# 각 노드에 있는 양 또는 늑대에 대한 정보가 담긴 배열 info,
# 2진 트리의 각 노드들의 연결 관계를 담은 2차원 배열 edges가 매개변수로 주어질 때,
# 문제에 제시된 조건에 따라 각 노드를 방문하면서 모을 수 있는 양은 최대 몇 마리인지 return 하도록 solution 함수를 완성해주세요.
def solution(info, edges):
    def nextNodes(v):
        temp = list()
        for e in edges:
            # i는 부모노드, j는 자식노드
            i, j = e
            # 부모노드 번호 비교
            if v == i:
                temp.append(j)
        return temp

    def dfs(sheep, wolf, current, path):
        # 지금 노드 확인, 양 늑대 판별
        if info[current]:
            wolf += 1
        else:
            sheep += 1

        # 늑대가 다 잡아먹음, 무시
        if sheep <= wolf:
            return 0

        # 아니라면 임시 변수에 값 갱신
        maxSheep = sheep

        # 서칭 시작
        for p in path:
            for n in nextNodes(p):
                if n not in path:
                    path.append(n)
                    # 최대 양 판별
                    maxSheep = max(maxSheep, dfs(sheep, wolf, n, path))
                    path.pop()
        return maxSheep
    answer = dfs(0, 0, 0, [0])
    return answer

# 디스크 컨트롤러
# 하드디스크는 한 번에 하나의 작업만 수행할 수 있습니다.
# 디스크 컨트롤러를 구현하는 방법은 여러 가지가 있습니다. 가장 일반적인 방법은 요청이 들어온 순서대로 처리하는 것입니다.
# 각 작업에 대해 [작업이 요청되는 시점, 작업의 소요시간]을 담은 2차원 배열 jobs가 매개변수로 주어질 때,
# 작업의 요청부터 종료까지 걸린 시간의 평균을 가장 줄이는 방법으로 처리하면
# 평균이 얼마가 되는지 return 하도록 solution 함수를 작성해주세요. (단, 소수점 이하의 수는 버립니다)
from heapq import heappush, heappop
def solution(jobs):
    jobs.sort()
    num = len(jobs)
    waiting = [] # (소요시간, 요청시점)
    count = [] # 각 작업이 몇초 걸렸는지
    now = 0 #현재 시각
    while len(count) != num : 
        while jobs and now >= jobs[0][0] : 
            top = jobs.pop(0)
            heappush(waiting, (top[1], top[0]))

        if jobs and waiting == []:
            top = jobs.pop(0)
            now = top[0]
            heappush(waiting, (top[1], top[0]))
        x,y = heappop(waiting)
        now += x 
        count.append(now-y)

    return sum(count)//num

# 불량 사용자
# 개발팀 내에서 이벤트 개발을 담당하고 있는 "무지"는 최근 진행된 카카오이모티콘 이벤트에 비정상적인 방법으로 당첨을 시도한 응모자들을 발견하였습니다.
# 이런 응모자들을 따로 모아 불량 사용자라는 이름으로 목록을 만들어서 당첨 처리 시 제외하도록 이벤트 당첨자 담당자인 "프로도" 에게 전달하려고 합니다.
# 이 때 개인정보 보호을 위해 사용자 아이디 중 일부 문자를 '*' 문자로 가려서 전달했습니다.
# 가리고자 하는 문자 하나에 '*' 문자 하나를 사용하였고 아이디 당 최소 하나 이상의 '*' 문자를 사용하였습니다.
# "무지"와 "프로도"는 불량 사용자 목록에 매핑된 응모자 아이디를 제재 아이디 라고 부르기로 하였습니다.
# 이벤트 응모자 아이디 목록이 담긴 배열 user_id와 불량 사용자 아이디 목록이 담긴 배열 banned_id가 매개변수로 주어질 때,
# 당첨에서 제외되어야 할 제재 아이디 목록은 몇가지 경우의 수가 가능한 지 return 하도록 solution 함수를 완성해주세요.
from itertools import permutations
from collections import Counter
def is_match(uid, bid):
    """ 아이디가 banned_id에 매칭되는지 여부를 리턴 """
    if len(uid) != len(bid):
        return False
    for u, b in zip(uid, bid):
        if b=="*" or u==b:
            continue
        else:
            return False
    return True

def is_available_case(user_id, banned_id):
    """ user_id리스트와 banned_id 리스트 (동일한 길이) 가 매칭되는지 여부를 리턴 """
    for uid, bid in zip(user_id, banned_id):
        if not is_match(uid, bid):
            return False
    return True

def solution(user_id, banned_id):
    answer = []
    # 순열을 통해 user_id에서 banned_id의 길이만큼 샘플링한 리스트에 대하여
    for u_ids in permutations(user_id, len(banned_id)):
        # 두 리스트가 매칭되는 경우
        if is_available_case(u_ids, banned_id):
            cnt = Counter(u_ids)
            # 중복되지 않는다면 answer에 추가한다.
            if cnt not in answer:
                answer.append(cnt)
    return len(answer)

# 자물쇠와 열쇠
# 잠겨있는 자물쇠는 격자 한 칸의 크기가 1 x 1인 N x N 크기의 정사각 격자 형태이고
# 특이한 모양의 열쇠는 M x M 크기인 정사각 격자 형태로 되어 있습니다.
# 자물쇠에는 홈이 파여 있고 열쇠 또한 홈과 돌기 부분이 있습니다.
# 열쇠는 회전과 이동이 가능하며 열쇠의 돌기 부분을 자물쇠의 홈 부분에 딱 맞게 채우면 자물쇠가 열리게 되는 구조입니다.
# 자물쇠 영역을 벗어난 부분에 있는 열쇠의 홈과 돌기는 자물쇠를 여는 데 영향을 주지 않지만,
# 자물쇠 영역 내에서는 열쇠의 돌기 부분과 자물쇠의 홈 부분이 정확히 일치해야 하며 열쇠의 돌기와 자물쇠의 돌기가 만나서는 안됩니다.
# 또한 자물쇠의 모든 홈을 채워 비어있는 곳이 없어야 자물쇠를 열 수 있습니다.
# 열쇠를 나타내는 2차원 배열 key와 자물쇠를 나타내는 2차원 배열 lock이 매개변수로 주어질 때,
# 열쇠로 자물쇠를 열수 있으면 true를, 열 수 없으면 false를 return 하도록 solution 함수를 완성해주세요.
# NxN 2차원 리스트 d도 회전
# 회전 각도 d => 1: 90도, 2: 180도, 3: 270도
def rotate(array, d):
    n = len(array)  # 배열의 길이
    result = [[0] * n for _ in range(n)]

    if d % 4 == 1:
        for r in range(n):
            for c in range(n):
                result[c][n - r - 1] = array[r][c]
    elif d % 4 == 2:
        for r in range(n):
            for c in range(n):
                result[n - r - 1][n - c - 1] = array[r][c]
    elif d % 4 == 3:
        for r in range(n):
            for c in range(n):
                result[n - c - 1][r] = array[r][c]
    else:
        for r in range(n):
            for c in range(n):
                result[r][c] = array[r][c]

    return result


# 자물쇠 중간 NxN 부분이 모두 1인지 확인
def check(new_lock):
    n = len(new_lock) // 3
    for i in range(n, n * 2):
        for j in range(n, n * 2):
            if new_lock[i][j] != 1:
                return False
    return True


def solution(key, lock):
    m = len(key)
    n = len(lock)
    # 기존 자물쇠보다 3배 큰 자물쇠
    new_lock = [[0] * (n * 3) for _ in range(n * 3)]
    # 새로운 자물쇠의 중앙 부분에 기존 자물쇠 넣기
    for i in range(n):
        for j in range(n):
            new_lock[n + i][n + j] = lock[i][j]

    # 열쇠를 (1, 1)부터 (N*2, N*2)까지 이동시키며 확인
    for i in range(1, n * 2):
        for j in range(1, n * 2):
            # 열쇠를 0, 90, 180, 270도로 회전시키며 확인
            for d in range(4):
                r_key = rotate(key, d)  # key를 d만큼 회전시킨 리스트
                for x in range(m):
                    for y in range(m):
                        new_lock[i + x][j + y] += r_key[x][y]

                if check(new_lock):
                    return True

                for x in range(m):
                    for y in range(m):
                        new_lock[i + x][j + y] -= r_key[x][y]

    return False

# 스티커 모으기(2)
# 원형으로 연결된 스티커에서 몇 장의 스티커를 뜯어내어 뜯어낸 스티커에 적힌 숫자의 합이 최대가 되도록 하고 싶습니다.
# 단 스티커 한 장을 뜯어내면 양쪽으로 인접해있는 스티커는 찢어져서 사용할 수 없게 됩니다.
# 스티커에 적힌 숫자가 배열 형태로 주어질 때, 스티커를 뜯어내어 얻을 수 있는 숫자의 합의 최댓값을 return 하는 solution 함수를 완성해 주세요.
# 원형의 스티커 모양을 위해 배열의 첫 번째 원소와 마지막 원소가 서로 연결되어 있다고 간주합니다.
def solution(sticker):
    answer = 0
    if len(sticker) == 1:
        return sticker.pop()
    size = len(sticker)
    # 1번 선택하는 경우 -> 1..n-1번 배열에 대한 DP
    dp1 = [0] + sticker[:-1]
    for i in range(2, size):
        dp1[i] = max(dp1[i-1], dp1[i-2] + dp1[i])
    
    # 2번 선택하는 경우 -> 2...n번 배열에 대한 DP
    dp2 = [0] + sticker[1:]
    for i in range(2, size):
        dp2[i] = max(dp2[i-1], dp2[i-2] + dp2[i])
        
    answer = max(dp1[-1], dp2[-1])
    return answer

# 보석 쇼핑
# 어느 날 스트레스를 풀기 위해 보석 매장에 쇼핑을 하러 간 어피치는 이전처럼 진열대의 특정 범위의 보석을 모두 구매하되 특별히 아래 목적을 달성하고 싶었습니다.
#   진열된 모든 종류의 보석을 적어도 1개 이상 포함하는 가장 짧은 구간을 찾아서 구매
# 진열대 번호 순서대로 보석들의 이름이 저장된 배열 gems가 매개변수로 주어집니다.
# 이때 모든 보석을 하나 이상 포함하는 가장 짧은 구간을 찾아서 return 하도록 solution 함수를 완성해주세요.
# 가장 짧은 구간의 시작 진열대 번호와 끝 진열대 번호를 차례대로 배열에 담아서 return 하도록 하며,
# 만약 가장 짧은 구간이 여러 개라면 시작 진열대 번호가 가장 작은 구간을 return 합니다.
def solution(gems):
    answer = [0, len(gems)]
    size = len(set(gems))   # 보석 종류 갯수
    left, right = 0, 0 # left는 보석 빼 줄 포인터, right는 보석 더해 줄 포인터
    gem_dict = {gems[0] : 1}
    
    while left < len(gems) and right < len(gems):   # 투 포인터가 범위를 벗어나면 무한루프 종료
        # 딕셔너리에 보석 종류가 다 들어오는 경우
        if len(gem_dict) == size:
            if right - left < answer[1] - answer[0]:    # 최소 크기 확인
                answer = [left, right]       
            else:
                gem_dict[gems[left]] -= 1
                if gem_dict[gems[left]] == 0:
                    del gem_dict[gems[left]]    # count가 0이 되면 key가 없어야하므로 반드시 del
                left += 1
                
        else:
            right += 1
            
            if right == len(gems):
                break
                
            if gems[right] in gem_dict: # 딕셔너리 key에 있으면 count
                gem_dict[gems[right]] += 1
                
            else:   # 없으면 추가
                gem_dict[gems[right]] = 1
    
    return [answer[0]+1, answer[1]+1] # 시작 인덱스가 1번 진열대 부터 라서 1 증가

# 합승 택시 요금
# 지점의 개수 n, 출발지점을 나타내는 s, A의 도착지점을 나타내는 a, B의 도착지점을 나타내는 b,
# 지점 사이의 예상 택시요금을 나타내는 fares가 매개변수로 주어집니다.
# 이때, A, B 두 사람이 s에서 출발해서 각각의 도착 지점까지 택시를 타고 간다고 가정할 때,
# 최저 예상 택시요금을 계산해서 return 하도록 solution 함수를 완성해 주세요.
# 만약, 아예 합승을 하지 않고 각자 이동하는 경우의 예상 택시요금이 더 낮다면, 합승을 하지 않아도 됩니다.
import heapq
def solution(n, s, a, b, fares):
    INF = 10000000
    answer = INF
    graph = [[] * (n + 1) for _ in range(n + 1)]

    for f in fares:
        node1, node2, fee = f
        # node1 - > node2 가는 요금이 fee
        graph[node1].append((node2, fee))
        # node2 - > node1 가는 요금이 fee
        graph[node2].append((node1, fee))

    def dijkstra(s):
        q = []
        # 최단거리 테이블을 무한으로 초기화
        distance = [INF] * (n + 1)
        # 거리(금액), 노드번호 순서
        heapq.heappush(q, (0, s))
        # 시작노드로 가는 최단거리는 0
        distance[s] = 0
        while q:
            dist, now = heapq.heappop(q)
            # 현재 노드가 이미 처리된 노드면 무시
            if distance[now] < dist:
                continue
            for g in graph[now]:
                cost = dist + g[1]
                if cost < distance[g[0]]:
                    distance[g[0]] = cost
                    heapq.heappush(q, (cost, g[0]))
        return distance

    distance_list = [[]] + [dijkstra(i) for i in range(1, n + 1)]

    for i in range(1, n + 1):
        answer = min(distance_list[s][i] + distance_list[i][a] + distance_list[i][b], answer)

    return answer

# N으로 표현
# 이처럼 숫자 N과 number가 주어질 때, N과 사칙연산만 사용해서
# 표현 할 수 있는 방법 중 N 사용횟수의 최솟값을 return 하도록 solution 함수를 작성하세요.
def solution(N, number):
    # s[i] : 주어진 수 N을 i+1번 사용해서 만들 수 있는 수들의 집합
    s = [set() for x in range(8)] # set 8개 초기화, 왜 8개를 만드냐? N 사용횟수가 8보다 크면 -1을 return하므로 N을 1개부터 8개 까지 사용하여 만든 값들이 number가 안될 경우 -1을 return한다.
    for i, x in enumerate(s, start = 1): # 보통 첫번째 원소의 idx는 0인데 여기서는 첫번째 원소의 idx를 1로 시작한다.
        x.add(int(str(N) * i)) # 8개의 set 각각 초기화, s[0] = N, s[1] = NN ... s[7] = NNNNNNNN (8개)
    # s[i] 즉 N을 i+1개 사용했을 때 만들 수 있는 숫자 구하기.
    for i in range(len(s)): 
        for j in range(i): 
            for op1 in s[j]: # op1 : 피연산자1, N을 j+1번 사용하여 만들 수 있는 숫자들
                for op2 in s[i-j-1]: # op2 : 피연산자2, N을 i-j번 사용하여 만들 수 있는 숫자들
                    # op1과 op2를 사칙연산 --> 즉 N을 i+1번 사용하여 만들 수 있는 숫자를 구하게 되고 이를 s[i]에 대입
                    s[i].add(op1 + op2)
                    s[i].add(op1 - op2)
                    s[i].add(op1 * op2)
                    if op2 != 0:
                        s[i].add(op1 // op2)
        if number in s[i]: # N을 i+1번 사용했을 때 찾고자하는 값 number가 존재한다면 i+1 return
            answer = i + 1
            break
        else: # N을 8번 사용했는데도 찾고자하는 값 number가 존재하지 않는다면 -1 return
            answer = -1
    return answer

# 경주로 건설
# 제공된 경주로 설계 도면에 따르면 경주로 부지는 N x N 크기의 정사각형 격자 형태이며 각 격자는 1 x 1 크기입니다.
# 설계 도면에는 각 격자의 칸은 0 또는 1 로 채워져 있으며, 0은 칸이 비어 있음을 1은 해당 칸이 벽으로 채워져 있음을 나타냅니다.
# 경주로의 출발점은 (0, 0) 칸(좌측 상단)이며, 도착점은 (N-1, N-1) 칸(우측 하단)입니다.
# 죠르디는 출발점인 (0, 0) 칸에서 출발한 자동차가 도착점인 (N-1, N-1) 칸까지 무사히 도달할 수 있게 중간에 끊기지 않도록 경주로를 건설해야 합니다.
# 경주로는 상, 하, 좌, 우로 인접한 두 빈 칸을 연결하여 건설할 수 있으며, 벽이 있는 칸에는 경주로를 건설할 수 없습니다.
# 이때, 인접한 두 빈 칸을 상하 또는 좌우로 연결한 경주로를 직선 도로 라고 합니다.
# 또한 두 직선 도로가 서로 직각으로 만나는 지점을 코너 라고 부릅니다.
# 건설 비용을 계산해 보니 직선 도로 하나를 만들 때는 100원이 소요되며, 코너를 하나 만들 때는 500원이 추가로 듭니다.
# 죠르디는 견적서 작성을 위해 경주로를 건설하는 데 필요한 최소 비용을 계산해야 합니다.
# 도면의 상태(0은 비어 있음, 1은 벽)을 나타내는 2차원 배열 board가 매개변수로 주어질 때,
# 경주로를 건설하는데 필요한 최소 비용을 return 하도록 solution 함수를 완성해주세요.
from collections import deque
def solution(board):
    def bfs(start):
        direc = {0:[-1, 0], 1:[0, 1], 2:[1, 0], 3:[0, -1]} # 북,동,남,서 순서
        length = len(board)
        visited = [[987654321]*length for _ in range(length)]
        visited[0][0] = 0

        q = deque([start]) # x, y, cost, dir
        while q:
            x, y, cost, d = q.popleft()
            for i in range(4): # 북,동,남,서 순서
                nx = x + direc[i][0]
                ny = y + direc[i][1]

                # board 안에 있고, 벽이 아닌지 확인
                if 0 <= nx < length and 0 <= ny < length and board[nx][ny] == 0:
                    
                    # 비용계산
                    if i == d : ncost = cost + 100
                    else : ncost =  cost + 600
                    # 최소 비용이면 갱신 후 endeque!
                    if ncost < visited[nx][ny]:
                        visited[nx][ny] = ncost
                        q.append([nx, ny, ncost, i])
                        
        return visited[-1][-1]
    
    return min([bfs((0, 0, 0, 1)), bfs((0, 0, 0, 2))])

# 기둥과 보 설치
# 프로그램은 2차원 가상 벽면에 기둥과 보를 이용한 구조물을 설치할 수 있는데, 기둥과 보는 길이가 1인 선분으로 표현되며 다음과 같은 규칙을 가지고 있습니다.
#   기둥은 바닥 위에 있거나 보의 한쪽 끝 부분 위에 있거나, 또는 다른 기둥 위에 있어야 합니다.
#   보는 한쪽 끝 부분이 기둥 위에 있거나, 또는 양쪽 끝 부분이 다른 보와 동시에 연결되어 있어야 합니다.
#   단, 바닥은 벽면의 맨 아래 지면을 말합니다.
# 2차원 벽면은 n x n 크기 정사각 격자 형태이며, 각 격자는 1 x 1 크기입니다. 맨 처음 벽면은 비어있는 상태입니다.
# 기둥과 보는 격자선의 교차점에 걸치지 않고, 격자 칸의 각 변에 정확히 일치하도록 설치할 수 있습니다.
# 벽면의 크기 n, 기둥과 보를 설치하거나 삭제하는 작업이 순서대로 담긴 2차원 배열 build_frame이 매개변수로 주어질 때,
# 모든 명령어를 수행한 후 구조물의 상태를 return 하도록 solution 함수를 완성해주세요.
def solution(n, build_frame):
    bow = [[0 for _ in range(n+1)] for _ in range(n)]
    gidung = [[0 for _ in range(n)] for _ in range(n+1)]

    def check_gidung(x, y):
        # 바닥 위 or 보의 한쪽 끝 or 또 다른 기둥 위
        return y == 0 or gidung[x][y-1] or (x < n and bow[x][y]) or (x > 0 and bow[x-1][y])

    def check_bow(x, y):
        # 한쪽 끝이 기둥 위 or 양쪽 끝이 보와 연결
        return gidung[x][y-1] or gidung[x+1][y-1] or (1 <= x < n-1 and bow[x-1][y] and bow[x+1][y])

    for x, y, a, b in build_frame:
        # 기둥
        if a == 0:
            # 기둥 설치
            if b == 1:
                # 설치 가능 조건인가
                if check_gidung(x, y):
                    gidung[x][y] = 1
            # 기둥 삭제
            else:
                # 일단 삭제
                gidung[x][y] = 0
                # 기둥 삭제 후 문제가 생기면 되돌리기
                
                # 연결된 기둥 확인
                    # 윗기둥만
                if y < n-1 and gidung[x][y+1] and not check_gidung(x, y+1):
                    gidung[x][y] = 1
                # 연결된 보 확인
                    # 위로 연결된 보만
                elif x < n and bow[x][y+1] and not check_bow(x, y+1):
                    gidung[x][y] = 1
                elif x-1>=0 and bow[x-1][y+1] and not check_bow(x-1, y+1):
                    gidung[x][y] = 1
            
                
        # 보
        else:
            # 보 설치
            if b == 1:
                # 보 설치 가능 조건인가?
                if check_bow(x, y):
                    bow[x][y] = 1
            # 보 삭제
            else:
                # 일단 삭제
                bow[x][y] = 0
                # 보 삭제 후 문제가 생기면 되돌리기
                
                # 연결된 보 확인
                if x > 0 and bow[x-1][y] and not check_bow(x-1, y):
                    bow[x][y] = 1
                elif x + 1 < n and bow[x+1][y] and not check_bow(x+1, y):
                    bow[x][y] = 1
                
                # 연결된 기둥 확인
                elif y < n and gidung[x][y] and not check_gidung(x, y):
                    bow[x][y] = 1
                elif y < n and gidung[x+1][y] and not check_gidung(x+1, y):
                    bow[x][y] = 1

    
    # 설치된 기둥, 보 찾기
    answer = []
    for x in range(n+1):
        for y in range(n+1):
            if y < n and gidung[x][y]:
                answer.append([x, y, 0])
            if x < n and bow[x][y]:
                answer.append([x, y, 1])

    return answer

# 순위검색
# 지원자가 지원서에 입력한 4가지의 정보와 획득한 코딩테스트 점수를 하나의 문자열로 구성한 값의 배열 info,
# 개발팀이 궁금해하는 문의조건이 문자열 형태로 담긴 배열 query가 매개변수로 주어질 때,
# 각 문의조건에 해당하는 사람들의 숫자를 순서대로 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
from itertools import combinations
from bisect import bisect_left


def solution(info, query):
    answer = []
    info_dict = {}

    for i in range(len(info)):
        infol = info[i].split()  # info안의 문자열을 공백을 기준으로 분리
        mykey = infol[:-1]  # info의 점수제외부분을 key로 분류
        myval = infol[-1]  # info의 점수부분을 value로 분류

        for j in range(5):  # key들로 만들 수 있는 모든 조합 생성
            for c in combinations(mykey, j):
                tmp = ''.join(c)
                if tmp in info_dict:
                    info_dict[tmp].append(int(myval))  # 그 조합의 key값에 점수 추가
                else:
                    info_dict[tmp] = [int(myval)]

    for k in info_dict:
        info_dict[k].sort()  # dict안의 조합들을 점수순으로 정렬

    for qu in query:  # query도 마찬가지로 key와 value로 분리
        myqu = qu.split(' ')
        qu_key = myqu[:-1]
        qu_val = myqu[-1]

        while 'and' in qu_key:  # and 제거
            qu_key.remove('and')
        while '-' in qu_key:  # - 제거
            qu_key.remove('-')
        qu_key = ''.join(qu_key)  # dict의 key처럼 문자열로 변경

        if qu_key in info_dict:  # query의 key가 info_dict의 key로 존재하면
            scores = info_dict[qu_key]

            if scores:  # score리스트에 값이 존재하면
                enter = bisect_left(scores, int(qu_val))

                answer.append(len(scores) - enter)
        else:
            answer.append(0)

    return answer

# 미로 탈출 명령어
# n x m 격자 미로가 주어집니다. 당신은 미로의 (x, y)에서 출발해 (r, c)로 이동해서 탈출해야 합니다.
# 단, 미로를 탈출하는 조건이 세 가지 있습니다.
#   격자의 바깥으로는 나갈 수 없습니다.
#   (x, y)에서 (r, c)까지 이동하는 거리가 총 k여야 합니다. 이때, (x, y)와 (r, c)격자를 포함해, 같은 격자를 두 번 이상 방문해도 됩니다.
#   미로에서 탈출한 경로를 문자열로 나타냈을 때, 문자열이 사전 순으로 가장 빠른 경로로 탈출해야 합니다.
# 이동 경로는 다음과 같이 문자열로 바꿀 수 있습니다.
#   l: 왼쪽으로 한 칸 이동
#   r: 오른쪽으로 한 칸 이동
#   u: 위쪽으로 한 칸 이동
#   d: 아래쪽으로 한 칸 이동
# 격자의 크기를 뜻하는 정수 n, m, 출발 위치를 뜻하는 정수 x, y, 탈출 지점을 뜻하는 정수 r, c,
# 탈출까지 이동해야 하는 거리를 뜻하는 정수 k가 매개변수로 주어집니다.
# 이때, 미로를 탈출하기 위한 경로를 return 하도록 solution 함수를 완성해주세요.
# 단, 위 조건대로 미로를 탈출할 수 없는 경우 "impossible"을 return 해야 합니다.
from collections import deque
def solution(n, m, x, y, r, c, k):
    answer = ''
    # 남은 거리 탐색 자주 해주어야 하므로 함수로 빼주기
    def manhattan(x1, y1):
        return abs(x1 - (r-1)) + abs(y1-(c-1))

    # k가 최단 거리보다 작거나, 최단 거리 - k가 홀수라면 도착지에 k번만에 도착 불가
    if manhattan(x-1, y-1) > k or (manhattan(x-1, y-1) - k) % 2:
        return 'impossible'
    # 탐색 방향 사전순으로 - d l r u
    direct = {(1,0):'d', (0,-1):'l', (0,1):'r', (-1,0):'u'}
    q = deque()
    q.append((x-1, y-1, 0, ''))
    while q:
        si, sj, cnt, route = q.popleft()
        # 도착했는데 남은 거리가 홀수라면 도착지에 k만큼 오지 못한다!
        if (si, sj) == (r-1, c-1) and (k-cnt) % 2:
            return 'impossible'
        elif (si, sj) == (r-1, c-1) and cnt == k:
            return route
        for di, dj in direct:
            ni, nj = si+di, sj+dj
            if 0<=ni<n and 0<=nj<m:
                # 다음 이동 자리를 보는 것이므로 +1 을 해주어야 함
                if manhattan(ni, nj) + cnt + 1 > k:
                    continue
                q.append((ni, nj, cnt+1, route+direct[(di, dj)]))
                break

    return answer

# 등산코스 정하기
# XX산은 n개의 지점으로 이루어져 있습니다. 각 지점은 1부터 n까지 번호가 붙어있으며,
# 출입구, 쉼터, 혹은 산봉우리입니다. 각 지점은 양방향 통행이 가능한 등산로로 연결되어 있으며,
# 서로 다른 지점을 이동할 때 이 등산로를 이용해야 합니다.
# 이때, 등산로별로 이동하는데 일정 시간이 소요됩니다.
# 등산코스는 방문할 지점 번호들을 순서대로 나열하여 표현할 수 있습니다.
# 등산코스를 따라 이동하는 중 쉼터 혹은 산봉우리를 방문할 때마다 휴식을 취할 수 있으며,
# 휴식 없이 이동해야 하는 시간 중 가장 긴 시간을 해당 등산코스의 intensity라고 부르기로 합니다.
# 당신은 XX산의 출입구 중 한 곳에서 출발하여 산봉우리 중 한 곳만 방문한 뒤 다시 원래의 출입구로 돌아오는 등산코스를 정하려고 합니다.
# 다시 말해, 등산코스에서 출입구는 처음과 끝에 한 번씩, 산봉우리는 한 번만 포함되어야 합니다.
# 당신은 이러한 규칙을 지키면서 intensity가 최소가 되도록 등산코스를 정하려고 합니다.
# XX산의 지점 수 n, 각 등산로의 정보를 담은 2차원 정수 배열 paths, 출입구들의 번호가 담긴 정수 배열 gates,
# 산봉우리들의 번호가 담긴 정수 배열 summits가 매개변수로 주어집니다.
# 이때, intensity가 최소가 되는 등산코스에 포함된 산봉우리 번호와 intensity의 최솟값을 차례대로 정수 배열에 담아 return 하도록 solution 함수를 완성해주세요.
# intensity가 최소가 되는 등산코스가 여러 개라면 그중 산봉우리의 번호가 가장 낮은 등산코스를 선택합니다.
from collections import defaultdict
from heapq import heappop, heappush
# n: 노드 수
# gates: 출입구, sumits: 산봉우리
def solution(n, paths, gates, summits):
    def get_min_intensity():
        pq = []  # (intensity, 현재 위치)
        visited = [10000001] * (n + 1)

        # 모든 출발지를 우선순위 큐에 삽입
        for gate in gates:
            heappush(pq, (0, gate))
            visited[gate] = 0

        # 산봉우리에 도착할 때까지 반복
        while pq:
            intensity, node = heappop(pq)

            # 산봉우리이거나 더 큰 intensity라면 더 이상 이동하지 않음
            if node in summits_set or intensity > visited[node]:
                continue

            # 이번 위치에서 이동할 수 있는 곳으로 이동
            for weight, next_node in graph[node]:
                # next_node 위치에 더 작은 intensity로 도착할 수 있다면 큐에 넣지 않음
                # (출입구는 이미 0으로 세팅되어있기 때문에 방문하지 않음)
                new_intensity = max(intensity, weight)
                if new_intensity < visited[next_node]:
                    visited[next_node] = new_intensity
                    heappush(pq, (new_intensity, next_node))

        # 구한 intensity 중 가장 작은 값 반환
        min_intensity = [0, 10000001]
        for summit in summits:
            if visited[summit] < min_intensity[1]:
                min_intensity[0] = summit
                min_intensity[1] = visited[summit]

        return min_intensity

    summits.sort()
    summits_set = set(summits)
    # graph: 등산로 정보
    graph = defaultdict(list)
    for i, j, w in paths:
        graph[i].append((w, j))
        graph[j].append((w, i))

    return get_min_intensity()

# 택배 배달과 수거하기
# 당신은 일렬로 나열된 n개의 집에 택배를 배달하려 합니다. 배달할 물건은 모두 크기가 같은 재활용 택배 상자에 담아 배달하며,
# 배달을 다니면서 빈 재활용 택배 상자들을 수거하려 합니다.
# 배달할 택배들은 모두 재활용 택배 상자에 담겨서 물류창고에 보관되어 있고,
# i번째 집은 물류창고에서 거리 i만큼 떨어져 있습니다. 또한 i번째 집은 j번째 집과 거리 j - i만큼 떨어져 있습니다. (1 ≤ i ≤ j ≤ n)
# 트럭에는 재활용 택배 상자를 최대 cap개 실을 수 있습니다. 트럭은 배달할 재활용 택배 상자들을 실어 물류창고에서 출발해 각 집에 배달하면서,
# 빈 재활용 택배 상자들을 수거해 물류창고에 내립니다. 각 집마다 배달할 재활용 택배 상자의 개수와 수거할 빈 재활용 택배 상자의 개수를 알고 있을 때,
# 트럭 하나로 모든 배달과 수거를 마치고 물류창고까지 돌아올 수 있는 최소 이동 거리를 구하려 합니다.
# 각 집에 배달 및 수거할 때, 원하는 개수만큼 택배를 배달 및 수거할 수 있습니다.
# 트럭에 실을 수 있는 재활용 택배 상자의 최대 개수를 나타내는 정수 cap, 배달할 집의 개수를 나타내는 정수 n,
# 각 집에 배달할 재활용 택배 상자의 개수를 담은 1차원 정수 배열 deliveries와
# 각 집에서 수거할 빈 재활용 택배 상자의 개수를 담은 1차원 정수 배열 pickups가 매개변수로 주어집니다.
# 이때, 트럭 하나로 모든 배달과 수거를 마치고 물류창고까지 돌아올 수 있는 최소 이동 거리를 return 하도록 solution 함수를 완성해 주세요.
# 맨 마지막 집부터 방문하면서 방문할 때 모든 배달과 수거가 0이 되도록
def solution(cap, n, deliveries, pickups):
    answer = 0
    deliver = 0   # 남은 배달 가능 개수
    pick = 0   # 남은 수거 가능 개수
    for i in range(n-1, -1, -1):
        cnt = 0
        while deliver < deliveries[i] or pick < pickups[i]:
            cnt += 1
            deliver += cap
            pick += cap
        deliver -= deliveries[i]
        pick -= pickups[i]
        answer += (i + 1) * cnt
    return answer * 2

# 가장 많이 받은 선물
# 선물을 직접 전하기 힘들 때 카카오톡 선물하기 기능을 이용해 축하 선물을 보낼 수 있습니다.
# 당신의 친구들이 이번 달까지 선물을 주고받은 기록을 바탕으로 다음 달에 누가 선물을 많이 받을지 예측하려고 합니다.
#   두 사람이 선물을 주고받은 기록이 있다면, 이번 달까지 두 사람 사이에 더 많은 선물을 준 사람이 다음 달에 선물을 하나 받습니다.
#   두 사람이 선물을 주고받은 기록이 하나도 없거나 주고받은 수가 같다면, 선물 지수가 더 큰 사람이 선물 지수가 더 작은 사람에게 선물을 하나 받습니다.
# 위에서 설명한 규칙대로 다음 달에 선물을 주고받을 때, 당신은 선물을 가장 많이 받을 친구가 받을 선물의 수를 알고 싶습니다.
# 친구들의 이름을 담은 1차원 문자열 배열 friends 이번 달까지 친구들이 주고받은 선물 기록을 담은 1차원 문자열 배열 gifts가 매개변수로 주어집니다.
# 이때, 다음달에 가장 많은 선물을 받는 친구가 받을 선물의 수를 return 하도록 solution 함수를 완성해 주세요.
def solution(friends, gifts):
    gifted = {} # gifted = {"friend 이름": {"선물 준 친구 이름": 이 친구에게 준 선물 개수}}
    gift_idx = {} # 선물 지수 
    # 딕셔너리 초기화
    for friend in friends:
        gifted[friend] = {}
        gift_idx[friend] = 0
    
    for gift in gifts:
        t, f = gift.split(' ') # t: 선물을 준 사람, f: 받은 사람
        if f in gifted[t]:
            gifted[t][f] += 1
        else:
            gifted[t][f] = 1
        # 선물 지수 반영
        gift_idx[t] += 1
        gift_idx[f] -= 1
    
    # 각자 받게 될 선물 개수
    will_get = [0 for _ in friends] # friends 리스트 순서대로 저장
    for i in range(len(friends)):
        curr = friends[i] # 인덱스 i에 해당하는 친구
        for j in range(i+1, len(friends)):
            another = friends[j] # 인덱스 j에 해당하는 친구
            # curr가 another에게 준 선물 개수
            a = gifted[curr][another] if another in gifted[curr] else 0 
            # another가 curr에게 준 선물 개수
            b = gifted[another][curr] if curr in gifted[another] else 0 
            
            if a > b: # curr가 선물을 더 많이 줬다면
                will_get[i] += 1
            elif a < b: # another가 선물을 더 많이 줬다면
                will_get[j] += 1
            elif a == b: # 둘이 선물을 주고 받은 개수가 같다면 선물 지수 확인
                ai, bi = gift_idx[curr], gift_idx[another]
                if ai > bi:
                    will_get[i] += 1
                elif ai < bi:
                    will_get[j] += 1
    
    answer = max(will_get)
    return answer

# 이모티콘 할인행사
# 카카오톡에서는 이모티콘을 무제한으로 사용할 수 있는 이모티콘 플러스 서비스 가입자 수를 늘리려고 합니다.
# 이를 위해 카카오톡에서는 이모티콘 할인 행사를 하는데, 목표는 다음과 같습니다.
#   1. 이모티콘 플러스 서비스 가입자를 최대한 늘리는 것.
#   2. 이모티콘 판매액을 최대한 늘리는 것.
# 1번 목표가 우선이며, 2번 목표가 그 다음입니다.
# 이모티콘 할인 행사는 다음과 같은 방식으로 진행됩니다.
#   n명의 카카오톡 사용자들에게 이모티콘 m개를 할인하여 판매합니다.
#   이모티콘마다 할인율은 다를 수 있으며, 할인율은 10%, 20%, 30%, 40% 중 하나로 설정됩니다.
# 카카오톡 사용자들은 다음과 같은 기준을 따라 이모티콘을 사거나, 이모티콘 플러스 서비스에 가입합니다.
#   각 사용자들은 자신의 기준에 따라 일정 비율 이상 할인하는 이모티콘을 모두 구매합니다.
#   각 사용자들은 자신의 기준에 따라 이모티콘 구매 비용의 합이 일정 가격 이상이 된다면, 이모티콘 구매를 모두 취소하고 이모티콘 플러스 서비스에 가입합니다.
# 카카오톡 사용자 n명의 구매 기준을 담은 2차원 정수 배열 users, 이모티콘 m개의 정가를 담은 1차원 정수 배열 emoticons가 주어집니다.
# 이때, 행사 목적을 최대한으로 달성했을 때의 이모티콘 플러스 서비스 가입 수와 이모티콘 매출액을 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def solution(users, emoticons):
    answer = [0, 0]
    data = [10, 20, 30, 40]
    discount = []
    def dfs(tmp, d): # 모든 경우의 할인율 조합을 구함
        if d == len(tmp):
            discount.append(tmp[:])
            return
        else:
            for i in data:
                tmp[d] += i
                dfs(tmp, d+1)
                tmp[d] -= i
    dfs([0]*len(emoticons), 0)
    
    for disc in discount: # 만들어진 모든 조합을 하나씩 살펴봄
        cnt = 0
        get = 0
        for i in users:
            pay = 0
            for j in range(len(disc)):
                if i[0] <= disc[j]:
                    pay += emoticons[j] * (100 - disc[j])/100
                if pay >= i[1]:
                    break
            if pay >= i[1]: # 만약 유저의 제한금액 초과시 플러스 구매
                pay = 0
                cnt += 1
            get += pay
        if cnt >= answer[0]: # 현재 최대값을 넘어가면 갱신
            if cnt == answer[0]:
                answer[1] = max(answer[1], get)
            else:
                answer[1] = get
            answer[0] = cnt

    return answer

# 붕대감기
# 붕대 감기는 t초 동안 붕대를 감으면서 1초마다 x만큼의 체력을 회복합니다.
# t초 연속으로 붕대를 감는 데 성공한다면 y만큼의 체력을 추가로 회복합니다.
# 게임 캐릭터에는 최대 체력이 존재해 현재 체력이 최대 체력보다 커지는 것은 불가능합니다.
# 기술을 쓰는 도중 몬스터에게 공격을 당하면 기술이 취소되고, 공격을 당하는 순간에는 체력을 회복할 수 없습니다.
# 몬스터에게 공격당해 기술이 취소당하거나 기술이 끝나면 그 즉시 붕대 감기를 다시 사용하며, 연속 성공 시간이 0으로 초기화됩니다.
# 몬스터의 공격을 받으면 정해진 피해량만큼 현재 체력이 줄어듭니다.
# 이때, 현재 체력이 0 이하가 되면 캐릭터가 죽으며 더 이상 체력을 회복할 수 없습니다.
# 당신은 붕대감기 기술의 정보, 캐릭터가 가진 최대 체력과 몬스터의 공격 패턴이 주어질 때 캐릭터가 끝까지 생존할 수 있는지 궁금합니다.
# 붕대 감기 기술의 시전 시간, 1초당 회복량, 추가 회복량을 담은 1차원 정수 배열 bandage와
# 최대 체력을 의미하는 정수 health, 몬스터의 공격 시간과 피해량을 담은 2차원 정수 배열 attacks가 매개변수로 주어집니다.
# 모든 공격이 끝난 직후 남은 체력을 return 하도록 solution 함수를 완성해 주세요.
# 만약 몬스터의 공격을 받고 캐릭터의 체력이 0 이하가 되어 죽는다면 -1을 return 해주세요.
def solution(bandage, health, attacks):
    t, x, y = bandage
    max_health = health
    end_time = attacks[-1][0]
    attacks = {attack[0]:attack[1] for attack in attacks}
    cur_t = 0
    cur_health = health
    for i in range(end_time + 1):
        # 공격
        if i in attacks:
            cur_t = 0
            cur_health -= attacks[i]
            
            # 사망
            if cur_health <= 0:
                return -1
            continue
        
        # 공격받지 않음
        cur_t += 1
        cur_health += x
        
        # 추가 회복
        if cur_t == t:
            cur_health += y
            cur_t = 0
            
        cur_health = min(cur_health, max_health)
    
    return cur_health

# [PCCP 기출문제] 2번 / 석유 시추
# 세로길이가 n 가로길이가 m인 격자 모양의 땅 속에서 석유가 발견되었습니다.
# 석유는 여러 덩어리로 나누어 묻혀있습니다. 당신이 시추관을 수직으로 단 하나만 뚫을 수 있을 때,
# 가장 많은 석유를 뽑을 수 있는 시추관의 위치를 찾으려고 합니다.
# 시추관은 열 하나를 관통하는 형태여야 하며, 열과 열 사이에 시추관을 뚫을 수 없습니다.
# 석유가 묻힌 땅과 석유 덩어리를 나타내는 2차원 정수 배열 land가 매개변수로 주어집니다.
# 이때 시추관 하나를 설치해 뽑을 수 있는 가장 많은 석유량을 return 하도록 solution 함수를 완성해 주세요.
from collections import deque
def solution(land):
    answer = 0
    n, m = len(land), len(land[0])
    # 방문배열
    visited = [[False] * m for _ in range(n)]
    # 각 열의 기름의 총량을 저장하는 리스트
    oil = [0] * m
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    def bfs(row, col):
        queue = deque()
        queue.append([row, col])
        visited[row][col] = True
        cnt = 1
        # bfs 탐색중 석유가 있는 열, 중복을 방지하기 위한 set
        oil_covered = {col}

        while queue:
            pr, pc = queue.popleft()
            for dr, dc in directions:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < n and 0 <= nc < m and land[nr][nc] == 1 and not visited[nr][nc]:
                    queue.append([nr, nc])
                    visited[nr][nc] = True
                    cnt += 1
                    # 현재 석유를 발견한 열 추가
                    oil_covered.add(nc)

        # 각 열을 돌면서 석유량 추가
        for c in oil_covered:
            oil[c] += cnt

    # bfs 탐색
    for i in range(n):
        for j in range(m):
            if land[i][j] == 1 and not visited[i][j]:
                bfs(i, j)

    answer = max(oil)
    return answer

# 파괴되지 않은 건물
# N x M 크기의 행렬 모양의 게임 맵이 있습니다. 이 맵에는 내구도를 가진 건물이 각 칸마다 하나씩 있습니다.
# 적은 이 건물들을 공격하여 파괴하려고 합니다. 건물은 적의 공격을 받으면 내구도가 감소하고 내구도가 0이하가 되면 파괴됩니다.
# 반대로, 아군은 회복 스킬을 사용하여 건물들의 내구도를 높이려고 합니다.
# 적의 공격과 아군의 회복 스킬은 항상 직사각형 모양입니다.
# 건물의 내구도를 나타내는 2차원 정수 배열 board와 적의 공격 혹은 아군의 회복 스킬을 나타내는 2차원 정수 배열 skill이 매개변수로 주어집니다.
# 적의 공격 혹은 아군의 회복 스킬이 모두 끝난 뒤 파괴되지 않은 건물의 개수를 return하는 solution함수를 완성해 주세요.
def solution(board, skill):
    R, C = len(board), len(board[0])
    delta = [[0] * (C+1) for _ in range(R+1)]
    
    for op, rmin, cmin, rmax, cmax, degree in skill:
        degree = -degree if op == 1 else degree
        
        delta[rmin][cmin] += degree
        delta[rmax+1][cmin] -= degree
        delta[rmin][cmax+1] -= degree
        delta[rmax+1][cmax+1] += degree
        
    for r in range(R):
        for c in range(1, C):
            delta[r][c] += delta[r][c-1]
    
    for c in range(C):
        for r in range(1, R):
            delta[r][c] += delta[r-1][c]

    return sum(board[r][c] + delta[r][c] > 0 for r in range(R) for c in range(C))

# 블록 이동하기
# 로봇개발자 "무지"는 한 달 앞으로 다가온 "카카오배 로봇경진대회"에 출품할 로봇을 준비하고 있습니다.
# 준비 중인 로봇은 2 x 1 크기의 로봇으로 "무지"는 "0"과 "1"로 이루어진 N x N 크기의 지도에서
# 2 x 1 크기인 로봇을 움직여 (N, N) 위치까지 이동 할 수 있도록 프로그래밍을 하려고 합니다.
# 로봇이 이동하는 지도는 가장 왼쪽, 상단의 좌표를 (1, 1)로 하며 지도 내에 표시된 숫자 "0"은 빈칸을 "1"은 벽을 나타냅니다.
# 로봇은 벽이 있는 칸 또는 지도 밖으로는 이동할 수 없습니다.
# 로봇은 처음에 아래 그림과 같이 좌표 (1, 1) 위치에서 가로방향으로 놓여있는 상태로 시작하며, 앞뒤 구분없이 움직일 수 있습니다.
# 로봇이 움직일 때는 현재 놓여있는 상태를 유지하면서 이동합니다.
# 위 그림과 같이 로봇은 90도씩 회전할 수 있습니다.
# 단, 로봇이 차지하는 두 칸 중, 어느 칸이든 축이 될 수 있지만,
# 회전하는 방향(축이 되는 칸으로부터 대각선 방향에 있는 칸)에는 벽이 없어야 합니다.
# 로봇이 한 칸 이동하거나 90도 회전하는 데는 걸리는 시간은 정확히 1초 입니다.
# "0"과 "1"로 이루어진 지도인 board가 주어질 때,
# 로봇이 (N, N) 위치까지 이동하는데 필요한 최소 시간을 return 하도록 solution 함수를 완성해주세요.
from collections import deque
def can_move(cur1, cur2, new_board):
    Y, X = 0, 1
    cand = []
    # 평행이동
    DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for dy, dx in DELTAS:
        nxt1 = (cur1[Y] + dy, cur1[X] + dx)
        nxt2 = (cur2[Y] + dy, cur2[X] + dx)
        if new_board[nxt1[Y]][nxt1[X]] == 0 and new_board[nxt2[Y]][nxt2[X]] == 0:
            cand.append((nxt1, nxt2))
    # 회전
    if cur1[Y] == cur2[Y]: # 가로방향 일 때
        UP, DOWN = -1, 1
        for d in [UP, DOWN]:
            if new_board[cur1[Y]+d][cur1[X]] == 0 and new_board[cur2[Y]+d][cur2[X]] == 0:
                cand.append((cur1, (cur1[Y]+d, cur1[X])))
                cand.append((cur2, (cur2[Y]+d, cur2[X])))
    else: # 세로 방향 일 때
        LEFT, RIGHT = -1, 1
        for d in [LEFT, RIGHT]:
            if new_board[cur1[Y]][cur1[X]+d] == 0 and new_board[cur2[Y]][cur2[X]+d] == 0:
                cand.append(((cur1[Y], cur1[X]+d), cur1))
                cand.append(((cur2[Y], cur2[X]+d), cur2))
                
    return cand

def solution(board):
    # board 외벽 둘러싸기
    N = len(board)
    new_board = [[1] * (N+2) for _ in range(N+2)]
    for i in range(N):
        for j in range(N):
            new_board[i+1][j+1] = board[i][j]

    # 현재 좌표 위치 큐 삽입, 확인용 set
    que = deque([((1, 1), (1, 2), 0)])
    confirm = set([((1, 1), (1, 2))])

    while que:
        cur1, cur2, count = que.popleft()
        if cur1 == (N, N) or cur2 == (N, N):
            return count
        for nxt in can_move(cur1, cur2, new_board):
            if nxt not in confirm:
                que.append((*nxt, count+1))
                confirm.add(nxt)
                
# 산 모양 타일링
# 한 변의 길이가 1인 정삼각형 2n+1개를 이어붙여 윗변의 길이가 n,
# 아랫변의 길이가 n+1인 사다리꼴을 만들 수 있습니다.
# 이때 사다리꼴의 윗변과 변을 공유하는 n개의 정삼각형 중 일부의 위쪽에 같은 크기의 정삼각형을 붙여 새로운 모양을 만들었습니다.
# 사다리꼴의 윗변의 길이를 나타내는 정수 n과 사다리꼴 윗변에 붙인 정삼각형을 나타내는 1차원 정수 배열 tops가 매개변수로 주어집니다.
# 이때 문제 설명에 따라 만든 모양을 정삼각형 또는 마름모 타일로 빈 곳이 없도록 채우는 경우의 수를 10007로 나눈 나머지를 return 하도록 solution 함수를 완성해 주세요.
def solution(n, tops):
    MOD = 10007
    dp1 = [0] * n
    dp2 = [0] * n
    dp1[0] = 1
    dp2[0] = 2 + tops[0]
    
    for i in range(1, n):
        dp1[i] = (dp1[i - 1] + dp2[i - 1]) % MOD
        dp2[i] = ((dp1[i - 1] * (1 + tops[i])) + \
                (dp2[i - 1] * (2 + tops[i]))) % MOD
        
    return (dp1[-1] + dp2[-1]) % MOD

# 다단계 칫솔 판매
# 민호는 다단계 조직을 이용하여 칫솔을 판매하고 있습니다.
# 판매원이 칫솔을 판매하면 그 이익이 피라미드 조직을 타고 조금씩 분배되는 형태의 판매망입니다. 
# 어느정도 판매가 이루어진 후, 조직을 운영하던 민호는 조직 내 누가 얼마만큼의 이득을 가져갔는지가 궁금해졌습니다.
# 민호는 center이며, 파란색 네모는 여덟 명의 판매원을 표시한 것입니다.
# 각각은 자신을 조직에 참여시킨 추천인에 연결되어 피라미드 식의 구조를 이루고 있습니다.
# 조직의 이익 분배 규칙은 간단합니다. 모든 판매원은 칫솔의 판매에 의하여 발생하는 이익에서 10% 를 계산하여
# 자신을 조직에 참여시킨 추천인에게 배분하고 나머지는 자신이 가집니다.
# 모든 판매원은 자신이 칫솔 판매에서 발생한 이익 뿐만 아니라, 자신이 조직에 추천하여 가입시킨 판매원에게서 발생하는 이익의 10% 까지 자신에 이익이 됩니다.
# 자신에게 발생하는 이익 또한 마찬가지의 규칙으로 자신의 추천인에게 분배됩니다.
# 단, 10% 를 계산할 때에는 원 단위에서 절사하며, 10%를 계산한 금액이 1 원 미만인 경우에는 이득을 분배하지 않고 자신이 모두 가집니다.
# 각 판매원의 이름을 담은 배열 enroll, 각 판매원을 다단계 조직에 참여시킨 다른 판매원의 이름을 담은 배열 referral,
# 판매량 집계 데이터의 판매원 이름을 나열한 배열 seller, 판매량 집계 데이터의 판매 수량을 나열한 배열 amount가 매개변수로 주어질 때,
# 각 판매원이 득한 이익금을 나열한 배열을 return 하도록 solution 함수를 완성해주세요.
# 판매원에게 배분된 이익금의 총합을 계산하여(정수형으로), 입력으로 주어진 enroll에 이름이 포함된 순서에 따라 나열하면 됩니다.
import math
def solution(enroll, referral, seller, amount):
    parentTree = dict(zip(enroll, referral))
    answer = dict(zip(enroll, [0 for i in range(len(enroll))]))

    for i in range(len(seller)):
        earn = amount[i] * 100
        target = seller[i]

        while True :
            if earn < 10 : #10원 단위 이하라면 모두 받고 레퍼럴 종료
                answer[target] += earn
                break
            else : #10% 레퍼럴을 제외하고 받는다
                answer[target] += math.ceil(earn * 0.9)
                if parentTree[target] == "-": #상위가 없다면 종료
                    break
                earn = math.floor(earn*0.1)
                target = parentTree[target]
                    
    return list(answer.values())

# 최적의 행렬 곱셈
# 크기가 a by b인 행렬과 크기가 b by c 인 행렬이 있을 때,
# 두 행렬을 곱하기 위해서는 총 a x b x c 번 곱셈해야합니다.
# 각 행렬의 크기 matrix_sizes 가 매개변수로 주어 질 때,
# 모든 행렬을 곱하기 위한 최소 곱셈 연산의 수를 return하는 solution 함수를 완성해 주세요.
def solution(sizes):
    dp = [[0 for j in range(len(sizes))] for i in range(len(sizes))]
    for gap in range(1, len(sizes)) : 
        for s in range(0, len(sizes)-gap) : 
            e = s+gap
            
            candidate = list()
            for m in range(s, e) :
                candidate.append(
                    dp[s][m]+dp[m+1][e]+
                    sizes[s][0]*sizes[m][1]*sizes[e][1])
            dp[s][e] = min(candidate)
            
    return dp[0][-1]

# 아이템 줍기
# 지형은 각 변이 x축, y축과 평행한 직사각형이 겹쳐진 형태로 표현하며,
# 캐릭터는 이 다각형의 둘레(굵은 선)를 따라서 이동합니다.
# 만약 직사각형을 겹친 후 다음과 같이 중앙에 빈 공간이 생기는 경우,
# 다각형의 가장 바깥쪽 테두리가 캐릭터의 이동 경로가 됩니다.
# 단, 서로 다른 두 직사각형의 x축 좌표 또는 y축 좌표가 같은 경우는 없습니다.
# 즉, 위 그림처럼 서로 다른 두 직사각형이 꼭짓점에서 만나거나, 변이 겹치는 경우 등은 없습니다.
# 다음 그림과 같이 지형이 2개 이상으로 분리된 경우도 없습니다.
# 지형을 나타내는 직사각형이 담긴 2차원 배열 rectangle, 초기 캐릭터의 위치 characterX, characterY,
# 아이템의 위치 itemX, itemY가 solution 함수의 매개변수로 주어질 때,
# 캐릭터가 아이템을 줍기 위해 이동해야 하는 가장 짧은 거리를 return 하도록 solution 함수를 완성해주세요.
from collections import deque
def solution(rectangle, characterX, characterY, itemX, itemY):
    answer = 0
    graph = [[-1 for _ in range(102)] for _ in range(102)]
    visited = [[1 for _ in range(102)] for _ in range(102)]
    direction = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    dq = deque()
    for r in rectangle:
        x1, y1, x2, y2 = map(lambda x: x*2, r)
        for i in range(x1, x2+1):
            for j in range(y1, y2+1):
                # x1, x2, y1, y2는 테두리이므로 제외하고 내부만 0으로 채움
                if x1 < i < x2 and y1 < j < y2:
                    graph[i][j] = 0
                # 다른 직사각형의 내부가 아니면서 테두리일 때 1로 채움
                elif graph[i][j] != 0:
                    graph[i][j] = 1                
    # 반복문을 마치면 테두리는 1, 내부는 0, 외부는 -1이 될 것이다   
    # 캐릭터와 아이템의 좌표도 2배씩 늘린다
    cx, cy, ix, iy = 2*characterX, 2*characterY, 2*itemX, 2*itemY
    dq.append((cx, cy))
    while dq:
        x, y = dq.popleft()
        if x == ix and y == iy:
        	# 답을 반환할 때 2로 나누어 반환해준다
            answer = visited[x][y] // 2
            break   
        for k in range(4):
            nx, ny = x + direction[k][0], y + direction[k][1]
            if graph[nx][ny] == 1 and visited[nx][ny] == 1:
                visited[nx][ny] += visited[x][y]
                dq.append((nx, ny))
    return answer

# 표현 가능한 이진트리
# 당신은 이진트리를 수로 표현하는 것을 좋아합니다.
# 이진트리를 수로 표현하는 방법은 다음과 같습니다.
#   1.이진수를 저장할 빈 문자열을 생성합니다.
#   2.주어진 이진트리에 더미 노드를 추가하여 포화 이진트리로 만듭니다. 루트 노드는 그대로 유지합니다.
#   3.만들어진 포화 이진트리의 노드들을 가장 왼쪽 노드부터 가장 오른쪽 노드까지, 왼쪽에 있는 순서대로 살펴봅니다. 노드의 높이는 살펴보는 순서에 영향을 끼치지 않습니다.
#   4.살펴본 노드가 더미 노드라면, 문자열 뒤에 0을 추가합니다. 살펴본 노드가 더미 노드가 아니라면, 문자열 뒤에 1을 추가합니다.
#   5.문자열에 저장된 이진수를 십진수로 변환합니다.
# 이진트리에서 리프 노드가 아닌 노드는 자신의 왼쪽 자식이 루트인 서브트리의 노드들보다 오른쪽에 있으며,
# 자신의 오른쪽 자식이 루트인 서브트리의 노드들보다 왼쪽에 있다고 가정합니다.
# 이진트리로 만들고 싶은 수를 담은 1차원 정수 배열 numbers가 주어집니다. numbers에 주어진 순서대로 하나의 이진트리로 해당 수를 표현할 수 있다면 1을,
# 표현할 수 없다면 0을 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def dfs(b, i, depth):
    if depth == 0:  	# 리프 노드에 도달했다면
        return True 	# 포화이진트리
    
    # 부모노드가 '0' 일때
    # 왼쪽 자식 노드가 '1' 이거나 오른쪽 자식 노드가 '1' 이라면 포화 이진트리가 될 수 없음
    elif b[i] == '0':   
        if b[i - depth] == '1' or b[i + depth] == '1': return False

    # 왼쪽 서브 트리 탐색
    left = dfs(b, i - depth, depth // 2)
    # 오른쪽 서브 트리 탐색
    right = dfs(b, i + depth, depth // 2)
    return left and right
    
    
def solution(numbers):
    answer = []
    for num in numbers:				# num = 42
        b = bin(num)[2:]  			# b = 101010 / len(b) = 6
        nodes = bin(len(b) + 1)[2:] 	# nodes = 7 = 111
        
        # 포화이진트리가 아닌 경우 더미노드(0추가)
        if '1' in nodes[1:]:       
            dummies = (1 << len(nodes)) - int(nodes, 2)
            b = '0' * dummies + b
            
        # 이미 포화이진트리일 경우
        result = dfs(b, len(b)//2, (len(b)+1)//4)
        answer.append(1 if result else 0)
        
    return answer

# 매칭 점수
# 프렌즈 대학교 조교였던 제이지는 허드렛일만 시키는 네오 학과장님의 마수에서 벗어나, 카카오에 입사하게 되었다.
# 평소에 관심있어하던 검색에 마침 결원이 발생하여, 검색개발팀에 편입될 수 있었고, 대망의 첫 프로젝트를 맡게 되었다.
# 그 프로젝트는 검색어에 가장 잘 맞는 웹페이지를 보여주기 위해 아래와 같은 규칙으로 검색어에 대한 웹페이지의 매칭점수를 계산 하는 것이었다.
#   한 웹페이지에 대해서 기본점수, 외부 링크 수, 링크점수, 그리고 매칭점수를 구할 수 있다.
#   한 웹페이지의 기본점수는 해당 웹페이지의 텍스트 중, 검색어가 등장하는 횟수이다. (대소문자 무시)
#   한 웹페이지의 외부 링크 수는 해당 웹페이지에서 다른 외부 페이지로 연결된 링크의 개수이다.
#   한 웹페이지의 링크점수는 해당 웹페이지로 링크가 걸린 다른 웹페이지의 기본점수 ÷ 외부 링크 수의 총합이다.
#   한 웹페이지의 매칭점수는 기본점수와 링크점수의 합으로 계산한다.
# 검색어 word와 웹페이지의 HTML 목록인 pages가 주어졌을 때, 매칭점수가 가장 높은 웹페이지의 index를 구하라.
# 만약 그런 웹페이지가 여러 개라면 그중 번호가 가장 작은 것을 구하라.
import re
def solution(word, pages):
    webpage = []
    webpageName = []
    webpageGraph = dict() # 나를 가리키는 외부 링크
    
    for page in pages:
        url = re.search('<meta property="og:url" content="(\S+)"', page).group(1)
        basicScore = 0
        for f in re.findall(r'[a-zA-Z]+', page.lower()):
            if f == word.lower():
                basicScore += 1
        exiosLink = re.findall('<a href="(https://[\S]*)"', page)
        
        for link in exiosLink:
            if link not in webpageGraph.keys():
                webpageGraph[link] = [url]
            else:
                webpageGraph[link].append(url)
        
        webpageName.append(url)
        webpage.append([url, basicScore, len(exiosLink)]) # 내가 가진 외부 링크 (개수)
        
    # 링크점수 = 해당 웹페이지로 링크가 걸린 다른 웹페이지의 기본점수 ÷ 외부 링크 수의 총합
    # 매칭점수 = 기본점수 + 링크점수
    maxValue = 0
    result = 0
    for i in range(len(webpage)):
        url = webpage[i][0]
        score = webpage[i][1]
        
        if url in webpageGraph.keys():
            # 나를 가리키는 다른 링크의 기본점수 ÷ 외부 링크 수의 총합을 구하기 위해
            for link in webpageGraph[url]: 
                a, b, c = webpage[webpageName.index(link)]
                score += (b / c)
        
        if maxValue < score:
            maxValue = score
            result = i
    
    return result

# 표 편집
# 위 그림에서 파란색으로 칠해진 칸은 현재 선택된 행을 나타냅니다. 단, 한 번에 한 행만 선택할 수 있으며, 표의 범위(0행 ~ 마지막 행)를 벗어날 수 없습니다.
# 이때, 다음과 같은 명령어를 이용하여 표를 편집합니다.
#   "U X": 현재 선택된 행에서 X칸 위에 있는 행을 선택합니다.
#   "D X": 현재 선택된 행에서 X칸 아래에 있는 행을 선택합니다.
#   "C" : 현재 선택된 행을 삭제한 후, 바로 아래 행을 선택합니다. 단, 삭제된 행이 가장 마지막 행인 경우 바로 윗 행을 선택합니다.
#   "Z" : 가장 최근에 삭제된 행을 원래대로 복구합니다. 단, 현재 선택된 행은 바뀌지 않습니다.
# 처음 표의 행 개수를 나타내는 정수 n, 처음에 선택된 행의 위치를 나타내는 정수 k,
# 수행한 명령어들이 담긴 문자열 배열 cmd가 매개변수로 주어질 때,
# 모든 명령어를 수행한 후 표의 상태와 처음 주어진 표의 상태를 비교하여 삭제되지 않은 행은 O,
# 삭제된 행은 X로 표시하여 문자열 형태로 return 하도록 solution 함수를 완성해주세요.
N = 10**9
class Node:
    # 활성, 비활성
    live = True
    # 이전 노드와 다음 노드
    def __init__(self, p, n):
        self.prev = p if p >= 0 else None
        self.next = n if n < N else None
def solution(n, k, camaand):
    global N
    N = n
    # linked list
    table = {i: Node(i-1, i+1) for i in range(n)}
    # 현재 선택된 행
    now = k
    # 삭제된 번호
    stack = []
    for cmd in camaand:
        # 삭제
        if cmd[0] == 'C':
            # 비활성
            table[now].live = False
            stack.append(now)
            prev, next = table[now].prev, table[now].next
            
            # 이전 노드가 있다면 현재 노드의 다음 노드와 연결
            if prev is not None:
                table[prev].next = next
            # 다음 노드가 있다면 이전 노드를 다음 노드와 연결
            if next is not None:
                table[next].prev = prev
            # 다음 노드가 없다면 이전 노드 선택, 아니면 다음 노드 선택택
            if table[now].next is None:
                now = table[now].prev
            else:
                now = table[now].next

        # 복구
        elif cmd[0] == 'Z':
            # 활성
            re = stack.pop()
            table[re].live = True
            prev, next = table[re].prev, table[re].next

            # 이전 노드가 있다면 복구 행과 이전노드 연결
            if prev is not None:
                table[prev].next = re
            # 다음 노드가 있다면 복구 행과 다음 노드 연결
            if next is not None:
                table[next].prev = re
        else:
            c, amout = cmd.split()
            # 위
            if c == 'U':
                # 연결된 이전 노드로 계속 변경
                for _ in range(int(amout)):
                    now = table[now].prev

            # 아래
            else:
                # 연결된 다음 노드로 계속 이동
                for _ in range(int(amout)):
                    now = table[now].next

    return ''.join('O' if table[i].live else 'X' for i in range(n))

# 길 찾기 게임
# 그냥 지도를 주고 게임을 시작하면 재미가 덜해지므로,
# 라이언은 방문할 곳의 2차원 좌표 값을 구하고 각 장소를 이진트리의 노드가 되도록 구성한 후,
# 순회 방법을 힌트로 주어 각 팀이 스스로 경로를 찾도록 할 계획이다.
# 라이언은 아래와 같은 특별한 규칙으로 트리 노드들을 구성한다.
# 트리를 구성하는 모든 노드의 x, y 좌표 값은 정수이다.
#   모든 노드는 서로 다른 x값을 가진다.
#   같은 레벨(level)에 있는 노드는 같은 y 좌표를 가진다.
#   자식 노드의 y 값은 항상 부모 노드보다 작다.
#   임의의 노드 V의 왼쪽 서브 트리(left subtree)에 있는 모든 노드의 x값은 V의 x값보다 작다.
#   임의의 노드 V의 오른쪽 서브 트리(right subtree)에 있는 모든 노드의 x값은 V의 x값보다 크다.
# 곤경에 빠진 카카오 프렌즈를 위해 이진트리를 구성하는 노드들의 좌표가 담긴 배열 nodeinfo가 매개변수로 주어질 때,
# 노드들로 구성된 이진트리를 전위 순회, 후위 순회한 결과를 2차원 배열에 순서대로 담아 return 하도록 solution 함수를 완성하자.
import sys; sys.setrecursionlimit(10001)
from collections import namedtuple
Node = namedtuple('Node', ['x', 'y', 'id'])
def solution(nodeinfo):
    # 특정 노드를 기준으로 partition을 잘 하면 될 것 같습니다.
    # 우선 nodeinfo를 (x, y, id)의 배열로 바꿉니다.
    nodeinfo = [Node(x, y, id) for id, (x, y) in enumerate(nodeinfo, 1)]
    
    def preorder(arr):
        if len(arr) < 1:
            return [node.id for node in arr]
        
        # 전위순회부터 해보면, 배열에서 y좌표가 가장 큰 값을 pivot으로 두고
        # 먼저 pivot의 id를 리턴 배열에 넣은 뒤
        # x좌표의 대소를 기준으로 나뉜 두 배열에서
        # 전위순회를 재귀적으로 돌려주면 됩니다.
        pivot = max(arr, key=lambda node: node.y)
        
        res = [pivot.id]
        res += preorder([node for node in arr if node.x < pivot.x])
        res += preorder([node for node in arr if node.x > pivot.x])
        
        return res
    
    def postorder(arr):
        if len(arr) < 1:
            return [node.id for node in arr]
        
        # 후위순회는, 배열에서 y좌표가 가장 큰 값을 pivot으로 두고
        # x좌표의 대소를 기준으로 나뉜 두 배열에서
        # 후위순회를 재귀적으로 돌린 다음 pivot의 id를 리턴 배열에 넣으면 됩니다.
        pivot = max(arr, key=lambda node: node.y)
        
        res = []
        res += postorder([node for node in arr if node.x < pivot.x])
        res += postorder([node for node in arr if node.x > pivot.x])
        res.append(pivot.id)
        
        return res
        
    return preorder(nodeinfo), postorder(nodeinfo)

# 추석 트래픽
# 장애 대비용 서버 증설 여부를 결정하기 위해 작년 추석 기간인 9월 15일 로그 데이터를 분석한 후 초당 최대 처리량을 계산해보기로 했다.
# 초당 최대 처리량은 요청의 응답 완료 여부에 관계없이 임의 시간부터 1초(=1,000밀리초)간 처리하는 요청의 최대 개수를 의미한다.
#   solution 함수에 전달되는 lines 배열은 N(1 ≦ N ≦ 2,000)개의 로그 문자열로 되어 있으며,
#       각 로그 문자열마다 요청에 대한 응답완료시간 S와 처리시간 T가 공백으로 구분되어 있다.
#   응답완료시간 S는 작년 추석인 2016년 9월 15일만 포함하여 고정 길이 2016-09-15 hh:mm:ss.sss 형식으로 되어 있다.
#   처리시간 T는 0.1s, 0.312s, 2s 와 같이 최대 소수점 셋째 자리까지 기록하며 뒤에는 초 단위를 의미하는 s로 끝난다.
#       예를 들어, 로그 문자열 2016-09-15 03:10:33.020 0.011s은
#       "2016년 9월 15일 오전 3시 10분 33.010초"부터 "2016년 9월 15일 오전 3시 10분 33.020초"까지 "0.011초" 동안 처리된 요청을 의미한다.
#       (처리시간은 시작시간과 끝시간을 포함)
#   서버에는 타임아웃이 3초로 적용되어 있기 때문에 처리시간은 0.001 ≦ T ≦ 3.000이다.
#   lines 배열은 응답완료시간 S를 기준으로 오름차순 정렬되어 있다.
def solution(lines):
    answer = 0
    log = [] # log의 (시작시간, 끝시간) 저장
    for line in lines:
        date, s, t = line.split() # 날짜, 응답완료시간, 처리시간
        s = s.split(':')
        t = t.replace('s', '')
        end = (int(s[0])*3600 + int(s[1])*60 + float(s[2])) * 1000 # 끝시간을 msec 단위로 저장
        start = end - float(t)*1000 + 1 # 시작 시간을 msec 단위로 저장
        log.append([start, end])
    for x in log:
    	# 최대 초당 처리량 구하기
        answer = max(answer, throughput(log, x[0], x[0] + 1000), throughput(log, x[1], x[1] + 1000))

    return answer

# 초당 처리량을 구하는 함수
def throughput(log, start, end):
    cnt = 0
    for x in log:
        if x[0] < end and x[1] >= start:
            cnt += 1
    return cnt

# 공 이동 시뮬레이션
# n행 m열의 격자가 있습니다. 격자의 각 행은 0, 1, ..., n-1번의 번호,
# 그리고 각 열은 0, 1, ..., m-1번의 번호가 순서대로 매겨져 있습니다.
# 당신은 이 격자에 공을 하나 두고, 그 공에 다음과 같은 쿼리들을 날리고자 합니다.
#   열 번호가 감소하는 방향으로 dx칸 이동하는 쿼리 (query(0, dx))
#   열 번호가 증가하는 방향으로 dx칸 이동하는 쿼리 (query(1, dx))
#   행 번호가 감소하는 방향으로 dx칸 이동하는 쿼리 (query(2, dx))
#   행 번호가 증가하는 방향으로 dx칸 이동하는 쿼리 (query(3, dx))
# 단, 공은 격자 바깥으로 이동할 수 없으며, 목적지가 격자 바깥인 경우 공은 이동하다가 더 이상 이동할 수 없을 때 멈추게 됩니다.
#   예를 들어, 5행 × 4열 크기의 격자 내의 공이 3행 2열에 있을 때 query(3, 10) 쿼리를 받은 경우 공은 4행 2열에서 멈추게 됩니다.
#   (격자의 크기가 5행 × 4열이므로, 0~4번 행과 0~3번 열로 격자가 구성되기 때문입니다.)
# 격자의 행의 개수 n, 열의 개수 m, 정수 x와 y, 그리고 쿼리들의 목록을 나타내는 2차원 정수 배열 queries가 매개변수로 주어집니다.
# n × m개의 가능한 시작점에 대해서 해당 시작점에 공을 두고 queries 내의 쿼리들을 순서대로 시뮬레이션했을 때,
# x행 y열에 도착하는 시작점의 개수를 return 하도록 solution 함수를 완성해주세요.
def solution(n, m, x, y, queries):
    answer = 0
    x_min, x_max, y_min, y_max = x, x, y, y
    for idx in range(len(queries) - 1, -1, -1):
        direc, dist = queries[idx]
        if direc == 0:  # 좌(오른쪽에서 왔음)
            y_max += dist  # 오른쪽으로 늘리기
            if y_max > m - 1: # 범위 벗어나면
                y_max = m - 1  # 끝값으로
            if y_min != 0:  # 왼쪽 값이 끝이 아니면 범위 축소
                y_min += dist

        elif direc == 1:  # 우(왼쪽에서 왔음)
            y_min -= dist
            if y_min < 0:
                y_min = 0
            if y_max != m - 1:
                y_max -= dist

        elif direc == 2:  # 상(아래서 왔음)
            x_max += dist
            if x_max > n - 1:
                x_max = n - 1
            if x_min != 0:
                x_min += dist

        else:  # 하(위에서 왔음)
            x_min -= dist
            if x_min < 0:
                x_min = 0
            if x_max != n - 1:
                x_max -= dist
        if y_min > m - 1 or y_max < 0 or x_min > n - 1 or x_max < 0:
            return answer
    else:
         answer = (y_max - y_min + 1) * (x_max - x_min + 1)
    return answer

# 퍼즐 조각 채우기
# 테이블 위에 놓인 퍼즐 조각을 게임 보드의 빈 공간에 적절히 올려놓으려 합니다. 게임 보드와 테이블은 모두 각 칸이 1x1 크기인 정사각 격자 모양입니다.
# 이때, 다음 규칙에 따라 테이블 위에 놓인 퍼즐 조각을 게임 보드의 빈칸에 채우면 됩니다.
#   조각은 한 번에 하나씩 채워 넣습니다.
#   조각을 회전시킬 수 있습니다.
#   조각을 뒤집을 수는 없습니다.
#   게임 보드에 새로 채워 넣은 퍼즐 조각과 인접한 칸이 비어있으면 안 됩니다.
# 현재 게임 보드의 상태 game_board, 테이블 위에 놓인 퍼즐 조각의 상태 table이 매개변수로 주어집니다.
# 규칙에 맞게 최대한 많은 퍼즐 조각을 채워 넣을 경우,
# 총 몇 칸을 채울 수 있는지 return 하도록 solution 함수를 완성해주세요.
def solution(game_board, table):

    # game_board의 빈칸, table의 블럭 찾는 함수
    def finding(mat, num):
        # return할 블럭 및 빈칸들이 들어있는 배열
        arr = []
        visit = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                # 현재 찾는 것(블럭 / 빈칸)과 같고 방문하지 않았으면 BFS
                if mat[i][j] != num or visit[i][j]:
                    continue

                Q = [[i, j]]
                visit[i][j] = 1
                k = 0
                while k < len(Q):
                    r, c = Q[k]
                    for d in range(4):
                        nr, nc = r + dr[d], c + dc[d]
                        if 0 <= nr < n and 0 <= nc < n:
                            if not visit[nr][nc] and mat[nr][nc] == num:
                                Q.append([nr, nc])
                                visit[nr][nc] = 1
                    k += 1

                # 도형 추가
                arr.append(Q)
        return arr

    # 도형 위치 인덱스를 1개의 문자열로 이어서 반환
    def hashing(group):
        # 해당 도형 위치의 가장 왼쪽 모서리
        min_r, min_c = 50, 50
        for r, c in group:
            min_r = min(min_r, r)
            min_c = min(min_c, c)

        # 도형을 왼쪽 구석에 넣었을때 위치로 갱신
        for i in range(len(group)):
            group[i][0] -= min_r
            group[i][1] -= min_c

        # 최선 행, 차선 열 오름차순으로 정렬
        group.sort()


        # 문자열로 바꿔주고 return
        arr = []
        for r, c in group:
            arr.append(str(r))
            arr.append(str(c))

        return ''.join(arr)

    # 빈칸의 위치를 90도 회전시키는 함수
    def rotate(shape):
        for i in range(len(shape)):
            r, c = shape[i]
            shape[i] = [c, -r]

    n = len(game_board)
    dr, dc = [0, 0, 1, -1], [1, -1, 0, 0]
    answer = 0

    # 블럭과 빈칸 찾음
    block = finding(table, 1)
    blank = finding(game_board, 0)

    # 블럭들 문자열로 해싱후 딕셔너리에 저장
    for i in range(len(block)):
        block[i] = hashing(block[i])

    temp = dict()
    for i in block:
        temp[i] = temp.get(i, 0) + 1
    block = temp

    # 빈칸들을 순회하면서
    # 해당 빈칸을 4번 90도 회전하며, 맞는 블럭이 있는지 탐색
    # 있으면 블럭 딕셔너리에 개수 -1
    for i in range(len(blank)):
        for _ in range(4):
            rotate(blank[i])
            hash_blank = hashing(blank[i])
            if block.get(hash_blank):
                block[hash_blank] -= 1

                # 인덱스가 행열행열....식으로 들어가기 때문에
                # 문자열의 길이 //2 해서 answer에 추가
                answer += len(hash_blank)//2
                break

    return answer

# 금과 은 운반하기
# 어느 왕국에 하나 이상의 도시들이 있습니다. 왕국의 왕은 새 도시를 짓기로 결정하였습니다.
# 해당 도시를 짓기 위해서는 도시를 짓는 장소에 금 a kg과 은 b kg이 전달되어야 합니다.
# 각 도시에는 번호가 매겨져 있는데, i번 도시에는 금 g[i] kg, 은 s[i] kg, 그리고 트럭 한 대가 있습니다.
# i번 도시의 트럭은 오직 새 도시를 짓는 건설 장소와 i번 도시만을 왕복할 수 있으며, 편도로 이동하는 데 t[i] 시간이 걸리고,
# 최대 w[i] kg 광물을 운반할 수 있습니다. (광물은 금과 은입니다. 즉, 금과 은을 동시에 운반할 수 있습니다.)
# 모든 트럭은 같은 도로를 여러 번 왕복할 수 있으며 연료는 무한대라고 가정합니다.
# 정수 a, b와 정수 배열 g, s, w, t가 매개변수로 주어집니다. 주어진 정보를 바탕으로 각 도시의 트럭을 최적으로 운행했을 때,
# 새로운 도시를 건설하기 위해 금 a kg과 은 b kg을 전달할 수 있는 가장 빠른 시간을 구해 return 하도록 solution 함수를 완성해주세요.
def solution(a, b, g, s, w, t):
    start = 0
    # 최악의 경우
    # 광물의 최대무게 : 10**9 * 2(금,은)
    # 도시가 1개만 있고 소요시간이 최대, 1씩 옮길 수 있을 때 : 10**5 * 2(왕복)
    answer = end = (10 ** 9) * (10 ** 5) * 4

    # 이진 탐색
    while start <= end:
        mid = (start + end) // 2
        gold = 0
        silver = 0
        total = 0

        for i in range(len(g)):
            # 현재 정보
            now_gold = g[i]
            now_silver = s[i]
            now_weight = w[i]
            now_time = t[i]

            # 주어진 시간내에서 이동할 수 있는 횟수 (왕복으로 계산)
            move_cnt = mid // (now_time * 2)

            # 편도 추가
            if mid % (now_time * 2) >= now_time:
                move_cnt += 1

            # 주어지 시간 내 최대 적재 가능량 누적하기
            possible_move_weight = move_cnt * now_weight
            gold += now_gold if (now_gold < possible_move_weight) else possible_move_weight
            silver += now_silver if (now_silver < possible_move_weight) else possible_move_weight
            total += now_gold + now_silver if (now_gold + now_silver < possible_move_weight) else possible_move_weight

        # total이 a+b 보다 크거나 같으면서 금과 은의 누적 최대값이 a와 b보다 크거나 같아야 한다.
        # 만약 금과 은의 누적 최대값이 보내야만 하는 a,b보다 작다면 현재 시간 내 처리 불가능하다.
        if total >= a + b and gold >= a and silver >= b:
            end = mid - 1
            answer = min(answer, mid)
        else:
            start = mid + 1
    return answer

# 광고 삽입
#  "죠르디"의 동영상 재생시간 길이 play_time, 공익광고의 재생시간 길이 adv_time,
# 시청자들이 해당 동영상을 재생했던 구간 정보 logs가 매개변수로 주어질 때,
# 시청자들의 누적 재생시간이 가장 많이 나오는 곳에 공익광고를 삽입하려고 합니다.
# 이때, 공익광고가 들어갈 시작 시각을 구해서 return 하도록 solution 함수를 완성해주세요.
# 만약, 시청자들의 누적 재생시간이 가장 많은 곳이 여러 곳이라면,
# 그 중에서 가장 빠른 시작 시각을 return 하도록 합니다.
def str2int(time):
    hour = int(time[:2]) * 3600
    minute = int(time[3:5]) * 60
    second = int(time[6:])
    
    return hour+minute+second

def int2str(time):
    hour = str(time // 3600).zfill(2)
    minute = str(time % 3600 // 60).zfill(2)
    second = str(time % 3600 % 60).zfill(2)
    
    return hour+":"+minute+":"+second

def solution(play_time, adv_time, logs):
    dp = [0] * (str2int(play_time) + 1)
    
    for i in logs:
        temp = i.split('-')
        start = str2int(temp[0])
        end = str2int(temp[1])
        dp[start] += 1
        dp[end] -= 1
        
    for i in range(1, str2int(play_time)):
        dp[i] = dp[i] + dp[i-1]
    for i in range(1, str2int(play_time)):
        dp[i] = dp[i] + dp[i-1]
    
    max_value = -1
    answer = 0
    for i in range(str2int(adv_time)-1, str2int(play_time)):
        temp = dp[i] - dp[i-str2int(adv_time)] 
        if temp > max_value:
            max_value = temp
            answer = i-str2int(adv_time) + 1
            
    return int2str(answer)

# 카드 짝 맞추기
# 게임 개발자인 베로니는 개발 연습을 위해 다음과 같은 간단한 카드 짝맞추기 보드 게임을 개발해 보려고 합니다.
# 게임이 시작되면 화면에는 카드 16장이 뒷면을 위로하여 4 x 4 크기의 격자 형태로 표시되어 있습니다.
# 각 카드의 앞면에는 카카오프렌즈 캐릭터 그림이 그려져 있으며, 8가지의 캐릭터 그림이 그려진 카드가 각기 2장씩 화면에 무작위로 배치되어 있습니다.
# 유저가 카드를 2장 선택하여 앞면으로 뒤집었을 때 같은 그림이 그려진 카드면 해당 카드는 게임 화면에서 사라지며,
# 같은 그림이 아니라면 원래 상태로 뒷면이 보이도록 뒤집힙니다. 이와 같은 방법으로 모든 카드를 화면에서 사라지게 하면 게임이 종료됩니다.
# 게임에서 카드를 선택하는 방법은 다음과 같습니다.
#   카드는 커서를 이용해서 선택할 수 있습니다.
#       커서는 4 x 4 화면에서 유저가 선택한 현재 위치를 표시하는 "굵고 빨간 테두리 상자"를 의미합니다.
#   커서는 [Ctrl] 키와 방향키에 의해 이동되며 키 조작법은 다음과 같습니다.
#       방향키 ←, ↑, ↓, → 중 하나를 누르면, 커서가 누른 키 방향으로 1칸 이동합니다.
#       [Ctrl] 키를 누른 상태에서 방향키 ←, ↑, ↓, → 중 하나를 누르면, 누른 키 방향에 있는 가장 가까운 카드로 한번에 이동합니다.
#           만약, 해당 방향에 카드가 하나도 없다면 그 방향의 가장 마지막 칸으로 이동합니다.
#       만약, 누른 키 방향으로 이동 가능한 카드 또는 빈 공간이 없어 이동할 수 없다면 커서는 움직이지 않습니다.
#   커서가 위치한 카드를 뒤집기 위해서는 [Enter] 키를 입력합니다.
#       [Enter] 키를 입력해서 카드를 뒤집었을 때
#           앞면이 보이는 카드가 1장 뿐이라면 그림을 맞출 수 없으므로 두번째 카드를 뒤집을 때 까지 앞면을 유지합니다.
#           앞면이 보이는 카드가 2장이 된 경우, 두개의 카드에 그려진 그림이 같으면 해당 카드들이 화면에서 사라지며,
#           그림이 다르다면 두 카드 모두 뒷면이 보이도록 다시 뒤집힙니다.
# "베로니"는 게임 진행 중 카드의 짝을 맞춰 몇 장 제거된 상태에서 카드 앞면의 그림을 알고 있다면,
# 남은 카드를 모두 제거하는데 필요한 키 조작 횟수의 최솟값을 구해 보려고 합니다.
# 키 조작 횟수는 방향키와 [Enter] 키를 누르는 동작을 각각 조작 횟수 1로 계산하며,
# [Ctrl] 키와 방향키를 함께 누르는 동작 또한 조작 횟수 1로 계산합니다.
# 현재 카드가 놓인 상태를 나타내는 2차원 배열 board와 커서의 처음 위치 r, c가 매개변수로 주어질 때,
# 모든 카드를 제거하기 위한 키 조작 횟수의 최솟값을 return 하도록 solution 함수를 완성해 주세요.
from itertools import permutations
from collections import deque

size = 4
myboard = [[] for i in range(4)]
card_pos = {}
d = [[0,1], [1,0], [0,-1], [-1,0]]
INF = int(1e4)
answer = INF
orders = []

# 전역 변수를 이용한 보드(myboard), 카드 2장의 위치(card_pos) 초기화 
# 지우는 순서에 대한 순열(orders) 초기화
# card_pos 예시: card_pos[1] = [[0,0], [1,2]] // 카드 1은 보드의 [0,0], [1,2]에 존재 
def init(board):
    global myboard, card_pos, orders
    for i in range(size):
        for j in range(size):
            if board[i][j] != 0:
                card = board[i][j]
                if card not in card_pos.keys(): card_pos[card] = [[i,j]]
                else: card_pos[card].append([i,j])

            myboard[i].append(board[i][j])
    
    orders = [key for key in card_pos.keys()]
    orders = list(permutations(orders))
    
# 이동한 결과가 보드 범위내 있는지 판단하는 함수            
def isin(y,x):
    if -1<y<size:
        if -1<x<size: return True
        
    return False

# ctrl + 방향키
def move(y, x, mv):
    global myboard
    ny, nx = y, x

    while True:
        _ny = ny + mv[0]
        _nx = nx + mv[1]
        if not isin(_ny, _nx): return ny, nx
        if myboard[_ny][_nx] != 0: return _ny, _nx
            
        ny = _ny
        nx = _nx

# 카드 1장을 찾을 때 나오는 거리를 반환하는 함수(목표 위치도 반환함)
# 시작 위치: myboard[sy, sx], 찾아야 할 위치: myboard[ey, ex] 
def bfs(sy, sx, ey, ex):
    if [sy, sx] == [ey, ex]: return sy, sx, 1
    global myboard
    q = []
    q = deque(q)
    table = [[0 for j in range(size)] for i in range(size)]
    visit = [[False for j in range(size)] for i in range(size)]
    q.append([sy, sx])
    visit[sy][sx] = True

    while q:
        y, x = q.popleft()

        for i in range(4):
            ny = y + d[i][0]
            nx = x + d[i][1]

            if isin(ny, nx):
                if not visit[ny][nx]:
                    visit[ny][nx] = True
                    table[ny][nx] = table[y][x] + 1
                    if [ny,nx] == [ey,ex]:
                        return ny, nx, table[ny][nx] + 1

                    q.append([ny, nx])

            ny, nx = move(y, x, d[i])

            if not visit[ny][nx]:
                visit[ny][nx] = True      
                table[ny][nx] = table[y][x] + 1
                if [ny,nx] == [ey,ex]:
                    return ny, nx, table[ny][nx] + 1

                q.append([ny, nx])

    return sy, sx, INF

# 찾은 2장의 카드를 myboard에서 지워주는 함수           
def remove(card):
    global myboard, card_pos
    for y, x in card_pos[card]: myboard[y][x] = 0

# 지워진 2장의 카드를 myboard에서 복원해주는 함수
def restore(card):
    global myboard, card_pos
    for y, x in card_pos[card]: myboard[y][x] = card

# backtracking
def backtrack(sy, sx, k, m, count):
    global orders, myboard, answer, card_pos

    if k == len(card_pos.keys()):
        if answer > count: answer = count
        return

    card = orders[m][k]
    left_y, left_x = card_pos[card][0][0], card_pos[card][0][1]
    right_y, right_x = card_pos[card][1][0], card_pos[card][1][1]

    ry1, rx1, res1 = bfs(sy, sx, left_y, left_x)
    ry2, rx2, res2 = bfs(ry1, rx1, right_y, right_x)
    
    remove(card)
    backtrack(ry2, rx2, k+1, m, count + res1 + res2)
    restore(card)

    ry1, rx1, res1 = bfs(sy, sx, right_y, right_x)
    ry2, rx2, res2 = bfs(ry1, rx1, left_y, left_x)

    remove(card)
    backtrack(ry2, rx2, k+1, m, count + res1 + res2)
    restore(card)

def solution(board, r, c):
    global answer
    init(board)

    for i in range(len(orders)):
        backtrack(r, c, 0, i, 0)

    return answer

# 코딩 테스트 공부
# 알고리즘에 대한 지식은 알고력, 코드를 구현하는 능력은 코딩력이라고 표현합니다. 알고력과 코딩력은 0 이상의 정수로 표현됩니다.
# 문제를 풀기 위해서는 문제가 요구하는 일정 이상의 알고력과 코딩력이 필요합니다.
# 풀 수 없는 문제를 해결하기 위해서는 알고력과 코딩력을 높여야 합니다. 알고력과 코딩력을 높이기 위한 다음과 같은 방법들이 있습니다.
#   알고력을 높이기 위해 알고리즘 공부를 합니다. 알고력 1을 높이기 위해서 1의 시간이 필요합니다.
#   코딩력을 높이기 위해 코딩 공부를 합니다. 코딩력 1을 높이기 위해서 1의 시간이 필요합니다.
#   현재 풀 수 있는 문제 중 하나를 풀어 알고력과 코딩력을 높입니다. 각 문제마다 문제를 풀면 올라가는 알고력과 코딩력이 정해져 있습니다.
#   문제를 하나 푸는 데는 문제가 요구하는 시간이 필요하며 같은 문제를 여러 번 푸는 것이 가능합니다.
# 당신은 주어진 모든 문제들을 풀 수 있는 알고력과 코딩력을 얻는 최단시간을 구하려 합니다.
# 초기의 알고력과 코딩력을 담은 정수 alp와 cop, 문제의 정보를 담은 2차원 정수 배열 problems가 매개변수로 주어졌을 때,
# 모든 문제들을 풀 수 있는 알고력과 코딩력을 얻는 최단시간을 return 하도록 solution 함수를 작성해주세요.
def solution(alp, cop, problems):
    max_alp_req, max_cop_req = [0, 0]  # 목표값
    
    for problem in problems:
        max_alp_req = max(max_alp_req, problem[0])
        max_cop_req = max(max_cop_req, problem[1])
    
    dp = [[float('inf')] * (max_cop_req+1) for _ in range(max_alp_req+1)]
    
    alp = min(alp, max_alp_req)  # 둘중 하나라도 목표값을 넘어가면 안된다.
    cop = min(cop, max_cop_req)
    
    dp[alp][cop] = 0  # dp[i][j]의 의미 : 알고력 i, 코딩력 j을 도달 할 수 있는 최단시간
    
    for i in range(alp, max_alp_req+1):
        for j in range(cop, max_cop_req+1):
            if i < max_alp_req:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + 1)
            if j < max_cop_req:
                dp[i][j+1] = min(dp[i][j+1], dp[i][j] + 1)
            
            for alp_req, cop_req, alp_rwd, cop_rwd, cost in problems:
                if i >= alp_req and j >= cop_req:
                    new_alp = min(i+alp_rwd, max_alp_req)  # 둘중 하나라도 목표값을 넘어가면 안된다.
                    new_cop = min(j+cop_rwd, max_cop_req)
                    dp[new_alp][new_cop] = min(dp[new_alp][new_cop], dp[i][j] + cost)
                    
    return dp[max_alp_req][max_cop_req]

# 사라지는 발판
# 플레이어 A와 플레이어 B가 서로 게임을 합니다.
# 당신은 이 게임이 끝날 때까지 양 플레이어가 캐릭터를 몇 번 움직이게 될지 예측하려고 합니다.
# 각 플레이어는 자신의 캐릭터 하나를 보드 위에 올려놓고 게임을 시작합니다.
# 게임 보드는 1x1 크기 정사각 격자로 이루어져 있으며, 보드 안에는 발판이 있는 부분과 없는 부분이 있습니다.
# 발판이 있는 곳에만 캐릭터가 서있을 수 있으며, 처음 캐릭터를 올려놓는 곳은 항상 발판이 있는 곳입니다.
# 캐릭터는 발판이 있는 곳으로만 이동할 수 있으며, 보드 밖으로 이동할 수 없습니다.
# 밟고 있던 발판은 그 위에 있던 캐릭터가 다른 곳으로 이동하여 다른 발판을 밞음과 동시에 사라집니다.
# 양 플레이어는 번갈아가며 자기 차례에 자신의 캐릭터를 상하좌우로 인접한 4개의 칸 중에서 발판이 있는 칸으로 옮겨야 합니다.
# 다음과 같은 2가지 상황에서 패자와 승자가 정해지며, 게임이 종료됩니다.
#   움직일 차례인데 캐릭터의 상하좌우 주변 4칸이 모두 발판이 없거나 보드 밖이라서 이동할 수 없는 경우, 해당 차례 플레이어는 패배합니다.
#   두 캐릭터가 같은 발판 위에 있을 때, 상대 플레이어의 캐릭터가 다른 발판으로 이동하여 자신의 캐릭터가 서있던 발판이 사라지게 되면 패배합니다.
# 게임은 항상 플레이어 A가 먼저 시작합니다. 양 플레이어는 최적의 플레이를 합니다.
# 즉, 이길 수 있는 플레이어는 최대한 빨리 승리하도록 플레이하고,
# 질 수밖에 없는 플레이어는 최대한 오래 버티도록 플레이합니다. '이길 수 있는 플레이어'는 실수만 하지 않는다면 항상 이기는 플레이어를 의미하며,
# '질 수밖에 없는 플레이어'는 최선을 다해도 상대가 실수하지 않으면 항상 질 수밖에 없는 플레이어를 의미합니다.
# 최대한 오래 버틴다는 것은 양 플레이어가 캐릭터를 움직이는 횟수를 최대화한다는 것을 의미합니다.
# 게임 보드의 초기 상태를 나타내는 2차원 정수 배열 board와 플레이어 A의 캐릭터 초기 위치를 나타내는 정수 배열 aloc,
# 플레이어 B의 캐릭터 초기 위치를 나타내는 정수 배열 bloc이 매개변수로 주어집니다.
# 양 플레이어가 최적의 플레이를 했을 때, 두 캐릭터가 움직인 횟수의 합을 return 하도록 solution 함수를 완성해주세요.
n,m = 0,0
move = [(0,1),(0,-1),(1,0),(-1,0)]
visit = [[0]*5 for _ in range(5)]
def OOB(x,y):
    return x < 0 or x >= n or y < 0 or y >= m
# 결과값이 
def play(board,curx,cury,opx,opy):
    global visit
    if visit[curx][cury]: return 0
    canWin = 0
    for mov in move:
        dx, dy = mov
        nx, ny = curx + dx, cury + dy
        if OOB(nx,ny) or visit[nx][ny] or board[nx][ny] == 0 : continue
        # 방문처리
        visit[curx][cury] = 1
        opResult = play(board,opx,opy,nx,ny)+1
        # 방문처리 끝
        visit[curx][cury] = 0

        # 현재 저장된 값 패배인데 상대가 졌다고 하면 이기는 경우로 저장
        if canWin % 2 == 0 and opResult % 2 == 1 : canWin = opResult
        # 현재 저장된 값 패배인데 상대가 이겼다고 하면 최대한 늦게 지는 횟수 선택
        elif canWin % 2 == 0 and opResult % 2 == 0 : canWin = max(canWin,opResult)
        # 현재 저장된 값 승리인데 상대가 졌다고 하면 최대한 빨리 이기는 횟수 선택
        elif canWin % 2 == 1 and opResult % 2 == 1 : canWin = min(canWin,opResult)
    return canWin

def solution(board, aloc, bloc):
    global n,m
    n, m = len(board), len(board[0])
    return play(board,aloc[0],aloc[1],bloc[0],bloc[1])

# 도둑질
# 도둑이 어느 마을을 털 계획을 하고 있습니다. 이 마을의 모든 집들은 아래 그림과 같이 동그랗게 배치되어 있습니다.
# 각 집들은 서로 인접한 집들과 방범장치가 연결되어 있기 때문에 인접한 두 집을 털면 경보가 울립니다.
# 각 집에 있는 돈이 담긴 배열 money가 주어질 때, 도둑이 훔칠 수 있는 돈의 최댓값을 return 하도록 solution 함수를 작성하세요.
def solution(money):
    dp1 = [0] * len(money)
    dp2 = [0] * len(money)
    # 1번 집을 터는 경우
    dp1[0] = money[0]
    for i in range(1, len(money) - 1):  # 마지막 집은 털지 못함
        dp1[i] = max(dp1[i - 1], dp1[i - 2] + money[i])
    # 1번 집을 안터는 경우
    dp1[0] = 0
    for i in range(1, len(money)):
        dp2[i] = max(dp2[i - 1], dp2[i - 2] + money[i])

    return max(dp1[-2], dp2[-1])

# 무지의 먹방 라이브
# 회전판에 먹어야 할 N 개의 음식이 있다.
# 각 음식에는 1부터 N 까지 번호가 붙어있으며, 각 음식을 섭취하는데 일정 시간이 소요된다.
# 무지는 다음과 같은 방법으로 음식을 섭취한다.
#   무지는 1번 음식부터 먹기 시작하며, 회전판은 번호가 증가하는 순서대로 음식을 무지 앞으로 가져다 놓는다.
#   마지막 번호의 음식을 섭취한 후에는 회전판에 의해 다시 1번 음식이 무지 앞으로 온다.
#   무지는 음식 하나를 1초 동안 섭취한 후 남은 음식은 그대로 두고, 다음 음식을 섭취한다.
#       다음 음식이란, 아직 남은 음식 중 다음으로 섭취해야 할 가장 가까운 번호의 음식을 말한다.
#   회전판이 다음 음식을 무지 앞으로 가져오는데 걸리는 시간은 없다고 가정한다.
# 무지가 먹방을 시작한 지 K 초 후에 네트워크 장애로 인해 방송이 잠시 중단되었다.
# 무지는 네트워크 정상화 후 다시 방송을 이어갈 때, 몇 번 음식부터 섭취해야 하는지를 알고자 한다.
# 각 음식을 모두 먹는데 필요한 시간이 담겨있는 배열 food_times, 네트워크 장애가 발생한 시간 K 초가 매개변수로 주어질 때
# 몇 번 음식부터 다시 섭취하면 되는지 return 하도록 solution 함수를 완성하라.
import heapq
def solution(food_times, k):
    
    #전체 음식을 먹는 시간보다 k가 크거나 같다면 -1
    if sum(food_times)<=k:
        return -1
    
    q=[]
    for i in range(len(food_times)):
        #(음식 시간, 음식 번호) 형태로 heapq에 삽입
        heapq.heappush(q, (food_times[i], i+1))
    
    sum_value = 0 #먹기 위해 사용한 시간
    previous = 0 #직전에 다 먹은 음식 시간
    
    length = len(food_times) #남은 음식의 개수
    
    #sum_value + (현재의 음식 시간 - 이전 음식 시간) * 현재 남은 음식 개수와 k 비교
    #위가 현재 전체 남은 시간(k)보다 커질때까지 반복한다
    while sum_value+((q[0][0]-previous)*length)<=k:
        now=heapq.heappop(q)[0] #현재의 음식 시간
        sum_value += (now-previous)*length 
        length -= 1 #다 먹은 음식 제외
        previous = now #이전 음식 시간 재설정
    
    #남은 음식 중에서 몇 번째 음식인지 확인하여 출력
    result = sorted(q, key=lambda x:x[1]) #음식의 번호 기준으로 정렬
    return result[(k-sum_value)%length][1] #번호 기준으로, 남은 초수(k-sum_value) 만큼 반복
    answer = -1
    return answer

# 올바른 괄호의 갯수
# 올바른 괄호란 (())나 ()와 같이 올바르게 모두 닫힌 괄호를 의미합니다. )(나 ())() 와 같은 괄호는 올바르지 않은 괄호가 됩니다.
# 괄호 쌍의 개수 n이 주어질 때,
# n개의 괄호 쌍으로 만들 수 있는 모든 가능한 괄호 문자열의 갯수를 반환하는 함수 solution을 완성해 주세요.
def solution(n):
    answer = 1
    dp = [0 for i in range(n+1)]
    dp[0], dp[1] = 1, 1
    for i in range(2, n+1):
        for j in range(i):
            dp[i] += dp[j] * dp[i-1-j]

    answer = dp[n]
    return answer

# 징검다리
# 출발지점부터 distance만큼 떨어진 곳에 도착지점이 있습니다.
# 그리고 그사이에는 바위들이 놓여있습니다. 바위 중 몇 개를 제거하려고 합니다.
# 출발지점부터 도착지점까지의 거리 distance, 바위들이 있는 위치를 담은 배열 rocks, 제거할 바위의 수 n이 매개변수로 주어질 때,
# 바위를 n개 제거한 뒤 각 지점 사이의 거리의 최솟값 중에 가장 큰 값을 return 하도록 solution 함수를 작성해주세요.
def solution(distance, rocks, n):
	# 커트라인 최솟값 = 1
    left = 1
    # 커트라인 최댓값 = distance
    right = distance
    # 바위 위치를 정렬하고, 도착지점 append
    rocks.sort()
    rocks.append(distance)
    while left <= right:
        mid = (left+right)//2
        # 제거한 바위 개수
        delete = 0
        # 이전 바위의 위치
        prev_rock = 0
        for rock in rocks:
            dist = rock - prev_rock
            # 거리가 커트라인 보다 작다면, 바위를 제거
            if dist < mid:
                delete += 1
                # 제거한 바위가 너무 많다면 break
                if delete > n:
                    break
            # 바위를 제거하지 않았다면, prev_rock을 갱신
            else:
                prev_rock = rock
                
        # 초과 제거한 경우 : 커트라인이 너무 높음
        if delete > n:
            right = mid -1
        # 이하 제거한 경우 : 커트라인이 적절하거나 너무 낮음
        else:
            answer = mid
            left = mid + 1
    return answer

# 호텔 방 배정
# 스노우타운"에서 호텔을 운영하고 있는 "스카피"는 호텔에 투숙하려는 고객들에게 방을 배정하려 합니다. 호텔에는 방이 총 k개 있으며, 각각의 방은 1번부터 k번까지 번호로 구분하고 있습니다.
# 처음에는 모든 방이 비어 있으며 "스카피"는 다음과 같은 규칙에 따라 고객에게 방을 배정하려고 합니다.
#   한 번에 한 명씩 신청한 순서대로 방을 배정합니다.
#   고객은 투숙하기 원하는 방 번호를 제출합니다.
#   고객이 원하는 방이 비어 있다면 즉시 배정합니다.
#   고객이 원하는 방이 이미 배정되어 있으면 원하는 방보다 번호가 크면서 비어있는 방 중 가장 번호가 작은 방을 배정합니다.
# 전체 방 개수 k와 고객들이 원하는 방 번호가 순서대로 들어있는 배열 room_number가 매개변수로 주어질 때,
# 각 고객에게 배정되는 방 번호를 순서대로 배열에 담아 return 하도록 solution 함수를 완성해주세요.
import sys
sys.setrecursionlimit(10000000) # 설정해주지 않으면 재귀가 많이 일어나면서 런타임에러 등이 나타날 수 있다.

def findEmptyRoom(number, rooms): # 빈방을 찾는 함수
    if number not in rooms:        
        rooms[number] = number + 1
        return number
    
    empty = findEmptyRoom(rooms[number], rooms)
    rooms[number] = empty + 1
    return empty


def solution(k, room_number):
    answer = []
    rooms = dict() # 몇번 방이 비어있는지 체크하는 딕셔너리

    for number in room_number:
        emptyRoom = findEmptyRoom(number, rooms)
        answer.append(emptyRoom)
    
    return answer

# 자동완성
# 포털 다음에서 검색어 자동완성 기능을 넣고 싶은 라이언은 한 번 입력된 문자열을 학습해서 다음 입력 때 활용하고 싶어 졌다.
#   예를 들어, go 가 한 번 입력되었다면, 다음 사용자는 g 만 입력해도 go를 추천해주므로 o를 입력할 필요가 없어진다!
#   단, 학습에 사용된 단어들 중 앞부분이 같은 경우에는 어쩔 수 없이 다른 문자가 나올 때까지 입력을 해야 한다.
# 효과가 얼마나 좋을지 알고 싶은 라이언은 학습된 단어들을 찾을 때 몇 글자를 입력해야 하는지 궁금해졌다.
# 라이언을 도와 위와 같이 문자열이 입력으로 주어지면 학습을 시킨 후, 학습된 단어들을 순서대로 찾을 때 몇 개의 문자를 입력하면 되는지 계산하는 프로그램을 만들어보자.
def solution(words):
    N = len(words)
    words.sort() # 단어를 사전순으로 정렬
    result = [0] * N # 단어마다 입력해야 하는 문자 수
    for i in range(N - 1):
        # 인접하는 두 단어 비교
        a = len(words[i])
        b = len(words[i + 1])
        for j in range(min(a, b)):
            if words[i][j] != words[i + 1][j]:
                j -= 1 # 일치하지 않으면 일치하는 최대 인덱스로 저장 후 break
                break

        # 일치하는 인덱스 + 1만큼 문자를 입력해야 한다.
        # 단, 입력하는 문자 수가 단어 길이를 넘지 말아야 한다.
        result[i] = max(result[i], min(a, j + 2))
        result[i + 1] = max(result[i + 1], min(b, j + 2))

    # 단어마다 입력해야 하는 문자 수를 합해서 반환
    return sum(result)

# 쿠키 구입
# 과자를 바구니 단위로 파는 가게가 있습니다.
# 이 가게는 1번부터 N번까지 차례로 번호가 붙은 바구니 N개가 일렬로 나열해 놨습니다.
# 철수는 두 아들에게 줄 과자를 사려합니다. 첫째 아들에게는 l번 바구니부터 m번 바구니까지,
# 둘째 아들에게는 m+1번 바구니부터 r번 바구니까지를 주려합니다.
# 단, 두 아들이 받을 과자 수는 같아야 합니다(1 <= l <= m, m+1 <= r <= N).
# 즉, A[i] 를 i번 바구니에 들어있는 과자 수라고 했을 때, A[l]+..+A[m] = A[m+1]+..+A[r] 를 만족해야 합니다.
# 각 바구니 안에 들은 과자 수가 차례로 들은 배열 cookie가 주어질 때,
# 조건에 맞게 과자를 살 경우 한 명의 아들에게 줄 수 있는 가장 많은 과자 수를 return 하는 solution 함수를 완성해주세요.
# (단, 조건에 맞게 과자를 구매할 수 없다면 0을 return 합니다)
def solution(cookie):
    answer = 0
    n = len(cookie)
    for i in range(n - 1):
        left_sum, left_idx = cookie[i], i
        right_sum, right_idx = cookie[i + 1], i + 1

        while True:
            if left_sum == right_sum:
                answer = max(answer, left_sum)
                # answer = max(answer, right_sum)
                
            if left_idx > 0 and left_sum <= right_sum:
                left_idx -= 1
                left_sum += cookie[left_idx]
            elif right_idx < n - 1 and right_sum <= left_sum:
                right_idx += 1
                right_sum += cookie[right_idx]
            else:
                break

    return answer

# 등대
# 인천 앞바다에는 1부터 n까지 서로 다른 번호가 매겨진 등대 n개가 존재합니다.
# 등대와 등대 사이를 오가는 뱃길이 n-1개 존재하여, 어느 등대에서 출발해도 다른 모든 등대까지 이동할 수 있습니다.
# 등대 관리자 윤성이는 전력을 아끼기 위하여, 이 중 몇 개의 등대만 켜 두려고 합니다.
# 하지만 등대를 아무렇게나 꺼버리면, 뱃길을 오가는 배들이 위험할 수 있습니다.
# 한 뱃길의 양쪽 끝 등대 중 적어도 하나는 켜져 있도록 등대를 켜 두어야 합니다.
# 등대의 개수 n과 각 뱃길이 연결된 등대의 번호를 담은 이차원 배열 lighthouse가 매개변수로 주어집니다.
# 윤성이가 켜 두어야 하는 등대 개수의 최솟값을 return 하도록 solution 함수를 작성해주세요.
import sys
from collections import defaultdict
sys.setrecursionlimit(1000001)
A = defaultdict(list)
vis = [False] * 1000001

# 자신을 포함한 subtree에서, 내가 켜졌을 때의 최소 점등 등대 개수와
# 내가 꺼졌을 때의 최소 점등 등대 개수를 반환합니다.
def dfs(u):
    vis[u] = True
    if not A[u]:
        # u가 leaf라면 내가 켜졌을 떄의 최소 점등 등대 개수는 1
        # 내가 꺼졌을 때의 최소 점등 등대 개수는 0
        return 1, 0
    
    # u가 leaf가 아니라면
    on, off = 1, 0
    for v in [v for v in A[u] if not vis[v]]:
        # 내가 켜졌다면 child들은 켜지든 꺼지든 상관 없습니다. -> 킨 것과 끈 것중 최소값을 취함
        # 내가 꺼졌다면 child들은 무조건 켜져야 합니다.
        # 이 점을 생각해서 leaf들의 정보를 취합, 정리합니다.
        child_on, child_off = dfs(v)
        on += min(child_on, child_off)
        off += child_on
    return on, off

def solution(n, lighthouse):
    for u, v in lighthouse:
        A[u].append(v)
        A[v].append(u)
        
    on, off = dfs(1)
    return min(on, off)

# 아날로그 시계
# 시침, 분침, 초침이 있는 아날로그시계가 있습니다. 시계의 시침은 12시간마다,
# 분침은 60분마다, 초침은 60초마다 시계를 한 바퀴 돕니다.
# 따라서 시침, 분침, 초침이 움직이는 속도는 일정하며 각각 다릅니다.
# 이 시계에는 초침이 시침/분침과 겹칠 때마다 알람이 울리는 기능이 있습니다.
# 당신은 특정 시간 동안 알람이 울린 횟수를 알고 싶습니다.
# 알람이 울리는 횟수를 센 시간을 나타내는 정수 h1, m1, s1, h2, m2, s2가 매개변수로 주어집니다.
# 이때, 알람이 울리는 횟수를 return 하도록 solution 함수를 완성해주세요.
def solution(h1, m1, s1, h2, m2, s2):
    answer = 0
    
    # 시작시간과 끝시간을 초단위로 변환
    startTime = h1 * 3600 + m1 * 60 + s1
    endTime = h2 * 3600 + m2 * 60 + s2  
    
    # next기준으로 계산할거니 포함되지 않는 시작시간 00시, 12시 미리 카운팅
    if startTime == 0 * 3600 or startTime == 12 * 3600:
        answer += 1
        
    while startTime < endTime:
        # 시침 1시간 = 30도 -> 1초에 30/3600도 즉, 1/120도 이동
        # 분침 1분 = 6도 -> 1초에 6/60도 즉, 1/10도 이동
        # 초침 1초 = 6도 -> 1초에 6도 이동 
        hCurAngle = startTime / 120 % 360
        mCurAngle = startTime / 10 % 360
        sCurAngle = startTime * 6 % 360

        # 다음 위치가 360도가 아닌 0도로 계산되어 카운팅에 포함되지 않는 경우 방지
        # 이동했을 때 지나쳤거나 같아졌는지를 비교하는 것이므로 현재위치는 해줄 필요없음
        hNextAngle = 360 if (startTime + 1) / 120 % 360 == 0 else (startTime + 1) / 120 % 360
        mNextAngle = 360 if (startTime + 1) / 10 % 360 == 0 else (startTime + 1) / 10 % 360
        sNextAngle = 360 if (startTime + 1) * 6 % 360 == 0 else (startTime + 1) * 6 % 360

        if sCurAngle < hCurAngle and sNextAngle >= hNextAngle:
            answer += 1
        if sCurAngle < mCurAngle and sNextAngle >= mNextAngle:
            answer += 1
        # 시침/분침과 동시에 겹쳤을 때 중복카운팅 제외 
        if sNextAngle == hNextAngle and hNextAngle == mNextAngle:
            answer -= 1

        startTime += 1
    
    return answer

# 주사위 고르기
# A와 B가 n개의 주사위를 가지고 승부를 합니다. 주사위의 6개 면에 각각 하나의 수가 쓰여 있으며,
# 주사위를 던졌을 때 각 면이 나올 확률은 동일합니다.
# 각 주사위는 1 ~ n의 번호를 가지고 있으며, 주사위에 쓰인 수의 구성은 모두 다릅니다.
# A가 먼저 n / 2개의 주사위를 가져가면 B가 남은 n / 2개의 주사위를 가져갑니다.
# 각각 가져간 주사위를 모두 굴린 뒤, 나온 수들을 모두 합해 점수를 계산합니다.
# 점수가 더 큰 쪽이 승리하며, 점수가 같다면 무승부입니다.
# A는 자신이 승리할 확률이 가장 높아지도록 주사위를 가져가려 합니다.
# 주사위에 쓰인 수의 구성을 담은 2차원 정수 배열 dice가 매개변수로 주어집니다.
# 이때, 자신이 승리할 확률이 가장 높아지기 위해
# A가 골라야 하는 주사위 번호를 오름차순으로 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해 주세요.
# 승리할 확률이 가장 높은 주사위 조합이 유일한 경우만 주어집니다.
from itertools import combinations, product
def solution(dice):
    n = len(dice)
    A, result = [], []
    cases = list(combinations(range(n), n//2)) # 뽑을 수 있는 경우의 수 (인덱스)
    for case in cases:
        dices = [dice[c] for c in case] # 인덱스에 해당하는 실제 주사위
        nums = sorted([sum(i) for i in product(*dices)]) # 뽑은 주사위의 합의 경우의 수
        A.append(nums)
        
    a, p = 0, len(A)
    for i in range(p):
        B = A[p-i-1] # B와 A는대칭의 구조를 가지므로 B = A[p-i-1]
        
        #각 조합이 이기는 횟수 이분탐색
        temp = 0
        for t in A[i]:
            left, right = 0, len(B)-1
            while left <= right:
                mid = (left + right) // 2
                if B[mid] < t:
                    left = mid + 1
                else :
                    right = mid - 1
            temp += left
        
        # 가장 승리 확률이 높은 경우의 수 반환
        if a < temp:
            a = temp
            result = [x + 1 for x in cases[i]]
    
    return result

# 지형이동
# N x N 크기인 정사각 격자 형태의 지형이 있습니다. 각 격자 칸은 1 x 1 크기이며,
# 숫자가 하나씩 적혀있습니다. 격자 칸에 적힌 숫자는 그 칸의 높이를 나타냅니다.
# 이 지형의 아무 칸에서나 출발해 모든 칸을 방문하는 탐험을 떠나려 합니다. 
# 칸을 이동할 때는 상, 하, 좌, 우로 한 칸씩 이동할 수 있는데, 현재 칸과 이동하려는 칸의 높이 차가 height 이하여야 합니다.
# 높이 차가 height 보다 많이 나는 경우에는 사다리를 설치해서 이동할 수 있습니다. 이때, 사다리를 설치하는데 두 격자 칸의 높이차만큼 비용이 듭니다.
# 따라서, 최대한 적은 비용이 들도록 사다리를 설치해서 모든 칸으로 이동 가능하도록 해야 합니다.
# 설치할 수 있는 사다리 개수에 제한은 없으며, 설치한 사다리는 철거하지 않습니다.
# 각 격자칸의 높이가 담긴 2차원 배열 land와 이동 가능한 최대 높이차 height가 매개변수로 주어질 때,
# 모든 칸을 방문하기 위해 필요한 사다리 설치 비용의 최솟값을 return 하도록 solution 함수를 완성해주세요.
import heapq
def solution(land, height):
    dx=[1,-1,0,0]
    dy=[0,0,1,-1]
    n=len(land)
    visited=[[False]*n for x in range(n)]
    #시작점은 0,0 말고 아무거나 해도 됨
    h=[(0,0,0)]
    res=0
    while h:
        cnt,x,y=heapq.heappop(h) 
        
        if visited[x][y]:
            continue
            
        res+=cnt        
        visited[x][y]=True
        for i in range(4):
            xx=dx[i]+x
            yy=dy[i]+y
            #방문하지 않은 경우만 check
            if 0<=xx<n and 0<=yy<n and not visited[xx][yy]:
                temp=abs(land[x][y]-land[xx][yy])
                #height보다 차이가 크다면 그 차이를 넣어줘야함(사다리 건설비용)
                #height보다 작다면 사다리 건설비용이 들지 않기 때문에 0을 삽입
                if temp>height:
                    heapq.heappush(h,(temp,xx,yy))
                else:
                    heapq.heappush(h,(0,xx,yy))
    return res

# 방의 개수
# 원점(0,0)에서 시작해서 아래처럼 숫자가 적힌 방향으로 이동하며 선을 긋습니다.
# 그림을 그릴 때, 사방이 막히면 방하나로 샙니다.
# 이동하는 방향이 담긴 배열 arrows가 매개변수로 주어질 때,
# 방의 갯수를 return 하도록 solution 함수를 작성하세요.
import collections
def solution(arrows):
    answer = 0
    move = [(-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)]
    now = (0, 0)

    # visited : 노드 방문 체크
    # visited_dir : 노드 방문 경로 체크 ((A, B)는 A -> B 경로를 의미)
    visited = collections.defaultdict(int)
    visited_dir = collections.defaultdict(int)

    # arrows 따라 노드 좌표를 큐에 추가
    queue = collections.deque([now])
    for i in arrows:
        # 모래 시계 형태 예외를 처리하기 위해 해당 방향으로 2칸씩 추가한다.
        for _ in range(2):
            next = (now[0] + move[i][0], now[1] + move[i][1])
            queue.append(next)

            now = next

    now = queue.popleft()
    visited[now] = 1

    while queue:
        next = queue.popleft()

        # 이미 방문한 노드(visited[x]==1)인 경우
        if visited[next] == 1:
            # 해당 경로로 처음 들어온 경우인 경우 answer++
            # 처음 들어온 경우에 방이 처음 생성되므로!
            if visited_dir[(now, next)] == 0:
                answer += 1
        # 처음 방문한 노드인 경우 방문 체크를 한다.
        else:
            visited[next] = 1

        # 해당 노드로 들어온 경로를 방문 체크해준다.
        visited_dir[(now, next)] = 1
        visited_dir[(next, now)] = 1
        now = next

    return answer

# 행렬과 연산
# 당신은 행렬에 적용할 수 있는 두 가지 연산을 만들었습니다.
#   ShiftRow
#       모든 행이 아래쪽으로 한 칸씩 밀려납니다. 즉, 모든 행에 대해서 i번째 행은 i+1번째 행이 됩니다. (마지막 행은 1번째 행이 됩니다.)
#   Rotate
#       행렬의 바깥쪽에 있는 원소들을 시계 방향으로 한 칸 회전시킵니다.
#       행렬의 바깥쪽에 있는 원소들은 첫 행, 첫 열, 끝 행, 끝 열에 포함되는 원소들입니다.
#       한 칸 회전시킨다는 것은 이 원소들이 시계 방향으로 한 칸씩 밀려난다는 것을 의미합니다. 즉, 다음 4개의 연산이 동시에 시행됩니다.
#           첫 행에서 끝 열에 있는 원소를 제외한 첫 행의 모든 원소는 오른쪽으로 한 칸 이동합니다.
#           끝 열에서 끝 행에 있는 원소를 제외한 끝 열의 모든 원소는 아래쪽으로 한 칸 이동합니다.
#           끝 행에서 첫 열에 있는 원소를 제외한 끝 행의 모든 원소는 왼쪽으로 한 칸 이동합니다.
#           첫 열에서 첫 행에 있는 원소를 제외한 첫 열의 모든 원소는 위쪽으로 한 칸 이동합니다.
# 당신은 행렬에 연산을 여러 번 시행하려고 합니다.
# 행렬의 초기 상태를 담고 있는 2차원 정수 배열 rc,
# 시행할 연산을 순서대로 담고 있는 문자열 배열 operations가 매개변수로 주어졌을 때,
# 연산을 차례대로 시행한 후의 행렬 상태를 return 하도록 solution 함수를 완성해주세요.
from collections import deque
def solution(rc, operations):
    # 행 수, 열 수
    r_len, c_len = len(rc), len(rc[0])
    
    # ShiftRow 연산을 위해 행별로 관리 [양쪽 원소를 제외한 행들, ...]
    rows = deque(deque(row[1:-1]) for row in rc)
    # Rotate 연산을 위해 바깥쪽 원소들(열별)을 관리 [첫열, 마지막열]
    out_cols = [deque(rc[r][0] for r in range(r_len)),
                deque(rc[r][c_len - 1] for r in range(r_len))]

    # 연산
    for operation in operations:
        # ShiftRow 연산
        if operation[0] == "S":
            # 마지막(가장 아래) 행을 처음(가장 위)로 이동
            rows.appendleft(rows.pop())
            # 첫 열과 마지막 열의 마지막(가장 아래) 원소를 처음(가장 위)으로 이동
            out_cols[0].appendleft(out_cols[0].pop())
            out_cols[1].appendleft(out_cols[1].pop())
        
        # Rotate 연산
        else:
            # << rows가 비어있을 수 있기 때문에 순서 주의 >>
            # 마지막 열의 마지막(가장 아래) 원소를 마지막 행의 마지막(가장 오른쪽)으로 이동
            rows[r_len - 1].append(out_cols[1].pop())
            # 마지막 행의 첫(가장 왼쪽) 원소를 첫 열의 마지막(가장 아래)으로 이동
            out_cols[0].append(rows[r_len - 1].popleft())
            # 첫 열의 첫(가장 위) 원소를 첫 행의 처음(가장 왼쪽)으로 이동
            rows[0].appendleft(out_cols[0].popleft())
            # 첫 행의 마지막(가장 오른쪽) 원소를 마지막 열의 처음(가장 위)으로 이동
            out_cols[1].appendleft(rows[0].pop())
            
    answer = []
    for r in range(r_len):
        answer.append([])
        answer[r].append(out_cols[0][r])
        answer[r].extend(rows[r])
        answer[r].append(out_cols[1][r]) 
    
    return answer

# 가사검색
# 친구들로부터 천재 프로그래머로 불리는 "프로도"는 음악을 하는 친구로부터 자신이 좋아하는 노래 가사에
# 사용된 단어들 중에 특정 키워드가 몇 개 포함되어 있는지 궁금하니 프로그램으로 개발해 달라는 제안을 받았습니다.
# 그 제안 사항 중, 키워드는 와일드카드 문자중 하나인 '?'가 포함된 패턴 형태의 문자열을 뜻합니다.
# 와일드카드 문자인 '?'는 글자 하나를 의미하며, 어떤 문자에도 매치된다고 가정합니다.
#   예를 들어 "fro??"는 "frodo", "front", "frost" 등에 매치되지만 "frame", "frozen"에는 매치되지 않습니다.
# 가사에 사용된 모든 단어들이 담긴 배열 words와 찾고자 하는 키워드가 담긴 배열 queries가 주어질 때,
# 각 키워드 별로 매치된 단어가 몇 개인지 순서대로 배열에 담아 반환하도록 solution 함수를 완성해 주세요.
from bisect import bisect_left,bisect_right
def solution(words, queries):
    answer = []
    # 같은 길이의 단어끼리 모으기
    w = [[] for _ in range(10001)]
    reverse_w = [[] for _ in range(10001)]
    for word in words:
        w[len(word)].append(word)
        reverse_w[len(word)].append(word[::-1])
        
    # 정렬
    for i in range(10001):
        w[i].sort()
        reverse_w[i].sort()
            
    for q in queries:
        # 이분 탐색을 통해 범위 구하기
        # froaa <= fro?? <= frozz
        # ????o : oaaaa <= o???? <= ozzzz
        qA = q.replace('?','a')
        qZ = q.replace('?','z')
        
        # ?가 접두사에 있는가? -> 단어를 뒤집어서 범위 구하기
        if q[0] == '?':
            s = bisect_left(reverse_w[len(q)],qA[::-1])
            e = bisect_right(reverse_w[len(q)],qZ[::-1])
            answer.append(e-s)
        # ?가 접미사에 있는가?
        else:
            s = bisect_left(w[len(q)],qA)
            e = bisect_right(w[len(q)],qZ)
            answer.append(e-s)
    
    return answer

# 고고학 최고의 발견
# 퍼즐은 시계들이 행렬을 이루는 구조물인데 하나의 시계에 시곗바늘은 하나씩만 있습니다.
# 각 시곗바늘은 시계방향으로만 돌릴 수 있고 한 번의 조작으로 90도씩 돌릴 수 있습니다.
# 시계들은 기계장치에 의해 연결되어 있어 어떤 시계의 시곗바늘을 돌리면 그 시계의 상하좌우로 인접한 시계들의 시곗바늘도 함께 돌아갑니다.
# 행렬의 모서리에 위치한 시계의 시곗바늘을 돌리는 경우에는 인접한 세 시계의 시곗바늘들이 함께 돌아가며,
# 꼭짓점에 위치한 시계의 시곗바늘을 돌리는 경우에는 인접한 두 시계의 시곗바늘들이 함께 돌아갑니다.
# 각 시계는 12시, 3시, 6시, 9시 방향 중의 한 방향을 가리키고 있습니다.
# 각 시계의 시곗바늘을 적절히 조작하여 모든 시곗바늘이 12시 방향을 가리키면 퍼즐이 해결되어 성궤를 봉인하고 있는 잠금장치가 열릴 것입니다.
# 노후화된 퍼즐 기계장치가 걱정되었던 혜선은 가능한 최소한의 조작으로 퍼즐을 해결하려고 합니다.
# 시곗바늘들의 행렬 clockHands가 매개변수로 주어질 때, 퍼즐을 해결하기 위한 최소한의 조작 횟수를 return 하도록 solution 함수를 완성해주세요.
from itertools import product
def solution(clockHands):
    answer = 9876543210
    n = len(clockHands)
    dy = [-1, 1, 0, 0, 0]
    dx = [0, 0, -1, 1, 0]

    def rotate(a, b, t, arr):
        for k in range(5):
            y, x = a + dy[k], b + dx[k]
            if 0 <= y < n and 0 <= x < n:
                arr[y][x] = (arr[y][x] + t) % 4

    for case in product(range(4), repeat=n):    # 첫째줄 최대4번까지 회전 한다는 가정 하에 모든 경우의 수를 만든다.
        arr = [i[:] for i in clockHands]    # 깊은 복사는 deepcopy 보다 slicing 이 빠름

        for j in range(n):    # case 를 가지고 첫번째 줄만 회전 시킨다
            rotate(0, j, case[j], arr)

        result = sum(case)    # 첫번째 줄 조작 횟수의 합

        for i in range(1, n):    # 두번째 줄부터 체크
            for j in range(n):
                if arr[i-1][j]:    # 12시 가있지 않은 시계만 조작
                    temp = 4 - arr[i-1][j]    # 12시에 가도록 하기 위한 조작 횟수
                    rotate(i, j, temp, arr)    # 회전
                    result += temp    # 조작 횟수 누적

        if sum(arr[n-1]):    # 마지막 라인에 12시를 향하지 않는 시계가 존재
            continue    # pass
        answer = min(answer, result)    # 시계가 모두 12시를 가리킨다면 answer을 최솟값으로 갱신

    return answer

# 짝수 행 세기
# 모든 수가 0 또는 1로 이루어진 2차원 배열 a가 주어집니다.
# 다음 조건을 모두 만족하는 2차원 배열 b의 경우의 수를 (107 + 19)로 나눈 나머지를 return 하도록 solution 함수를 완성해주세요.
#   b의 모든 원소는 0 아니면 1입니다.
#   a의 행/열의 개수와 b의 행/열의 개수가 같습니다. (= a와 b의 크기가 같습니다.)
#   i = 1, 2, ..., (a의 열의 개수)에 대해서 a의 i번째 열과 b의 i번째 열에 들어 있는 1의 개수가 같습니다.
#   b의 각 행에 들어 있는 1의 개수가 짝수입니다. (0도 짝수입니다.)
from collections import defaultdict
from math import comb
from functools import lru_cache
@lru_cache(maxsize=None)
def C(n,k): return comb(n,k)
def solution(a):
    one_cnt = [sum(ones) for ones in zip(*a)] # 각열 1의 갯수
    DP = defaultdict(int,{(rows := len(a))-one_cnt[0]:comb(rows, one_cnt[0])}) # 1열까지 계산한 DP
    for ones in one_cnt[1:]: # DP[2][j] 부터 계산
        next_DP = defaultdict(int)
        for even_rows in DP:
            odd_rows = rows-even_rows
            for add_one in range(max(0,ones-odd_rows), min(ones,even_rows)+1): # range 범위는 미만이기때문에 +1
                next_DP[even_rows+ones-2*add_one] += DP[even_rows] * C(even_rows, add_one) * C(odd_rows, ones-add_one)%(10**7+19)
        DP = next_DP
    return DP[rows]

# 아방가르드 타일링
# 정우는 예술적 감각이 뛰어난 타일공입니다.
# 그는 단순한 타일을 활용하여 불규칙하면서도 화려하게 타일링을 하곤 합니다.
# 어느 날 정우는 가로 길이 n, 세로 길이 3 인 판을 타일링하는 의뢰를 맡았습니다.
# 아방가르드한 디자인 영감이 떠오른 정우는 다음과 같은 두 가지 종류의 타일로 타일링을 하기로 결정했습니다.
# 타일
#       ㅇ
#  ㅇㅇ  ㅇ
#  ㅇ   ㅇ
# 각 타일은 90도씩 회전할 수 있으며 타일의 개수는 제한이 없습니다.
# n이 주어졌을 때, 이 두 가지 종류의 타일로 n x 3 크기의 판을 타일링하는 방법의 수를 return 하도록 solution 함수를 완성해주세요.
def solution(n):
    dp = [1,1,3,10] + [0]*(n-3)
    sp_c2, sp_c3 = 2, 5
    sp_cases = [12, 2, 4] #각각 6, 4, 5 일 때 초기에 더해지는 값들
    if n <= 3 : return dp[n]
    for idx in range(4,n+1):
        sp_c = idx%3
        total = sp_cases[sp_c]
        
        total += dp[idx-1] + sp_c2*dp[idx-2] + sp_c3*dp[idx-3]
        dp[idx] = total
        
        sp_cases[sp_c] += dp[idx-1]*2 + dp[idx-2]*2 + dp[idx-3]*4
        
    answer = dp[n]%1000000007
    
    return answer

# 쌍둥이 빌딩 숲
# 은비는 길을 걷다가 관광 명소인 쌍둥이 빌딩 숲을 보게 되었습니다.
# 쌍둥이 빌딩 숲은 일렬로 빌딩들이 줄지어 서있는 곳입니다.
# 쌍둥이 빌딩 숲에는 높이가 1부터 n까지 각각 2 채씩 총 2n채의 빌딩이 존재하기 때문에 그러한 이름이 붙게 되었으며,
# 같은 높이를 가지는 빌딩 사이에는 그보다 높은 빌딩이 존재하지 않습니다.
# 은비는 쌍둥이 빌딩 숲을 한쪽 측면에서(열 방향으로) 바라보고 있습니다.
# 이때 count 채의 빌딩이 구분되어 보였습니다.
# 은비의 세계는 안타깝게도 원근감이 존재하지 않지만,
# 다행히 서로 다른 높이를 가지는 빌딩들은 각각 고유한 색깔을 가지고 있어
# 어떤 빌딩이 다른 빌딩에 의해 전체가 가려지지 않는다면 볼 수 있습니다.
# n과 count가 주어졌을 때, 빌딩들이 배치될 수 있는 방법의 수를 return 하도록 solution 함수를 완성해주세요.
import sys; sys.setrecursionlimit(1000000)
cache = {(1, 1): 1}
def solve(i, j):
    if i == 0 or j == 0:
        return 0
    if (i, j) in cache:
        return cache[(i, j)]
    v = solve(i-1, j-1) + solve(i-1, j) * 2 * (i-1)
    cache[(i, j)] = v % 1000000007
    return cache[(i, j)]
    
def solution(n, count):
    # cache[i][j] : i종류의 빌딩을 가지고 j종류의 빌딩이 구분되게 세우는 경우의 수
    #
    # cache[i+1][j+1], 즉, i+1번째 종류의 빌딩이 보여서 j+1종류가 되려면 맨 앞에 세워야 함.
    # 따라서 cache[i+1][j+1] = cache[i][j]
    #
    # cache[i+1][j] : 즉, i+1번째 종류의 빌딩이 가려지게 세우려면,
    # 두 빌딩은 무조건 이어붙인 상태로 배치되고, 2*i 개의 큰 빌딩들 오른쪽 아무데나 들어가도 되므로
    # 2*i 종류.
    # 따라서 cache[i+1][j] = cache[i][j] * 2*i
    #
    # 정리하면
    # cache[i][j] = cache[i-1][j-1] + cache[i-1][j] * 2*(i-1)
    return solve(n, count)

# 표 병합
# 당신은 표 편집 프로그램을 작성하고 있습니다.
# 표의 크기는 50 × 50으로 고정되어있고 초기에 모든 셀은 비어 있습니다.
# 각 셀은 문자열 값을 가질 수 있고, 다른 셀과 병합될 수 있습니다.
# 위에서 r번째, 왼쪽에서 c번째 위치를 (r, c)라고 표현할 때, 당신은 다음 명령어들에 대한 기능을 구현하려고 합니다.
#   "UPDATE r c value"
#       (r, c) 위치의 셀을 선택합니다.
#       선택한 셀의 값을 value로 바꿉니다.
#   "UPDATE value1 value2"
#       value1을 값으로 가지고 있는 모든 셀을 선택합니다.
#       선택한 셀의 값을 value2로 바꿉니다.
#   "MERGE r1 c1 r2 c2"
#       (r1, c1) 위치의 셀과 (r2, c2) 위치의 셀을 선택하여 병합합니다.
#       선택한 두 위치의 셀이 같은 셀일 경우 무시합니다.
#       선택한 두 셀은 서로 인접하지 않을 수도 있습니다.
#           이 경우 (r1, c1) 위치의 셀과 (r2, c2) 위치의 셀만 영향을 받으며, 그 사이에 위치한 셀들은 영향을 받지 않습니다.
#       두 셀 중 한 셀이 값을 가지고 있을 경우 병합된 셀은 그 값을 가지게 됩니다.
#       두 셀 모두 값을 가지고 있을 경우 병합된 셀은 (r1, c1) 위치의 셀 값을 가지게 됩니다.
#       이후 (r1, c1) 와 (r2, c2) 중 어느 위치를 선택하여도 병합된 셀로 접근합니다.
#   "UNMERGE r c"
#       (r, c) 위치의 셀을 선택하여 해당 셀의 모든 병합을 해제합니다.
#       선택한 셀이 포함하고 있던 모든 셀은 프로그램 실행 초기의 상태로 돌아갑니다.
#       병합을 해제하기 전 셀이 값을 가지고 있었을 경우 (r, c) 위치의 셀이 그 값을 가지게 됩니다.
#   "PRINT r c"
#       (r, c) 위치의 셀을 선택하여 셀의 값을 출력합니다.
#       선택한 셀이 비어있을 경우 "EMPTY"를 출력합니다.
# 실행할 명령어들이 담긴 1차원 문자열 배열 commands가 매개변수로 주어집니다.
# commands의 명령어들을 순서대로 실행하였을 때, "PRINT r c" 명령어에 대한 실행결과를 순서대로
# 1차원 문자열 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def solution(commands):
    answer = []  # 결과를 저장할 리스트
    cell = [['EMPTY' for _ in range(51)] for _ in range(51)]  # 'EMPTY'로 초기화
    pointer = [[[i, j] for j in range(51)] for i in range(51)]  # 자기 자신을 가리키도록 초기화 
    
    def insert(r, c, value):
        rr, cc = pointer[r][c]  # 포인터에서 실제 위치(rr, cc) 가져옴
        cell[rr][cc] = value
        
    def update(value1, value2):
        for i in range(51):
            for j in range(51):
                if cell[i][j] == value1:
                    cell[i][j] = value2
                    
    def merge(r1, c1, r2, c2):
        rr1, cc1 = pointer[r1][c1]  # 1번 셀의 실제 위치
        rr2, cc2 = pointer[r2][c2]  # 2번 셀의 실제 위치
        value1, value2 = [cell[rr1][cc1], cell[rr2][cc2]]  # 각 셀 값을 가져옴
        cell[rr1][cc1] = value2 if value1 == 'EMPTY' else value1  # 값 변경
        for i in range(51):
            for j in range(51):
                if pointer[i][j] == [rr2, cc2]: # 2번 셀의 포인터를 1번 셀의 위치로 변경
                    pointer[i][j] = [rr1, cc1]  
        
    def unmerge(r, c):
        rr, cc = pointer[r][c]  # 실제 위치 가져옴
        value = cell[rr][cc]  # 셀 값 가져옴
        for i in range(51):
            for j in range(51):
                if pointer[i][j] == [rr, cc]:
                    pointer[i][j] = [i, j]  # 포인터 초기화
                    cell[i][j] = 'EMPTY'  # 값 초기화
        cell[r][c] = value  # 원래 위치의 셀 값을 복원
        
    def pprint(r, c):
        rr, cc = pointer[r][c]  # 포인터에서 실제 위치(rr, cc) 가져옴
        answer.append(cell[rr][cc])  # 셀 값 결과 리스트에 추가
        
    # 메인 루프
    for i in commands:
        a = i.split(' ')  # 공백을 기준으로 커맨드와 인자들을 분리
        if a[0] == 'UPDATE':
            if len(a) == 4:
                insert(int(a[1]), int(a[2]), a[3]) 
            else:
                update(a[1], a[2]) 
        elif a[0] == 'MERGE':
            merge(int(a[1]), int(a[2]), int(a[3]), int(a[4]))  
        elif a[0] == 'UNMERGE':
            unmerge(int(a[1]), int(a[2]))  
        else:
            pprint(int(a[1]), int(a[2]))  
            
    return answer

# 상담원 인원
# 현대모비스는 우수한 SW 인재 채용을 위해 상시로 채용 설명회를 진행하고 있습니다.
# 채용 설명회에서는 채용과 관련된 상담을 원하는 참가자에게 멘토와 1:1로 상담할 수 있는 기회를 제공합니다.
# 채용 설명회에는 멘토 n명이 있으며, 1~k번으로 분류되는 상담 유형이 있습니다. 각 멘토는 k개의 상담 유형 중 하나만 담당할 수 있습니다.
# 멘토는 자신이 담당하는 유형의 상담만 가능하며, 다른 유형의 상담은 불가능합니다.
# 멘토는 동시에 참가자 한 명과만 상담 가능하며, 상담 시간은 정확히 참가자가 요청한 시간만큼 걸립니다.
# 참가자가 상담 요청을 하면 아래와 같은 규칙대로 상담을 진행합니다.
#   상담을 원하는 참가자가 상담 요청을 했을 때, 참가자의 상담 유형을 담당하는 멘토 중 상담 중이 아닌 멘토와 상담을 시작합니다.
#   만약 참가자의 상담 유형을 담당하는 멘토가 모두 상담 중이라면, 자신의 차례가 올 때까지 기다립니다.
#       참가자가 기다린 시간은 참가자가 상담 요청했을 때부터 멘토와 상담을 시작할 때까지의 시간입니다.
#   모든 멘토는 상담이 끝났을 때 자신의 상담 유형의 상담을 받기 위해 기다리고 있는 참가자가 있으면 즉시 상담을 시작합니다.
#       이때, 먼저 상담 요청한 참가자가 우선됩니다.
#   참가자의 상담 요청 정보가 주어질 때, 참가자가 상담을 요청했을 때부터 상담을 시작하기까지 기다린 시간의 합이 최소가 되도록 각 상담 유형별로 멘토 인원을 정하려 합니다.
# 단, 각 유형별로 멘토 인원이 적어도 한 명 이상이어야 합니다.
# 상담 유형의 수를 나타내는 정수 k, 멘토의 수를 나타내는 정수 n과 참가자의 상담 요청을 담은 2차원 정수 배열 reqs가 매개변수로 주어집니다.
# 멘토 인원을 적절히 배정했을 때 참가자들이 상담을 받기까지 기다린 시간을 모두 합한 값의 최솟값을 return 하도록 solution 함수를 완성해 주세요.
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
def solution(k, n, reqs):
    # 큰 수
    answer = 999999999
    # 중복조합
    comb = combinations_with_replacement([i for i in range(k)], r=n-k)
    # 상담원-유형 케이스목록 구하기
    cases = []
    for case in comb:
        base = [1 for i in range(k)]
        for c in case:
            base[c] += 1
        cases.append(base)
    # 참가자 데이터 가공 문제유형 : [시작시간, 종료시간]
    participants = defaultdict(list)
    for start_time, minutes, category in reqs:
        participants[category].append([start_time, start_time + minutes])
    # 각 케이스에 대해 wait_time을 구하고 최소값 찾기
    for case in cases:
        wait_time = 0 
        # 각 유형에 대해  wait_time 계산하여 더하기
        for i in range(k):
            p_list = sorted(participants[i+1], key=lambda x:x[0])
            mento_list = [0 for _ in range(case[i])]
            for start_time, end_time in p_list:
                mento_list = sorted(mento_list)
                if mento_list[0] <= start_time:
                    mento_list[0] = end_time
                else:
                    temp_t = mento_list[0] - start_time
                    mento_list[0] = end_time + temp_t
                    wait_time += temp_t
            #수행 중 만약 현재 최소값보다 길어지면 break
            if wait_time > answer:
                break
        if wait_time < answer:
            answer = wait_time
    return answer

# 에어컨
# 현대모비스에서 개발한 실내공조 제어 시스템은 차내에 승객이 탑승 중일 때
# 항상 쾌적한 실내온도(t1 ~ t21)를 유지할 수 있도록 합니다. 현재(0분) 실내온도는 실외온도와 같습니다.
# 실내공조 제어 시스템은 실내온도를 조절하기 위해 에어컨의 전원을 켜 희망온도를 설정합니다.
# 희망온도는 에어컨의 전원이 켜져 있는 동안 원하는 값으로 변경할 수 있습니다.
# 실내온도와 희망온도가 다르다면 1분 뒤 실내온도가 희망온도와 같아지는 방향으로 1도 상승 또는 하강합니다.
# 실내온도가 희망온도와 같다면 에어컨이 켜져 있는 동안은 실내온도가 변하지 않습니다.
# 에어컨의 전원을 끄면 실내온도가 실외온도와 같아지는 방향으로 매 분 1도 상승 또는 하강합니다.
# 실내온도와 실외온도가 같다면 실내온도는 변하지 않습니다.
# 에어컨의 소비전력은 현재 실내온도에 따라 달라집니다.
# 에어컨의 희망온도와 실내온도가 다르다면 매 분 전력을 a만큼 소비하고,
# 희망온도와 실내온도가 같다면 매 분 전력을 b만큼 소비합니다. 에어컨이 꺼져 있다면 전력을 소비하지 않으며,
# 에어컨을 켜고 끄는데 필요한 시간과 전력은 0이라고 가정합니다.
# 실내공조 제어 시스템은 차내에 승객이 탑승 중일 때 실내온도를 t1 ~ t2도 사이로 유지하면서, 에어컨의 소비전력을 최소화합니다.
# 실외온도를 나타내는 정수 temperature, 쾌적한 실내온도의 범위를 나타내는 정수 t1, t2,
# 에어컨의 소비전력을 나타내는 정수 a, b와 승객이 탑승 중인 시간을 나타내는 1차원 정수 배열 onboard가 매개변수로 주어집니다.
# 승객이 탑승 중인 시간에 쾌적한 실내온도를 유지하기 위한 최소 소비전력을 return 하도록 solution 함수를 완성해 주세요.
import math
def solution(temperature, t1, t2, a, b, onboard):
    # 0으로 정규화
    diff = temperature - t2 if temperature > t2 else t1 - temperature
    # 시간0에 대해 초기 비용으로 초기화
    result = [[ 1000000 for i in range(diff+1) ]+[0]]
    last_passenger_index = 0
    for i in range(1, len(onboard)):
        arr = []
        for j in range(diff+2):
            temp = [result[i-1][j]+b]
            # 할수있는 행동(비용)은 온도유지(b), 온도 내리기(a), 온도올리기(0)
            if j == diff + 1:
                temp.append(result[i-1][j])
            if j <= diff:
                temp.append(result[i-1][j+1] + a)
            if j > 0:
                temp.append(result[i-1][j-1])
            min_cost = min(temp)
            # 시간i에 대해 승객에 탑승 중이고, 적정온도 범위 밖이라면.
            if onboard[i] == 1:
                last_passenger_index = i
                if j > 1:
                    min_cost = 100000 # 100 * 1000
            arr.append(min_cost)
        result.append(arr)
    return min(result[last_passenger_index])

# 안티 세포
import math
import bisect
MOD = int(1e9 + 7)

def binary_search(arr, x):
    idx = bisect.bisect_left(arr, x)
    if idx == 0 and arr[idx] != x:
        return -1
    elif idx == len(arr):
        return -1
    else:
        return idx

def solve(b):
    maxb, n = max(b), len(b)
    ps = [0, b[0]]
    for i in range(1, n):
        ps.append(ps[-1] + b[i])
    maxe = math.ceil(math.log2(ps[-1]))
    dp = [[0] * (maxe + 1) for _ in range(n)]
    dp[0][0] = 1
    for i in range(1, n):
        # e = 0인 기저 케이스들을 채웁니다.
        dp[i][0] = sum(dp[i-1])
        
        # 나머지 케이스들을 채웁니다.
        for e in range(1, maxe + 1):
            if dp[i][e-1] == 0:
                continue
            
            # jhalf를 찾습니다.
            target = ps[i + 1] - b[i] * 2**(e-1)
            jhalf = binary_search(ps, target)
            if jhalf == -1:
                continue
            eprime = math.log2(b[i] / b[jhalf - 1]) + (e - 1)
            if not eprime.is_integer() or eprime < 0:
                continue
            if not dp[jhalf - 1][int(eprime)] > 0:
                continue
            
            # jfull을 찾습니다.
            target = ps[i + 1] - b[i] * 2**e
            jfull = binary_search(ps, target)
            if jfull == -1:
                continue
            if jfull == 0:
                dp[i][e] = 1
            else:
                dp[i][e] = sum(dp[jfull - 1]) % MOD
    return sum(dp[-1]) % MOD

def solution(a, s):
    answer = []
    start = 0
    for l in s:
        b = a[start:start+l]
        answer.append( solve(b) )
        start += l
    return answer

# 단어 퍼즐
# 단어 퍼즐은 주어진 단어 조각들을 이용해서 주어진 문장을 완성하는 퍼즐입니다.
# 이때, 주어진 각 단어 조각들은 각각 무한개씩 있다고 가정합니다.
# 사용 가능한 단어 조각들을 담고 있는 배열 strs와 완성해야 하는 문자열 t가 매개변수로 주어질 때,
# 주어진 문장을 완성하기 위해 사용해야 하는 단어조각 개수의 최솟값을 return 하도록 solution 함수를 완성해 주세요. 
# 만약 주어진 문장을 완성하는 것이 불가능하면 -1을 return 하세요.
def solution(strs, t):
    INF = 99999
    dp = [INF] * len(t) + [0]
    for i in range(len(t)-1, -1, -1):
        for j in range(1, min(6, len(t)-i+1)):
            if t[i:i+j] in strs:
                dp[i] = min(dp[i], dp[i+j]+1)
    return -1 if dp[0]==INF else dp[0]

# 지형 편집
# XX 게임에서는 지형 편집 기능을 이용하여 플레이어가 직접 게임 속 지형을 수정할 수 있습니다. 
# 이 게임에서는 1 x 1 x 1 크기의 정육면체 블록을 쌓아 게임 속 지형을 표현합니다.
# 이때, 블록이 공중에 떠 있거나, 블록 하나가 여러 개의 칸에 걸쳐 놓일 수는 없습니다. 
# 따라서 지형을 편집하기 위해서는 각 칸의 제일 위에 블록 1개를 새로 추가하거나,
# 제일 위에 있는 블록 한 개를 삭제하는 방식으로 지형을 수정해야 합니다. 
# 이때, 블록 한 개를 새로 추가하거나 삭제하기 위해서는 게임머니를 사용해야 하므로 몇 개의 블록을 추가하고 삭제할지 신중한 선택이 필요합니다.
# 이 게임을 즐기던 한 플레이어는 N x N 크기의 지역에 자신만의 별장을 만들고 싶어졌습니다. 
# 이를 위해서는 울퉁불퉁한 지형의 모든 칸의 높이가 같아지도록 만들어야 합니다.
# 이때, 블록 한 개를 추가하려면 P의 비용이, 제거하려면 Q의 비용이 들게 됩니다.
# 현재 지형의 상태를 나타내는 배열 land와 블록 한 개를 추가하는 데 필요한 비용 P,
# 블록 한 개를 제거하는 데 필요한 비용 Q가 매개변수로 주어질 때,
# 모든 칸에 쌓여있는 블록의 높이가 같아지도록 하는 데 필요한 비용의 최솟값을 return 하도록 solution 함수를 완성해 주세요.
from itertools import chain
def solution(land, P, Q):
    # 일렬로 세우기
    line = list(chain.from_iterable(land))
    line.sort()
    # print(line)
    n = len(line)

    # 가장 낮은 층으로 편집 (무조건 0이 아니라 가지고 있는 층중에 맨 밑)
    # 가장 낮은 층은 지형이 다 있으므로 그 위의 블록을 모두 제거하면 된다.
    cost = (sum(line) - line[0] * n) * Q
    answer = cost

    # 한층씩 쌓기
    for i in range(1, n):
        if line[i] != line[i - 1]:
            # print(f'cost: {cost}, line[i-1]: {line[i - 1]}, line[i]: {line[i]} ')
            cost = cost + ((line[i] - line[i - 1]) * i * P) - ((line[i] - line[i - 1]) * (n - i) * Q)
            if answer < cost:  # 시간 단축 - 변곡점
                break
            answer = min(answer, cost)
    return answer

# 사칙연산
# 사칙연산에서 더하기(+)는 결합법칙이 성립하지만, 빼기(-)는 결합법칙이 성립하지 않습니다.
# 문자열 형태의 숫자와, 더하기 기호("+"), 뺄셈 기호("-")가 들어있는 배열 arr가 매개변수로 주어질 때,
# 서로 다른 연산순서의 계산 결과 중 최댓값을 return 하도록 solution 함수를 완성해 주세요.
#   M[(i, j)]
#   nums[i] 부터 nums[j]까지 연산했을 때 나올 수 있는 최댓값
#   m[(i, j)]
#   nums[i] 부터 nums[j]까지 연산했을 때 나올 수 있는 최솟값
M, m = {}, {}
#   점화식
#   i~j를 두 부분으로 분할하는 모든 k에 대해서 (k=i+1, i+2, ..., j)
#   --> i~k-1 / k~j로 나뉜다고 하자
#   나올 수 있는 연산의 최댓값, 최솟값을 저장해 두어야 한다.
#   ops[k-1]의 경우에 따라 나뉜다
#   ops[k-1] == '-'인 경우,
#   최댓값을 위해서는 M[(i, k-1)] - m[(k, j)] 를 기억해둔다.
#   최솟값을 위해서는 m[(i, k-1)] - M[(k, j)] 를 기억해둔다.
#   ops[k-1] == '+'인 경우,
#   최댓값을 위해서는 M[(i, k-1)] + M[(k, j)] 를 기억해둔다.
#   최솟값을 위해서는 m[(i, k-1)] + m[(k, j)] 를 기억해둔다.
def solution(arr):
    nums = [int(x) for x in arr[::2]]
    ops = [x for x in arr[1::2]]
    for i in range(len(nums)):
        M[(i, i)] = nums[i]
        m[(i, i)] = nums[i]
    for d in range(1, len(nums)):
        for i in range(len(nums)):
            j = i + d
            if j >= len(nums):
                continue
            
            maxcandidates, mincandidates = [], []
            for k in range(i+1, j+1):
                if ops[k-1] == '-':
                    mx = M[(i, k-1)] - m[(k, j)]
                    mn = m[(i, k-1)] - M[(k, j)]
                    maxcandidates.append(mx)
                    mincandidates.append(mn)
                else:
                    mx = M[(i, k-1)] + M[(k, j)]
                    mn = m[(i, k-1)] + m[(k, j)]
                    maxcandidates.append(mx)
                    mincandidates.append(mn)
            
            M[(i, j)] = max(maxcandidates)
            m[(i, j)] = min(mincandidates)
                    
    return M[(0, len(nums) - 1)]