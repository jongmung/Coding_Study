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
