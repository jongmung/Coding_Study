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
