# 부분 문자열 이어 붙여 문자열 만들기
# 길이가 같은 문자열 배열 my_strings와 이차원 정수 배열 parts가 매개변수로 주어집니다.
# parts[i]는 [s, e] 형태로, my_string[i]의 인덱스 s부터 인덱스 e까지의 부분 문자열을 의미합니다.
# 각 my_strings의 원소의 parts에 해당하는 부분 문자열을 순서대로 이어 붙인 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_strings, parts):
    answer = ''
    for idx, val in enumerate(parts):
        answer += my_strings[idx][val[0]:val[1]+1]
    return answer

# 조건에 맞게 수열 변환하기1
# 정수 배열 arr가 주어집니다
# . arr의 각 원소에 대해 값이 50보다 크거나 같은 짝수라면 2로 나누고,
# 50보다 작은 홀수라면 2를 곱합니다. 그 결과인 정수 배열을 return 하는 solution 함수를 완성해 주세요.
def solution(arr):
    answer = []
    for i in arr:
        if i%2==0 and i>=50:
            answer.append(int(i)/2)
        elif i%2!=0 and i<=50:
            answer.append(int(i)*2)
        else:
            answer.append(i)
    return answer

# 조건에 맞게 수열 변환하기3
# 정수 배열 arr와 자연수 k가 주어집니다.
# 만약 k가 홀수라면 arr의 모든 원소에 k를 곱하고, k가 짝수라면 arr의 모든 원소에 k를 더합니다.
# 이러한 변환을 마친 후의 arr를 return 하는 solution 함수를 완성해 주세요.
def solution(arr, k):
    answer = []
    for i in arr:
        if k%2==0:
            answer.append(int(i)+k)
        else:
            answer.append(int(i)*k)
    return answer

# 수열과 구간 쿼리1
# 정수 배열 arr와 2차원 정수 배열 queries이 주어집니다.
# queries의 원소는 각각 하나의 query를 나타내며, [s, e] 꼴입니다.
# 각 query마다 순서대로 s ≤ i ≤ e인 모든 i에 대해 arr[i]에 1을 더합니다.
# 위 규칙에 따라 queries를 처리한 이후의 arr를 return 하는 solution 함수를 완성해 주세요.
def solution(arr, queries):
    for i in queries:
        increse = [x + 1 for x in arr[i[0]:i[1]+1]]
        arr[i[0]:i[1]+1] = increse
    return arr

# 수열과 구간 쿼리2
# 정수 배열 arr와 2차원 정수 배열 queries이 주어집니다.
# queries의 원소는 각각 하나의 query를 나타내며, [s, e, k] 꼴입니다.
# 각 query마다 순서대로 s ≤ i ≤ e인 모든 i에 대해 k보다 크면서 가장 작은 arr[i]를 찾습니다.
# 각 쿼리의 순서에 맞게 답을 저장한 배열을 반환하는 solution 함수를 완성해 주세요.
# 단, 특정 쿼리의 답이 존재하지 않으면 -1을 저장합니다.
def solution(arr, queries):
    answer = []
    tmp = []
    for s, e, k in queries:
        tmp = list(filter(lambda x: x > k, sorted(arr[s:e+1])))
        if len(tmp) > 0:
            answer.append(tmp[0])
        else:
            answer.append(-1)
    return answer

# 수열과 구간 쿼리3
# 정수 배열 arr와 2차원 정수 배열 queries이 주어집니다.
# queries의 원소는 각각 하나의 query를 나타내며, [i, j] 꼴입니다.
# 각 query마다 순서대로 arr[i]의 값과 arr[j]의 값을 서로 바꿉니다.
# 위 규칙에 따라 queries를 처리한 이후의 arr를 return 하는 solution 함수를 완성해 주세요.
def solution(arr, queries):
    for i in queries:
        arr[i[0]], arr[i[1]] = arr[i[1]], arr[i[0]]
    return arr

# 수열과 구간 쿼리4
# 정수 배열 arr와 2차원 정수 배열 queries이 주어집니다.
# queries의 원소는 각각 하나의 query를 나타내며, [s, e, k] 꼴입니다.
# 각 query마다 순서대로 s ≤ i ≤ e인 모든 i에 대해 i가 k의 배수이면 arr[i]에 1을 더합니다.
# 위 규칙에 따라 queries를 처리한 이후의 arr를 return 하는 solution 함수를 완성해 주세요.
def solution(arr, queries):
    for s, e, k in queries:
        for i in range(s, e+1):
            if i % k == 0:
                arr[i] += 1
    return arr

# 등차수열의 특정한 항만 더하기
# 두 정수 a, d와 길이가 n인 boolean 배열 included가 주어집니다.
# 첫째항이 a, 공차가 d인 등차수열에서 included[i]가 i + 1항을 의미할 때,
# 이 등차수열의 1항부터 n항까지 included가 true인 항들만 더한 값을 return 하는 solution 함수를 작성해 주세요.
def solution(a, d, included):
    answer = 0
    tmp = []
    for i in range(len(included)):
        tmp.append(a + (d * i))
    for idx, val in enumerate(included):
        if val:
            answer += tmp[idx]
    return answer

# 배열의 원소만큼 추가하기
# 아무 원소도 들어있지 않은 빈 배열 X가 있습니다.
# 양의 정수 배열 arr가 매개변수로 주어질 때,
# arr의 앞에서부터 차례대로 원소를 보면서 원소가 a라면 X의 맨 뒤에 a를 a번 추가하는 일을 
# 반복한 뒤의 배열 X를 return 하는 solution 함수를 작성해 주세요.
def solution(arr):
    answer = []
    for i in range(len(arr)):
        for _ in range(arr[i]):
            answer.append(arr[i])
    return answer

# 배열의 길이에 따라 다른 연산하기
# 정수 배열 arr과 정수 n이 매개변수로 주어집니다.
# arr의 길이가 홀수라면 arr의 모든 짝수 인덱스 위치에 n을 더한 배열을,
# arr의 길이가 짝수라면 arr의 모든 홀수 인덱스 위치에 n을 더한 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(arr, n):
    for idx, val in enumerate(arr):
        if len(arr) % 2 == 0:
            if idx % 2 == 1:
                arr[idx] += n
        else:
            if idx % 2 == 0:
                arr[idx] += n
    return arr

# 배열의 길이를 2의 거듭제곱으로 만들기
# 정수 배열 arr이 매개변수로 주어집니다.
# arr의 길이가 2의 정수 거듭제곱이 되도록 arr 뒤에 정수 0을 추가하려고 합니다.
# arr에 최소한의 개수로 0을 추가한 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(arr):
    count = 0 
    length = len(arr)
    while length > 1:
        length = length / 2
        count += 1
    return arr + [0] * (2 ** count - len(arr))

# 특정 문자열로 끝나는 가장 긴 부분 문자열 찾기
# 문자열 myString과 pat가 주어집니다.
# myString의 부분 문자열중 pat로 끝나는 가장 긴 부분 문자열을 찾아서 return 하는 solution 함수를 완성해 주세요.
def solution(myString, pat):
    return myString.rsplit(pat,1)[0]+pat

# 조건에 맞게 수열 변환하기 2
# 정수 배열 arr가 주어집니다. arr의 각 원소에 대해 값이 50보다 크거나 같은 짝수라면 2로 나누고,
# 50보다 작은 홀수라면 2를 곱하고 다시 1을 더합니다.
# 이러한 작업을 x번 반복한 결과인 배열을 arr(x)라고 표현했을 때,
# arr(x) = arr(x + 1)인 x가 항상 존재합니다.
# 이러한 x 중 가장 작은 값을 return 하는 solution 함수를 완성해 주세요.
# 단, 두 배열에 대한 "="는 두 배열의 크기가 서로 같으며,
# 같은 인덱스의 원소가 각각 서로 같음을 의미합니다.
def solution(arr):
    idx = 0
    prev = arr
    while True:
        change = []
        for i in prev:
            if i >= 50 and i % 2 == 0: change.append(int(i / 2))
            elif i < 50 and i % 2 == 1: change.append(i * 2 + 1)
            else: change.append(i)
        same = all(a == b for a, b in zip(prev, change))
        if same:
            break
        idx += 1
        prev = change
    return idx

# 정수를 나선형으로 배치하기
# 양의 정수 n이 매개변수로 주어집니다. n × n 배열에 1부터 n2 까지
# 정수를 인덱스 [0][0]부터 시계방향 나선형으로 배치한 이차원 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(n):
    array = [[0] * n for _ in range(n)]

    count = 1

    startRow = 0
    endRow = n - 1
    startCol = 0
    endCol = n - 1

    while count <= n * n:
        for i in range(startCol, endCol + 1):
            array[startRow][i] = count
            count += 1
        startRow += 1

        for i in range(startRow, endCol + 1):
            array[i][endCol] = count
            count += 1
        endCol -= 1

        for i in range(endCol, startCol - 1, -1):
            array[endRow][i] = count
            count += 1
        endRow -= 1

        for i in range(endRow, startRow - 1, -1):
            array[i][startCol] = count
            count += 1
        startCol += 1

    return array

# 부분 문자열
# 어떤 문자열 A가 다른 문자열 B안에 속하면 A를 B의 부분 문자열이라고 합니다.
# 예를 들어 문자열 "abc"는 문자열 "aabcc"의 부분 문자열입니다.
# 문자열 str1과 str2가 주어질 때, str1이 str2의 부분 문자열이라면 1을
# 부분 문자열이 아니라면 0을 return하도록 solution 함수를 완성해주세요.
def solution(str1, str2):
    answer = 0
    if str1 in str2:
        answer = 1
    else:
        answer
    return answer

# 배열에서 문자열 대소문자 변환하기
# 문자열 배열 strArr가 주어집니다. 모든 원소가 알파벳으로만 이루어져 있을 때,
# 배열에서 홀수번째 인덱스의 문자열은 모든 문자를 대문자로,
# 짝수번째 인덱스의 문자열은 모든 문자를 소문자로 바꿔서 반환하는 solution 함수를 완성해 주세요.
def solution(strArr):
    answer = []
    for idx, val in enumerate(strArr):
        if idx % 2 == 0:
            answer.append(val.lower())
        else:
            answer.append(val.upper())    
    return answer

# x 사이의 개수
# 문자열 myString이 주어집니다. myString을 문자 "x"를 기준으로 나눴을 때
# 나눠진 문자열 각각의 길이를 순서대로 저장한 배열을 return 하는 solution 함수를 완성해 주세요.
def solution(myString):
    tmp = myString.split("x")
    return [len(i) for i in tmp]

# 콜라츠 수열 만들기
# 모든 자연수 x에 대해서 현재 값이 x이면 x가 짝수일 때는 2로 나누고,
# x가 홀수일 때는 3 * x + 1로 바꾸는 계산을 계속해서 반복하면
# 언젠가는 반드시 x가 1이 되는지 묻는 문제를 콜라츠 문제라고 부릅니다.
# 그리고 위 과정에서 거쳐간 모든 수를 기록한 수열을 콜라츠 수열이라고 부릅니다.
# 계산 결과 1,000 보다 작거나 같은 수에 대해서는 전부 언젠가 1에 도달한다는 것이 알려져 있습니다.
# 임의의 1,000 보다 작거나 같은 양의 정수 n이 주어질 때
# 초기값이 n인 콜라츠 수열을 return 하는 solution 함수를 완성해 주세요.
def solution(n):
    res = [n]
    while n != 1:
        if n%2:
            n = 3 * n + 1
        else:
            n //= 2
        res.append(n)
    return res

# 무작위로 k개의 수 뽑기
# 랜덤으로 서로 다른 k개의 수를 저장한 배열을 만드려고 합니다.
# 적절한 방법이 떠오르지 않기 때문에 일정한 범위 내에서 무작위로 수를 뽑은 후, 
# 지금까지 나온적이 없는 수이면 배열 맨 뒤에 추가하는 방식으로 만들기로 합니다.
# 이미 어떤 수가 무작위로 주어질지 알고 있다고 가정하고, 실제 만들어질 길이 k의 배열을 예상해봅시다.
# 정수 배열 arr가 주어집니다. 문제에서의 무작위의 수는 arr에 저장된 순서대로 주어질 예정이라고 했을 때,
# 완성될 배열을 return 하는 solution 함수를 완성해 주세요.
# 단, 완성될 배열의 길이가 k보다 작으면 나머지 값을 전부 -1로 채워서 return 합니다.
def solution(arr, k):
    answer = []
    arr = [x for i, x in enumerate(arr) if x not in arr[:i]]
    if len(arr)>=k:
        answer = arr[:k]
    else:
        answer = arr[:k]
        answer += [-1]*(k-len(arr))
    return answer

# l로 만들기
# 알파벳 소문자로 이루어진 문자열 myString이 주어집니다.
# 알파벳 순서에서 "l"보다 앞서는 모든 문자를 "l"로 바꾼 문자열을 return 하는 solution 함수를 완성해 주세요.
def solution(myString):
    chars = { "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25 }
    char = ""
    for i in myString:
        if chars[i] < 11:
            i = "l"
        char += i
    return char

# 할 일 목록
# 오늘 해야 할 일이 담긴 문자열 배열 todo_list와
# 각각의 일을 지금 마쳤는지를 나타내는 boolean 배열 finished가 매개변수로 주어질 때,
# todo_list에서 아직 마치지 못한 일들을 순서대로 담은 문자열 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(todo_list, finished):
    answer = dict(zip(todo_list, finished))
    return [key for key, val in answer.items() if val is False]

# 리스트 자르기
# 정수 n과 정수 3개가 담긴 리스트 slicer 그리고 정수 여러 개가 담긴 리스트 num_list가 주어집니다. slicer에 담긴 정수를
# 차례대로 a, b, c라고 할 때, n에 따라 다음과 같이 num_list를 슬라이싱 하려고 합니다.
# n = 1 : num_list의 0번 인덱스부터 b번 인덱스까지
# n = 2 : num_list의 a번 인덱스부터 마지막 인덱스까지
# n = 3 : num_list의 a번 인덱스부터 b번 인덱스까지
# n = 4 : num_list의 a번 인덱스부터 b번 인덱스까지 c 간격으로
# 올바르게 슬라이싱한 리스트를 return하도록 solution 함수를 완성해주세요.
def solution(n, slicer, num_list):
    a,b,c=slicer
    if n==1: return num_list[:b+1]
    if n==2: return num_list[a:]
    if n==3: return num_list[a:b+1]
    return num_list[a:b+1:c]

# 접미사 배열
# 어떤 문자열에 대해서 접미사는 특정 인덱스부터 시작하는 문자열을 의미합니다.
# 예를 들어, "banana"의 모든 접미사는 "banana", "anana", "nana", "ana", "na", "a"입니다.
# 문자열 my_string이 매개변수로 주어질 때,
# my_string의 모든 접미사를 사전순으로 정렬한 문자열 배열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string):
    answer = []
    for i in range(len(my_string)):
        answer.append(my_string[i:])
    answer.sort()
    return answer
# 다른사람 풀이
def solution(my_string):
    return sorted(my_string[i:] for i in range(len(my_string)))

# 배열 만들기3
# 정수 배열 arr와 2개의 구간이 담긴 배열 intervals가 주어집니다.
# intervals는 항상 [[a1, b1], [a2, b2]]의 꼴로 주어지며 각 구간은 닫힌 구간입니다.
# 닫힌 구간은 양 끝값과 그 사이의 값을 모두 포함하는 구간을 의미합니다.
# 이때 배열 arr의 첫 번째 구간에 해당하는 배열과 두 번째 구간에 해당하는 배열을 앞뒤로 붙여
# 새로운 배열을 만들어 return 하는 solution 함수를 완성해 주세요.
def solution(arr, intervals):
    answer = []
    a = []
    for i in range(len(intervals)):
        answer = intervals[i]
        a += arr[answer[0]:answer[1]+1]
    return a
# 다른사람 풀이
def solution(arr, intervals):
    s1, e1 = intervals[0]
    s2, e2 = intervals[1]
    return arr[s1:e1+1] + arr[s2:e2+1]

# 빈 배열에 추가, 삭제하기
# 아무 원소도 들어있지 않은 빈 배열 X가 있습니다.
# 길이가 같은 정수 배열 arr과 boolean 배열 flag가 매개변수로 주어질 때,
# flag를 차례대로 순회하며 flag[i]가 true라면 X의 뒤에 arr[i]를 arr[i] × 2 번 추가하고,
# flag[i]가 false라면 X에서 마지막 arr[i]개의 원소를 제거한 뒤 X를 return 하는 solution 함수를 작성해 주세요.
def solution(arr, flag):
    answer = []
    for i in range(len(flag)):
        if flag[i] == 1:
            for j in range(arr[i]):
                answer.append(arr[i])
                answer.append(arr[i])
        elif flag[i] == 0:
             for k in range(arr[i]):
                answer.pop(-1)
    return answer
# 다른사람 풀이
def solution(arr, flag):
    X = []
    for i, f in enumerate(flag):
        if f:
            X += [arr[i]] * (arr[i]*2)
        else:
            for _ in range(arr[i]):
                X.pop()
    return X

# 글자 지우기
# 문자열 my_string과 정수 배열 indices가 주어질 때,
# my_string에서 indices의 원소에 해당하는 인덱스의 글자를 지우고
# 이어 붙인 문자열을 return 하는 solution 함수를 작성해 주세요.
def solution(my_string, indices):
    answer = list(my_string)
    for i in indices:
        answer[i]=""
    return "".join(answer)
    
# 전국 대회 선발 고사
# 0번부터 n - 1번까지 n명의 학생 중 3명을 선발하는 전국 대회 선발 고사를 보았습니다.
# 등수가 높은 3명을 선발해야 하지만, 개인 사정으로 전국 대회에 참여하지 못하는 학생들이 있어
# 참여가 가능한 학생 중 등수가 높은 3명을 선발하기로 했습니다.
# 각 학생들의 선발 고사 등수를 담은 정수 배열 rank와
# 전국 대회 참여 가능 여부가 담긴 boolean 배열 attendance가 매개변수로 주어집니다.
# 전국 대회에 선발된 학생 번호들을 등수가 높은 순서대로 각각 a, b, c번이라고 할 때
# 10000 × a + 100 × b + c를 return 하는 solution 함수를 작성해 주세요.
def solution(rank, attendance):
    n = len(rank)
    answer =0
    r_a = []
    for i in range(n):
        if attendance[i]:
            r_a.append([rank[i], i])
    r_a.sort(key = lambda v : v[0])
    return 10000 * r_a[0][1] + 100 * r_a[1][1] + r_a[2][1]

# 최소직사각형
# 모든 명함의 가로 길이와 세로 길이를 나타내는 2차원 배열 sizes가 매개변수로 주어집니다.
# 모든 명함을 수납할 수 있는 가장 작은 지갑을 만들 때, 지갑의 크기를 return 하도록 solution 함수를 완성해주세요.
def solution(sizes):
    w = []
    h = []
    for i in range(len(sizes)):
        if sizes[i][0] >= sizes[i][1]:
            w.append(sizes[i][0])
            h.append(sizes[i][1])
        else:
            h.append(sizes[i][0])
            w.append(sizes[i][1])
    return max(w) * max(h)

# 옹알이2
# 머쓱이는 태어난 지 11개월 된 조카를 돌보고 있습니다.
# 조카는 아직 "aya", "ye", "woo", "ma" 네 가지 발음과
# 네 가지 발음을 조합해서 만들 수 있는 발음밖에 하지 못하고 연속해서 같은 발음을 하는 것을 어려워합니다.
# 문자열 배열 babbling이 매개변수로 주어질 때,
# 머쓱이의 조카가 발음할 수 있는 단어의 개수를 return하도록 solution 함수를 완성해주세요.
def solution(babbling):
    answer = 0
    for i in babbling:
        for j in ['aya','ye','woo','ma']:
            if j*2 not in i:
                i=i.replace(j,' ')
                print("i =", i)
        if len(i.strip())==0:
            answer +=1
    return answer

# 추억 점수
# 사진들을 보며 추억에 젖어 있던 루는 사진별로 추억 점수를 매길려고 합니다.
# 사진 속에 나오는 인물의 그리움 점수를 모두 합산한 값이 해당 사진의 추억 점수가 됩니다.
#   예를 들어 사진 속 인물의 이름이 ["may", "kein", "kain"]이고
#   각 인물의 그리움 점수가 [5점, 10점, 1점]일 때
#   해당 사진의 추억 점수는 16(5 + 10 + 1)점이 됩니다.
#   다른 사진 속 인물의 이름이 ["kali", "mari", "don", "tony"]이고
#   ["kali", "mari", "don"]의 그리움 점수가 각각 [11점, 1점, 55점]]이고,
#   "tony"는 그리움 점수가 없을 때, 이 사진의 추억 점수는 3명의 그리움 점수를 합한 67(11 + 1 + 55)점입니다.
# 그리워하는 사람의 이름을 담은 문자열 배열 name,
# 각 사람별 그리움 점수를 담은 정수 배열 yearning,
# 각 사진에 찍힌 인물의 이름을 담은 이차원 문자열 배열 photo가 매개변수로 주어질 때,
# 사진들의 추억 점수를 photo에 주어진 순서대로 배열에 담아 return하는 solution 함수를 완성해주세요.
def solution(name, yearning, photo):
    answer = []
    for people in photo:
        score = 0
        for n in people:
            if n in name:
                score += yearning[name.index(n)]
        answer.append(score)
    return answer

# 가장 가까운 같은 글자
# 문자열 s가 주어졌을 때, s의 각 위치마다 자신보다 앞에 나왔으면서,
# 자신과 가장 가까운 곳에 있는 같은 글자가 어디 있는지 알고 싶습니다.
# 예를 들어, s="banana"라고 할 때,
# 각 글자들을 왼쪽부터 오른쪽으로 읽어 나가면서 다음과 같이 진행할 수 있습니다.
# banana = [-1,-1,-1,2,2,2]
# 문자열 s이 주어질 때, 위와 같이 정의된 연산을 수행하는 함수 solution을 완성해주세요.
def solution(s):
    answer = []
    s_dict = dict()
    for i in range(len(s)):
        if s[i] not in s_dict:
            answer.append(-1)
        else:
            answer.append(i-s_dict[s[i]])
        s_dict[s[i]] = i
    return answer

# 푸드 파이트 대회
# 수웅이는 매달 주어진 음식을 빨리 먹는 푸드 파이트 대회를 개최합니다.
# 이 대회에서 선수들은 1대 1로 대결하며, 매 대결마다 음식의 종류와 양이 바뀝니다.
# 대결은 준비된 음식들을 일렬로 배치한 뒤, 한 선수는 제일 왼쪽에 있는 음식부터 오른쪽으로,
# 다른 선수는 제일 오른쪽에 있는 음식부터 왼쪽으로 순서대로 먹는 방식으로 진행됩니다.
# 중앙에는 물을 배치하고, 물을 먼저 먹는 선수가 승리하게 됩니다.
# 이때, 대회의 공정성을 위해 두 선수가 먹는 음식의 종류와 양이 같아야 하며,
# 음식을 먹는 순서도 같아야 합니다.
# 또한, 이번 대회부터는 칼로리가 낮은 음식을 먼저 먹을 수 있게 배치하여
# 선수들이 음식을 더 잘 먹을 수 있게 하려고 합니다.
# 이번 대회를 위해 수웅이는 음식을 주문했는데,
# 대회의 조건을 고려하지 않고 음식을 주문하여 몇 개의 음식은 대회에 사용하지 못하게 되었습니다.
# 수웅이가 준비한 음식의 양을 칼로리가 적은 순서대로 나타내는 정수 배열 food가 주어졌을 때,
# 대회를 위한 음식의 배치를 나타내는 문자열을 return 하는 solution 함수를 완성해주세요.
def solution(food):
    temp = '' # 왼쪽 선수 음식
    for i in range(1, len(food)):
        temp += str(i) * (food[i]//2)
    return temp + '0' + temp[::-1]

# 카드 뭉치
#  원하는 카드 뭉치에서 카드를 순서대로 한 장씩 사용합니다.
#  한 번 사용한 카드는 다시 사용할 수 없습니다.
#  카드를 사용하지 않고 다음 카드로 넘어갈 수 없습니다.
#  기존에 주어진 카드 뭉치의 단어 순서는 바꿀 수 없습니다.
# 문자열로 이루어진 배열 cards1, cards2와 원하는 단어 배열 goal이 매개변수로 주어질 때,
# cards1과 cards2에 적힌 단어들로 goal를 만들 있다면 "Yes"를,
# 만들 수 없다면 "No"를 return하는 solution 함수를 완성해주세요.
def solution(cards1, cards2, goal):
    for g in goal:
        if len(cards1) > 0 and g == cards1[0]:
            cards1.pop(0)       
        elif len(cards2) > 0 and g == cards2[0]:
            cards2.pop(0)
        else:
            return "No"
    return "Yes"

# 과일 장수
# 과일 장수가 사과 상자를 포장하고 있습니다.
# 사과는 상태에 따라 1점부터 k점까지의 점수로 분류하며,
# k점이 최상품의 사과이고 1점이 최하품의 사과입니다.
# 사과 한 상자의 가격은 다음과 같이 결정됩니다.
#   한 상자에 사과를 m개씩 담아 포장합니다.
#   상자에 담긴 사과 중 가장 낮은 점수가 p (1 ≤ p ≤ k)점인 경우, 사과 한 상자의 가격은 p * m 입니다.
# 과일 장수가 가능한 많은 사과를 팔았을 때,
# 얻을 수 있는 최대 이익을 계산하고자 합니다.(사과는 상자 단위로만 판매하며, 남는 사과는 버립니다)
#   예를 들어, k = 3, m = 4, 사과 7개의 점수가 [1, 2, 3, 1, 2, 3, 1]이라면, 다음과 같이 [2, 3, 2, 3]으로 구성된 사과 상자 1개를 만들어 판매하여 최대 이익을 얻을 수 있습니다.
#   (최저 사과 점수) x (한 상자에 담긴 사과 개수) x (상자의 개수) = 2 x 4 x 1 = 8
# 사과의 최대 점수 k, 한 상자에 들어가는 사과의 수 m, 사과들의 점수 score가 주어졌을 때,
# 과일 장수가 얻을 수 있는 최대 이익을 return하는 solution 함수를 완성해주세요.
def solution(k, m, score): # 사과 최대 점수, 한 상자 사과 개수, 사과 점수
    answer = 0 # 이익
    score.sort(reverse=True)
    for i in range(0, len(score), m):
        if len(score[i:i+m]) == m:
            answer += min(score[i:i+m]) * m
    return answer

# 덧칠하기
# 넓은 벽 전체에 페인트를 새로 칠하는 대신, 구역을 나누어 일부만 페인트를 새로 칠 함으로써 예산을 아끼려 합니다.
# 이를 위해 벽을 1미터 길이의 구역 n개로 나누고, 각 구역에 왼쪽부터 순서대로 1번부터 n번까지 번호를 붙였습니다.
# 그리고 페인트를 다시 칠해야 할 구역들을 정했습니다.
# 벽에 페인트를 칠하는 롤러의 길이는 m미터이고, 롤러로 벽에 페인트를 한 번 칠하는 규칙은 다음과 같습니다.
#   롤러가 벽에서 벗어나면 안 됩니다.
#   구역의 일부분만 포함되도록 칠하면 안 됩니다.
# 즉, 롤러의 좌우측 끝을 구역의 경계선 혹은 벽의 좌우측 끝부분에 맞춘 후 롤러를 위아래로 움직이면서 벽을 칠합니다.
# 현재 페인트를 칠하는 구역들을 완전히 칠한 후 벽에서 롤러를 떼며, 이를 벽을 한 번 칠했다고 정의합니다.
# 한 구역에 페인트를 여러 번 칠해도 되고 다시 칠해야 할 구역이 아닌 곳에 페인트를 칠해도 되지만 다시 칠하기로 정한 구역은 적어도 한 번 페인트칠을 해야 합니다.
# 예산을 아끼기 위해 다시 칠할 구역을 정했듯 마찬가지로 롤러로 페인트칠을 하는 횟수를 최소화하려고 합니다.
# 정수 n, m과 다시 페인트를 칠하기로 정한 구역들의 번호가 담긴 정수 배열 section이 매개변수로 주어질 때
# 롤러로 페인트칠해야 하는 최소 횟수를 return 하는 solution 함수를 작성해 주세요.
def solution(n, m, section):
    answer = 1
    paint = section[0]
    for i in range(1, len(section)):
        if section[i] - paint >= m:
            answer += 1
            paint = section[i]
    return answer

# 콜라 문제
# 콜라를 받기 위해 마트에 주어야 하는 병 수 a,
# 빈 병 a개를 가져다 주면 마트가 주는 콜라 병 수 b,
# 상빈이가 가지고 있는 빈 병의 개수 n이 매개변수로 주어집니다.
# 상빈이가 받을 수 있는 콜라의 병 수를 return 하도록 solution 함수를 작성해주세요.
# 보유 중인 빈 병이 a개 미만이면, 추가적으로 빈 병을 받을 순 없습니다
def solution(a, b, n):
    answer = 0
    # 가지고 있는 콜라병이 교환 가능한 동안에
    while n >= a:
        # 빈 병 개수 추가
        answer += (n // a * b)
        # 교환하지 못하고 남은 병 + 새로 얻은 콜라
        n = (n % a) + (n // a * b) 
    return answer

# 폰켓몬
# 홍 박사님은 당신에게 자신의 연구실에 있는 총 N 마리의 폰켓몬 중에서 N/2마리를 가져가도 좋다고 했습니다.
# 홍 박사님 연구실의 폰켓몬은 종류에 따라 번호를 붙여 구분합니다.
# 따라서 같은 종류의 폰켓몬은 같은 번호를 가지고 있습니다.
# 예를 들어 연구실에 총 4마리의 폰켓몬이 있고,
# 각 폰켓몬의 종류 번호가 [3번, 1번, 2번, 3번]이라면 이는 3번 폰켓몬 두 마리,
# 1번 폰켓몬 한 마리, 2번 폰켓몬 한 마리가 있음을 나타냅니다.
# 이때, 4마리의 폰켓몬 중 2마리를 고르는 방법은 다음과 같이 6가지가 있습니다.
#   1.첫 번째(3번), 두 번째(1번) 폰켓몬을 선택
#   2.첫 번째(3번), 세 번째(2번) 폰켓몬을 선택
#   3.첫 번째(3번), 네 번째(3번) 폰켓몬을 선택
#   4.두 번째(1번), 세 번째(2번) 폰켓몬을 선택
#   5.두 번째(1번), 네 번째(3번) 폰켓몬을 선택
#   6.세 번째(2번), 네 번째(3번) 폰켓몬을 선택
# 이때, 첫 번째(3번) 폰켓몬과 네 번째(3번) 폰켓몬을 선택하는 방법은 한 종류(3번 폰켓몬 두 마리)의 폰켓몬만 가질 수 있지만,
# 다른 방법들은 모두 두 종류의 폰켓몬을 가질 수 있습니다.
# 따라서 위 예시에서 가질 수 있는 폰켓몬 종류 수의 최댓값은 2가 됩니다.
# 당신은 최대한 다양한 종류의 폰켓몬을 가지길 원하기 때문에, 최대한 많은 종류의 폰켓몬을 포함해서 N/2마리를 선택하려 합니다.
# N마리 폰켓몬의 종류 번호가 담긴 배열 nums가 매개변수로 주어질 때, N/2마리의 폰켓몬을 선택하는 방법 중,
# 가장 많은 종류의 폰켓몬을 선택하는 방법을 찾아, 그때의 폰켓몬 종류 번호의 개수를 return 하도록 solution 함수를 완성해주세요.
def solution(nums):
    answer = len(set(nums))
    if len(nums) / 2 > answer:
        return answer
    else:
        return len(nums) / 2

# 명예의 전당(1)
# 매일 출연한 가수의 점수가 지금까지 출연 가수들의 점수 중 상위 k번째 이내이면 해당 가수의 점수를 명예의 전당이라는 목록에 올려 기념합니다.
# 즉 프로그램 시작 이후 초기에 k일까지는 모든 출연 가수의 점수가 명예의 전당에 오르게 됩니다.
# k일 다음부터는 출연 가수의 점수가 기존의 명예의 전당 목록의 k번째 순위의 가수 점수보다 더 높으면,
# 출연 가수의 점수가 명예의 전당에 오르게 되고 기존의 k번째 순위의 점수는 명예의 전당에서 내려오게 됩니다.
# 예를 들어, k = 3이고, 7일 동안 진행된 가수의 점수가 [10, 100, 20, 150, 1, 100, 200]이라면,
# 명예의 전당에서 발표된 점수는 아래의 그림과 같이 [10, 10, 10, 20, 20, 100, 100]입니다.
# 명예의 전당 목록의 점수의 개수 k, 1일부터 마지막 날까지 출연한 가수들의 점수인 score가 주어졌을 때,
# 매일 발표된 명예의 전당의 최하위 점수를 return하는 solution 함수를 완성해주세요.
def solution(k, score):
    answer = []
    a=[]
    for i in score:
        if len(a)<k:
            a.append(i)
        else:
            if min(a)<i:
                a.remove(min(a))
                a.append(i)
        answer.append(min(a))
    return answer

# 기사단원의 무기
# 각 기사는 자신의 기사 번호의 약수 개수에 해당하는 공격력을 가진 무기를 구매하려 합니다.
# 단, 이웃나라와의 협약에 의해 공격력의 제한수치를 정하고,
# 제한수치보다 큰 공격력을 가진 무기를 구매해야 하는 기사는 협약기관에서 정한 공격력을 가지는 무기를 구매해야 합니다.
# 무기를 만들 때, 무기의 공격력 1당 1kg의 철이 필요합니다.
# 그래서 무기점에서 무기를 모두 만들기 위해 필요한 철의 무게를 미리 계산하려 합니다.
# 기사단원의 수를 나타내는 정수 number와 이웃나라와 협약으로 정해진 공격력의 제한수치를 나타내는 정수 limit와
# 제한수치를 초과한 기사가 사용할 무기의 공격력을 나타내는 정수 power가 주어졌을 때,
# 무기점의 주인이 무기를 모두 만들기 위해 필요한 철의 무게를 return 하는 solution 함수를 완성하시오.
def solution(number, limit, power):
    answer = 0
    kg=[]
    for i in range(1,number+1):
        cnt=0
        for j in range(1,int(i**0.5)+1):
            if(i%j==0):
                cnt+=2
                if j**2==i: cnt-=1
            if cnt>limit:
                cnt=power
                break
        kg.append(cnt)
    return sum(kg)

# 체육복
# 학생들의 번호는 체격 순으로 매겨져 있어, 바로 앞번호의 학생이나 바로 뒷번호의 학생에게만 체육복을 빌려줄 수 있습니다.
# 전체 학생의 수 n, 체육복을 도난당한 학생들의 번호가 담긴 배열 lost,
# 여벌의 체육복을 가져온 학생들의 번호가 담긴 배열 reserve가 매개변수로 주어질 때,
# 체육수업을 들을 수 있는 학생의 최댓값을 return 하도록 solution 함수를 작성해주세요.
def solution(n, lost, reserve):
    # 정렬
    lost.sort()
    reserve.sort()
    # lost, reserve에 공통으로 있는 요소 제거
    for i in reserve[:]:
        if i in lost:
            reserve.remove(i)
            lost.remove(i)
    # 체육복 빌려주기(나의 앞 번호부터 확인)
    for i in reserve:
        if i-1 in lost:
            lost.remove(i-1)
        elif i+1 in lost:
            lost.remove(i+1)
    return n-len(lost)

# 문자열 나누기
# 문자열 s가 입력되었을 때 다음 규칙을 따라서 이 문자열을 여러 문자열로 분해하려고 합니다.
#   먼저 첫 글자를 읽습니다. 이 글자를 x라고 합시다.
#   이제 이 문자열을 왼쪽에서 오른쪽으로 읽어나가면서, x와 x가 아닌 다른 글자들이 나온 횟수를 각각 셉니다.
#   처음으로 두 횟수가 같아지는 순간 멈추고, 지금까지 읽은 문자열을 분리합니다.
#   s에서 분리한 문자열을 빼고 남은 부분에 대해서 이 과정을 반복합니다. 남은 부분이 없다면 종료합니다.
#   만약 두 횟수가 다른 상태에서 더 이상 읽을 글자가 없다면, 역시 지금까지 읽은 문자열을 분리하고, 종료합니다.
# 문자열 s가 매개변수로 주어질 때, 위 과정과 같이 문자열들로 분해하고, 분해한 문자열의 개수를 return 하는 함수 solution을 완성하세요.
def solution(s):
    answer = 0
    cnt1 = 0
    cnt2 = 0
    for i in s:
        if cnt1 == cnt2:
            answer += 1
            x = i
        if i == x:
            cnt1 += 1
        else:
            cnt2 += 1            
    return answer

# 예산
# 물품을 구매해 줄 때는 각 부서가 신청한 금액만큼을 모두 지원해 줘야 합니다.
# 예를 들어 1,000원을 신청한 부서에는 정확히 1,000원을 지원해야 하며,
# 1,000원보다 적은 금액을 지원해 줄 수는 없습니다.
# 부서별로 신청한 금액이 들어있는 배열 d와 예산 budget이 매개변수로 주어질 때,
# 최대 몇 개의 부서에 물품을 지원할 수 있는지 return 하도록 solution 함수를 완성해주세요.
def solution(d, budget):
    d.sort()
    while budget < sum(d):
        d.pop()
    return len(d)

# 소수 만들기
# 주어진 숫자 중 3개의 수를 더했을 때 소수가 되는 경우의 개수를 구하려고 합니다.
# 숫자들이 들어있는 배열 nums가 매개변수로 주어질 때,
# nums에 있는 숫자들 중 서로 다른 3개를 골라 더했을 때 
# 소수가 되는 경우의 개수를 return 하도록 solution 함수를 완성해주세요.
from itertools import combinations
import math
def is_prime_number(x):
    #2부터 x의 제곱근까지의 모든 수를 확인하며
    for i in range(2, int(math.sqrt(x))+1):
        #x가 해당 수로 나누어 떨어진다면
        if x % i == 0:
            return False # 소수가 아님
    return True # 소수임
def solution(nums):
    answer = 0
    for x in combinations(nums, 3):
        if is_prime_number(sum(x)):
            answer += 1
    return answer

# 둘만의 암호
# 두 문자열 s와 skip, 그리고 자연수 index가 주어질 때, 다음 규칙에 따라 문자열을 만들려 합니다. 암호의 규칙은 다음과 같습니다.
#   문자열 s의 각 알파벳을 index만큼 뒤의 알파벳으로 바꿔줍니다.
#   index만큼의 뒤의 알파벳이 z를 넘어갈 경우 다시 a로 돌아갑니다.
#   skip에 있는 알파벳은 제외하고 건너뜁니다.
#   예를 들어 s = "aukks", skip = "wbqd", index = 5일 때, a에서 5만큼 뒤에 있는 알파벳은 f지만
#   [b, c, d, e, f]에서 'b'와 'd'는 skip에 포함되므로 세지 않습니다.
#   따라서 'b', 'd'를 제외하고 'a'에서 5만큼 뒤에 있는 알파벳은 [c, e, f, g, h] 순서에 의해 'h'가 됩니다.
#   나머지 "ukks" 또한 위 규칙대로 바꾸면 "appy"가 되며 결과는 "happy"가 됩니다.
# 두 문자열 s와 skip, 그리고 자연수 index가 매개변수로 주어질 때 위 규칙대로 s를 변환한 결과를 return하도록 solution 함수를 완성해주세요.
def solution(s, skip, index):
    answer = ""
    alpha = "abcdefghijklmnopqrstuvwxyz" # 알파벳
    for ch in skip: # ch => skip의 문자 하나하나
        if ch in alpha:
            alpha = alpha.replace(ch, "") # 알파벳 안에 skip 문자들 제거
    for i in s:
        change = alpha[(alpha.index(i) + index) % len(alpha)] # s의 문자 인덱스 + index를 alpha의 길이로 나눈 나머지를 알파벳으로 변환
        answer += change
    return answer

# 햄버거 만들기
# 함께 일을 하는 다른 직원들이 햄버거에 들어갈 재료를 조리해 주면 조리된 순서대로 상수의 앞에 아래서부터 위로 쌓이게 되고,
# 상수는 순서에 맞게 쌓여서 완성된 햄버거를 따로 옮겨 포장을 하게 됩니다.
# 상수가 일하는 가게는 정해진 순서(아래서부터, 빵 – 야채 – 고기 - 빵)로 쌓인 햄버거만 포장을 합니다.
# 상수는 손이 굉장히 빠르기 때문에 상수가 포장하는 동안 속 재료가 추가적으로 들어오는 일은 없으며,
# 재료의 높이는 무시하여 재료가 높이 쌓여서 일이 힘들어지는 경우는 없습니다.
#   예를 들어, 상수의 앞에 쌓이는 재료의 순서가 [야채, 빵, 빵, 야채, 고기, 빵, 야채, 고기, 빵]일 때,
#   상수는 여섯 번째 재료가 쌓였을 때, 세 번째 재료부터 여섯 번째 재료를 이용하여 햄버거를 포장하고, 아홉 번째 재료가 쌓였을 때,
#   두 번째 재료와 일곱 번째 재료부터 아홉 번째 재료를 이용하여 햄버거를 포장합니다. 즉, 2개의 햄버거를 포장하게 됩니다.
# 상수에게 전해지는 재료의 정보를 나타내는 정수 배열 ingredient가 주어졌을 때,
# 상수가 포장하는 햄버거의 개수를 return 하도록 solution 함수를 완성하시오.
def solution(ingredient):
    answer = 0
    stack=[]
    for i in ingredient:
        stack.append(i)
        if stack[-4:]==[1,2,3,1]:
            answer+=1
            for k in range(4):
                stack.pop()
    return answer

# 대충만든 자판
# 키 하나에 여러 문자가 할당된 경우, 동일한 키를 연속해서 빠르게 누르면 할당된 순서대로 문자가 바뀝니다.
#   예를 들어, 1번 키에 "A", "B", "C" 순서대로 문자가 할당되어 있다면
#   1번 키를 한 번 누르면 "A", 두 번 누르면 "B", 세 번 누르면 "C"가 되는 식입니다.
# 같은 규칙을 적용해 아무렇게나 만든 휴대폰 자판이 있습니다. 이 휴대폰 자판은 키의 개수가 1개부터 최대 100개까지 있을 수 있으며,
# 특정 키를 눌렀을 때 입력되는 문자들도 무작위로 배열되어 있습니다.
# 또, 같은 문자가 자판 전체에 여러 번 할당된 경우도 있고, 키 하나에 같은 문자가 여러 번 할당된 경우도 있습니다.
# 심지어 아예 할당되지 않은 경우도 있습니다.
# 따라서 몇몇 문자열은 작성할 수 없을 수도 있습니다.
# 이 휴대폰 자판을 이용해 특정 문자열을 작성할 때, 키를 최소 몇 번 눌러야 그 문자열을 작성할 수 있는지 알아보고자 합니다.
# 1번 키부터 차례대로 할당된 문자들이 순서대로 담긴 문자열배열 keymap과 입력하려는 문자열들이 담긴 문자열 배열 targets가 주어질 때,
# 각 문자열을 작성하기 위해 키를 최소 몇 번씩 눌러야 하는지 순서대로 배열에 담아 return 하는 solution 함수를 완성해 주세요.
# 단, 목표 문자열을 작성할 수 없을 때는 -1을 저장합니다.
def solution(keymap, targets):
    answer = []
    for word in targets:    # targets에 있는 한 단어에 대해서
        times = 0           # 누른 키 총합

        for char in word:   # 한 단어의 개별 문자에 대해
            flag = False    # 목표 문자열을 작성할수 있는지 없는지 판단하기 위함
            time = 101      # keymap의 원소의 길이가 최대 100이기 때문에 101로 설정
            # keymap에 있는 모든 원소를 반복하면서 가장 적게 누를수 있는 char(문자)를 찾음
            for key in keymap:      
                if char in key:	# key(문자열)안에 char(문자)가 존재하면
                    time = min(key.index(char)+1, time)
                    flag = True    # 목표 문자열을 작성할 수 있음
                    
            if not flag:           # 만약 keymap을 전부다 돌았는데도 문자를 찾지 못하면
                times = -1;break   # 목표 문자열을 작성할 수 없음 
            
            times += time          # 하나의 문자에 대해 누른 키를 총합(키)에 더해줌          
        answer.append(times)        
        # targets에 있는 하나의 단어에 대해 수행이 끝났으면 answer에 누른 키 총합을 넣어줌
    return answer

# 실패율
# 실패율을 구하는 부분에서 위기에 빠지고 말았다. 오렐리를 위해 실패율을 구하는 코드를 완성하라.
#   실패율은 다음과 같이 정의한다.
#   스테이지에 도달했으나 아직 클리어하지 못한 플레이어의 수 / 스테이지에 도달한 플레이어 수
# 전체 스테이지의 개수 N, 게임을 이용하는 사용자가 현재 멈춰있는 스테이지의 번호가 담긴 배열 stages가 매개변수로 주어질 때,
# 실패율이 높은 스테이지부터 내림차순으로 스테이지의 번호가 담겨있는 배열을 return 하도록 solution 함수를 완성하라.
def solution(N, stages):
    People = len(stages)
    faillist = {}
    for i in range(1, N + 1):
        if People != 0:
            faillist[i] = stages.count(i) / People
            People -= stages.count(i)
        else:
            faillist[i] = 0
    return sorted(faillist, key=lambda i: faillist[i], reverse=True)

# 로또의 최고 순위와 최저 순위
# 알아볼 수 없는 번호를 0으로 표기하기로 하고, 
# 민우가 구매한 로또 번호를 담은 배열 lottos, 당첨 번호를 담은 배열 win_nums가 매개변수로 주어집니다.
# 이때, 당첨 가능한 최고 순위와 최저 순위를 차례대로 배열에 담아서 return 하도록 solution 함수를 완성해주세요.
def solution(lottos, win_nums):
    rank=[6,6,5,4,3,2,1] # 딕셔너리 쓰지않고 순서대로 당첨 내용 리스트로!
    cnt_0 = lottos.count(0) # 찾고자 하는 항목이 파이썬의 리스트에 몇개나 들어있는지 확인하는 count 함수
    ans = 0
    for x in win_nums:
        if x in lottos:
            ans += 1
    return rank[cnt_0 + ans],rank[ans]

# 키패드 누르기
def solution(numbers, hand):
    num = { 1 : [0,0], 2 : [0,1], 3 : [0,2],
           4 : [1,0], 5 : [1,1], 6 : [1,2],
           7 : [2,0], 8 : [2,1], 9 : [2,2],
          '*' : [3,0], 0 : [3,1], '#' : [3,2] }
    answer = ''
    r = num['#']
    l = num['*']
    for i in numbers:
        now = num[i]
        
        if i in [1, 4, 7]:
            l = now
            answer += 'L'
        elif i in [3, 6, 9]:
            r = now
            answer += 'R'
        else:
            l_d = 0
            r_d = 0
            
            for x, y, z in zip(l, r, now):
                l_d += abs(x - z)
                r_d += abs(y - z)
            
            if l_d < r_d:
                l = now
                answer += 'L'
            elif l_d > r_d:
                r = now
                answer += 'R'
            else:
                if hand == 'right':
                    r = now
                    answer += 'R'
                else:
                    l = now
                    answer += 'L'
    return answer

# 달리기 경주
def solution(players, callings):
    result = {player: i for i, player in enumerate(players)} # 선수: 등수
    for who in callings:
        idx = result[who] # 호명된 선수의 현재 등수
        result[who] -= 1 # 하나 앞 등수로 바꿔줌 -1
        result[players[idx-1]] += 1 # 앞에 위치했던 선수의 등수 +1
        players[idx-1], players[idx] = players[idx], players[idx-1] # 위치 변경
    return players

# 성격 유형 검사하기
def solution(survey, choices):
    answer = ''
    dic= {"R" : 0,"T" : 0,"C" : 0,"F" : 0,"J" : 0,"M" : 0,"A" : 0,"N" : 0 }
    for s,c in zip(survey,choices):
        if c>4:     dic[s[1]] += c-4
        elif c<4:   dic[s[0]] += 4-c
    
    li = list(dic.items())
    for i in range(0,8,2):
        if li[i][1] < li[i+1][1]: answer += li[i+1][0]
        else:   answer += li[i][0]
    return answer

# [1차] 비밀지도
def solution(n, arr1, arr2):
    answer = []
    
    for i in range(n):
        tmp = bin(arr1[i] | arr2[i])
        # tmp결과 ex) '0b1101'
        tmp = tmp[2:].zfill(n)
        # tmp결과 ex) '01101'
        tmp = tmp.replace('1','#').replace('0',' ')
        # tmp결과 ex) ' ## #'
        answer.append(tmp)
    
    return answer

# 크레인 인형뽑기 게임
def solution(board, moves):
    stacklist = []
    answer = 0
    for i in moves:
        for j in range(len(board)):
            if board[j][i-1] != 0:
                stacklist.append(board[j][i-1])
                board[j][i-1] = 0

                if len(stacklist) > 1:
                    if stacklist[-1] == stacklist[-2]:
                        stacklist.pop(-1)
                        stacklist.pop(-1)
                        answer += 2     
                break
    return answer

# 최댓값과 최솟값
# 문자열 s에는 공백으로 구분된 숫자들이 저장되어 있습니다.
# str에 나타나는 숫자 중 최소값과 최대값을 찾아 이를 "(최소값) (최대값)"형태의 문자열을 반환하는 함수,
# solution을 완성하세요.
def solution(s):
    arr = list(map(int, s.split(' ')))
    return str(min(arr)) + " " + str(max(arr))

# JadenCase 문자열 만들기
# JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다.
# 단, 첫 문자가 알파벳이 아닐 때에는 이어지는 알파벳은 소문자로 쓰면 됩니다. (첫 번째 입출력 예 참고)
# 문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.
def solution(s):
    answer = ''
    s=s.split(' ')
    for i in range(len(s)):
        # capitalize 내장함수를 사용하면 첫 문자가 알파벳일 경우 대문자로 만들고
        # 두번째 문자부터는 자동으로 소문자로 만든다
        # 첫 문자가 알파벳이 아니면 그대로 리턴한다
        s[i]=s[i].capitalize()
    answer=' '.join(s)
    return answer

# 최솟값 만들기
# 길이가 같은 배열 A, B 두개가 있습니다. 각 배열은 자연수로 이루어져 있습니다.
# 배열 A, B에서 각각 한 개의 숫자를 뽑아 두 수를 곱합니다.
# 이러한 과정을 배열의 길이만큼 반복하며, 두 수를 곱한 값을 누적하여 더합니다.
# 이때 최종적으로 누적된 값이 최소가 되도록 만드는 것이 목표입니다.
# (단, 각 배열에서 k번째 숫자를 뽑았다면 다음에 k번째 숫자는 다시 뽑을 수 없습니다.)
def solution(A,B):
    answer = 0
    A.sort()
    B.sort(reverse=True)
    for i in range(len(A)):
        answer += A[i]*B[i]
    return answer

# 숫자의 표현
# Finn은 요즘 수학공부에 빠져 있습니다.
# 수학 공부를 하던 Finn은 자연수 n을 연속한 자연수들로 표현 하는 방법이 여러개라는 사실을 알게 되었습니다.
# 예를들어 15는 다음과 같이 4가지로 표현 할 수 있습니다.
#   1 + 2 + 3 + 4 + 5 = 15
#   4 + 5 + 6 = 15
#   7 + 8 = 15
#   15 = 15
# 자연수 n이 매개변수로 주어질 때, 연속된 자연수들로
# n을 표현하는 방법의 수를 return하는 solution를 완성해주세요.
def solution(n):
    answer = 0
    for i in range(1, n+1):
        sum_val = 0
        for j in range(i, n+1):
            sum_val += j
            if sum_val == n:
                answer += 1
            elif sum_val > n:
                break      
    return answer

# 다음 큰 숫자
# 자연수 n이 주어졌을 때, n의 다음 큰 숫자는 다음과 같이 정의 합니다.
#   조건 1. n의 다음 큰 숫자는 n보다 큰 자연수 입니다.
#   조건 2. n의 다음 큰 숫자와 n은 2진수로 변환했을 때 1의 갯수가 같습니다.
#   조건 3. n의 다음 큰 숫자는 조건 1, 2를 만족하는 수 중 가장 작은 수 입니다.
#   예를 들어서 78(1001110)의 다음 큰 숫자는 83(1010011)입니다.
# 자연수 n이 매개변수로 주어질 때, n의 다음 큰 숫자를 return 하는 solution 함수를 완성해주세요.
def solution(n):
    #자연수 n을 이진수로 변환했을 때 1의 갯수
    cnt1 = bin(n)[2:].count("1")
    #자연수 n을 1씩 증가시킨다.
    while True:
        n += 1
        if cnt1 == bin(n)[2:].count("1"):
            break
    return n

# 짝지어 제거하기
# 짝지어 제거하기는, 알파벳 소문자로 이루어진 문자열을 가지고 시작합니다.
# 먼저 문자열에서 같은 알파벳이 2개 붙어 있는 짝을 찾습니다.
# 그다음, 그 둘을 제거한 뒤, 앞뒤로 문자열을 이어 붙입니다.
# 이 과정을 반복해서 문자열을 모두 제거한다면 짝지어 제거하기가 종료됩니다.
# 문자열 S가 주어졌을 때, 짝지어 제거하기를 성공적으로 수행할 수 있는지 반환하는 함수를 완성해 주세요.
# 성공적으로 수행할 수 있으면 1을, 아닐 경우 0을 리턴해주면 됩니다.
#   예를 들어, 문자열 S = baabaa 라면
#   b aa baa → bb aa → aa →
#   의 순서로 문자열을 모두 제거할 수 있으므로 1을 반환합니다.
def solution(s): 
    stack = []
    for i in s:
        if len(stack) == 0: stack.append(i)
        elif stack[-1] == i: stack.pop()
        else: stack.append(i)
    if len(stack) == 0: return 1
    else: return 0
    
# 피보나치 수
# 피보나치 수는 F(0) = 0, F(1) = 1일 때, 1 이상의 n에 대하여 F(n) = F(n-1) + F(n-2) 가 적용되는 수 입니다.
#   예를들어
#   F(2) = F(0) + F(1) = 0 + 1 = 1
#   F(3) = F(1) + F(2) = 1 + 1 = 2
#   F(4) = F(2) + F(3) = 1 + 2 = 3
#   F(5) = F(3) + F(4) = 2 + 3 = 5
#   와 같이 이어집니다.
# 2 이상의 n이 입력되었을 때, n번째 피보나치 수를 1234567으로 나눈 나머지를 리턴하는 함수, solution을 완성해 주세요.
def solution(n):
    fibo = [0, 1]
    for i in range(2, n+1):
        fibo.append(fibo[i-1] + fibo[i-2])
            
    answer = fibo[-1]%1234567
    return answer

# 멀리 뛰기
# 효진이는 멀리 뛰기를 연습하고 있습니다. 효진이는 한번에 1칸, 또는 2칸을 뛸 수 있습니다.
# 칸이 총 4개 있을 때, 효진이는
# (1칸, 1칸, 1칸, 1칸)
# (1칸, 2칸, 1칸)
# (1칸, 1칸, 2칸)
# (2칸, 1칸, 1칸)
# (2칸, 2칸)
# 의 5가지 방법으로 맨 끝 칸에 도달할 수 있습니다.
# 멀리뛰기에 사용될 칸의 수 n이 주어질 때, 효진이가 끝에 도달하는 방법이 몇 가지인지 알아내,
# 여기에 1234567를 나눈 나머지를 리턴하는 함수, solution을 완성하세요.
# 예를 들어 4가 입력된다면, 5를 return하면 됩니다.
def solution(n):
    if n<3:
        return n
    d=[0]*(n+1)
    d[1]=1
    d[2]=2
    for i in range(3,n+1):
        d[i]=d[i-1]+d[i-2]
    return d[n]%1234567

# 귤 고르기
from collections import Counter
def solution(k, tangerine):
    answer = 0
    counter=Counter(tangerine)
    # {3: 2, 2: 2, 5: 2, 1: 1, 4: 1}
    sort_=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    #정렬된 딕셔너리로 귤 개수 맞추기
    cnt=0
    for i in sort_:
        k-=i[1]
        answer+=1
        if k<=0:
            break
    return answer

# 이진 변환 반복하기
# 0과 1로 이루어진 어떤 문자열 x에 대한 이진 변환을 다음과 같이 정의합니다.
#   x의 모든 0을 제거합니다.
#   x의 길이를 c라고 하면, x를 "c를 2진법으로 표현한 문자열"로 바꿉니다.
# 예를 들어, x = "0111010"이라면, x에 이진 변환을 가하면
#   x = "0111010" -> "1111" -> "100" 이 됩니다.
# 0과 1로 이루어진 문자열 s가 매개변수로 주어집니다.
# s가 "1"이 될 때까지 계속해서 s에 이진 변환을 가했을 때,
# 이진 변환의 횟수와 변환 과정에서 제거된 모든 0의 개수를
# 각각 배열에 담아 return 하도록 solution 함수를 완성해주세요.
def solution(s):
    a, b = 0, 0
    while s != '1':
        a += 1
        num = s.count('1')
        b += len(s) - num
        s = bin(num)[2:]
    return [a, b]

# 카펫
# Leo는 집으로 돌아와서 아까 본 카펫의 노란색과 갈색으로 색칠된 격자의 개수는 기억했지만,
# 전체 카펫의 크기는 기억하지 못했습니다.
# Leo가 본 카펫에서 갈색 격자의 수 brown, 노란색 격자의 수 yellow가 매개변수로 주어질 때
# 카펫의 가로, 세로 크기를 순서대로 배열에 담아 return 하도록 solution 함수를 작성해주세요.
def solution(brown, yellow):
    answer = []
    total = brown + yellow                  # a * b = total
    for b in range(1,total+1):
        if (total / b) % 1 == 0:            # total / b = a
            a = total / b
            if a >= b:                      # a >= b
                if 2*a + 2*b == brown + 4:  # 2*a + 2*b = brown + 4 
                    return [a,b]
    return answer

# n개의 최소공배수
# 두 수의 최소공배수(Least Common Multiple)란 입력된 두 수의 배수 중 공통이 되는 가장 작은 숫자를 의미합니다.
# 예를 들어 2와 7의 최소공배수는 14가 됩니다.
# 정의를 확장해서, n개의 수의 최소공배수는 n 개의 수들의 배수 중 공통이 되는 가장 작은 숫자가 됩니다.
# n개의 숫자를 담은 배열 arr이 입력되었을 때 이 수들의 최소공배수를 반환하는 함수, solution을 완성해 주세요.
def solution(arr):
    from math import gcd                # 최대공약수를 구하는 gcd() import
    answer = arr[0]                     # answer을 arr[0]으로 초기화

    for num in arr:                     # 반복문을 처음부터 끝까지 돈다.
        #1. (arr[0],arr[1])의 최소공배수를 구한 후 answer에 저장
        #2. (#1에서 구한 최소공배수, arr[2])의 최소공배수를 구한 후 answer에 저장
        #3. 모든 배열을 돌면서 최소공배수를 구하고, 저장하고 하는 방식을 진행
        answer = answer*num // gcd(answer, num)     

    return answer

# 점프와 순간이동
# OO 연구소는 한 번에 K 칸을 앞으로 점프하거나, (현재까지 온 거리) x 2 에 해당하는 위치로
# 순간이동을 할 수 있는 특수한 기능을 가진 아이언 슈트를 개발하여 판매하고 있습니다.
# 이 아이언 슈트는 건전지로 작동되는데, 순간이동을 하면 건전지 사용량이 줄지 않지만,
# 앞으로 K 칸을 점프하면 K 만큼의 건전지 사용량이 듭니다.
# 그러므로 아이언 슈트를 착용하고 이동할 때는 순간 이동을 하는 것이 더 효율적입니다.
# 아이언 슈트 구매자는 아이언 슈트를 착용하고 거리가 N 만큼 떨어져 있는 장소로 가려고 합니다.
# 단, 건전지 사용량을 줄이기 위해 점프로 이동하는 것은 최소로 하려고 합니다.
# 아이언 슈트 구매자가 이동하려는 거리 N이 주어졌을 때,
# 사용해야 하는 건전지 사용량의 최솟값을 return하는 solution 함수를 만들어 주세요.
#   예를 들어 거리가 5만큼 떨어져 있는 장소로 가려고 합니다.
#   아이언 슈트를 입고 거리가 5만큼 떨어져 있는 장소로 갈 수 있는 경우의 수는 여러 가지입니다.
#   처음 위치 0 에서 5 칸을 앞으로 점프하면 바로 도착하지만, 건전지 사용량이 5 만큼 듭니다.
#   처음 위치 0 에서 2 칸을 앞으로 점프한 다음 순간이동 하면 (현재까지 온 거리 : 2) x 2에 해당하는 위치로 이동할 수 있으므로 위치 4로 이동합니다. 이때 1 칸을 앞으로 점프하면 도착하므로 건전지 사용량이 3 만큼 듭니다.
#   처음 위치 0 에서 1 칸을 앞으로 점프한 다음 순간이동 하면 (현재까지 온 거리 : 1) x 2에 해당하는 위치로 이동할 수 있으므로 위치 2로 이동됩니다. 이때 다시 순간이동 하면 (현재까지 온 거리 : 2) x 2 만큼 이동할 수 있으므로 위치 4로 이동합니다. 이때 1 칸을 앞으로 점프하면 도착하므로 건전지 사용량이 2 만큼 듭니다.
# 위의 3가지 경우 거리가 5만큼 떨어져 있는 장소로 가기 위해서 3번째 경우가 건전지 사용량이 가장 적으므로 답은 2가 됩니다.
def solution(N):
    answer = 0
    while N > 0:
        answer += N % 2
        N //= 2
    return answer

# 구명보트
# 무인도에 갇힌 사람들을 구명보트를 이용하여 구출하려고 합니다.
# 구명보트는 작아서 한 번에 최대 2명씩 밖에 탈 수 없고, 무게 제한도 있습니다.
#   예를 들어, 사람들의 몸무게가 [70kg, 50kg, 80kg, 50kg]이고
#   구명보트의 무게 제한이 100kg이라면 2번째 사람과 4번째 사람은 같이 탈 수 있지만
#   1번째 사람과 3번째 사람의 무게의 합은 150kg이므로 구명보트의 무게 제한을 초과하여 같이 탈 수 없습니다.
# 구명보트를 최대한 적게 사용하여 모든 사람을 구출하려고 합니다.
# 사람들의 몸무게를 담은 배열 people과 구명보트의 무게 제한 limit가 매개변수로 주어질 때,
# 모든 사람을 구출하기 위해 필요한 구명보트 개수의 최솟값을 return 하도록 solution 함수를 작성해주세요.
def solution(people, limit) :
    answer = 0
    people.sort()
    a = 0
    b = len(people) - 1
    while a < b :
        if people[b] + people[a] <= limit :
            a += 1
            answer += 1
        b -= 1
    return len(people) - answer

# 예상 대진표
# △△ 게임대회가 개최되었습니다. 이 대회는 N명이 참가하고, 토너먼트 형식으로 진행됩니다.
# N명의 참가자는 각각 1부터 N번을 차례대로 배정받습니다.
# 그리고, 1번↔2번, 3번↔4번, ... , N-1번↔N번의 참가자끼리 게임을 진행합니다.
# 각 게임에서 이긴 사람은 다음 라운드에 진출할 수 있습니다.
# 이때, 다음 라운드에 진출할 참가자의 번호는 다시 1번부터 N/2번을 차례대로 배정받습니다.
# 만약 1번↔2번 끼리 겨루는 게임에서 2번이 승리했다면 다음 라운드에서 1번을 부여받고,
# 3번↔4번에서 겨루는 게임에서 3번이 승리했다면 다음 라운드에서 2번을 부여받게 됩니다.
# 게임은 최종 한 명이 남을 때까지 진행됩니다.
# 이때, 처음 라운드에서 A번을 가진 참가자는 경쟁자로 생각하는 B번 참가자와 몇 번째 라운드에서 만나는지 궁금해졌습니다.
# 게임 참가자 수 N, 참가자 번호 A, 경쟁자 번호 B가 함수 solution의 매개변수로 주어질 때,
# 처음 라운드에서 A번을 가진 참가자는 경쟁자로 생각하는 B번 참가자와 몇 번째 라운드에서 만나는지
# return 하는 solution 함수를 완성해 주세요.
# 단, A번 참가자와 B번 참가자는 서로 붙게 되기 전까지 항상 이긴다고 가정합니다.
def solution(n,a,b): 
	answer = 0
	while a != b: 
		answer += 1
        # 1을 더해서 2로 나누었을 때, 자리수를 맞춰줌
        # 예) 1, 2의 경우는 2, 3으로 해서 나눴을때 몫이 1이 되도록
		a, b = (a+1)//2, (b+1)//2
	return answer

# 연속 부분 수열 합의 개수
# 철호는 수열을 가지고 놀기 좋아합니다.
# 어느 날 철호는 어떤 자연수로 이루어진 원형 수열의 연속하는 부분 수열의 합으로 만들 수 있는 수가 모두 몇 가지인지 알아보고 싶어졌습니다.
# 원형 수열이란 일반적인 수열에서 처음과 끝이 연결된 형태의 수열을 말합니다.
# 예를 들어 수열 [7, 9, 1, 1, 4] 로 원형 수열을 만들면 다음과 같습니다.
# 원형 수열은 처음과 끝이 연결되어 끊기는 부분이 없기 때문에 연속하는 부분 수열도 일반적인 수열보다 많아집니다.
# 원형 수열의 모든 원소 elements가 순서대로 주어질 때,
# 원형 수열의 연속 부분 수열 합으로 만들 수 있는 수의 개수를 return 하도록 solution 함수를 완성해주세요.
def solution(elements):
    result = set()
    
    elementLen = len(elements)
    # [7, 9, 1, 1, 4, 7, 9, 1, 1, 4]
    elements = elements * 2
    
    for i in range(elementLen):
        for j in range(elementLen):
            result.add(sum(elements[j:j+i+1]))
    return len(result)