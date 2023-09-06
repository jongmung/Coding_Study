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