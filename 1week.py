# 369게임
# 머쓱이는 친구들과 369게임을 하고 있습니다.
# 369게임은 1부터 숫자를 하나씩 대며
# 3, 6, 9가 들어가는 숫자는
# 숫자 대신 3, 6, 9의 개수만큼 박수를 치는 게임입니다.
# 머쓱이가 말해야하는 숫자 order가 매개변수로 주어질 때,
# 머쓱이가 쳐야할 박수 횟수를 return 하도록 solution 함수를 완성해보세요.
def solution(order):
    answer = 0
    for i in str(order):
        if i in ["3","6","9"]:
            answer+=1
    return answer

# 몫 구하기
# 정수 num1, num2가 매개변수로 주어질 때,
# num1을 num2로 나눈 몫을 return 하도록 solution 함수를 완성해주세요.
def solution(num1, num2):
    answer = num1//num2
    return answer

# 숫자 비교하기
# 정수 num1과 num2가 매개변수로 주어집니다.
# 두 수가 같으면 1 다르면 -1을 retrun하도록 solution 함수를 완성해주세요.
def solution(num1, num2):
    answer = 0
    if num1 == num2:
        answer = 1
    else:
        answer = -1
    return answer

# 나머지 구하기
# 정수 num1, num2가 매개변수로 주어질 때,
# num1를 num2로 나눈 나머지를 return 하도록 solution 함수를 완성해주세요.
def solution(num1, num2):
    answer = num1 % num2
    return answer

# 나이 출력
# 머쓱이는 40살인 선생님이 몇 년도에 태어났는지 궁금해졌습니다.
# 나이 age가 주어질 때, 2022년을 기준 출생 연도를 return 하는 solution 함수를 완성해주세요.
def solution(age):
    answer = 2023 - int(age)
    return answer

# 두 수의 합
# 정수 num1과 num2가 주어질 때, num1과 num2의 합을 return하도록 soltuion 함수를 완성해주세요.
def solution(num1, num2):
    answer = num1 + num2
    return answer

# 각도기
# 각에서 0도 초과 90도 미만은 예각,
# 90도는 직각, 90도 초과 180도 미만은 둔각 180도는 평각으로 분류합니다.
# 각 angle이 매개변수로 주어질 때
# 예각일 때 1, 직각일 때 2, 둔각일 때 3, 평각일 때 4를 return하도록 solution 함수를 완성해주세요.
def solution(angle):
    answer = 0
    if 0 < angle < 90:
        answer = 1
    elif angle == 90:
        answer = 2
    elif 90 < angle < 180:
        answer = 3
    elif angle == 180:
        answer = 4
    return answer
# 다른 사람 풀이
def solution(angle):
    answer = (angle // 90) * 2 + (angle % 90 > 0) * 1 # t of f로 계산되어 1 or 0
    return answer

# 두 수의 나눗셈
# 정수 num1과 num2가 매개변수로 주어질 때,
# num1을 num2로 나눈 값에
# 1,000을 곱한 후 정수 부분을 return 하도록 soltuion 함수를 완성해주세요.
def solution(num1, num2):
    answer = (num1/num2) * 1000
    return int(answer)

# 배열의 평균값
# 정수 배열 numbers가 매개변수로 주어집니다.
# numbers의 원소의 평균값을 return하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = sum(numbers)/len(numbers)
    return answer

# 짝수의 합
# 정수 n이 주어질 때, n이하의 짝수를 모두 더한 값을 return 하도록 solution 함수를 작성해주세요.
def solution(n):
    answer = 0
    i = 1
    for i in range(i, n+1):
        if (i) % 2 == 0:
            answer += i
    return answer

# 양꼬치
# 머쓱이네 양꼬치 가게는 10인분을 먹으면 음료수 하나를 서비스로 줍니다.
# 양꼬치는 1인분에 12,000원, 음료수는 2,000원입니다.
# 정수 n과 k가 매개변수로 주어졌을 때,
# 양꼬치 n인분과 음료수 k개를 먹었다면
# 총얼마를 지불해야 하는지 return 하도록 solution 함수를 완성해보세요.
def solution(n, k):
    answer = 12000*n + 2000*k
    if n // 10 > 0:
        answer -= ((int(n//10))*2000) 
    return answer

# 배열 원소의 길이
# 문자열 배열 strlist가 매개변수로 주어집니다.
# strlist 각 원소의 길이를 담은 배열을 retrun하도록 solution 함수를 완성해주세요.
def solution(strlist):
    answer = []
    for i in strlist: # 반복해서 문자열을 입력 받고 배열에 추가한다. 
        answer.append(len(i))
    return answer

# 점의 위치 구하기
# 사분면은 한 평면을 x축과 y축을 기준으로 나눈 네 부분입니다.
# 사분면은 아래와 같이 1부터 4까지 번호를매깁니다.
#   x 좌표와 y 좌표가 모두 양수이면 제1사분면에 속합니다.
#   x 좌표가 음수, y 좌표가 양수이면 제2사분면에 속합니다.
#   x 좌표와 y 좌표가 모두 음수이면 제3사분면에 속합니다.
#   x 좌표가 양수, y 좌표가 음수이면 제4사분면에 속합니다.
# x 좌표 (x, y)를 차례대로 담은 정수 배열 dot이 매개변수로 주어집니다.
# 좌표 dot이 사분면 중 어디에 속하는지 1, 2, 3, 4 중 하나를 return 하도록 solution 함수를 완성해주세요.
def solution(dot):
    answer = 1
    if dot[0]*dot[1] > 0:   # if elif else 사용
        if dot[0] > 0: 
            return answer
        else:
            answer = 3
    if dot[0] < 0 and dot[1] > 0:
        answer = 2
    elif dot[0] > 0 and dot[1 < 0]:
        answer = 4
    return answer
# 다른사람 풀이
def solution(dot):
    quad = [(3,2),(4,1)]
    return quad[dot[0] > 0][dot[1] > 0]