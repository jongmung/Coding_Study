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