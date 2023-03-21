# 유한소수 판별하기
# 소수점 아래 숫자가 계속되지 않고 유한개인 소수를 유한소수라고 합니다.
# 분수를 소수로 고칠 때 유한소수로 나타낼 수 있는 분수인지 판별하려고 합니다.
# 유한소수가 되기 위한 분수의 조건은 다음과 같습니다.
#  기약분수로 나타내었을 때, 분모의 소인수가 2와 5만 존재해야 합니다.
# 두 정수 a와 b가 매개변수로 주어질 때, a/b가 유한소수이면 1을,
# 무한소수라면 2를 return하도록 solution 함수를 완성해주세요.
from math import gcd
def solution(a, b):
    b //= gcd(a,b)
    while b%2==0:
        b//=2
    while b%5==0:
        b//=5
    return 1 if b==1 else 2

# 다른사람 풀이
def solution(a, b):
    return 1 if a/b * 1000 % 1 == 0 else 2

# 연속된 수의 합
# 연속된 세 개의 정수를 더해 12가 되는 경우는 3, 4, 5입니다.
# 두 정수 num과 total이 주어집니다. 연속된 수 num개를 더한 값이 total이 될 때,
# 정수 배열을 오름차순으로 담아 return하도록 solution함수를 완성해보세요.
def solution(num, total):
    average = total // num
    return [i for i in range(average - (num-1)//2, average + (num + 2)//2)]

