# 특정 문자 제거하기
# 문자열 my_string과 문자 letter이 매개변수로 주어집니다.
# my_string에서 letter를 제거한 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(my_string, letter):
    answer = ''
    answer = my_string.replace(letter,'') # .replace로 제거할 수 있다.
    return answer

# 피자 나눠 먹기 (1)
# 머쓱이네 피자가게는 피자를 일곱 조각으로 잘라 줍니다.
# 피자를 나눠먹을 사람의 수 n이 주어질 때,
# 모든 사람이 피자를 한 조각 이상 먹기 위해 필요한 피자의 수를 return 하는 solution 함수를 완성해보세요.
def solution(n):
    answer = 0
    if n%7 == 0:
        answer = n//7
    else:
        answer = (n//7)+1 
    return answer
# 다른사람 풀이
def solution(n):
    return (n - 1) // 7 + 1

# 배열 자르기
# 정수 배열 numbers와 정수 num1, num2가 매개변수로 주어질 때,
# numbers의 num1번 째 인덱스부터 num2번째 인덱스까지 자른 정수 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(numbers, num1, num2):
    answer = []
    for i in range(num1, num2+1):
        answer.append(numbers[i])
    return answer
# 다른사람 풀이
def solution(numbers, num1, num2):
    return numbers[num1:num2+1]

# 옷가게 할인 받기
# 머쓱이네 옷가게는 10만 원 이상 사면 5%, 30만 원 이상 사면 10%,
# 50만 원 이상 사면 20%를 할인해줍니다.
# 구매한 옷의 가격 price가 주어질 때,
# 지불해야 할 금액을 return 하도록 solution 함수를 완성해보세요.
def solution(price):
    answer = 0
    if price >= 100000 and price < 300000:
        answer = price * 0.95
    elif price >= 300000 and price < 500000:
        answer = price * 0.9
    elif price >= 500000:
        answer = price * 0.8
    else:
        answer = price
    return int(answer)
# 다른사람 풀이
def solution(price):
    discount_rates = {500000: 0.8, 300000: 0.9, 100000: 0.95, 0: 1}
    for discount_price, discount_rate in discount_rates.items():
        if price >= discount_price:
            return int(price * discount_rate)