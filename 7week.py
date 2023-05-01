# 두 개 뽑아서 더하기
# 정수 배열 numbers가 주어집니다.
# numbers에서 서로 다른 인덱스에 있는 두 개의 수를 뽑아 더해서
# 만들 수 있는 모든 수를 배열에 오름차순으로 담아 return 하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = set()
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            answer.add(numbers[i] + numbers[j])
    answer = list(answer)
    answer.sort()
    return answer
# 다른사람 풀이
def solution(numbers):
    answer = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            answer.append(numbers[i] + numbers[j])
    return sorted(list(set(answer)))

from itertools import combinations
def solution(numbers):
    return sorted(set(sum(i) for i in list(combinations(numbers, 2))))

# 2016년
# 2016년 1월 1일은 금요일입니다. 2016년 a월 b일은 무슨 요일일까요?
# 두 수 a ,b를 입력받아 2016년 a월 b일이 무슨 요일인지 리턴하는 함수,
# solution을 완성하세요.
# 요일의 이름은 일요일부터 토요일까지 각각 SUN,MON,TUE,WED,THU,FRI,SAT
def solution(a, b):
    answer = 0
    days = ['FRI','SAT','SUN','MON','TUE','WED','THU']
    months = [31, 29, 31, 30, 31, 30,31, 31, 30, 31, 30, 31]    
    for i in range(a-1):
        answer += months[i]
    answer += b-1
    answer = answer%7    
    return days[answer]
# 다른사람 풀이
import datetime
def getDayName(a,b):
    t = 'MON TUE WED THU FRI SAT SUN'.split()
    return t[datetime.datetime(2016, a, b).weekday()]