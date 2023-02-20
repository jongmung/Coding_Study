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
    quad = [(3,2),(4,1)]  # quad..사용
    return quad[dot[0] > 0][dot[1] > 0]

# 분수의 덧셈
# 첫 번째 분수의 분자와 분모를 뜻하는 numer1, denom1,
# 두 번째 분수의 분자와 분모를 뜻하는 numer2, denom2가 매개변수로 주어집니다.
# 두 분수를 더한 값을 기약 분수로 나타냈을 때
# 분자와 분모를 순서대로 담은 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(numer1, denom1, numer2, denom2):
    import fractions   # 분수를 사용하기위함
    answer = []
    a = fractions.Fraction(numer1, denom1) # 앞이 분자, 뒤가 분모
    b = fractions.Fraction(numer2, denom2)
    c = a+b      # 각각 분자와 분모라는 뜻
    answer.extend([c.numerator,c.denominator]) # 리스트에 한번에 넣음
    return answer

# 배열 두배 만들기
# 정수 배열 numbers가 매개변수로 주어집니다.
# numbers의 각 원소에 두배한 원소를 가진 배열을 return하도록 solution 함수를 완성해주세요.
def solution(numbers):
    answer = []
    for i in range(0, len(numbers)): # 배열의 길이 만큼 반복
        answer.append(2*numbers[i]) # 값을 곱해서 넣어준다.
    return answer
# 다른사람 풀이
def solution(numbers):
    return [num*2 for num in numbers]
# [표현식 for 항목 in 반복가능객체 if 조건문] 형태를 리스트 컴프리헨션이라 한다.

# 중앙값 구하기
# 중앙값은 어떤 주어진 값들을 크기의 순서대로 정렬했을 때
# 가장 중앙에 위치하는 값을 의미합니다.
# 예를 들어 1, 2, 7, 10, 11의 중앙값은 7입니다.
# 정수 배열 array가 매개변수로 주어질 때, 중앙값을 return 하도록 solution 함수를 완성해보세요.
def solution(array):
    array = sorted(array) # 오름차순 재배열
    length = len(array)//2 # 길이 나눠준다. 홀수니깐
    return array[length]

# 문자열 뒤집기
# 문자열 my_string이 매개변수로 주어집니다.
# my_string을 거꾸로 뒤집은 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(my_string):                  # reversed()는 반대방향으로 순회하는 iterator를 리턴합니다.
    answer = ''.join(reversed(my_string)) # join()으로 리턴된 iterator의 데이터를 하나의 
    return answer                         # string으로 만들면, 뒤집어진 문자열을 만들 수 있습니다.

# 문자 반복 출력하기
# 문자열 my_string과 정수 n이 매개변수로 주어질 때,
# my_string에 들어있는 각 문자를 n만큼 반복한 문자열을 return 하도록 solution 함수를 완성해보세요.
def solution(my_string, n):
    answer = []
    for i in range(len(my_string)):
        for j in range(n):
            answer.append(my_string[i])
    return ''.join(answer)
# 다른사람 풀이
def solution(my_string, n):
    return ''.join(i*n for i in my_string)
# [표현식 for 항목 in 반복가능객체 if 조건문] 형태를 리스트 컴프리헨션이라 한다.

# 머쓱이보다 키 큰 사람
# 머쓱이는 학교에서 키 순으로 줄을 설 때 몇 번째로 서야 하는지 궁금해졌습니다.
# 머쓱이네 반 친구들의 키가 담긴 정수 배열 array와 머쓱이의 키 height가 매개변수로 주어질 때,
# 머쓱이보다 키 큰 사람 수를 return 하도록 solution 함수를 완성해보세요.
def solution(array, height):
    answer = 0
    for i in range(len(array)):
        if height < array[i]:
            answer += 1
    return answer

# 짝수 홀수 개수
# 정수가 담긴 리스트 num_list가 주어질 때,
# num_list의 원소 중 짝수와 홀수의 개수를 담은 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(num_list):
    answer = []
    a = 0
    b = 0
    for i in range(len(num_list)):
        if num_list[i]%2 == 0:
            a+=1
        else:
            b+=1
    answer.extend([a,b])
    return answer
# 다른사람 풀이
def solution(num_list):
    answer = [0,0]
    for n in num_list:
        answer[n%2]+=1 # n값을 나눠서 짝수냐 홀수냐 구별한 뒤 값을 올려준다.
    return answer

# 배열 뒤집기
# 정수가 들어 있는 배열 num_list가 매개변수로 주어집니다.
# num_list의 원소의 순서를 거꾸로 뒤집은 배열을 return하도록 solution 함수를 완성해주세요.
def solution(num_list):
    return num_list[::-1]
     
# 짝수는 싫어요
# 정수 n이 매개변수로 주어질 때, n 이하의 홀수가 오름차순으로 담긴 배열을 return하도록 solution 함수를 완성해주세요.
def solution(n):
    answer = []
    for i in range(1,n+1):
        if i%2 == 1:
            answer.append(i)
    return answer

