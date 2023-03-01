# 제곱수 판별하기
# 어떤 자연수를 제곱했을 때 나오는 정수를 제곱수라고 합니다.
# 정수 n이 매개변수로 주어질 때, n이 제곱수라면 1을 아니라면 2를 return하도록 solution 함수를 완성해주세요.
import math
def solution(n):
    x = math.sqrt(n)
    if x % 1 == 0:
        answer = 1
    else:
        answer = 2
    return answer
# 다른사람 풀이
def solution(n):
    return 1 if (n ** 0.5).is_integer() else 2

# 가위바위보
# 가위는 2 바위는 0 보는 5로 표현합니다.
# 가위 바위 보를 내는 순서대로 나타낸 문자열 rsp가 매개변수로 주어질 때,
# rsp에 저장된 가위 바위 보를 모두 이기는 경우를 순서대로 나타낸 문자열을 return하도록 solution 함수를 완성해보세요.
def solution(rsp):
    result = {'2':'0','0':'5','5':'2'}
    answer = ''
    for i in rsp:  # 딕셔너리를 만들어서 get()을 사용하여 value 값을 찾아낸다.
        answer += result.get(i)
    return answer
# 다른사람 풀이
def solution(rsp):
    d = {'0':'5','2':'0','5':'2'}
    return ''.join(d[i] for i in rsp) # '',join을 이용하여 하나로 합쳐, dict[key]를 사용하여 value 값을 가져온다.

# 암호 해독
#  암호화된 문자열 cipher를 주고받습니다.
#  그 문자열에서 code의 배수 번째 글자만 진짜 암호입니다.
# 문자열 cipher와 정수 code가 매개변수로 주어질 때
# 해독된 암호 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(cipher, code):
    answer = ''
    for i in range(code,len(cipher)+1):
        if i % code == 0:
            answer +=cipher[i-1]
    return answer

# 직각삼각형 출력하기
# "*"의 높이와 너비를 1이라고 했을 때,
# "*"을 이용해 직각 이등변 삼각형을 그리려고합니다.
# 정수 n 이 주어지면 높이와 너비가 n 인 직각 이등변 삼각형을 출력하도록 코드를 작성해보세요.
n = int(input())
for i in range(1,n+1):
    print('*'*i,end='\n')

# 세균 증식
# 어떤 세균은 1시간에 두배만큼 증식한다고 합니다.
# 처음 세균의 마리수 n과 경과한 시간 t가 매개변수로 주어질 때
# t시간 후 세균의 수를 return하도록 solution 함수를 완성해주세요.
def solution(n, t):
    answer = n
    for i in range(t):
        answer = 2*answer
    return answer
# 다른사람 풀이
def solution(n, t):
    return n << t  # 비트시프트를 사용한다.

# 대문자와 소문자
# 문자열 my_string이 매개변수로 주어질 때,
# 대문자는 소문자로 소문자는 대문자로 변환한 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(my_string):
    return my_string.swapcase() # 대문자를 소문자로, 소문자는 대문자로 -> .swapcase()

# n의 배수 고르기
# 정수 n과 정수 배열 numlist가 매개변수로 주어질 때, numlist에서 n의 배수가 아닌 수들을 제거한 배열을 return하도록 solution 함수를 완성해주세요.
def solution(n, numlist):
    return ([i for i in numlist if i % n ==0])


