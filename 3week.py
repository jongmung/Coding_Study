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
    answer = []
    for i in range(len(numlist)):
        if numlist[i]%n == 0:
            answer.append(numlist[i])
    return answer
# 다른사람 풀이
def solution(n, numlist):
    return ([i for i in numlist if i % n ==0])

# 문자열 정렬하기 (1)
# 문자열 my_string이 매개변수로 주어질 때,
# my_string 안에 있는 숫자만 골라 오름차순 정렬한 리스트를
# return 하도록 solution 함수를 작성해보세요.
def solution(my_string):
    answer = []
    for i in my_string:
        try:
            answer.append(int(i))
        except:
            continue
    answer.sort()
    return answer # try except를 사용해 i가 int로 형변환이 가능하면 answer에 추가

# 주사위의 개수
# 머쓱이는 직육면체 모양의 상자를 하나 가지고 있는데 이 상자에
# 정육면체 모양의 주사위를 최대한 많이 채우고 싶습니다.
# 상자의 가로, 세로, 높이가 저장되어있는 배열 box와 주사위 모서리의 길이 정수 n이 매개변수로 주어졌을 때,
# 상자에 들어갈 수 있는 주사위의 최대 개수를 return 하도록 solution 함수를 완성해주세요.
def solution(box, n):
    answer = 1
    for i in range(len(box)):
        answer = answer*(box[i]//n)
    return answer
# 다른사람 풀이
def solution(box, n):
    x, y, z = box
    return (x // n) * (y // n) * (z // n )

# 가장 큰 수 찾기
# 정수 배열 array가 매개변수로 주어질 때,
# 가장 큰 수와 그 수의 인덱스를 담은 배열을 return 하도록 solution 함수를 완성해보세요.
def solution(array):
    answer = []
    a = 0
    b = 0
    for i in range(len(array)):
        if array[i]>a:
            a = array[i]
            b = i
    return answer.extend((a,b))
# 다른사람 풀이
def solution(array):
    return [max(array), array.index(max(array))]

# 인덱스 바꾸기
# 문자열 my_string과 정수 num1, num2가 매개변수로 주어질 때,
# my_string에서 인덱스 num1과 인덱스 num2에 해당하는 문자를 바꾼 문자열을 return 하도록 solution 함수를 완성해보세요.
def solution(my_string, num1, num2):
    answer = ''
    my_string = list(my_string)
    my_string[num1],my_string[num2]=my_string[num2],my_string[num1]
    return answer.join(my_string)

# 배열 회전시키기
# 정수가 담긴 배열 numbers와 문자열 direction가 매개변수로 주어집니다.
# 배열 numbers의 원소를 direction방향으로 한 칸씩 회전시킨 배열을 return하도록 solution 함수를 완성해주세요.
def solution(numbers, direction):
    answer = []
    for i in range(len(numbers)):
        if direction == "right":
            answer = [numbers[-1]] + numbers[:len(numbers)-1]
        else:
            answer = numbers[1:] + [numbers[0]]
    return answer
# 다른사람 풀이
def solution(numbers, direction):
    return [numbers[-1]] + numbers[:-1] if direction == 'right' else numbers[1:] + [numbers[0]]

# 외계행성의 나이
#  a는 0, b는 1, c는 2, ..., j는 9입니다.
# 예를 들어 23살은 cd, 51살은 fb로 표현합니다.
# 나이 age가 매개변수로 주어질 때 PROGRAMMER-962식 나이를 return하도록 solution 함수를 완성해주세요.
def solution(age):
    answer = ''
    a = ["a","b","c","d","e","f","g","h","i","j"]
    for i in str(age):
        answer += a[int(i)]
    return answer

# 약수 구하기
# 정수 n이 매개변수로 주어질 때, n의 약수를 오름차순으로 담은 배열을 return하도록 solution 함수를 완성해주세요.
def solution(n):
    answer = []
    for i in range(1, n+1):
        if n%i == 0:
            answer.append(i)
    return answer

# 숫자 찾기
# 정수 num과 k가 매개변수로 주어질 때,
# num을 이루는 숫자 중에 k가 있으면 num의 그 숫자가 있는 자리 수를 return하고 없으면 -1을 return 하도록 solution 함수를 완성해보세요.
def solution(num, k):
    a = str(num).find(str(k))
    return (a if a == -1 else a+1)
# 다른사람 풀이
def solution(num, k):
    return -1 if str(k) not in str(num) else str(num).find(str(k)) + 1

# 중복된 문자 제거
# 문자열 my_string이 매개변수로 주어집니다.
# my_string에서 중복된 문자를 제거하고 하나의 문자만 남긴 문자열을 return하도록 solution 함수를 완성해주세요.
def solution(my_string):
    return ''.join(dict.fromkeys(my_string))

