import jwt   # PyJWT
import uuid
import requests
import pandas as pd


# 업비트로 API 요청
payload = {
    'access_key': 'mHQTcgqBxOfCuNHS5QYAyRGg8jBTnERFDxPKUR4u',
    'nonce': str(uuid.uuid4()),
}

jwt_token = jwt.encode(payload, 'cHhK52GlQMvbo1Wr2I7XHL39H0uDrqcyhAMa0u8B',).decode('utf8')
authorization_token = 'Bearer {}'.format(jwt_token)


# 값을 가져올 URL 링크
url = "https://api.upbit.com/v1/candles/days"

num = 4    # 가져올 날짜의 수 (최대 200)

querystring = {"count" : num,  # 지정한 개수의 데이터
               "market" : "KRW-ETH",   # 이더리움 한화 가격
               "to" : "2020-11-13 00:00:00"}    # 언제까지의 데이터를 가져오겠다

response = requests.request("GET", url, params=querystring)


# 데이터를 저장할 CSV 파일
csv_file = open("chart_data.csv", "a")

df = pd.read_json(response.text)
df.to_csv(csv_file, sep = ',', na_rep = 'NaN')
