from flask import Flask, render_template
import pandas as pd
from fbprophet import Prophet
import openai
import requests
import os

app = Flask(__name__)

# Cấu hình API Key (OpenRouter / OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Hàm lấy dữ liệu giá nhôm
def get_aluminum_prices():
    # Dữ liệu mẫu vì không có API key Metals
    dates = pd.date_range(start="2024-01-01", periods=100)
    prices = 2200 + (pd.Series(range(100)).apply(lambda x: x * 0.5)).values
    df = pd.DataFrame({'ds': dates, 'y': prices})
    return df

# Hàm dự đoán giá
def forecast_prices(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)
    return forecast

# Hàm gọi OpenAI viết phân tích
def generate_report(latest_price):
    prompt = f"""
    Hôm nay giá nhôm LME đóng cửa ở mức {latest_price:.2f} USD/tấn.
    Viết phân tích ngắn gọn 150 từ về xu hướng giá và tác động địa chính trị.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia phân tích thị trường nhôm."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

@app.route('/')
def index():
    df = get_aluminum_prices()
    forecast = forecast_prices(df)
    latest_price = df['y'].iloc[-1]
    report = generate_report(latest_price)

    return render_template('index.html', forecast=forecast[['ds', 'yhat']], report=report)

if __name__ == '__main__':
    app.run(debug=True)
