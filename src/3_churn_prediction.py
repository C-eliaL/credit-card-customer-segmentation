import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_churn_model():
    """訓練流失預測模型並儲存"""
    print("--- 開始訓練流失預測模型 ---")

    # 1. 載入已分群的數據
    df = pd.read_csv('../data/processed/churn_segmented.csv')

    # 2. 特徵工程
    # 將目標變數轉換為 0/1
    df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    # 將類別變數進行 One-Hot Encoding
    df_model = pd.get_dummies(df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1), drop_first=True)

    # 3. 定義特徵 (X) 與目標 (y)
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']

    # 4. 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 5. 訓練邏輯斯迴歸模型
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    print("模型訓練完成。")

    # 6. 評估模型成效
    predictions = log_model.predict(X_test)
    print("\n模型評估報告 (Classification Report):")
    print(classification_report(y_test, predictions))
    
    # 7. 儲存訓練好的模型
    # 建立 models 資料夾 (如果不存在)
    import os
    os.makedirs('../models', exist_ok=True)
    joblib.dump(log_model, '../models/churn_model.pkl')
    print("訓練好的模型已儲存至 models/churn_model.pkl")
    print("--- 流失預測模型訓練完畢 ---\n")

if __name__ == '__main__':
    train_churn_model()