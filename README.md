# 信用卡客戶分群分析專案

## 📋 專案概述

這是一個完整的信用卡客戶分群分析專案，使用機器學習技術對客戶進行分群，並提供豐富的視覺化分析報告。

## 📊 資料來源

本專案使用的資料集來自 Kaggle：

**信用卡客戶資料集** - [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data)

- **資料集描述**: 包含信用卡客戶的行為和特徵資料
- **資料筆數**: 8,950 筆客戶記錄
- **特徵數量**: 18 個特徵欄位
- **主要特徵**: 帳戶餘額、消費金額、信用額度、付款行為等
- **授權**: 公開資料集，可供學習和研究使用

## 🎯 專案目標

- 分析信用卡客戶行為模式
- 使用 K-Means 演算法進行客戶分群
- 提供多維度的視覺化分析
- 識別不同客戶群體的特徵和行為模式

## 🛠️ 技術棧

- **Python 3.x**
- **pandas** - 數據處理
- **numpy** - 數值計算
- **scikit-learn** - 機器學習
- **matplotlib** - 基礎繪圖
- **seaborn** - 進階視覺化
- **jupyter** - 互動式分析

## 📁 專案結構

```
Credit Card Dataset for Clustering/
├── data/                          # 數據檔案
│   ├── BankChurners.csv          # 原始數據 (來自 Kaggle)
│   └── processed/                # 處理後數據
│       ├── creditcard_cleaned.csv
│       └── creditcard_segmented.csv
├── src/                          # 原始碼
│   └──customer_segmentation.py  # 客戶分群核心模組
├── notebooks/                    # Jupyter 筆記本
│   └── 1_data_exploration.ipynb  # 數據探索
├── reports/                      # 分析報告
│   └── figures/                  # 圖表檔案
│       ├── customer_segmentation_comprehensive.png
│       ├── customer_segmentation_pca.png
│       ├── cluster_balance_distribution.png
│       ├── cluster_spending_behavior.png
│       └── cluster_radar_chart.png
├── main.py                       # 主程式入口
├── requirements.txt              # 依賴套件
└── README.md                     # 專案說明
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 複製專案
git clone https://github.com/C-eliaL/credit-card-customer-segmentation.git
cd Credit-Card-Dataset-for-Clustering

# 下載資料集 (可選，專案已包含資料)
# 從 Kaggle 下載: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data
# 將下載的檔案放置到 data/ 資料夾中

# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (Linux/Mac)
source venv/bin/activate

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 執行分析

```bash
# 執行完整的客戶分群分析
python main.py
```

### 3. 查看結果

執行完成後，您可以在 `reports/figures/` 資料夾中找到以下圖表：

- **customer_segmentation_comprehensive.png** - 完整分析報告 (4合1)
- **customer_segmentation_pca.png** - PCA 分群視覺化
- **cluster_balance_distribution.png** - 群體餘額分佈
- **cluster_spending_behavior.png** - 群體消費行為
- **cluster_radar_chart.png** - 群體特徵雷達圖

## 📊 分析結果

### 客戶分群結果

專案將客戶分為 4 個群體：

- **群體 0**: 高餘額、低消費的保守型客戶
- **群體 1**: 低餘額、低消費的基礎型客戶
- **群體 2**: 高消費、高信用額度的優質客戶
- **群體 3**: 中等消費、穩定還款的標準客戶

### 分群品質指標

- **Silhouette Score**: 0.407 (良好品質)
- **總客戶數量**: 8,950 人
- **分群數量**: 4 個群體

## 🔧 主要功能

### 1. 數據預處理
- 缺失值處理
- 數據標準化
- 特徵工程

### 2. 機器學習分群
- PCA 降維
- K-Means 分群
- 分群品質評估

### 3. 視覺化分析
- 多維度圖表展示
- 群體特徵比較
- 行為模式分析

## 📈 圖表說明

### 完整分析報告 (4合1)
- **PCA 分群圖**: 展示客戶在降維空間中的分群結果
- **特徵比較圖**: 比較各群體的重要特徵
- **群體分佈圖**: 顯示各群體的客戶數量比例
- **相關性熱圖**: 展示特徵間的相關性

### 詳細分析圖表
- **餘額分佈圖**: 各群體帳戶餘額的分佈情況
- **消費行為圖**: 消費與現金預借的關係分析
- **雷達圖**: 多維度展示各群體特徵輪廓

## 🤝 貢獻指南

1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📝 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- **資料提供者**: 感謝 [Kaggle](https://www.kaggle.com/) 和資料集作者 [arjunbhasin2013](https://www.kaggle.com/arjunbhasin2013) 提供優質的信用卡客戶資料集

**注意**: 本專案僅供學習和研究使用，請勿用於商業用途。
