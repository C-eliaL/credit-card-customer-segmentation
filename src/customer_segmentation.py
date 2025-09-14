import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np  

def perform_segmentation():
    """執行客戶分群並儲存結果"""
    print("--- 開始執行信用卡客戶分群 ---")

    
    # 1. 取得當前腳本檔案 (customer_segmentation.py) 所在的資料夾路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 建立目標檔案的完整路徑
    
    file_path = os.path.join(script_dir, '..', 'data', 'processed', 'creditcard_cleaned.csv')
    
    print(f"正在從以下絕對路徑讀取檔案：{file_path}") 
    
    # 3. 讀取 CSV
    df = pd.read_csv(file_path)
    # --- END: 修正檔案路徑 ---

    # 2. 數據標準化 
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    print("數據標準化完成。")

    # 3. 使用 PCA 進行降維
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
    print("PCA 降維完成，已將數據降至 2 維。")

    # 4. 執行 K-Means 分群
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(df_pca)
    print("K-Means 分群完成。")

    # 結合 PCA 結果以利視覺化
    df_plot = pd.concat([df, df_pca], axis=1)

    # 5. 計算分群品質指標
    silhouette_avg = silhouette_score(df_pca, df['Cluster'])
    print(f"分群品質指標 (Silhouette Score): {silhouette_avg:.3f}")
    
    # 6. 多圖表展示
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 建立 2x2 子圖布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('信用卡客戶分群完整分析報告', fontsize=20, fontweight='bold', y=0.98)
    
    # 子圖 1: PCA 分群結果
    sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='Cluster', 
                   palette='viridis', s=50, alpha=0.7, ax=axes[0,0])
    axes[0,0].set_title('客戶分群結果 (PCA 視覺化)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('主成分 1')
    axes[0,0].set_ylabel('主成分 2')
    axes[0,0].legend(title='客戶群體', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子圖 2: 各群體特徵比較 (重要欄位)
    important_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    cluster_profile = df.groupby('Cluster')[important_features].mean()
    
    # 標準化數據以便比較
    cluster_profile_scaled = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
    cluster_profile_scaled.plot(kind='bar', ax=axes[0,1], width=0.8)
    axes[0,1].set_title('各群體特徵比較 (標準化)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('客戶群體')
    axes[0,1].set_ylabel('標準化數值')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # 子圖 3: 群體大小分佈
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    wedges, texts, autotexts = axes[1,0].pie(cluster_counts.values, 
                                            labels=[f'群體 {i}' for i in cluster_counts.index],
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
    axes[1,0].set_title('各群體客戶數量分佈', fontsize=14, fontweight='bold')
    
    # 子圖 4: 特徵相關性熱圖
    correlation_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
    correlation_matrix = df[correlation_features].corr()
    
    im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(correlation_features)))
    axes[1,1].set_yticks(range(len(correlation_features)))
    axes[1,1].set_xticklabels(correlation_features, rotation=45, ha='right')
    axes[1,1].set_yticklabels(correlation_features)
    axes[1,1].set_title('特徵相關性熱圖', fontsize=14, fontweight='bold')
    
    # 在熱圖上添加數值
    for i in range(len(correlation_features)):
        for j in range(len(correlation_features)):
            text = axes[1,1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=8)
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=axes[1,1], shrink=0.8)
    cbar.set_label('相關係數', rotation=270, labelpad=15)
    
    # 調整子圖間距
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # 儲存多圖表
    multi_chart_path = os.path.join(script_dir, '..', 'reports', 'figures', 'customer_segmentation_comprehensive.png')
    plt.savefig(multi_chart_path, dpi=300, bbox_inches='tight')
    print(f"\n完整分析報告已儲存至: {multi_chart_path}")
    
    # 儲存原始 PCA 分群圖 (保持原有功能)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=50, alpha=0.7)
    plt.title('信用卡客戶分群結果 (PCA 視覺化)', fontsize=16)
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.savefig(os.path.join(script_dir, '..', 'reports', 'figures', 'customer_segmentation_pca.png'), dpi=300, bbox_inches='tight')
    print("原始 PCA 分群圖已儲存至 reports/figures/")
    
    # 7. 分析各群體輪廓
    cluster_profile = df.groupby('Cluster').mean().round(2)
    print("\n各客群輪廓分析 (部分重要欄位):")
    print(cluster_profile[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']])
    
    # 8. 生成額外的詳細分析圖表
    # 8.1 各群體餘額分佈箱線圖
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Cluster', y='BALANCE', hue='Cluster', palette='viridis', legend=False)
    plt.title('各群體帳戶餘額分佈比較', fontsize=16, fontweight='bold')
    plt.xlabel('客戶群體')
    plt.ylabel('帳戶餘額')
    plt.savefig(os.path.join(script_dir, '..', 'reports', 'figures', 'cluster_balance_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print("群體餘額分佈圖已儲存至 reports/figures/")
    
    # 8.2 各群體消費行為散點圖
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='PURCHASES', y='CASH_ADVANCE', hue='Cluster', 
                   palette='viridis', s=60, alpha=0.7)
    plt.title('各群體消費行為分析 (消費 vs 現金預借)', fontsize=16, fontweight='bold')
    plt.xlabel('消費總額')
    plt.ylabel('現金預借')
    plt.legend(title='客戶群體', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(script_dir, '..', 'reports', 'figures', 'cluster_spending_behavior.png'), 
                dpi=300, bbox_inches='tight')
    print("群體消費行為圖已儲存至 reports/figures/")
    
    # 8.3 各群體特徵雷達圖
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 選擇重要特徵進行雷達圖展示
    radar_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # 完成圓圈
    
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    
    for cluster_id in range(4):
        values = cluster_profile.loc[cluster_id, radar_features].values.tolist()
        values += values[:1]  # 完成圓圈
        
        # 標準化數值到 0-1 範圍
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min())
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'群體 {cluster_id}', color=colors[cluster_id])
        ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features)
    ax.set_ylim(0, 1)
    ax.set_title('各群體特徵雷達圖', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.savefig(os.path.join(script_dir, '..', 'reports', 'figures', 'cluster_radar_chart.png'), 
                dpi=300, bbox_inches='tight')
    print("群體特徵雷達圖已儲存至 reports/figures/")

    # 9. 儲存附帶分群標籤的數據
    df.to_csv(os.path.join(script_dir, '..', 'data', 'processed', 'creditcard_segmented.csv'), index=False)
    print("已分群的數據儲存至 data/processed/creditcard_segmented.csv")
    
    # 10. 輸出總結報告
    print("\n" + "="*60)
    print("           客戶分群分析總結報告")
    print("="*60)
    print(f"總客戶數量: {len(df):,}")
    print(f"分群數量: 4 個群體")
    print(f"分群品質指標 (Silhouette Score): {silhouette_avg:.3f}")
    print(f"生成的圖表數量: 6 個")
    print("\n生成的圖表檔案:")
    print("1. customer_segmentation_comprehensive.png - 完整分析報告 (4合1)")
    print("2. customer_segmentation_pca.png - PCA 分群視覺化")
    print("3. cluster_balance_distribution.png - 群體餘額分佈")
    print("4. cluster_spending_behavior.png - 群體消費行為")
    print("5. cluster_radar_chart.png - 群體特徵雷達圖")
    print("6. 原有圖表: balance_distribution.png, purchases_distribution.png")
    print("="*60)
    print("--- 客戶分群執行完畢 ---\n")