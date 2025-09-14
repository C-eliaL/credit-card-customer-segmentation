import sys
import os

# 將專案的根目錄加入到 Python 的模組搜尋路徑中
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.customer_segmentation import perform_segmentation

def main():
    """專案主流程控制器"""
    print("================================")
    print("  信用卡客戶分群專案啟動！")
    print("================================")

    # 執行客戶分群
    perform_segmentation()

    print("================================")
    print("  所有分析流程執行完畢！")
    print("================================")

if __name__ == '__main__':
    main()