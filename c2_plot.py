
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 讀取CSV檔案
# 假設 'handwriting_comparison_results.csv' 是已經生成好並包含 'Student', 'LPIPS', 'SSIM' 列的CSV檔案
df = pd.read_csv('D:/the/place/you/save/excel')

# 篩選出 SSIM 值在 0.3 到 0.7 之間的數據
df_filtered = df[(df['SSIM'] >= 0.56) & (df['SSIM'] <= 0.7) & 
                 (df['LPIPS'] >= 0.3) & (df['LPIPS'] <= 0.7)]

# 設定圖片大小和標題
plt.figure(figsize=(20, 10))
plt.title('scatter - SSIM LPIPS similarity')

# 使用Seaborn繪製散點圖，並將學生ID設為每個點的標籤
scatter = sns.scatterplot(data=df_filtered, x='SSIM', y='LPIPS', hue='Student', legend=False)

# 在每個點旁邊添加學生ID標籤
for index, row in df_filtered.iterrows():
     scatter.text(row['SSIM'] + 0.001, row['LPIPS'], 
                  row['Student'], horizontalalignment='left', 
                  size='small', color='black', weight='semibold')

# 設定x軸和y軸的刻度間距為0.02，範圍為0.3到0.7
plt.xticks(ticks=np.arange(0.56, 0.72, 0.02)) ##SSIM
plt.yticks(ticks=np.arange(0.38, 0.6, 0.02))  ##LPIPS


# 設定x軸和y軸標籤
plt.xlabel('SSIM')
plt.ylabel('LPIPS')

# 儲存圖片
plt.savefig('hsieh.png', bbox_inches='tight')

# 顯示圖片
plt.show()