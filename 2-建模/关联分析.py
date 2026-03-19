import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('清洗.csv')

# 选择需要分析的特征列
features = ['区域', '户型', '面积', '朝向', '装修', '楼层', '楼层类型', '总价(万)', '单价(元/平)']
df = df[features]

# 对分类变量进行标签编码
label_encoders = {}
for column in ['区域', '户型', '朝向', '装修', '楼层', '楼层类型']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 计算相关系数矩阵
corr_matrix = df.corr()

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,
            annot_kws={"size": 10})
plt.title('特征关联度热力图', fontsize=15)
plt.xticks(ha='right')
plt.tight_layout()
plt.savefig('关联热力图.png')