import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('清洗.csv')
print(df['总价(万)'])
# 选择特征和目标列
features = ['区域','户型', '面积', '朝向', '装修', '楼层', '楼层类型']
target = '总价(万)'

# 将文本类型的特征替换为数值
label_encoders = {}
for col in features:
    print(f"\n处理列: {col} | 类型: {df[col].dtype}")

    if df[col].dtype == 'object':
        # 处理空值（可选：用特定值填充NaN）
        col_data = df[col].fillna('未知').astype(str)

        # 初始化编码器
        le = LabelEncoder()

        # 转换并存储
        encoded_values = le.fit_transform(col_data)
        df[col] = encoded_values
        label_encoders[col] = le

        # 打印调试信息
        print(f"类别映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        print(f"编码结果: {encoded_values}")
    else:
        print("跳过数值列")
print("\n编码后的DataFrame:")
print(df)
print("\nlabel_encoders 内容:")
print({k: v.classes_ for k, v in label_encoders.items()})
# 划分训练集和测试集
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型优化
dt_param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
dt_model = DecisionTreeRegressor(random_state=42)
dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid,
                            cv=3, scoring='r2', n_jobs=-1)
dt_grid_search.fit(X_train_scaled, y_train)
best_dt_model = dt_grid_search.best_estimator_
y_pred_dt = best_dt_model.predict(X_test_scaled)

# 评估模型
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} 评估结果：")
    print(f"R²分数: {r2:.2f}")
    return r2

# 评估模型
r2_dt = evaluate_model(y_test, y_pred_dt, "决策树")
print(f"最佳参数: {dt_grid_search.best_params_}")

# 绘制预测结果对比图
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_dt, color='green', alpha=0.5, label='决策树预测')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='真实数据')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.legend()
plt.savefig('实际值vs预测值.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存模型和编码器
joblib.dump(best_dt_model, 'decision_tree_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')


print("\n模型训练和评估完成！")