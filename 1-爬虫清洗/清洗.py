import pandas as pd
import pymysql

df = pd.read_csv('数据.csv')

# 删除链接重复的数据
df.drop_duplicates(subset=['链接'], keep='first', inplace=True)

df['面积'] = df['面积'].astype(str).str.replace('平米', '').str.strip()

df['单价(元/平)'] = df['单价(元/平)'].astype(str).str.replace('"', '').str.replace(',', '')

df['总价(万)'] = df['总价(万)'].astype(str).str.replace(',', '')
df['朝向'] = df['朝向'].astype(str).str[0]
# 统一转换为数值类型
numeric_cols = ['面积', '总价(万)', '单价(元/平)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # 转换失败设为NaN
    df = df.dropna(subset=[col])  # 删除转换失败的行
print(df)
# 处理楼层列-保留括号前的数据
df['楼层'] = df['楼层'].str.split('(').str[0]
floor_counts = df['楼层'].value_counts(dropna=False)
print("原始楼层统计:")
print(floor_counts)

# 确定常见楼层
common_floors = floor_counts[floor_counts >= 200].index
df['楼层'] = df['楼层'].apply(lambda x: x if x in common_floors else '中楼层')

# 打印分类变量的类型和计数
categorical_cols = ['朝向', '楼层类型', '装修', '户型', '楼层']

for col in categorical_cols:
    print(f"\n{col}原始类型统计:")
    value_counts = df[col].value_counts(dropna=False)
    print(value_counts)
    valid_categories = value_counts[value_counts >= 100].index
    df[col] = df[col].where(df[col].isin(valid_categories), other='其他')

# 打印处理后的统计
for col in categorical_cols:
    print(f"\n{col}处理后的类型统计:")
    print(df[col].value_counts(dropna=False))

# 打印每列缺失值
print("\n各列缺失值统计:")
print(df.isnull().sum())
'''

朝向类型统计:
朝向
南    11974
东      645
北      201朝向处理后的类型统计:
朝向
南    11974
东      645
北      201
西      150
Name: count, dtype: int64

楼层类型处理后的类型统计:
楼层类型
板楼      7594
暂无数据    2257
板塔结合    1665
塔楼      1099
平房       355
Name: count, dtype: int64

装修处理后的类型统计:
装修
精装    6043
简装    2855
毛坯    2186
其他    1886
Name: count, dtype: int64

户型处理后的类型统计:
户型
3室2厅    5172
2室2厅    2653
2室1厅    1741
4室2厅    1161
1室1厅     921
3室1厅     802
其他       153
5室2厅     148
4室1厅     113
1室0厅     106
Name: count, dtype: int64
'''

try:
    # 创建数据库连接
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='house',
        charset='utf8mb4'
    )

    with conn.cursor() as cursor:
        # 创建表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            区域 VARCHAR(255),
            链接 VARCHAR(512),
            标题 VARCHAR(512),
            户型 VARCHAR(50),
            面积 FLOAT,
            朝向 VARCHAR(50),
            装修 VARCHAR(50),
            楼层 VARCHAR(50),
            楼层类型 VARCHAR(50),
            `总价` FLOAT,
            `单价` FLOAT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        cursor.execute(create_table_sql)

        # 准备插入数据
        insert_sql = """
        INSERT INTO data (
            区域, 链接, 标题,户型, 面积, 朝向, 装修, 楼层, 楼层类型, `总价`, `单价`
        ) VALUES (%s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # 批量插入
        data_to_insert = [
            (
                row['区域'], row['链接'], row['标题'], row['户型'],
                row['面积'], row['朝向'], row['装修'], row['楼层'], row['楼层类型'],
                row['总价(万)'], row['单价(元/平)']
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(insert_sql, data_to_insert)
        conn.commit()
        print(f"\n成功插入 {len(data_to_insert)} 条数据到MySQL数据库!")

except Exception as e:
    print(f"\n数据库操作出错: {e}")
    if 'conn' in locals():
        conn.rollback()
finally:
    if 'conn' in locals():
        conn.close()

# 显示清洗后的数据
print("\n清洗后的数据:")
print(df)
df.to_csv('清洗.csv',index=False)