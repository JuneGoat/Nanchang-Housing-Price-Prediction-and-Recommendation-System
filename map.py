import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map
import pymysql

# 连接数据库
con = pymysql.connect(host='localhost', port=3306, user='root', password='root', db='house', charset='utf8')
cur = con.cursor()
sql = 'select `单价`,`区域` from `data`'
cur.execute(sql)
res = cur.fetchall()

df = pd.DataFrame(res, columns=['单价', '区域'])
df['单价'] = df['单价'].astype(float)

# 计算不同区域的平均单价，保留两位小数
avg_price = df.groupby('区域')['单价'].mean().round(2).reset_index()

# 数据转换
region_price = avg_price[['区域', '单价']].values.tolist()

# 绘制地图
c = (
    Map()
    .add(
        series_name="平均单价",
        data_pair=region_price,
        maptype="南昌",
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="南昌各区平均单价"),
        visualmap_opts=opts.VisualMapOpts(
            min_=avg_price['单价'].min(),
            max_=avg_price['单价'].max(),
            is_piecewise=True,  # 是否分段显示
        ),
    )
)

# 渲染地图
c.render("map.html")
# 打印平均单价
print(avg_price)
# 关闭数据库连接
cur.close()
con.close()
