import json
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import until
import pandas as pd
import pymysql
import joblib
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from functools import wraps

app = Flask(__name__)
app.secret_key = '123456'


def query(sql, args=None, limit=None):
    con = pymysql.connect(host='localhost', port=3306, user='root', password='123456789', db='house', charset='utf8')
    cur = con.cursor()
    try:
        if args is None:
            cur.execute(sql)
        else:
            if limit is not None:
                args = list(args) + [limit]
                sql += " LIMIT %s"
            cur.execute(sql, args)
        res = cur.fetchall()
        return res
    finally:
        cur.close()
        con.close()


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        data = request.form.get('data')
        data = json.loads(data)
        username = data['username']
        password = data['password']
        sql = 'select * from `tb_user` where user_name = "{0}" and password = "{1}" limit 0,1'.format(username,
                                                                                                      password)
        res = until.query(sql)
        if res == ():
            data = {
                'info': '账号或密码错误'
            }
            return json.dumps(data)
        else:
            session['username'] = username
            session['user_id'] = res[0][0]
            data = {
                'info': "登录成功"
            }
            return json.dumps(data)


@app.route('/register', methods=['GET', 'POST'])
def res():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        data = request.form.get('data')
        data = json.loads(data)
        username = data['username']
        password = data['password']
        sql = 'select * from tb_user where user_name = "{0}" and password = "{1}" limit 0,1'.format(username, password)
        res = until.query(sql)
        if res != ():
            data = {
                'info': '该用户已经注册'
            }
            return json.dumps(data)
        else:
            sql = 'insert into tb_user(`user_name`,`password`) values ("%s","%s")' % (username, password)
            until.insert(sql)
            data = {
                'info': "注册成功"
            }
            return json.dumps(data)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/all_house', methods=['GET', 'POST'])
@login_required
def all_house():
    # 获取分页参数和搜索关键词
    page = request.args.get('page', 1, type=int)  # 当前页码
    page_size = 10  # 每页显示的数据量
    offset = (page - 1) * page_size

    # 获取搜索关键词
    region = request.args.get('region', '').strip()  # 区域
    layout = request.args.get('layout', '').strip()  # 户型
    address = request.args.get('address', '').strip()  # 标题

    # 查询
    sql = '''
        SELECT `id`, `区域`, `链接`, `标题`, `户型`, `面积`, `朝向`, `装修`, `楼层`, `楼层类型`, `总价`, `单价`
        FROM `data`
        WHERE `区域` LIKE %s AND `户型` LIKE %s AND `标题` LIKE %s
        LIMIT %s OFFSET %s
    '''
    # 查询数据
    res = query(sql, (f'%{region}%', f'%{layout}%', f'%{address}%', page_size, offset))
    data = [list(i) for i in res]

    # 查询总数据量
    count_sql = '''
        SELECT COUNT(*)
        FROM `data`
        WHERE `区域` LIKE %s AND `户型` LIKE %s AND `标题` LIKE %s
    '''
    total = query(count_sql, (f'%{region}%', f'%{layout}%', f'%{address}%'))[0][0]
    total_pages = (total + page_size - 1) // page_size  # 计算总页数
    # 计算分页范围
    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)

    return render_template(
        'all_house.html',
        data=data,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page,
        region=region,
        layout=layout,
        address=address
    )


# 添加收藏
@app.route('/add_favorite', methods=['POST'])
@login_required
def add_favorite():
    if request.method == 'POST':
        data = request.get_json()
        user_id = session.get('user_id')
        data_id = data.get('data_id')

        # 检查是否已收藏
        check_sql = 'SELECT * FROM user_data WHERE user_id = %s AND data_id = %s'
        res = query(check_sql, (user_id, data_id))

        if res:
            return jsonify({'success': False, 'message': '已收藏过该房源'})

        # 添加收藏
        sql = 'insert into user_data(`user_id`,`data_id`) values ("%s","%s")' % (user_id, data_id)
        try:
            until.insert(sql)
            return jsonify({'success': True, 'message': '收藏成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})


# 取消收藏
@app.route('/remove_favorite', methods=['POST'])
@login_required
def remove_favorite():
    if request.method == 'POST':
        data = request.get_json()
        user_id = session.get('user_id')
        data_id = data.get('data_id')

        delete_sql = 'DELETE FROM user_data WHERE user_id ={0} AND data_id = {1}'.format(user_id, data_id)
        try:
            until.insert(delete_sql)
            return jsonify({'success': True, 'message': '取消收藏成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})


# 用户收藏列表
@app.route('/user_house', methods=['GET'])
@login_required
def user_house():
    user_id = session.get('user_id')
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    page_size = 10
    offset = (page - 1) * page_size

    sql = '''
        SELECT d.`id`, d.`区域`, d.`标题`, d.`户型`, d.`面积`, d.`朝向`, d.`装修`, d.`楼层`, d.`楼层类型`, d.`总价`, d.`单价`
        FROM `data` d
        JOIN `user_data` ud ON d.id = ud.data_id
        WHERE ud.user_id = %s
        LIMIT %s OFFSET %s
    '''
    res = query(sql, (user_id, page_size, offset))
    data = [list(i) for i in res]

    # 查询总收藏数
    count_sql = 'SELECT COUNT(*) FROM user_data WHERE user_id = %s'
    total = query(count_sql, (user_id,))[0][0]
    total_pages = (total + page_size - 1) // page_size

    # 计算分页范围
    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)

    return render_template(
        'user_house.html',
        data=data,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page
    )


def content_based_recommendation(user_id, exclude_ids=[], limit=10):
    """
    基于内容的推荐算法
    :param user_id: 用户ID
    :param exclude_ids: 需要排除的房源ID列表
    :param limit: 返回推荐数量
    :return: 推荐房源列表
    """
    # 获取用户收藏的房源特征
    favorites_sql = """
        SELECT d.`id`, d.`区域`, d.`户型`, d.`总价`, d.`面积`, d.`楼层`, d.`朝向`
        FROM data d
        JOIN user_data ud ON d.id = ud.data_id
        WHERE ud.user_id = %s
    """
    favorites = query(favorites_sql, (user_id,))

    # 如果没有收藏，返回空列表
    if not favorites:
        return []

    # 计算用户偏好特征（收藏房源的平均特征）
    avg_features = {
        '区域': {},
        '户型': {},
        '总价': 0,
        '面积': 0,
        '楼层': {},
        '朝向': {}
    }

    # 计算各特征的频率或平均值
    for fav in favorites:
        avg_features['总价'] += fav[3]
        avg_features['面积'] += fav[4]

        for feature in ['区域', '户型', '楼层', '朝向']:
            idx = ['id', '区域', '户型', '总价', '面积', '楼层', '朝向'].index(feature)
            val = fav[idx]
            avg_features[feature][val] = avg_features[feature].get(val, 0) + 1

    # 计算平均值
    avg_features['总价'] /= len(favorites)
    avg_features['面积'] /= len(favorites)

    # 找出最常出现的区域、户型等
    preferred_region = max(avg_features['区域'].items(), key=lambda x: x[1])[0]
    preferred_layout = max(avg_features['户型'].items(), key=lambda x: x[1])[0]
    preferred_floor = max(avg_features['楼层'].items(), key=lambda x: x[1])[0]
    preferred_direction = max(avg_features['朝向'].items(), key=lambda x: x[1])[0]

    # 获取所有候选房源
    exclude_ids = exclude_ids + [fav[0] for fav in favorites]
    exclude_condition = "AND d.id NOT IN (%s)" % ','.join(['%s'] * len(exclude_ids)) if exclude_ids else ""

    candidates_sql = f"""
        SELECT d.`id`, d.`标题`, d.`区域`, d.`总价`, d.`单价`, d.`户型`, d.`面积`, d.`楼层`, d.`朝向`
        FROM data d
        WHERE 1=1 {exclude_condition}
    """
    candidates = query(candidates_sql, exclude_ids) if exclude_ids else query(candidates_sql)

    # 计算每个候选房源与用户偏好的相似度
    def calculate_similarity(house):
        """
        计算单个房源与用户偏好的相似度
        评分规则：
        - 区域匹配: +3分
        - 户型匹配: +2分
        - 朝向匹配: +1分
        - 楼层匹配: +1分
        - 价格相似度: (1 - 价格差异百分比) * 2
        - 面积相似度: (1 - 面积差异百分比) * 2
        """
        score = 0

        # 区域匹配
        if house[2] == preferred_region:
            score += 3

        # 户型匹配
        if house[5] == preferred_layout:
            score += 2

        # 朝向匹配
        if house[8] == preferred_direction:
            score += 1

        # 楼层匹配
        if house[7] == preferred_floor:
            score += 1

        # 价格相似度
        price_diff = abs(house[3] - avg_features['总价']) / avg_features['总价']
        score += (1 - min(price_diff, 1)) * 2  # 最大2分

        # 面积相似度
        area_diff = abs(house[6] - avg_features['面积']) / avg_features['面积']
        score += (1 - min(area_diff, 1)) * 2  # 最大2分

        return score

    scored_houses = []
    for house in candidates:
        similarity = calculate_similarity(house)
        scored_houses.append((house, similarity))
    # 为所有候选房源计算相似度
    scored_houses.sort(key=lambda x: x[1], reverse=True)
    recommended = []
    for house, score in scored_houses[:limit]:
        # 生成推荐理由
        reasons = []
        if house[2] == preferred_region:
            reasons.append("同区域")
        if house[5] == preferred_layout:
            reasons.append("同户型")
        if house[8] == preferred_direction:
            reasons.append("同朝向")
        if house[7] == preferred_floor:
            reasons.append("同楼层")

        price_diff = abs(house[3] - avg_features['总价']) / avg_features['总价']
        if price_diff < 0.1:
            reasons.append("价格相近")

        area_diff = abs(house[6] - avg_features['面积']) / avg_features['面积']
        if area_diff < 0.1:
            reasons.append("面积相近")

        # 将推荐理由添加到房源信息中
        house_list = list(house)
        house_list.append(", ".join(reasons) if reasons else "优质推荐")
        recommended.append(house_list)

    return recommended


@app.route('/tj_house', methods=['GET'])
@login_required
def tj_house():
    user_id = session.get('user_id')

    # 基于内容的推荐
    recommended_houses = content_based_recommendation(user_id)

    # 如果推荐不足10条，随机补全
    if len(recommended_houses) < 10:
        remaining = 10 - len(recommended_houses)
        exclude_ids = [house[0] for house in recommended_houses]

        # 获取用户已收藏的ID
        favorites_sql = "SELECT data_id FROM user_data WHERE user_id = %s"
        favorites = query(favorites_sql, (user_id,))
        favorite_ids = [fav[0] for fav in favorites] if favorites else []
        exclude_ids.extend(favorite_ids)



        # 构建SQL查询
        if exclude_ids:
            # 有排除ID的情况
            placeholders = ','.join(['%s'] * len(exclude_ids))
            random_sql = f"""
                    SELECT * FROM data 
                    WHERE id NOT IN ({placeholders})
                    ORDER BY RAND()
                    LIMIT %s
                """
            random_houses = query(random_sql, exclude_ids + [remaining])
        else:
            # 没有排除ID的情况
            random_sql = """
                    SELECT * FROM data 
                    ORDER BY RAND()
                    LIMIT %s
                """
            random_houses = query(random_sql, (remaining,))

        # 添加推荐理由
        for house in random_houses:
            house_list = list(house)
            house_list.append("为您精心挑选的优质房源")
            recommended_houses.append(house_list)

    # 冷启动情况：用户没有任何收藏，随机推荐10条
    if not recommended_houses:
        random_sql = "SELECT * FROM data ORDER BY RAND() LIMIT 10"
        random_houses = query(random_sql)
        recommended_houses = [list(house) + ["新用户专属推荐"] for house in random_houses]

    return render_template('tj_house.html', data=recommended_houses)


@app.route('/logout')
def logout():
    session.clear()  # 清空所有session
    return redirect(url_for('login'))


@app.route('/lc', methods=['GET', 'POST'])
def lc():
    sql = 'select 单价,楼层 from `data`'
    res = until.query(sql)
    columns = ['单价', '楼层']
    df = pd.DataFrame(res, columns=columns)
    df['楼层'] = df['楼层'].str.split('（').str[0]
    df['单价'] = df['单价'].astype(int)
    result = df.groupby(['楼层']).agg({'单价': 'mean'}).reset_index()
    data = [result['楼层'].tolist(), result['单价'].tolist()]
    return render_template('lc.html', data=data)


@app.route('/qy', methods=['GET', 'POST'])
def qy():
    sql = 'select 单价,区域 from `data`'
    res = until.query(sql)
    # 将查询结果转换为DataFrame
    columns = ['单价', '区域']
    df = pd.DataFrame(res, columns=columns)
    df['单价'] = df['单价'].astype(int)
    # 计算每个行政区的平均价格
    average_prices = df.groupby('区域')['单价'].mean().astype(int)
    # 将数据转换为饼图格式的字典列表
    pie_data = [{'name': district, 'value': int(price)} for district, price in
                zip(average_prices.index, average_prices.values)]
    return render_template('qy.html', data=pie_data)


@app.route('/hx', methods=['GET', 'POST'])
def hx():
    sql = 'select `总价`,`户型`,`区域` from `data`'
    res = until.query(sql)
    columns = ['总价', '户型', '区域']
    df = pd.DataFrame(res, columns=columns)
    df['总价'] = df['总价'].astype(float)

    local = ['红谷滩区', '新建区', '青山湖区', '青云谱区', '西湖区', '东湖区']
    hx_order = {
        '1室0厅': 0,
        '1室1厅': 1,
        '2室1厅': 3,
        '2室2厅': 4,
        '3室1厅': 5,
        '3室2厅': 6,
        '4室1厅': 7,
        '4室2厅': 8,
        '5室2厅': 9
    }
    existing_hx = sorted(
        df['户型'].unique().tolist(),
        key=lambda x: hx_order.get(x, 10)
    )

    heatmap_data = []
    for area in local:
        for hx in existing_hx:
            avg_price = df[(df['区域'] == area) & (df['户型'] == hx)]['总价'].mean()
            heatmap_data.append([area, hx, int(avg_price) if not pd.isna(avg_price) else 0])
    max_value = max([d[2] for d in heatmap_data]) if heatmap_data else 272

    heatmap = (
        HeatMap(init_opts=opts.InitOpts(width="100%", height="700px"))
            .add_xaxis(local)
            .add_yaxis(
            "总价",
            existing_hx,  # 使用排序后的户型列表
            heatmap_data,
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="不同区域不同户型的总价热力图"),
            xaxis_opts=opts.AxisOpts(name="区域"),
            yaxis_opts=opts.AxisOpts(
                name="户型",
                # 保证Y轴按定义的顺序显示
                type_="category",
                axislabel_opts=opts.LabelOpts(interval=0)),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=max_value,
                is_calculable=True,
                orient="horizontal",
                pos_left="center"),
        )
    )

    return render_template('hx.html', heatmap_render=heatmap.render_embed())


@app.route('/tz', methods=['GET', 'POST'])
def tz():
    return render_template('tz.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')


@app.route('/cx', methods=['GET', 'POST'])
def cx():
    sql = 'select `单价`,`朝向` from `data` where 朝向 != "未知"'
    res = until.query(sql)
    columns = ['单价', '朝向']
    df = pd.DataFrame(res, columns=columns)
    df['单价'] = df['单价'].astype(int)
    result = df.groupby(['朝向']).agg({'单价': 'mean'}).reset_index()
    return render_template('cx.html', data=[result['朝向'].tolist(), result['单价'].tolist()])


@app.route('/yc', methods=['GET', 'POST'])
def yc():
    if request.method == 'GET':
        return render_template('yc.html')

    # 1. 获取并验证表单数据
    form_data = {
        '区域': request.form.get('qy', '').strip(),
        '户型': request.form.get('huxing', '').strip(),
        '面积': float(request.form.get('mianji', 0)),
        '朝向': request.form.get('chaoxiang', '').strip(),
        '装修': request.form.get('zx', '').strip(),
        '楼层': request.form.get('louceng', '').strip(),
        '楼层类型': request.form.get('lclx', '').strip()
    }

    dt_model, label_encoders, scaler = load_artifacts()

    new_data = pd.DataFrame([form_data])

    dt_pred = predict_with_dt(new_data, dt_model, label_encoders, scaler)
    return jsonify({
        'dt_pred': float(dt_pred[0]),
        'status': 'success'
    })


def load_artifacts():
    """加载所有需要的模型和工具"""
    return (
        joblib.load('./static/decision_tree_model.pkl'),
        joblib.load('./static/label_encoders.pkl'),
        joblib.load('./static/scaler.pkl'),
    )


def predict_with_dt(data, model, label_encoders, scaler):
    """决策树预测流程"""
    data_encoded = encode_categorical(data, label_encoders)

    data_scaled = scaler.transform(data_encoded)

    return model.predict(data_scaled)


def encode_categorical(data, label_encoders):
    data_encoded = data.copy()
    for col, le in label_encoders.items():
        if col in data_encoded.columns:
            data_encoded[col] = data_encoded[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            data_encoded[col] = le.transform(data_encoded[col])
    return data_encoded


if __name__ == '__main__':
    app.run(debug=True)
