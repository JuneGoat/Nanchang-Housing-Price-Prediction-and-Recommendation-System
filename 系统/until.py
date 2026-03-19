import pymysql


def coon():  # 连接数据库
    con = pymysql.connect(host='localhost', port=3306, user='root', password='123456789', db='house',
                          charset='utf8')  # 连接数据库
    cur = con.cursor()
    return con, cur


def close():
    con, cur = coon()
    cur.close()
    con.close()


def query(sql):
    con, cur = coon()
    cur.execute(sql)
    res = cur.fetchall()
    close()
    return res


def insert(sql):
    con, cur = coon()
    cur.execute(sql)
    con.commit()
    close()