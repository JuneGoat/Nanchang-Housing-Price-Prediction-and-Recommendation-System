import csv
import time
import requests
from bs4 import BeautifulSoup


cookies = {
    'lianjia_uuid': '5ec6dfb8-6356-4805-a9fb-6d4679a4a2e7',
    'Hm_lvt_46bf127ac9b856df503ec2dbf942b67e': '1743259013',
    'HMACCOUNT': '96F3E50B4751E4E9',
    '_jzqa': '1.1267608448058595800.1743259013.1743259013.1743259013.1',
    '_jzqc': '1',
    '_jzqx': '1.1743259013.1743259013.1.jzqsr=cn%2Ebing%2Ecom|jzqct=/.-',
    '_jzqckmp': '1',
    'sajssdk_2015_cross_new_user': '1',
    'lianjia_ssid': 'aa695ede-8195-4d66-854e-6658d732520f',
    '_ga': 'GA1.2.84091446.1743259111',
    '_gid': 'GA1.2.861979712.1743259111',
    '_ga_TJZVFLS7KV': 'GS1.2.1743259181.1.0.1743259181.0.0.0',
    '_ga_WLZSQZX7DE': 'GS1.2.1743259181.1.0.1743259181.0.0.0',
    'select_city': '360100',
    '_qzjc': '1',
    'login_ucid': '2000000179296601',
    'lianjia_token': '2.0014c31cc973e9a87c056e35f87587f65b',
    'lianjia_token_secure': '2.0014c31cc973e9a87c056e35f87587f65b',
    'security_ticket': 'ahM8WkL1ErWXUBbCrsphoaxNpmiStB/mxWC93s8gfAjxm/Oh6yvPA+IGqpPOKM3Uhy/EeeJIgFiz1hBlrEgfIH1qkYJPCRwm/9ERJZeyxrsd+NibS/07IoG+TBkVVCuYwcvDRPOPklkjlxCoC4T0FfKJ4Pj+Hfm4oCbgwTjXTVg=',
    'ftkrc_': '162563af-cf82-45ec-9697-3a0d93ac478e',
    'lfrc_': '6eae08be-1acf-42c0-a5e4-afe56e09c76a',
    'sensorsdata2015jssdkcross': '%7B%22distinct_id%22%3A%22195e25590a6126-05884a973b220a-4c657b58-3686400-195e25590a71748%22%2C%22%24device_id%22%3A%22195e25590a6126-05884a973b220a-4c657b58-3686400-195e25590a71748%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D',
    'Hm_lpvt_46bf127ac9b856df503ec2dbf942b67e': '1743259799',
    '_jzqb': '1.25.10.1743259013.1',
    '_qzja': '1.1056813035.1743259187104.1743259187104.1743259187104.1743259403399.1743259798590.0.0.0.22.1',
    '_qzjb': '1.1743259187104.22.0.0.0',
    '_qzjto': '22.1.0',
    'srcid': 'eyJ0Ijoie1wiZGF0YVwiOlwiMzlkMmM3MGYyYTY3YzJjYjdiYzNmMzBkMjY4NjdmNGQ2NDY4Mzc0NTM5YTA4ZmNmYjNiZTk5NDdhN2QzMWYwMjkwNDMwMDMyMTljOWUyOTFlMWEyMGY3ZDNhZTNhOThjZmM4ZjM1NzY0ZmEzOTY3MGQxNjMwMzc0ZTM2MmJmNmU5N2FlOWEyNzAyYjQzNDM4NWM3MTg2ZGM5M2JmMDA3YTYzNTY1YmVlMzAwMTRhNDEyZmFjMjQ3NjAzYjViMDU0NzI1ZjdhZTc0ZDg2M2Q3OGMwYTcyZWJhZWMxYTFiNzJhNGViZmE4Njk2ZmIzMjZhMDBlNzJiNTAyZTgzOGY4M1wiLFwia2V5X2lkXCI6XCIxXCIsXCJzaWduXCI6XCJmN2JhZjNiY1wifSIsInIiOiJodHRwczovL25jLmxpYW5qaWEuY29tL2Vyc2hvdWZhbmcvZG9uZ2h1cXUvcGcyLyIsIm9zIjoid2ViIiwidiI6IjAuMSJ9',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Referer': 'https://nc.lianjia.com/ershoufang/donghuqu/',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0',
    'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Microsoft Edge";v="134"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}

codes = ['honggutanqu', 'xinjianqu', 'qingshanhuqu', 'qingyunpuqu', 'xihuqu', 'donghuqu']
names = ['红谷滩区', '新建区', '青山湖区', '青云谱区', '西湖区', '东湖区']


# 创建CSV文件
with open('数据1.csv', 'a', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    # csv_writer.writerow(
        # ['区域', '链接', '标题', '户型', '面积', '朝向', '装修', '楼层', 楼层类型,'总价(万)', '单价(元/平)'])

    failed_urls = []  # 存储失败的URL

    for code, name in zip(codes, names):
        for page in range(1, 101):
            try:

                time.sleep(2)  # 礼貌性暂停
                url = f'https://nc.lianjia.com/ershoufang/{code}/pg{page}/'
                print(f"正在抓取: {url}")

                response = requests.get(url, headers=headers, cookies=cookies)
                soup = BeautifulSoup(response.text, 'lxml')
                house_list = soup.find('ul', class_='sellListContent')

                if not house_list:
                    print(f"第 {page} 页没有找到房源列表")
                    continue

                for house in house_list.find_all('li'):
                    try:
                        # 基础信息
                        title = house.find('div', class_='title').a.text.strip()
                        link = house.find('div', class_='title').a['href']


                        # 价格信息
                        total_price = house.find('div', class_='totalPrice').span.text.strip()
                        unit_price = house.find('div', class_='unitPrice').span.text.strip().replace('单价',
                                                                                                     '').replace(
                            '元/平', '')

                        # 房屋详细信息
                        house_info = house.find('div', class_='houseInfo').text.strip()
                        house_info_parts = [x.strip() for x in house_info.split('|')]

                        # 确保信息完整
                        if len(house_info_parts) >= 6:
                            layout = house_info_parts[1]
                            area = house_info_parts[2]
                            direction = house_info_parts[3]
                            decoration = house_info_parts[4]
                            floor = house_info_parts[5]
                        else:
                            layout = area = direction = decoration = floor = '未知'

                        # 写入CSV
                        csv_writer.writerow([
                            name,  # 区域
                            link,  # 链接
                            title,  # 标题
                            layout,  # 户型
                            area,  # 面积
                            direction,  # 朝向
                            decoration,  # 装修
                            floor,  # 楼层
                            total_price.replace('万', ''),  # 总价(万)
                            unit_price  # 单价(元/平)
                        ])
                        print([
                            name,  # 区域
                            link,  # 链接
                            title,  # 标题
                            layout,  # 户型
                            area,  # 面积
                            direction,  # 朝向
                            decoration,  # 装修
                            floor,  # 楼层
                            total_price.replace('万', ''),  # 总价(万)
                            unit_price  # 单价(元/平)
                        ])
                    except Exception as e:
                        print(f"处理房源时出错: {e}")
                        continue

            except Exception as e:
                print(f"请求页面出错: {url}, 错误: {e}")
                failed_urls.append(url)
                continue

    # 打印所有失败的URL
    print("失败的URL列表:")
    for url in failed_urls:
        print(url)