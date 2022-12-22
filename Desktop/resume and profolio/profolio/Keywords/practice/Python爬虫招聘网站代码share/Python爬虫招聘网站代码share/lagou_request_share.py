
"""以下代码根据需求可以爬取拉勾网不同职位、城市等招聘信息，只需要修改url和header即可
如果遇到反爬，重新打开网站再更新header里面信息，或者修改sleep的时间，或者使用代理
"""


import csv
import requests
import re
import json
import pprint
from time import sleep
from random import randint
import datetime

key_word = input('请输入你想要搜索的岗位名字：')
pages = input('请输入你想要爬取的页数：')

with open(f'{key_word}_lagou.csv_{datetime.date.today()}','w',encoding='utf-8',newline='') as filename:
    csv_dictwriter = csv.DictWriter(filename,fieldnames=[
    '岗位名称',
    '公司名称',
    '薪资',
    '工作地点',
    '学历',
    '经验',
    '融资情况',
    '公司规模',
    '福利待遇',
    '发布日期',
    '技能要求',
    '详情页',
    ])
    csv_dictwriter.writeheader()
    #1、发送请求

    for page in range(1,int(pages)):
        sleep(randint(4,10))

        url = f'https://www.lagou.com/wn/jobs?fromSearch=true&kd={key_word}&city=%E6%B7%B1%E5%9C%B3&pn={page}'
        print(f'======================================正在爬取第{page}页============================================')

        headers = {
           #自行更新
        }
        response = requests.get(url=url,headers=headers)
        # print(response)  #查看连接状态，200为连接成功

        #2、获取数据
        # print(response.text)  #查看服务器返回response响应的文本内容是否正确
        html_data = re.findall('</div></div></div></div></div><script id="__NEXT_DATA__" type="application/json">(.*?)</script>',response.text)[0]
        # print(html_data) #输出是一个字符串
        json_data = json.loads(html_data)  #将json格式的字符串html_data转为Python字典
        # pprint.pprint(json_data)

        wanted_data = json_data['props']['pageProps']['initData']['content']['positionResult']['result']
        #3、解析数据
        try:
            for index in wanted_data:
                job_inf = index['positionDetail'].replace('<br />','').replace('<br>','').replace('<p>','')
                herf = f'https://www.lagou.com/wn/jobs/{index["positionId"]}.html'
                skill = index['positionLables']
                skill_tag = ','.join(skill)

                dict={
                    '岗位名称':index[ 'positionName'],
                    '公司名称':index['companyFullName'],
                    '薪资':index['salary'],
                    '工作地点':index[ 'district'],
                    '学历':index['education'],
                    '经验':index['workYear'],
                    '融资情况':index[ 'financeStage'],
                    '公司规模':index[ 'companySize'],
                    '福利待遇':index['positionAdvantage'],
                    '发布日期':index['createTime'],
                    '技能要求': skill_tag,
                    '详情页':herf
                }
                csv_dictwriter.writerow(dict)

                print(dict)
        except TypeError:
            pass