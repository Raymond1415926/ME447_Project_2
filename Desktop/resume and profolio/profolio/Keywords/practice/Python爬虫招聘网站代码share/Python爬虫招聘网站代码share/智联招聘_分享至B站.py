
import requests
import json
import re
import pprint
import csv
from time import sleep
from random import randint

keyword = input('请输入你想要搜索的岗位： ')
pages = input('请输入你想要爬取的页数： ')

with open(f'{keyword}zhilian.csv','w',encoding='utf-8',newline='') as filename:
    dictwriter = csv.DictWriter(filename,fieldnames=[
        '岗位名称',
        '公司名称',
        '学历',
        '公司规模',
        '地区',
        '福利待遇',
        '行业',
        '薪资',
        '经验',
        '发布时间',
        '详情页',
    ])
    dictwriter.writeheader()
    for page in range(1,int(pages)):
        sleep(randint(3, 8))
        print(f'========================================正在采集第{page}页的数据内容============================================')
        url = f'https://sou.zhaopin.com/?jl=765&kw={keyword}&p={pages}'
        headers = {
        #输入网址上的headers即可
        }
        response = requests.get(url=url,headers=headers)

        html_data = re.findall('"locationInfo":{},"selectCity":"","positionList":(.*?),"isSupportBatchDelivery":true,',response.text)[0]

        json_data = json.loads(html_data)

        for index in json_data:
            dit = {
            '岗位名称': index['name'],
            '公司名称':index['companyName'],
            '学历':index['education'],
            '公司规模':index['companySize'],
            '地区':index['cityDistrict'],
            '福利待遇':index['positionHighlight'],
            '行业':index['industryName'],
            '薪资':index['salary60'],
            '经验':index['workingExp'],
            '发布时间':index['publishTime'],
            '详情页':index['positionURL'],

            }
            dictwriter.writerow(dit)
            print(dit)