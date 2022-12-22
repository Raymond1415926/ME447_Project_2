
import requests
import re
import json
import pprint
import csv
from random import randint
from time import sleep

key_word = input('请输入你想要搜索的岗位名字：')
pages = input('请输入你想要爬取的页数：')


with open(f'{key_word}_前程无忧.csv',mode='w',encoding='ANSI',newline='') as f:

    csv_writer = csv.DictWriter(f,fieldnames=[
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
    '详情页'
    ])
    csv_writer.writeheader()

    for page in range(1,int(pages)):
        #1、发送请求
        print(f'========================================正在采集第{page}页的数据内容============================================')
        sleep(randint(5, 10))
        url = f'https://search.51job.com/list/040000,000000,0000,00,9,99,{key_word},2,{page}.html'

        headers = {
       #这个需要自己用网上的hearders~
        }

        response = requests.get(url=url,headers = headers)

        #3、解析数据

        html_data = re.findall('window.__SEARCH_RESULT__ = (.*?)</script>',response.text)[0]


        json_data = json.loads(html_data)

        for index in json_data['engine_jds']:
            try:
                dit={
                    '岗位名称':index['job_name'],
                    '公司名称':index['company_name'],
                    '薪资':index['providesalary_text'],
                    '工作地点':index['workarea_text'],
                    '学历':index['attribute_text'][2],
                    '经验':index['attribute_text'][1],
                    '融资情况':index['companytype_text'],
                    '公司规模': index['companysize_text'],
                    '福利待遇':index['jobwelf'],
                    '发布日期':index['updatedate'],
                    '详情页':index['job_href'],
                }
                csv_writer.writerow(dit)
                print(dit)
            except IndexError:
                pass





