
import json
import re
import requests
import pprint
import csv
import time
import random

with open('taobao.csv','w',encoding='ANSI',newline='') as filename:
    csvwriter = csv.DictWriter(filename,fieldnames=['标题','价格','店铺','购买人数','地点','商品详情页','店铺链接','图片链接'])
    csvwriter.writeheader()

    for page in range(1,6):
        time.sleep(random.randint(1,3))
        print(f"========================================正在爬取第{page}页=============================================")
        url = f'#根据视频上的操作粘贴网址以及修改page哦~'
        headers = {
        #粘贴网址上的headers就可以啦
        }
        response = requests.get(url=url,headers=headers)
        # print(response.text)
        html_data = re.findall('g_page_config = (.*);',response.text)[0]
        # print(html_data)
        json_data = json.loads(html_data)
        # pprint.pprint(json_data)

        for index in json_data[ 'mods']['itemlist']['data']['auctions']:
            dict = {'标题': index['raw_title'],
                    '价格':index['view_price'],
                    '店铺':index['nick'],
                    '购买人数':index['view_sales'],
                    '地点':index['item_loc'],
                    '商品详情页':index['detail_url'],
                    '店铺链接':index['shopLink'],
                    '图片链接':index['pic_url']
                    }
            csvwriter.writerow(dict)
            print(dict)
