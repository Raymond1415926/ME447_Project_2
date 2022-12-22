import requests
from bs4 import BeautifulSoup
import pandas as pd

page_indexs = range(0, 250, 25)
def download_all_htmls() :
    htmls = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
    for idx in page_indexs:
        url = f"https://movie.douban.com/top250?start={idx}&filter="
        print("crawl html:", url)
        r = requests.get(url, headers =headers)
        # if r.status_code != 200: #fail
        #     raise Exception("error")
        htmls.append(r.text)
    return htmls

def parse_single_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    article_items = (
        soup.find("div", class_= "article")
        .find("ol", class_= "grid_view")
        .find_all("div", class_="item")
    )
    data = []
    for article_item in article_items:
        rank = article_item.find("div", class_="pic").find("em").get_text()
        info = article_item.find("div", class_="info")
        title = info.find("div", class_="hd").find("span", class_="title").get_text()
        stars = (
            info.find("div", class_= "bd")
            .find("div", class_= "star")
            .find_all("span")
        )
        rating_star = stars[0]["class"][0]
        rating_num = stars[1].get_text()
        comments =  stars[3].get_text()
        data.append({
            "rank":rank,
            "title":title,
            "rating_star":rating_star.replace("rating", "").replace("-t",""),
            "rating_num":rating_num,
            "comments":comments.replace("人评价", "")
        })

    return data

htmls = download_all_htmls()

all_data = []
for html in htmls:
    all_data.extend(parse_single_html(html))
df = pd.DataFrame(all_data)
df.to_excel("Douban Top 250.xlsx")