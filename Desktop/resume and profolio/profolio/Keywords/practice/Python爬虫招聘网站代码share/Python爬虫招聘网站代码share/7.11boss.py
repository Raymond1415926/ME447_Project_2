import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options
import random
import csv

with open(f'boss_{datetime.date.today()}.csv','w',encoding='ANSI',newline='') as filename:
    csvwriter = csv.DictWriter(filename,fieldnames=[
        '岗位名称',
        '公司名称',
        '薪资',
        '工作地点',
        '详情页链接'
    ])
    csvwriter.writeheader()

    driver = webdriver.Chrome()
    driver.get('https://www.zhipin.com/shenzhen/?ka=city-sites-101280600')
    driver.implicitly_wait(random.randint(5,8))
    driver.find_element(By.CLASS_NAME,'ipt-search').send_keys('python')
    driver.find_element(By.CLASS_NAME,'btn-search').click()

    def job_info():
        lis = driver.find_elements(By.CSS_SELECTOR, '.job-card-body')
        for li in lis:
            company_name = li.find_element(By.CLASS_NAME,'company-name').text
            job_name = li.find_element(By.CLASS_NAME,'job-name').text
            salary = li.find_element(By.CLASS_NAME,'salary').text
            job_area = li.find_element(By.CLASS_NAME,'job-area').text
            # company_type = li.find_element(By.CLASS_NAME,'company-tag-list').text
            href = li.find_element(By.CSS_SELECTOR,'.job-card-body a').get_attribute('href')

            print(company_name,salary,job_area,href,sep='|')
            dict = {
                '岗位名称':job_name,
                '公司名称':company_name,
                '薪资':salary,
                '工作地点':job_area,
                '详情页链接':href
            }
            csvwriter.writerow(dict)

        driver.find_element(By.CLASS_NAME, 'ui-icon-arrow-right').click()

    for page in range(1,11):
        time.sleep(1)
        job_info()


    driver.quit()




