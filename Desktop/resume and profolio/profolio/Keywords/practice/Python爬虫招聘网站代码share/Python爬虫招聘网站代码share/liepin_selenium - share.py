from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
import csv
import time
import datetime



with open(f'{key_word}_{datetime.date.today()}_liepin.csv','w',encoding='ANSI',newline='') as filename:
    csvwriter = csv.DictWriter(filename,fieldnames=[
    '岗位名称',
    '公司名称',
    '薪资',
    '工作地点',
    '学历',
    '经验',
    '行业',
    '详情页',
    '技能要求',
    ])
    csvwriter.writeheader()

    driver = webdriver.Chrome()
    #深圳网址
    for page in range(0,11):
        driver.get(f'#输入要爬取的网址')

        # driver.find_element(By.XPATH,'//*[@id="home-search-bar-container"]/div/div/div/div/div/div[1]/div[1]/div/div/div/input').send_keys(key_word)
        sleep(1)
        # driver.find_element(By.XPATH,'//*[@id="home-search-bar-container"]/div/div/div/div/div/div[1]/div[1]/div/div/div/span').click()
        driver.implicitly_wait(8)

        # driver.maximize_window()

        def get_job_info():
            lis = driver.find_elements(By.CSS_SELECTOR,'.left-list-box .job-list-item')
            # print(lis)

            for li in lis:
                job_name = li.find_element(By.CSS_SELECTOR,'.job-title-box div:nth-child(1)').text

                exp_list = []
                exp_lis = li.find_elements(By.CSS_SELECTOR,'.labels-tag')
                for e in exp_lis:
                    exp =e.text
                    exp_list.append(exp)
                exp = exp_list[0]
                edu = exp_list[1]
                tag = ','.join(exp_list[2:])

                salary = li.find_element(By.CSS_SELECTOR,'.job-salary').text
                company_name = li.find_element(By.CSS_SELECTOR,'.company-name').text

                company_tag_list = li.find_elements(By.CSS_SELECTOR,'.company-tags-box')
                company_tag = []
                for c in company_tag_list:
                    ct = c.text
                    company_tag.append(ct)
                    # print(ct)
                company_tag_str = ','.join(company_tag)

                district = li.find_element(By.CSS_SELECTOR,'.job-dq-box .ellipsis-1').text
                href = li.find_element(By.CSS_SELECTOR,'.job-detail-box a').get_attribute('href')
                print(job_name,company_name,salary,exp,edu,tag,company_tag_str,href)
                dict = {
                    '岗位名称':job_name,
                    '公司名称':company_name,
                    '薪资':salary,
                    '工作地点':district,
                    '经验':exp,
                    '学历':edu,
                    '技能要求':tag,
                    '行业':company_tag_str,
                    '详情页':href
                }
                csvwriter.writerow(dict)
            # driver.find_element(By.XPATH,'/html/body/div/div/section[1]/div/div/ul/li[8]/a/button').click()

        for page in range(0,11):
            sleep(1)
            get_job_info()

        driver.quit()