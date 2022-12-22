from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import lxml
import time

web = Chrome() #browser object
web.get("https://www.zhipin.com/web/geek/job?query=python&city=100010000") #this url can be any website

web.find_element(By.XPATH, '/html/body/div[10]/div[1]/div[2]/div[2]/button[4]').click()
print("clicked")
time.sleep(1)

web.find_element(By.XPATH, '//*[@id="search_input"]').send_keys("python", Keys.ENTER)
alst = web.find_elements(By.CLASS_NAME, 'item__10RTO') #find link by class name


# for a in alst:
#
#     a.find_element(By.CLASS_NAME, 'p-position__21iOS').click()
#     web.switch_to.window(web.window_handles[-1])
#     job_name = web.find_element(By.CLASS_NAME, "position-head-wrap-position-name").text
#     requirements = web.find_element(By.CLASS_NAME, 'job-detail').text
#     print(job_name)
#     # f = open("python_jobs/requirements for %s.txt" % job_name, 'w')
#     # f.write(requirements)
#     web.close()
#     web.switch_to.window(web.window_handles[0])
#     time.sleep(1)

input("press any key to exit")
