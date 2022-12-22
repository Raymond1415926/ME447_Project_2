from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests

driver = webdriver.Chrome()
driver.implicitly_wait(10) # this lets webdriver wait 10 seconds for the website to load
driver.get("https://www.google.com/")

driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input').send_keys('Apple phone', Keys.ENTER)

r = requests.get()




