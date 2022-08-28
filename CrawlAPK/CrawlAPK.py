# -*- coding: utf-8 -*-
from selenium import webdriver
import time
import pymysql
from selenium.webdriver import ActionChains
import xlrd
import random
import os


def getDriver():
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('headless')
    # prefs = {"profile.managed_default_content_settings.images": 2,'permissions.default.stylesheet':2}
    # chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(chrome_options=chrome_options)
    return driver


def getApk(appID):
    url = "https://apps.evozi.com/apk-downloader/"
    driver = getDriver()
    driver.get(url)
    time.sleep(3)
    name_div = driver.find_element_by_xpath("//input[@type='text']")
    name_div.clear()
    t = random.randint(2, 6)
    time.sleep(t)
    name_div.send_keys(appID)
    t = random.randint(3, 6)
    time.sleep(t)
    button = driver.find_element_by_xpath("//button[@type='button']")
    button.click()
    t = random.randint(50,100)
    time.sleep(t)
    warning_buttons = driver.find_elements_by_xpath("//a[@class='btn btn-block btn-warning']")
    if len(warning_buttons) == 0:
        pass
    else:
        warning_buttons[0].click()
        t = random.randint(3, 7)
        time.sleep(t)
        t = random.randint(3, 7)
        time.sleep(t)
        button = driver.find_element_by_xpath("//button[@type='button']")
        button.click()
        t = random.randint(40, 70)
        time.sleep(t)


    a = driver.find_element_by_xpath("//a[@class='btn btn-success btn-block mt-4 mb-4']")
    style = a.get_attribute('style')
    if style != 'display: none;':
        a.click()
        print('click download')
    else:
        print ('---------------')
        print ('fail!!!!')
        print (i)
        print ('---------------')

    t = random.randint(2, 6)
    time.sleep(t)


if __name__ == "__main__":
    appID = 'appinventor.ai_kayipkayik.gps_map'
    getApk(appID)