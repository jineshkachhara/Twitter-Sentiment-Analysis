from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import os
import codecs
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import re
import csv

intialurl = 'https://twitter.com/'

search_vairables = ['#trumprocks','#trumpsucks','#hillaryrocks','#hillarysucks']
sentiment_var = [1,0,1,0]
'''
search_vairables = ['#trumpsucks']
sentiment_var = [0]
'''
for i in range(len(search_vairables)):
    cromepath = 'chromedriver.exe'
    os.environ['webdriver.Chrome.driver'] = cromepath
    user_agent = {
        'User-Agent': 'Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; +http://www.google.com/bot.html) Safari/537.36.',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Language': 'en-US,en;q=0.8',
        'Accept-Encoding': 'none',
        'Connection': 'keep-alive'}
    driver = webdriver.Chrome(cromepath)
    driver.get(intialurl)
    #from_input = driver.find_element_by_xpath('//*[@id="root"]/div[2]/div[1]/div[1]/div[2]/div[1]/div')

    #from_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'StreamsCategoryBar')))

    button_bar = driver.find_element_by_class_name('StreamsCategoryBar')
    search_button = button_bar.find_element_by_tag_name('button')
    search_button.click()
    search_field = driver.find_element_by_class_name('form-search')
    search_field.click()

    actions = ActionChains(driver)
    actions.send_keys(search_vairables[i])
    actions.send_keys(Keys.ENTER)
    actions.perform()


    for j in range(0,2):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        j = j + 1

    # find all elements with a class that ends in 'tweet-text'
    tweets = driver.find_elements_by_css_selector("[data-item-type=tweet]")

    """
    for tweet in tweets:
        print (tweet.find_element_by_css_selector("[class$=tweet-text"))
    """

    # write the tweets to a file

    if 'trump' in search_vairables[i]:
        output_data = 'tweets_trump.txt'
    else:
        output_data = 'tweets_hillary.txt'

    #output_data = 'tweets' + str([i]) + '.txt'
    fw = codecs.open(output_data, 'a+', encoding='utf-8')
    for tweet in tweets:
        txt = ''

        try:
            txt = tweet.find_element_by_css_selector("[class$=tweet-text]").text
            '''
            txt = re.sub('(https?://[^\s])+', 'URL', txt)
            #txt = re.sub('(www\.[^\s]+)','URLMERA', txt)
            txt = re.sub('@[^\s]+', 'AT_USER', txt)
            #txt = re.sub(r'#([^\s]+)', r'\1', txt)
            '''
            txt = re.sub('(https?://[^\s])+', ' ', txt)
            txt = re.sub('(www\.[^\s]+)', ' ', txt)
            txt = re.sub('(ww\.[^\s]+)', ' ', txt)
            txt = re.sub('@[^\s]+', ' ', txt)
            txt = re.sub('#([^\s]+)', ' ', txt)
            txt = txt.lower().strip()
            if txt != '':
                txt = txt + '\t' + str(sentiment_var[i])
                fw.write(txt.replace('\n', ' ') + '\n')
        except:
            print('no text')
        #fw.write(txt.replace('\n', ' ') + '\n')
    #fw.write(txt.replace('\n', ' ') +  '\t' + str(retweets) + '\n')


fw.close()
driver.quit()