from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import time
import pandas as pd

options = Options()
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

driver.get("https://demagog.org.pl/wypowiedzi/")
button = driver.find_element(By.XPATH,"/html/body/div[2]/div/div/div/div[2]/button[3]")
while True:
    try:
        button = driver.find_element(By.XPATH, "/html/body/div[4]/main/section[2]/div[2]/div/div/button")
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", button)
        time.sleep(3)
    except (NoSuchElementException, ElementClickInterceptedException):
        print("Nie ma już więcej przycisków lub nie można kliknąć.")
        break

articles = driver.find_elements(By.XPATH, "//div[@class='views-field-title']//a")
links = [a.get_attribute("href") for a in articles]

driver.quit()

df = pd.DataFrame(links, columns=["link"])
df.to_csv("demagog_links.csv", index=False)
print(f"Zapisano {len(links)} linków do demagog_links.csv")
