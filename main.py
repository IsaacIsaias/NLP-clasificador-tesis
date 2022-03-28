import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

URL_CAREERS = 'http://oferta.unam.mx/indice-alfabetico.html'
URL_THESES = 'https://tesiunam.dgb.unam.mx/F/ABK5EA357ASAU9BV4ALHEQDFD697RG2JPAEXDG7R3HIBYE8UCG-24397?func=file&file_name=find-b'
PATH = 'C:\msedgedriver.exe'

autor = []
titulo = []
año = []
introduccion = []
conclusion = []
carrera = []

if __name__ == '__main__':
    response = requests.get(URL_CAREERS) # Petición al servidor para obtener todo el maquetado

    if response.status_code == 200:
        content = response.text

        soup = BeautifulSoup(content, 'html.parser')

        section = soup.find('section', {'id' : 'maincontent'})
        
        div = section.find('div', class_='row')

        careers = []
        for li in div.ul.find_all('div', class_='caption'):
            for a in li.p.find_all('a', class_='style1'):
                career = a.text.replace('• ', '')

                if career == 'Enseñanza de (Alemán) (Español) (Francés) (Inglés) (Italiano) Como Lengua Extranjera':
                    lang = ['Alemán', 'Español', 'Francés', 'Inglés', 'Italiano']
                    for l in lang:
                        e_lang = f'Enseñanza de {l} Como Lengua Extranjera'
                        careers.append(e_lang)

                elif career == 'Lengua y Literaturas Modernas (Letras Alemanas, Francesas, Inglesas, Italianas o Portuguesas)':
                    lang = ['Alemanas', 'Francesas', 'Inglesas', 'Italianas', 'Portuguesas']
                    for l in lang:
                        e_lang = f'Lengua y Literaturas Modernas (Letras {l})'
                        careers.append(e_lang)

                else:
                    careers.append(career)

    # Automatization
    s = Service('C:\msedgedriver.exe')
    driver = webdriver.Edge(service=s)

    URL = 'https://tesiunam.dgb.unam.mx/F?func=find-b-0&local_base=TES01'
    driver.implicitly_wait(10)
    driver.get(URL)
    print(driver.title)
    
    try:
        i = 0 # i = 9 -> Bibliotecología
        for career in careers:
            phrase = driver.find_element(By.XPATH, '//*[@id="palabra"]')
            phrase.clear()
            phrase.send_keys('Licenciatura en ' + careers[i])

            select_search = driver.find_element(By.ID, 'indiceWord')
            search_option = Select(select_search)
            search_option.select_by_visible_text('Grado')
            time.sleep(1)

            select_adjacency = driver.find_element(By.ID, 'adyacencia')
            adjacency_option = Select(select_adjacency)
            adjacency_option.select_by_visible_text('Buscar las palabras juntas')
            time.sleep(1)

            from_year = driver.find_element(By.NAME, 'filter_request_2')
            from_year.clear()
            from_year.send_keys('2017')
            time.sleep(1)

            to_year = driver.find_element(By.NAME, 'filter_request_3')
            to_year.clear()
            to_year.send_keys('2022')
            time.sleep(1)

            submit_button = driver.find_element(By.XPATH, '//*[@id="busquedaBasica"]/div/form/div[6]/div/button')
            submit_button.submit()
            time.sleep(1)

            NoNextPage = False
            while not NoNextPage:
                CURRENT_URL = driver.current_url
                response = requests.get(CURRENT_URL)
                content = response.text

                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', class_='table table-bordered table-hover table-sm')

                for row in table.tbody.find_all('tr'):
                    columns = row.find_all('td')

                    autores = columns[2].text.replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿')
                    autor.append(autores)

                    script_titulo = str(columns[3].script).replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿')
                    cond_titulo = re.search("title = '(.*?) /'", script_titulo)
                    cond_titulo1 = re.search("title = '(.*?)/'", script_titulo)
                    cond_titulo2 = re.search("title = '(.*?)', ", script_titulo)
                    cond_titulo3 = re.search("title = '(.*?)',", script_titulo)
                    cond_titulo4 = re.search("title = '(.*?)' ", script_titulo)
                    cond_titulo5 = re.search("title = '(.*?)'", script_titulo)
                    cond_titulo6 = re.search("title = '(.*?)';", script_titulo)
                    cond_titulo7 = re.search("title = '(.*?)' ;", script_titulo)
                    if cond_titulo:
                        titulo_tesis = cond_titulo.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo1:
                        titulo_tesis = cond_titulo1.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo2:
                        titulo_tesis = cond_titulo2.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo3:
                        titulo_tesis = cond_titulo3.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo4:
                        titulo_tesis = cond_titulo4.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo5:
                        titulo_tesis = cond_titulo5.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo6:
                        titulo_tesis = cond_titulo6.group(1)
                        titulo.append(titulo_tesis)
                    elif cond_titulo7:
                        titulo_tesis = cond_titulo7.group(1)
                        titulo.append(titulo_tesis)
                    else:
                        titulo_tesis = 'Error extracción'
                        titulo.append(titulo_tesis)

                    años = int(columns[4].text)
                    año.append(años)

                    carreras = careers[i]
                    carrera.append(carreras)

                print(autor, '\n', len(autor), '\n')
                print(titulo, '\n', len(titulo), '\n')
                print(año, '\n', len(año), '\n')
                print(carrera, '\n', len(carrera), '\n')
                print(driver.current_url, '\n')
                print(len(autor), len(titulo), len(año), len(carrera))
                time.sleep(2)


                body = soup.find('body')

                section = body.find_all('section', {'id' : 'shortButtons'})
        
                div = section[2].find('div', class_='col-md-12')
                strong = div.p.text.replace(' ', '')

                cond_reg = re.search("Registros(.*)-(.*)de(.*).", strong)
                cond_reg_2 = int(cond_reg.group(2))
                cond_reg_3 = int(cond_reg.group(3))
                print(cond_reg_2, type(cond_reg_2), cond_reg_3, type(cond_reg_3))


                if cond_reg_2 == cond_reg_3:
                    print("There were all the theses of this career.")
                    NoNextPage = True
                    break

                else:
                    try:
                        previous_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/img')))

                        driver.find_element(By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a/img').click()
                    except Exception as e:
                        next_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a[2]/img')))

                        driver.find_element(By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a[2]/img').click()

                        
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a/img')))


            i += 1
            driver.get(URL)
            NoNextPage = False

        #thesis = driver.find_element(By.XPATH, '//*[@id="resultados-formato-corto"]/div/div/div/div/table/tbody[1]/tr/td[6]/a')
        #thesis.click()

        #URL_THESIS = driver.current_url
        #response = requests.get(URL_THESIS)
        #content = response.text

        #soup = BeautifulSoup(content, 'html.parser')
        #div = soup.find('div', class_='pdfViewer')

        #table = soup.find('table', class_='table table-bordered table-hover table-sm')

    except Exception as e:
        print(e, 'Main error.')
