# Importar librerías
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import pandas as pd
import pdfplumber

# URL's necesarios
URL_CAREERS = 'http://oferta.unam.mx/indice-alfabetico.html'  # # Página donde se obtiene la lista completa de carreras que oferta la UNAM
# Base de Datos de las tesis de la UNAM
URL_DB = 'https://tesiunam.dgb.unam.mx/F?func=find-b-0&local_base=TES01'  # Base de Datos de las tesis de la UNAM
PATH = 'C:\msedgedriver.exe'  # Ruta del driver de selenium

# Listas para almacenar la introducción, el nombre y apellido del autor, el título, año y la carrera de c/u de las tesis
introduccion = []
autor_nombre = []
autor_apellido = []
titulo = []
año = []
carrera = []

total_tesis_carrera = []  # Lista del total de tesis por carrera
carreras_sin_tesis = []  # Lista de las carreras que no tienen tesis
# texto_tesis = []


def download_and_extract_theses_text():
    """
    Descarga una tesis en específico.

    Para ello, busca en primer lugar el atributo 'src' del segundo 'frame' con la finalidad de obtener el url del pdfViewer.
    Después, busca en la ventana de la tesis un botón de descarga y posteriormente hace clic en él.
    Cabe mencionar que el tiempo de espera puede variar de acuerdo al tamaño del archivo de descarga, así como de la velocidad del Internet con el que cuente cada persona.

    NOTA: Se encuentra un bloque comentado en la función, cuyo propósito es obtener el texto de la tesis.

    """
    try:
        pdf_url = driver.find_element(
            By.XPATH, '/html/frameset/frame[2]').get_attribute("src")
        driver.get(pdf_url)

        WebDriverWait(driver, 4).until(EC.number_of_windows_to_be(2))

        download_pdf = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="download"]')))
        download_pdf.click()
        time.sleep(20)
        driver.close()

    except:
        pdf_url = driver.find_element(
            By.XPATH, '/html/frameset/frameset/frame[2]').get_attribute("src")
        driver.get(pdf_url)

        WebDriverWait(driver, 4).until(EC.number_of_windows_to_be(2))

        download_pdf = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="download"]')))
        download_pdf.click()
        time.sleep(20)
        driver.close()

    # intro_pages = np.arange(4, 16)
    # local_filename = pdf_url.split('/')[-1]
    # pdf_path = './' + local_filename

    # with pdfplumber.open(pdf_path) as pdf:
    #     texto_sin_delimitar = []
    #     for page in intro_pages:
    #         texto_sin_delimitar.append(pdf.pages[page].extract_text().replace('\n', ' '))
    #
    # cond_texto_tesis = re.search('Introducción (.*?) Capítulo 1', texto_sin_delimitar)
    # texto_tesis = cond_texto_tesis.group(1)
    # introduccion.append(texto_sin_delimitar)
    # texto_tesis.append(arr_texto_tesis)


def create_dataset():
    """
    Crea el dataset con los datos obtenidos.

    Para ello, se crea un diccionario en la que cada llave corresponde a una lista de un dato en específico de la tesis, después se crea un dataframe y finalmente se almacena en un archivo csv.

    """

    dictionary = {'autor_nombre': autor_nombre, 'autor_apellido': autor_apellido, 'titulo': titulo, 'año': año, 'carrera': carrera}
    df = pd.DataFrame(dictionary)
    df.to_csv('tesis_carreras_UNAM.csv', index=False, encoding='utf-8')


def change_windows():
    parent_handle = driver.current_window_handle
    all_handles = driver.window_handles

    for handle in all_handles:
        if handle != parent_handle:
            driver.switch_to.window(handle)
            driver.maximize_window()

            # download_and_extract_theses_text()

            driver.switch_to.window(parent_handle)


if __name__ == '__main__':
    # Petición al servidor para obtener todo el maquetado
    response = requests.get(URL_CAREERS)

    if response.status_code == 200:
        print('La petición ha sido exitosa.\n')
        content = response.text

        # Se almacena en un documento html todas las licenciaturas de la UNAM
        with open('careers.html', 'w+') as f:
            f.write(content)

        soup = BeautifulSoup(content, 'html.parser')
        section = soup.find('section', {'id': 'maincontent'})
        div = section.find('div', class_='row')

        # Lista para almacenar las carreras
        careers = []
        for li in div.ul.find_all('div', class_='caption'):
            for a in li.p.find_all('a', class_='style1'):
                career = a.text.replace('• ', '').replace('Ã', 'Ó')

                # Correción del formato de algunas carreras
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

                elif career == 'Música - Canto':
                    musc = 'Música (Canto)'
                    careers.append(musc)

                elif career == 'Música - Composición':
                    musc = 'Música (Composición)'
                    careers.append(musc)

                elif career == 'Música - Educación Musical':
                    musc = 'Música (Educación Musical)'
                    careers.append(musc)

                elif career == 'Música - Instrumentista':
                    musc = 'Música Intrumentista'
                    careers.append(musc)
                
                elif career == 'Música - Piano':
                    musc = 'Música (Piano)'
                    careers.append(musc)

                else:
                    careers.append(career)
    print(careers)
    print('\nTotal Carreras (UNAM):', len(careers))

    # Automatización del Scraper
    s = Service(PATH)
    driver = webdriver.Edge(service=s)

    # Se usa la constante URL_DB
    driver.implicitly_wait(2)
    driver.get(URL_DB)
    driver.maximize_window()
    print(driver.title)

    ## La siguiente línea se puede comentar para poder obtener una extracción completa de cada una de las tesis de la UNAM
    # careers = ['Actuaría', 'Derecho', 'Economía', 'Psicología', 'Química Farmaceútico Biológica']
    try:
        i = 50
        # Proceso de búsqueda de tesis por carrera
        for career in careers:
            phrase = driver.find_element(By.XPATH, '//*[@id="palabra"]')
            phrase.clear()
            phrase.send_keys('Licenciatura en ' + careers[i])

            select_search = driver.find_element(By.ID, 'indiceWord')
            search_option = Select(select_search)
            search_option.select_by_visible_text('Grado')

            select_adjacency = driver.find_element(By.ID, 'adyacencia')
            adjacency_option = Select(select_adjacency)
            adjacency_option.select_by_visible_text('Buscar las palabras juntas')

            from_year = driver.find_element(By.NAME, 'filter_request_2')
            from_year.clear()
            from_year.send_keys('2015')

            to_year = driver.find_element(By.NAME, 'filter_request_3')
            to_year.clear()
            to_year.send_keys('2022')

            submit_button = driver.find_element(By.XPATH, '//*[@id="busquedaBasica"]/div/form/div[6]/div/button')
            submit_button.submit()

            try:
                try:
                    # Caso en el que no haya ninguna tésis disponible en una carrera en específico
                    NOT_THESES_URL = driver.current_url
                    print(NOT_THESES_URL)

                    driver.implicitly_wait(4)
                    response = requests.get(NOT_THESES_URL)
                    content = response.text

                    soup = BeautifulSoup(content, 'html.parser')
                    body = soup.find('body')
                    section = body.find('section', {'id': 'resultados-list-head'})
                    div = section.div.div.find('div', class_='col-md-12')
                    strong = div.p.strong.text.replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace(
                        'Ã­', 'í').replace('Ã', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ')
                    print(strong)
                    print(type(strong))
                    print(careers[i])

                    if strong == ('Licenciatura en ' + careers[i]):
                        print(f'Lo siento, no hay tesis disponibles para la Licenciatura en {careers[i]}.\n')

                        # Se usa la lista 'carreras_sin_tesis'
                        carreras_sin_tesis.append(careers[i])

                        driver.implicitly_wait(4)
                        driver.get(URL_DB)
                        driver.maximize_window()
                        i += 1
                except:
                    # Caso en el quee solo hay una tesis disponible en una carrera en específico
                    ONLY_THESIS_URL = driver.current_url
                    print(ONLY_THESIS_URL)

                    driver.implicitly_wait(4)
                    tags_name_link = driver.find_element(By.XPATH, '//*[@id="formatos"]/div/div/div/ul/li[5]/a')
                    tags_name_link.click()

                    response = requests.get(ONLY_THESIS_URL)
                    content = response.text

                    soup = BeautifulSoup(content, 'html.parser')
                    body = soup.find('body')
                    section = body.find('section', {'id': 'formato-completo'})
                    table = section.div.div.div.div.find('table', class_='table table-bordered table-hover table-sm')

                    datos_unica_tesis = []
                    for row in table.tbody.find_all('tr'):
                        column = row.find_all('td')

                        for r in column:
                            datos_unica_tesis.append((r.text).replace(
                                '\n', '').replace('\xa0', '').replace('     ', ''))

                    autor_unica_tesis = datos_unica_tesis[0].replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                        'Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                    cond_autor_ut = re.search("(.*?), (.*?),", autor_unica_tesis)
                    apellido_autor_unica_tesis = cond_autor_ut.group(1)
                    autor_apellido.append(apellido_autor_unica_tesis)
                    nombre_auto_unica_tesis = cond_autor_ut.group(2)
                    autor_nombre.append(nombre_auto_unica_tesis)

                    titulo_unica_tesis = datos_unica_tesis[1].replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                        'Ã­-', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                    cond_titulo_ut = re.search("(.*?) /", titulo_unica_tesis)
                    titulo_unica_tesis = cond_titulo_ut.group(1)
                    titulo.append(titulo_unica_tesis)

                    año_unica_tesis = int(datos_unica_tesis[2].replace(' ', ''))
                    año.append(año_unica_tesis)

                    carrera_unica_tesis = careers[i]
                    carrera.append(carrera_unica_tesis)

                    driver.implicitly_wait(4)
                    driver.back()
                    try:
                        t_des_com_env = WebDriverWait(driver, 4).until(EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="formato-completo"]/div/div/div/div/table/tbody/tr[17]/td/a')))
                        t_des_com_env.click()

                        change_windows()

                    except:
                        t_neu = WebDriverWait(driver, 4).until(EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="formato-completo"]/div/div/div/div/table/tbody/tr[15]/td/a')))
                        t_neu.click()

                        change_windows()

                    driver.get(URL_DB)
                    driver.maximize_window()
                    i += 1

            except:
                # Caso en el que haya dos o más tesis disponibles en una carrera en específico
                NoNextPage = False
                while not NoNextPage:
                    CURRENT_URL = driver.current_url

                    response = requests.get(CURRENT_URL)
                    content = response.text

                    soup = BeautifulSoup(content, 'html.parser')
                    table = soup.find('table', class_='table table-bordered table-hover table-sm')

                    j = 1
                    for row in table.tbody.find_all('tr'):
                        response = requests.get(CURRENT_URL)
                        content = response.text

                        soup = BeautifulSoup(content, 'html.parser')
                        table = soup.find('table', class_='table table-bordered table-hover table-sm')
                        columns = row.find_all('td')

                        try:
                            autores = columns[2].text.replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                                'Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                            cond_autores = re.search("(.*?), (.*?),", autores)
                            apellido_autores = cond_autores.group(1)
                            autor_apellido.append(apellido_autores)
                            nombre_autores = cond_autores.group(2)
                            autor_nombre.append(nombre_autores)

                        except:
                            autores = columns[2].text.replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                                'Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                            cond_autores = re.search("(.*?), (.*?)", autores)
                            apellido_autores = cond_autores.group(1)
                            autor_apellido.append(apellido_autores)
                            nombre_autores = cond_autores.group(2)
                            autor_nombre.append(nombre_autores)

                        años = int(columns[4].text)
                        año.append(años)

                        carreras = careers[i]
                        carrera.append(carreras)

                        COMPLETE_THESIS_XPATH = f'//*[@id="resultados-formato-corto"]/div/div/div/div/table/tbody[{j}]/tr/td[4]/a'
                        driver.find_element(By.XPATH, COMPLETE_THESIS_XPATH).click()
                        driver.implicitly_wait(4)
                        tags_name_link = driver.find_element(By.XPATH, '//*[@id="formatos"]/div/div/div/ul/li[5]/a')
                        tags_name_link.click()

                        response = requests.get(driver.current_url)
                        content = response.text

                        soup = BeautifulSoup(content, 'html.parser')
                        body = soup.find('body')
                        section = body.find('section', {'id': 'formato-completo'})
                        table = section.div.div.div.div.find('table', class_='table table-bordered table-hover table-sm')

                        datos_tesis = []
                        for row in table.tbody.find_all('tr'):
                            column = row.find_all('td')

                            for r in column:
                                datos_tesis.append((r.text).replace('\n', '').replace(
                                    '\xa0', '').replace('     ', ''))

                        try:
                            try:
                                titulos = datos_tesis[5].replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                                    'Ã­-', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                                cond_titulos = re.search(" (.*?)/", titulos)
                                titulos = cond_titulos.group(1)
                                titulo.append(titulos)
                            except:
                                titulos = datos_tesis[5].replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                                    'Ã­-', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                                cond_titulos = re.search(
                                    " (.*?)tesis que", titulos)
                                titulos = cond_titulos.group(1)
                                titulo.append(titulos)

                        except:
                            titulos = datos_tesis[3].replace('Ã', 'Á').replace('Ã¡', 'á').replace('Ã‰', 'É').replace('Ã©', 'é').replace('Ã', 'Í').replace('Ã­', 'í').replace(
                                'Ã­-', 'í').replace('Ã“', 'Ó').replace('Ã³', 'ó').replace('Ãš', 'Ú').replace('Ãº', 'ú').replace('Ã¼', 'ü').replace('Ã±', 'ñ').replace('Â¿', '¿').replace('Â´', '´')
                            cond_titulos = re.search(" (.*?)/", titulos)
                            titulos = cond_titulos.group(1)
                            titulo.append(titulos)

                        driver.implicitly_wait(4)
                        driver.back()
                        driver.back()

                        thesis_link = driver.find_element(By.XPATH, f'//*[@id="resultados-formato-corto"]/div/div/div/div/table/tbody[{j}]/tr/td[6]/a').click()

                        change_windows()
                        create_dataset()
                        j += 1

                    print(autor_apellido, '\n', len(autor_apellido), '\n')
                    print(autor_nombre, '\n', len(autor_nombre), '\n')
                    print(titulo, '\n', len(titulo), '\n')
                    print(año, '\n', len(año), '\n')
                    print(carrera, '\n', len(carrera), '\n')
                    print(driver.current_url, '\n')
                    print(len(autor_apellido), len(autor_nombre), len(titulo), len(año), len(carrera), '\n')

                    response = requests.get(CURRENT_URL)
                    content = response.text

                    soup = BeautifulSoup(content, 'html.parser')
                    body = soup.find('body')
                    section = body.find_all('section', {'id': 'shortButtons'})
                    div = section[2].find('div', class_='col-md-12')
                    strong = div.p.text.replace(' ', '')

                    cond_reg = re.search("Registros(.*)-(.*)de(.*).", strong)
                    cond_reg_2 = int(cond_reg.group(2))  # Registro máximo actual
                    cond_reg_3 = int(cond_reg.group(3))  # Total de Tesis
                    print(cond_reg_2, type(cond_reg_2), cond_reg_3, type(cond_reg_3))

                    # Condiciones para cambiar de carrera si se llega a un límite en específico o el número máximo de registros
                    ## Nota: El valor a igualar en la 'cond_reg_2' debe ser un múltiplo de 10
                    if (cond_reg_2 == 20) or (cond_reg_2 == cond_reg_3):
                        print(f"Estas fueron todas las tesis de la Licenciatura en {careers[i]}.\n")
                        NoNextPage = True
                        break

                    # Condición para cambiar de página si se han completado los diez registros
                    else:
                        try:
                            previous_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                                (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/img')))

                            driver.find_element(
                                By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a/img').click()
                        except Exception as e:
                            next_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                                (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a[2]/img')))

                            driver.find_element(
                                By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a[2]/img').click()

                            WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                                (By.XPATH, '//*[@id="jumpTo"]/div/div/div[2]/p/a/img')))

                driver.implicitly_wait(10)
                driver.get(URL_DB)
                driver.maximize_window()

                total_tesis_carrera.append(careers[i])
                total_tesis_carrera.append(cond_reg_3)
                print(total_tesis_carrera)
                print('Carreras sin tesis:', carreras_sin_tesis)
                create_dataset()

                NoNextPage = False
                i += 1

    # Excepción por si llegara a existir un error con el programa: se crea el dataset con los datos recabados y se cierra las ventanas de la automatización
    except Exception as e:
        print(e, 'El scraper ha terminado.\n')
        create_dataset()
        driver.quit()

print('El scraper ha terminado.\n')
create_dataset()
driver.quit()
