import os
import datetime
import time
from PySimpleGUI.PySimpleGUI import Spin
import pandas as pd
import PySimpleGUI as sg

from pandas.core.frame import DataFrame

TEST_PATH = r"C:/Users/tomasdfen/Documents/Universidad/work/segunda fase/interfaz/dataset.xlsx"

sg.theme('Dark Blue 3')  # please make your creations colorful
def valid_date(date, fmt = "%d-%m-%Y"):
    try:
        datetime.datetime.strptime(date,fmt)
        return True
    except ValueError:
        return False

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def open_main_window(path):
    
    load = sg.Window("Cargando conjunto de datos", [[sg.Text("Espere mientras se carga el conjunto de datos")]])
    data = dict()
    priceIn = False
    priceData = pd.DataFrame()
    while len(data.values()) == 0:
        load.read(timeout=1)
        print("Esperando")
        data : dict = pd.read_excel(path, index_col=0, parse_dates=True, sheet_name=None)
    load.close()
    
    
    for key in data.keys():
        if "PRECIO" in key.upper():
            priceIn = True
            priceData = data.pop(key)
            
            break

    if priceIn:
                     
        
        left = [[sg.Text("Variables a usar", key="new")],
                [sg.Listbox(list(data.keys()), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(None, 300),enable_events=True, key="-LIST-")]]

        right = [[sg.CalendarButton("Fecha de inicio", target='-START-', format="%d-%m-%Y", locale="ES"), sg.Input(key="-START-", enable_events=True)],
                [sg.CalendarButton("Fecha de fin", target='-END-', format="%d-%m-%Y", locale="ES"), sg.Input(key="-END-", enable_events=True)],
                [sg.Text("Ventana de pasos hacia atras"), sg.Input("1", key='-BACK-')],
                [sg.Text("Horizonte de predicción"), sg.Input("1", key='-FORWARD-')],
                [sg.Text("Iteraciones"), sg.Input("1", key='-ITER-')],
                [sg.Text("Batch de entrenamiento"), sg.Input("1", key='-TRAIN_BATCH-')],
                [sg.Text("Batch de test"), sg.Input("1", key='-TEST_BATCH-')],
                [sg.Button("OK", key="-GO-")]]

        errors = {
            "-START-":False,
            "-END-":False,
            "-BACK-":False,
            "-ITER-":False,
            "-TRAIN_BATCH-":False,
            "-TEST_BATCH-":False,
            "-FORDWARD-":False
        }
        
        layout = [[sg.Column(left), sg.VSeparator(), sg.Column(right, vertical_alignment="t")]]

        window = sg.Window("Selección de parámetros", layout, modal=True, size=(500, 500))

    else:
        
        window = sg.Window("No se ha encontrado variable de precio", [[sg.Text("En ninguna de las hojas del libro dado como entrada se ha encontrado la palabra 'precio'. Asegurese de que ha usado el libro correcto")]])

        
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-LIST-":
            print(values["-LIST-"])
            print(values)
        if event == "-START-":
            if len(values['-START-']) >= 10 and not valid_date(values['-START-']):
                window['-START-'].update(values['-START-'][:-1])  
            elif not all([c in "0123456789-" for c in values["-START-"]]):
                window['-START-'].update(background_color="#FF0000")
                errors['-START-'] = True
            else:
                window['-START-'].update(background_color="#FFFFFF")
                errors['-START-'] = False
        if event == "-END-":
            if len(values['-END-']) >= 10 and not valid_date(values['-END-']):
                window['-END-'].update(values['-END-'][:-1])  
            elif not all([c in "0123456789-" for c in values["-END-"]]):
                window['-END-'].update(background_color="#FF0000")
                errors['-END-'] = True
            else:
                window['-END-'].update(background_color="#FFFFFF")
                errors['-END-'] = False
                
        if event == "-TEST_BATCH-":
            if not isInt(values['-TEST_BATCH-']):
                window['-TEST_BATCH-'].update(background_color="#FF0000")
                errors['-TEST_BATCH-'] = True

            else:
                window['-TEST_BATCH-'].update(background_color="#FFFFFF")
                errors['-TEST_BATCH-'] = False

        if event == "-ITER-":
            if not isInt(values['-ITER-']):
                window['-ITER-'].update(background_color="#FF0000")
                errors['-ITER-'] = True

            else:
                window['-ITER-'].update(background_color="#FFFFFF")
                errors['-ITER-'] = False

        if event == "-TRAIN_BATCH-":
            if not isInt(values['-TRAIN_BATCH-']):
                window['-TRAIN_BATCH-'].update(background_color="#FF0000")
                errors['-TRAIN_BATCH-'] = True

            else:
                window['-TRAIN_BATCH-'].update(background_color="#FFFFFF")
                errors['-TRAIN_BATCH-'] = False

        if event == "-BACK-":
            if not isInt(values['-BACK-']):
                window['-BACK-'].update(background_color="#FF0000")
                errors['-BACK-'] = True

            else:
                window['-BACK-'].update(background_color="#FFFFFF")
                errors['-BACK-'] = False

        if event == "-FORWARD-":
            if not isInt(values['-FORWARD-']):
                window['-FORWARD-'].update(background_color="#FF0000")
                errors['-FORWARD-'] = True
            else:
                window['-FORWARD-'].update(background_color="#FFFFFF")
                errors['-FORWARD-'] = False
        if event == "-GO-":
            print(values["-LIST-"])
            foo = pd.DataFrame()
            for var in values['-LIST-']:
                fstcol_name = data[var].columns[0]
                fstcol = data[var][fstcol_name]
                for x in range(int(values['-BACK-'])):
                    if x > 0:
                        foo[f'{var}-{x}'] = fstcol.shift(x)
                    else:
                        foo[var] = fstcol    
            foo.dropna(inplace=True)
            print(foo)      
                
            
        
    window.close()

def exec():

    layout = [  [sg.Text('Carga un dataset')],
                [sg.Input(TEST_PATH,key="-PATH-"), sg.FileBrowse()],
                [sg.Button("OK", key="-OPEN-"), sg.Button("Cancelar", key="Exit")]         
    ] 

    window = sg.Window('Seleccione el dataset', layout)
    sg.Popup("Recomendación", "El uso óptimo de esta aplicación es con datos obtenidos del generador de datos electricos. Es posible especificar pasos temporales en esta aplicación, de modo que no es necesario que lo tengan los datos de entrada")
    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        elif event == "-OPEN-":
            path = values["-PATH-"]
            if os.path.exists(path):
                window.close()
                open_main_window(path)
            else:
                sg.popup("No existe la ruta especificada")
    window.close()
if __name__ == "__main__":
    exec()