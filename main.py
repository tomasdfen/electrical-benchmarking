import os
import json
import pickle
import xgboost
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import subprocess

TEST_PATH = r"C:/Users/tomasdfen/Documents/Universidad/work/segunda fase/interfaz/dataset.xlsx"

sg.theme('Dark Blue 3')
def valid_date(date, fmt = "%d-%m-%Y"):
    try:
        datetime.strptime(date,fmt)
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
    while len(data.values()) == 0:
        load.read(timeout=1)
        print("Esperando")
        if os.path.splitext(path)[1] == ".csv":
            data = pd.read_csv(path, index_col=0, parse_dates=True)
            data.index = data.index.tz_localize('UTC').tz_convert("ETC/GMT-2").tz_localize(None)
            vars = data.columns
        else:
            data = pd.read_excel(path, index_col=0, parse_dates=True, sheet_name=None)
            vars = data.keys()
            for k in vars:
                data[k].index = data[k].index.tz_localize('UTC').tz_convert("ETC/GMT-2").tz_localize(None)

    load.close()
                     
    left = [[sg.Text("Variables a usar", key="new")],
            [sg.Listbox(list(vars), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(None, 100),enable_events=True, key="-LIST-")]]

    right = [[sg.CalendarButton("Fecha de inicio", target='-START-', format="%d-%m-%Y", locale="ES"), sg.Input("01-06-2020",key="-START-", enable_events=True)],
            [sg.CalendarButton("Fecha de fin", target='-END-', format="%d-%m-%Y", locale="ES"), sg.Input("31-08-2020", key="-END-", enable_events=True)],
            [sg.Text("Variable objetivo"), sg.Combo(list(vars),default_value=list(vars)[0], enable_events=True, key="-TARGET-")],
            [sg.Text("Horizonte de predicción"), sg.Input("6", key='-FORWARD-')],
            [sg.Text("Avance por iteración"), sg.Input("8", key='-STEP-')],
            [sg.Text("Iteraciones"), sg.Input("15", key='-ITER-')],
            [sg.Check("Aprendizaje incremental", key='-INCREMENTAL-')],
            [sg.Text("Modelo"),sg.Radio("XGBoost", "models", key="-XGBOOST-", default=True),sg.Radio("SVM", "models", key="-SVM-", default=False)],
            [sg.Button("OK", key="-GO-")],
            [sg.Text("En la siguiente ventana podrá seleccionar \nparámetros para el modelo seleccionado y la \ndivisión temporal de las variables a usar", justification="left")]]

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
    
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-LIST-":
            pass
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
            if not all([c in "0123456789-," for c in values["-END-"]]):
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
            if values["-TARGET-"] in values["-LIST-"]:
                print("F")
                sg.Popup("Error", "La variable objetivo esta entre las variables a usar para el aprendizaje")
            elif len(values["-LIST-"]) == 0:
                print("F")
                sg.Popup("Error", "No se han seleccionado variables para el entrenamiento")
            elif values["-TARGET-"] == "":
                sg.Popup("Error", "No se han seleccionado variable objetivo")
            else:
                window.close()
                
                choose_params(values, data, path)
    window.close()

def choose_params(values, data, path):
    left = []
    for var in values['-LIST-']:
        left.append([sg.Text(var)])
        left.append([sg.Radio("Continuo", f"{var}-backtype", key=f"{var}-CONTINOUS-", default=True),sg.Radio("Selectivo", f"{var}-backtype", key=f"{var}-SELECTIVE-", default=False), sg.Input("36", key=f'{var}-BACK-')])
    left.append([sg.Button("+", key="-EXTRA-")])
    if values["-XGBOOST-"]:
        objectives = ["reg:squarederror", "reg:squaredlogerror", "reg:logistic", "reg:pseudohubererror"]
        right = [[sg.Text("Función objetivo"), sg.Combo(objectives,default_value=objectives[0], enable_events=True, key="objective")],
                  [sg.Text("Puntuación inicial"), sg.Slider((0,1),default_value=0.5, orientation="horizontal",resolution=0.1, enable_events=True, key="base_score")],
                  [sg.Input(r"C:\Users\tomasdfen\Downloads\test",key="-SAVE-",enable_events=True), sg.FolderBrowse()],
                  [sg.Button("OK", key="-GO-")]] 
        layout = [[sg.Column(left), sg.VSeparator(), sg.Column(right)]]
        window = sg.Window("Selección de parámetros: XGBoost", layout, modal=True, size=(None, None))
        while True:
            event, params = window.read()
            print(params)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "-EXTRA":
                print("Extra layer")
                
            if event == "-GO-":
                values['-START-'] = datetime.strptime(values['-START-'],"%d-%m-%Y")
                values['-END-'] = datetime.strptime(values['-END-'],"%d-%m-%Y")
                values["-FORWARD-"] = int(values["-FORWARD-"])
                values["-STEP-"] = values["-FORWARD-"] * int(values["-STEP-"])
                values["-ITER-"] = int(values["-ITER-"])
                path_to_save = params["-SAVE-"]
                x_data = pd.DataFrame()
                max_col = 0
                for var in values['-LIST-']:
                    fstcol_name = data[var].columns[0]
                    fstcol = data[var][fstcol_name]
                    if params[f"{var}-CONTINOUS-"]:
                        features = [i for i in range(int(params[f'{var}-BACK-'])+1)]
                    elif params[f"{var}-SELECTIVE-"]:
                        features = []
                        divs = params[f'{var}-BACK-'].split(",")
                        for div in divs:
                            if isInt(div):
                                features.append(int(div))
                            else:
                                startDiv, endDiv = div.split("-")
                                features += [i for i in range(int(startDiv), int(endDiv)+1)]
                    for x in features:
                        if x > 0:
                            x_data[f'{var}-{x}'] = fstcol.shift(x)
                        else:
                            x_data[var] = fstcol
                    if max_col < max(features):
                        max_col = max(features)
                print(x_data.columns)
                x_data.dropna(inplace=True)
                x_data = x_data.resample("H").last()
                
                y_data = data.pop((values['-TARGET-']))
                y_data = y_data[max_col:]
                
                batches = []
                for i in range(0, values['-ITER-']):
                    x_batch = x_data[(values['-START-']+timedelta(hours = values["-STEP-"]*i)):(values["-END-"]+timedelta(hours = values["-STEP-"]*i))][:-(values['-FORWARD-'])]
                    y_batch = y_data[(values['-START-']+timedelta(hours = values["-STEP-"]*i)):(values["-END-"]+timedelta(hours = values["-STEP-"]*i))][values['-FORWARD-']:]
                    batches.append((x_batch, y_batch))
                print(list(map(lambda x: f"{x[0].shape} vs {x[1].shape}", batches)))
        
                for var in values['-LIST-']:
                    params.pop(f"{var}-CONTINOUS-")
                    params.pop(f"{var}-BACK-")
                    params.pop(f"{var}-SELECTIVE-")
                    
        
                res = []
                performances = []
                batch = batches[0]
                to_pred = batches[1][0][-(values["-STEP-"]):] 
                real_pred = batches[1][1][-(values["-STEP-"]):]
                train_matrix = xgboost.DMatrix(batch[0],batch[1])
                pred_matrix = xgboost.DMatrix(to_pred, real_pred)
                model = xgboost.train(params, train_matrix)
                preds = model.predict(pred_matrix)
                res.append(preds)
                performances.append(mean_absolute_error(real_pred, preds))
                if not os.path.exists(params["-SAVE-"]):
                    os.mkdir(path_to_save)
                for i in range(len(batches[:-1])):
                    batch = batches[i]
                    to_pred = batches[i+1][0][-(values["-STEP-"]):] 
                    real_pred = batches[i+1][1][-(values["-STEP-"]):]
                    train_matrix = xgboost.DMatrix(batch[0], batch[1])
                    pred_matrix = xgboost.DMatrix(to_pred, real_pred)
                    if values['-INCREMENTAL-']:
                        model.save_model(os.path.join(path_to_save,f"checkpoint_{i}.model"))
                        model = xgboost.train(params, train_matrix, xgb_model=model)
                    else:
                        model.save_model(os.path.join(path_to_save,f"checkpoint.model"))
                        model = xgboost.train(params, train_matrix)
                    preds = model.predict(pred_matrix)
                    res.append(preds)
                    performances.append(mean_absolute_error(real_pred, preds))
                print(f"Performance {np.mean(performances)}")
                break
                
    if values["-SVM-"]:
        gamma = "scale"
        errors = {
            'degree':False,
            "-GAMMA-NUM-":False,
            "coef0": False
            
        }
        kernel = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        gammas = ['scale', 'auto', "Numerico"]
        right = [[sg.Text("Kernel"), sg.Combo(kernel,default_value="rbf", enable_events=True, key="objective")],
                  [sg.Text("Grado (kernel polinomial)"), sg.Input("3", enable_events=True, key='degree')],
                  [sg.Text("Gamma"), sg.Combo(gammas,default_value="scale", enable_events=True, key="-GAMMA-TYPE-")],
                  [sg.Text("Valor para Gamma"), sg.Input(0, disabled=True, key='-GAMMA-NUM-', enable_events=True, background_color='#3A3A3A')],
                  [sg.Text("Coef0"), sg.Input(0.0, key='coef0', enable_events=True)],
                  [sg.Text("Tolerancia"), sg.Input(1e-3, key='tol', enable_events=True)],
                  [sg.Text("C (regularización)"), sg.Input(1.0, key='C', enable_events=True)],
                  [sg.Text("Epsilon"), sg.Input(0.1, key='epsilon', enable_events=True)],
                  [sg.Check("Shrinking", key='shrinking', default=True)],
                  [sg.Input(r"C:\Users\tomasdfen\Downloads\test",key="-SAVE-",enable_events=True), sg.FolderBrowse()],
                  [sg.Button("OK", key="-GO-")]] 
        layout = [[sg.Column(left), sg.VSeparator(), sg.Column(right)]]

        window = sg.Window("Selección de parámetros: XGBoost", layout, modal=True, size=(None, None))
        while True:
                event, params = window.read()
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
                if event == "degree":
                    if not isInt(params['degree']):
                        window["degree"].update(background_color="#FF0000")
                        errors["degree"] = True
                    else:
                        window["degree"].update(background_color="#FFFFFF")
                        errors["degree"] = False
                        
                if event == "coef0":
                    if not all([c in "0123456789." for c in params["coef0"]]):
                        window["coef0"].update(background_color="#FF0000")
                        errors["coef0"] = True
                    else:
                        window["coef0"].update(background_color="#FFFFFF")
                        errors["coef0"] = False
                        
                if event == "tol":
                    if not all([c in "0123456789." for c in params["tol"]]):
                        window["tol"].update(background_color="#FF0000")
                        errors["tol"] = True
                    else:
                        window["tol"].update(background_color="#FFFFFF")
                        errors["tol"] = False
                        
                if event == "C":
                    if not all([c in "0123456789." for c in params["C"]]):
                        window["C"].update(background_color="#FF0000")
                        errors["C"] = True
                    else:
                        window["C"].update(background_color="#FFFFFF")
                        errors["C"] = False
                        
                if event == "epsilon":
                    if not all([c in "0123456789." for c in params["epsilon"]]):
                        window["epsilon"].update(background_color="#FF0000")
                        errors["epsilon"] = True
                    else:
                        window["epsilon"].update(background_color="#FFFFFF")
                        errors["epsilon"] = False
                        
                if event == "-GAMMA-TYPE-":
                    if params['-GAMMA-TYPE-'] not in ['scale', 'auto']:
                        window['-GAMMA-NUM-'].update(disabled = False, background_color = '#FFFFFF')
                        gamma = float(params["-GAMMA-NUM-"])
                    else:
                        window['-GAMMA-NUM-'].update(disabled=True, background_color = '#3A3A3A')
                        gamma = params['-GAMMA-TYPE-']
                        
                if event == "-GAMMA-NUM-":
                    if not all([c in "0123456789." for c in params["-GAMMA-NUM-"]]):
                        window["-GAMMA-NUM-"].update(background_color="#FF0000")
                        errors["-GAMMA-NUM-"] = True
                    else:
                        window["-GAMMA-NUM-"].update(background_color="#FFFFFF")
                        errors["-GAMMA-NUM-"] = False
                        if params["-GAMMA-NUM-"]:
                            gamma = float(params["-GAMMA-NUM-"])
                        else:
                            window["-GAMMA-NUM-"].update("0")
                if event == "-GO-":
                    if not "gamma" in params:
                        params["gamma"] = gamma
                    values['-START-'] = datetime.strptime(values['-START-'],"%d-%m-%Y")
                    values['-END-'] = datetime.strptime(values['-END-'],"%d-%m-%Y")
                    values["-FORWARD-"] = int(values["-FORWARD-"])
                    values["-STEP-"] = int(values["-STEP-"])
                    values["-ITER-"] = int(values["-ITER-"])
                    params["coef0"] = float(params["coef0"])
                    params["tol"] = float(params["tol"])
                    params["C"] = float(params["C"])
                    params["epsilon"] = float(params["epsilon"])
                    path_to_save = params["-SAVE-"]
                    x_data = pd.DataFrame()
                    for var in values['-LIST-']:
                        fstcol_name = data[var].columns[0]
                        fstcol = data[var][fstcol_name]
                        if values["-CONTINOUS-"]:
                            features = [i for i in range(int(values['-BACK-'])+1)]
                        elif values["-SELECTIVE-"]:
                            features = []
                            divs = values['-BACK-'].split(",")
                            for div in divs:
                                if isInt(div):
                                    features.append(int(div))
                                else:
                                    startDiv, endDiv = div.split("-")
                                    features += [i for i in range(int(startDiv), int(endDiv)+1)]
                        for x in features:
                            if x > 0:
                                x_data[f'{var}-{x}'] = fstcol.shift(x)
                            else:
                                x_data[var] = fstcol
                    x_data.dropna(inplace=True)
                    x_data = x_data.resample("H").last()
                    
                    y_data = data.pop((values['-TARGET-']))
                    y_data = y_data[max(features):]
                    
                    batches = []
                    for i in range(0, values['-ITER-']):
                        x_batch = x_data[(values['-START-']+timedelta(hours = values["-STEP-"]*i)):(values["-END-"]+timedelta(hours = values["-STEP-"]*i))][:-(values['-FORWARD-'])]
                        y_batch = y_data[(values['-START-']+timedelta(hours = values["-STEP-"]*i)):(values["-END-"]+timedelta(hours = values["-STEP-"]*i))][values['-FORWARD-']:]
                        batches.append((x_batch, y_batch))
            
                    res = []
                    performances = []
                    model = SVR()
                    if not os.path.exists(path_to_save):
                        os.mkdir(path_to_save)
                    for i in range(len(batches[:-1])):
                        batch = batches[i]
                        to_pred = batches[i+1][0][-(values["-STEP-"]):] 
                        real_pred = batches[i+1][1][-(values["-STEP-"]):]
                        model.fit(batch[0],batch[1].values.flatten())
                        preds = model.predict(to_pred)
                        res.append(preds)
                        performances.append(mean_absolute_error(real_pred, preds))
                    with open(os.path.join(path_to_save, "checkpoint.model"), 'wb') as f:
                        pickle.dump(model, f)
                    params.pop("-GAMMA-NUM-")
                    params.pop('-GAMMA-TYPE-')
                    break
    with open(os.path.join(path_to_save, "results.log"), "w+") as f:
        f.write(f"Resultados del entrenamiento con fecha: {datetime.now()}\n")
        if values["-XGBOOST-"]:
            f.write(f"Modelo: XGBoost \n")
        elif values["-SVM-"]:
            f.write(f"Modelo: SVR\n")
        params.pop("-SAVE-")
        params.pop("Browse")
        f.write(f"Parametros usados:\n")
        f.write(json.dumps(params)+"\n")
        f.write(f"Columnas usadas:\n")
        for column in x_data.columns:
            f.write(f"{column}\n")
        f.write(f"Rendimiento medio (mse): {np.mean(performances)}\n")
        f.write(f"Rendimiento por iteración (mse):\n")
        for i,p in enumerate(performances):
            f.write(f"\t{i} - {p}\n")
    subprocess.Popen(f'explorer //select, {path_to_save}')
    window.close()
    sg.Popup("Hecho")
    print(json.dumps(params))
    open_main_window(path)


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