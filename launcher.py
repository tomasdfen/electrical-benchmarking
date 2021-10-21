import os
import sys
import json
import pickle
import xgboost
import subprocess
import webbrowser
import numpy as np
import pandas as pd
from threading import Timer
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
plt.rcParams["figure.figsize"] = (8,5)
while True:
    try:
        from flask import Flask, render_template, request, redirect, session
        from flask.helpers import url_for, make_response

    except ModuleNotFoundError:
        print("Flask no instalado, se procede a instalar desde pip")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        continue
    break
while True:
    try:
        from flask_caching import Cache
    except ModuleNotFoundError:
        print("Flask-Caching no instalado, se procede a instalar desde pip")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask-Caching"])
        continue
    break

from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates")
if not os.path.exists("cache"):
    os.makedirs("cache")
    
app.config['UPLOAD_FOLDER'] = "cache"

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app.config.from_mapping(config)
cache = Cache(app)

data = None
vars = None
fmt = None
m_code = None

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def open_web_browser():
    webbrowser.open_new('http://localhost:5500/')

@app.route("/")
def init():
    return render_template("index.html")

@app.route("/load_dataset", methods = ["GET", "POST"])
def load_dataset():
    global data
    global vars
    global fmt
    error = ""
    if "dataset" not in request.files:
        error = "No hay fichero en la petición"
        return render_template("index.html", error = error)
    file = request.files["dataset"]
    if file.filename == "":
        error = "No se ha subido fichero"
        return render_template("index.html", error = error) 
    elif not os.path.splitext(file.filename)[1][1:] in ["csv", "xlsx"]:
        error = f"El tipo de archivo es '{os.path.splitext(file.filename)[1]}', el cual no es compatible con la aplicación"
        return render_template("index.html", error = error) 
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        fmt = request.form.get("format")
        date_parser = lambda x: datetime.strptime(x, fmt) if isinstance(x, str) else x
        if request.form.get("extension") == "xlsx-varioussheet":
            data = pd.read_excel(filename, index_col=0, parse_dates=True, sheet_name=None, date_parser=date_parser)
            vars = data.keys()
            for k in vars:
                data[k].index = data[k].index.tz_localize('UTC').tz_convert("ETC/GMT-2").tz_localize(None)
            
        return redirect("choose_params")
    
@app.route("/choose_params")
def choose_params():
    return render_template("exp_params.html", variables=list(vars))


@app.route("/submit_params", methods=["POST", "GET"])
def submit_params():
    selectedvars = request.form.getlist("selectedvars")
    incremental = bool(request.form.get("incremental"))
    savemodel = bool(request.form.get("savemodel"))
    start = datetime.strptime(request.form.get("start"), "%Y-%m-%d")
    end = datetime.strptime(request.form.get("end"), "%Y-%m-%d")
    forward = int(request.form.get("forward"))
    step = int(request.form.get("step")) * forward
    target = request.form.get("target")
    
    backs = []
    backtypes = []
    transformations = []
    for i in range(len(selectedvars)):
        backs.append(request.form.get(f"{i}-back"))
        backtypes.append(request.form.get(f"{i}-backtype"))
        transformations.append(request.form.get(f"{i}-transformation"))

    cache.set("selectedvars", selectedvars)
    cache.set("start", start)
    cache.set("target", target)
    cache.set("end", end)
    cache.set("step", step)
    cache.set("incremental", incremental)
    cache.set("savemodel", savemodel)
    cache.set("backs", backs)
    cache.set("forward", forward)
    cache.set("backtypes", backtypes)
    cache.set("transformations", transformations)
    
    if request.form.get("xgboost") and request.form.get("svm"):
        return render_template("model_params.html", models="xgboost_svm")
    elif request.form.get("svm"):
        return render_template("model_params.html", models="svm")
    elif request.form.get("xgboost"):
        return render_template("model_params.html", models="xgboost")

@app.route("/train/<model_code>", methods=["POST","GET"])
def train(model_code):
    global performances
    global img_paths
    global m_code
    m_code = model_code

    x_data = pd.DataFrame()
    max_col = 0
    for i,var in enumerate(cache.get("selectedvars")):
        var_df = pd.DataFrame()
        fstcol_name = data[var].columns[0]
        fstcol = data[var][fstcol_name]
        if cache.get("backtypes")[i] == "continous":
            features = [i for i in range(int(cache.get("backs")[i])+1)]
        elif cache.get("backtypes")[i] == "selective":
            features = []
            divs = cache.get("backs")[i].split(",")
            for div in divs:
                if isInt(div):
                    features.append(int(div))
                else:
                    startDiv, endDiv = div.split("-")
                    features += [i for i in range(int(startDiv), int(endDiv)+1)]
        for x in features:
            if x > 0:
                var_df[f'{var}-{x}'] = fstcol.shift(x)
            else:
                var_df[var] = fstcol

        if cache.get("transformations")[i] == "median":
            x_data[f"{var}-{cache.get('backs')[i]}-median"] = var_df.median(axis=1)
            if max_col < 1:
                max_col = 1 
        elif cache.get("transformations")[i] == "mean":
            x_data[f"{var}-{cache.get('backs')[i]}-mean"] = var_df.mean(axis=1)
            if max_col < 1:
                max_col = 1 
        else:
            x_data = pd.concat([x_data, var_df], axis=1)
            if max_col < max(features):
                max_col = max(features)            
    x_data.dropna(inplace=True)
    x_data = x_data.resample("H").last()
    x_data = x_data.loc[:,~x_data.columns.duplicated()]

    y_data = data[(cache.get("target"))]
    y_data = y_data[max_col:]
    
    batches = []
    if cache.get("incremental"):    
        for i in range(0, 16):
            x_batch = x_data[:(cache.get("end")+timedelta(hours = cache.get("step")*i))][:-(cache.get("forward"))]
            y_batch = y_data[:(cache.get("end")+timedelta(hours = cache.get("step")*i))][cache.get("forward"):]
            batches.append((x_batch, y_batch))
    else:
        for i in range(0, 16):
            x_batch = x_data[(cache.get("start")+timedelta(hours = cache.get("step")*i)):(cache.get("end")+timedelta(hours = cache.get("step")*i))][:-(cache.get("forward"))]
            y_batch = y_data[(cache.get("start")+timedelta(hours = cache.get("step")*i)):(cache.get("end")+timedelta(hours = cache.get("step")*i))][cache.get("forward"):]
            batches.append((x_batch, y_batch))
    print(list(map(lambda x: f"{x[0].shape} vs {x[1].shape}", batches)))

    for model_id in model_code.split("_"):
        res = []
        params = {}
        img_paths = []
        performances_mae = []
        performances_mse = []
        model_path = os.path.join("static",str(datetime.today().date()),model_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        if "xgboost" == model_id:
            params["objective"] = request.form.get("objective")
            params["base_score"] = request.form.get("base_score")
            
            batch = batches[0]
            to_pred = batches[1][0][-(cache.get("step")):] 
            real_pred = batches[1][1][-(cache.get("step")):]
            
            train_matrix = xgboost.DMatrix(batch[0],batch[1])
            pred_matrix = xgboost.DMatrix(to_pred, real_pred)
            model = xgboost.train(params, train_matrix)
            preds = model.predict(pred_matrix)
            res.append(preds)

            performances_mae.append(mean_absolute_error(real_pred, preds))
            performances_mse.append(mean_squared_error(real_pred, preds))
            plt.plot(real_pred.index, real_pred, label="Datos reales")
            plt.plot(real_pred.index,preds, label="Predcciones")
            plt.legend(loc="upper left")
            plt.savefig(os.path.join(model_path,f"{i}.svg"))
            img_paths.append(os.path.join(model_path,f"{i}.svg"))
            plt.clf()

            for i in range(1,len(batches)-1):
                batch = batches[i]
                print(batch[0].index)
                to_pred = batches[i+1][0][-(cache.get("step")):] 
                real_pred = batches[i+1][1][-(cache.get("step")):]
                
                train_matrix = xgboost.DMatrix(batch[0], batch[1])
                pred_matrix = xgboost.DMatrix(to_pred, real_pred)
                if cache.get("savemodel"):
                    model.save_model(os.path.join(model_path, f"checkpoint_{i}.model"))
                    model = xgboost.train(params, train_matrix, xgb_model=model)
                else:
                    model.save_model(os.path.join(model_path,f"checkpoint.model"))
                    model = xgboost.train(params, train_matrix)
                preds = model.predict(pred_matrix)
                performances_mae.append(mean_absolute_error(real_pred, preds))
                performances_mse.append(mean_squared_error(real_pred, preds))
                
                plt.plot(real_pred.index, real_pred, label="Datos reales")
                plt.plot(real_pred.index,preds, label="Predcciones")
                plt.legend(loc="upper left")
                plt.savefig(os.path.join(model_path,f"{i}.svg"))
                img_paths.append(os.path.join(model_path,f"{i}.svg"))
                plt.clf()
        
        elif "svm" == model_id:
            model = SVR()
            for i in range(len(batches)):
                batch = batches[i]
                to_pred = batches[i+1][0][-(cache.get("step")):] 
                real_pred = batches[i+1][1][-(cache.get("step")):]
                model.fit(batch[0],batch[1].values.flatten())
                preds = model.predict(to_pred)
                res.append(preds)
                performances_mae.append(mean_absolute_error(real_pred, preds))
                performances_mse.append(mean_squared_error(real_pred, preds))
                
                plt.plot(real_pred.index, real_pred, label="Datos reales")
                plt.plot(real_pred.index,preds, label="Predcciones")
                plt.legend(loc="upper left")
                plt.savefig(os.path.join(model_path,f"{i}.svg"))
                img_paths.append(os.path.join(model_path,f"{i}.svg"))
                plt.clf()
                
            with open(os.path.join(model_path, "checkpoint.model"), 'wb') as f:
                pickle.dump(model, f)


        cache.set(f"img_path_{model_id}", img_paths)
        cache.set(f"performance_mse_{model_id}", performances_mse)
        cache.set(f"performance_mae_{model_id}", performances_mae)
        cache.set(f"params_{model_id}", params)
        cache.set(f"preds_{model_id}", res)
      
        
    return redirect(url_for("results"))

@app.route("/results")
def results():
    keys=[]
    for model_id in m_code.split("_"):
        keys.append(f"img_path_{model_id}")
        keys.append(f"performance_mse_{model_id}")
        keys.append(f"performance_mae_{model_id}")
        keys.append(f"params_{model_id}")
        keys.append(f"preds_{model_id}")
    d = cache.get_dict(*keys)
    r = make_response(render_template("results.html",model_code = m_code, **d ))
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
    
@app.route("/downloadPerformance/<model>")
def downloadPerformance(model):
    path = os.path.join("static",str(datetime.today().date()), f"performances-{model}.log")
    with open(path, "w+") as f:
            f.write(f"Resultados del entrenamiento con fecha: {datetime.now()}\n")
            f.write(f"Parametros usados:\n")
            f.write(json.dumps(cache.get(f"params_{model}"))+"\n")
            f.write(f"Rendimiento medio (MSE): {np.mean(cache.get(f'performance_mse_{model}'))}\n")
            f.write(f"Rendimiento medio (MAE): {np.mean(cache.get(f'performance_mae_{model}'))}\n")
            f.write(f"Rendimiento por iteración (MSE):\n")
            for i,p in enumerate(cache.get(f"performance_mse_{model}")):
                f.write(f"\t{i} - {p}\n")
            f.write(f"Rendimiento por iteración (MAE):\n")
            for i,p in enumerate(cache.get(f"performance_mae_{model}")):
                f.write(f"\t{i} - {p}\n")
    r = make_response(path)
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r
    

@app.route("/downloadPreds/<model>")
def downloadPreds(model):
    path = os.path.join("static",str(datetime.today().date()), f"preds_{model}.csv")
    aux = pd.DataFrame(cache.get(f"preds_{model}"), columns = [f"{cache.get('target')}+{i+cache.get('forward')}" if i+cache.get('forward') > 0 else cache.get('target') for i in range(cache.get("step"))])
    aux.to_csv(path, index=False)
    r = make_response(path)
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r
        
# @app.errorhandler(Exception)
# def handle_exception(e):
#     return render_template("index.html", error=f"Ocurrio un error: {e}") 

if __name__ == '__main__':
    #Timer(1, open_web_browser).start()
    app.run(host="localhost", port=5500, debug=False)
    
    