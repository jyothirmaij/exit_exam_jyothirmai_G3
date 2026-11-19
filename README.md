# Enterprise Revenue Estimator

A three-page Flask application that predicts annual enterprise revenue from the
provided `raw_data/G3.csv` dataset. The supplied notebook and saved model are
for **net income**, so `train_model.py` trains a separate model whose target is
`revenue`.

## Project structure

```text
exit_jyothirmai/
|-- app.py
|-- model_utils.py
|-- train_model.py
|-- requirements.txt
|-- render.yaml
|-- raw_data/
|   `-- G3.csv
|-- models/                  # created by train_model.py
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- predict.html
|   `-- result.html
`-- static/
    `-- style.css
```

## Run locally on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

If PowerShell blocks virtual-environment activation, run the commands below
instead of activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000` in a browser. The app has an overview page, an
enterprise input form, and a prediction result page.

## Deploy on Render

1. Put this project and `raw_data/G3.csv` in a GitHub repository. Do not upload
   private or restricted data.
2. In Render, choose **New +** → **Blueprint** and connect that repository.
3. Render reads `render.yaml`: it installs `requirements.txt`, trains the model,
   then starts the Flask app with Gunicorn.
4. Open the Render URL after deployment completes.

The generated model files in `models/` are intentionally not committed; the
training script creates them during the local setup or Render build.

## Important limitations

- The app estimates `revenue` (not net income). Its model is trained when
  `python train_model.py` runs.
- The source data contains multiple currencies. The selected revenue currency
  is a model input; the application does **not** convert currencies. Enter
  net income and market capitalization in the currencies selected for those
  fields.
- Negative or missing revenue records are excluded from training. Predictions
  are statistical estimates, not audited financial figures or financial advice.
