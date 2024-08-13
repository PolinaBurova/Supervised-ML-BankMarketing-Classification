import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from pydantic import BaseModel
import json
import pandas as pd



from pydantic import BaseModel, Field

class Request(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


# create an app
app = fastapi.FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# function that is "called" when initiated
@app.on_event("startup")
async def startup_event():
# we set the tracking URI to point to the database file
# where the metadata of our registered models is stored
    with open(r'C:\Users\polin\OneDrive\Documents\GitHub\Supervised-ML-BankPrediction-Classification\config\app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(config["tracking_uri"])


# predict endpoint that will be called to receive requests with inputs for the model
# and will return the model's prediction in the response
@app.post("/predict")
async def root(input: Request):  
    # read app's config
    with open(r'C:\Users\polin\OneDrive\Documents\GitHub\Supervised-ML-BankPrediction-Classification\config\app.json') as f:
        config = json.load(f)
# we load the registered model
# according to the model name and model version read from the config
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{config['model_name']}/{config['model_version']}"
    )
    # we build a DataFrame with the model inputs that we received in the request
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
    # we call the model's predict function and obtain its prediction
    prediction = model.predict(input_df)
    # we return a dictionary as the response with the prediction associated with the key "prediction"
    return {"prediction": prediction.tolist()[0]}


uvicorn.run(app=app, port=5003)
