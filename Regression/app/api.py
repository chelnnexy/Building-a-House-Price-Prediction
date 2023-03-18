from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pandas
from utils import Preprocessor
import pandas as pd



#Create API
app = FastAPI()

#Load GB Model
with open('gb.pkl','rb') as f:
    model = dill.load(f)

#Type Checking Class Through Pydantic
class ScoringItem(BaseModel):
    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs: float
    PostCode: str

#Create Api end Point
@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}
