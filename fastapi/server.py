from typing import Optional
import joblib
from fastapi import FastAPI, Request
from starlette.responses import Response
from pydantic import BaseModel
import pandas as pd
import numpy as np


class Item(BaseModel):
    uuid: str                                 
    account_amount_added_12_24m: Optional[int] = None             
    account_days_in_dc_12_24m: Optional[float] = None               
    account_days_in_rem_12_24m: Optional[float] = None              
    account_days_in_term_12_24m: Optional[float] = None             
    account_incoming_debt_vs_paid_0_24m: Optional[float] = None     
    account_status: Optional[float] = None                          
    account_worst_status_0_3m: Optional[float] = None               
    account_worst_status_12_24m: Optional[float] = None             
    account_worst_status_3_6m: Optional[float] = None               
    account_worst_status_6_12m: Optional[float] = None              
    age: Optional[int] = None                                       
    avg_payment_span_0_12m: Optional[float] = None                  
    avg_payment_span_0_3m: Optional[float] = None                   
    merchant_category: Optional[str] = None                      
    merchant_group: Optional[str] = None                          
    has_paid: Optional[bool] = None                                  
    max_paid_inv_0_12m: Optional[float] = None                      
    max_paid_inv_0_24m: Optional[float] = None                      
    name_in_email: Optional[str] = None                           
    num_active_div_by_paid_inv_0_12m: Optional[float] = None        
    num_active_inv: Optional[int] = None                             
    num_arch_dc_0_12m: Optional[int] = None                          
    num_arch_dc_12_24m: Optional[int] = None                         
    num_arch_ok_0_12m: Optional[int] = None                          
    num_arch_ok_12_24m: Optional[int] = None                         
    num_arch_rem_0_12m: Optional[int] = None                         
    num_arch_written_off_0_12m: Optional[float] = None              
    num_arch_written_off_12_24m: Optional[float] = None             
    num_unpaid_bills: Optional[int] = None                           
    status_last_archived_0_24m: Optional[int] = None                 
    status_2nd_last_archived_0_24m: Optional[int] = None             
    status_3rd_last_archived_0_24m: Optional[int] = None             
    status_max_archived_0_6_months: Optional[int] = None             
    status_max_archived_0_12_months: Optional[int] = None            
    status_max_archived_0_24_months: Optional[int] = None            
    recovery_debt: Optional[int] = None                              
    sum_capital_paid_account_0_12m: Optional[int] = None             
    sum_capital_paid_account_12_24m: Optional[int] = None            
    sum_paid_inv_0_12m: Optional[int] = None                         
    time_hours: Optional[float] = None                              
    worst_status_active_inv: Optional[float] = None  


app = FastAPI()

model = joblib.load('model_best.pkl')

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/prediction/")
async def get_prediction(input_pred: Item):
	input_dict = input_pred.dict()
	np_ar = np.array(list(input_dict.values()))
	df_pred = pd.DataFrame(np.transpose(np_ar.reshape(len(np_ar), 1)), columns=input_dict.keys())
	result = model.predict_proba(df_pred)
	return {"uuid": input_dict["uuid"], "pd": result[0][1]}

