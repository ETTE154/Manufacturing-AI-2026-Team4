"""Replay frozen Apr30 hourly forecasts without target/May data access."""
from pathlib import Path
import sys,argparse,json
sys.dont_write_bytecode=True
import joblib
import numpy as np
from train_hourly import OUT,predict,prediction_frame

def forecast_saved(model='selected'):
    lock=json.loads((OUT/'selection_lock.json').read_text(encoding='utf-8'))
    if model=='selected':model=lock['selected_model']
    bundle=joblib.load(OUT/model/'saved_model/model.joblib');idx,p=predict(bundle)
    return prediction_frame(model,'hourly_multires745','saved_model_replay',0,idx[0],idx,np.full(len(idx),np.nan),p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['selected','xgboost','ridge','arima','baseline'],default='selected')
    p.add_argument('--output',required=True);args=p.parse_args();dest=Path(args.output)
    if OUT not in dest.resolve().parents:raise ValueError('Write only within new hourly directory')
    dest.parent.mkdir(parents=True,exist_ok=True);frame=forecast_saved(args.model)
    frame.to_csv(dest,index=False,encoding='utf-8-sig',float_format='%.17g')
    print(json.dumps({'output':str(dest),'model':str(frame.model.iloc[0]),'points':len(frame),'origin':str(frame.origin.iloc[0]),'May_source_read':False}))
if __name__=='__main__':main()
