"""Standalone frozen ARIMA/XGBoost inference; no raw data or May labels needed."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUT,prediction_frame
from arima_xgb import predict_saved_arima,predict_saved_xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['arima','xgboost'])
    parser.add_argument('--points',type=int,default=745,help='Number of future hourly values, 1 through 745 inclusive')
    parser.add_argument('--n-jobs',type=int,default=4)
    parser.add_argument('--output',default=None)
    args = parser.parse_args()
    if not 1<=args.points<=745:
        raise ValueError('The locked forecast supports 1 through 745 future hourly values')
    artifacts = OUT/'saved_models'
    extension = '.pkl' if args.model=='arima' else '.json'
    model_path = artifacts/f'final_{args.model}{extension}'
    metadata_path = artifacts/f'final_{args.model}_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    origin = pd.Timestamp(metadata['history_end'])+pd.Timedelta(hours=1)
    future = pd.date_range(origin,periods=args.points,freq='h')
    if args.model=='arima':
        path = predict_saved_arima(model_path,metadata_path,future)
    else:
        path = predict_saved_xgb(model_path,metadata_path,future,n_jobs=args.n_jobs)
    output = Path(args.output) if args.output else OUT/f'{args.model}_saved_inference.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    prediction_frame(args.model,'saved_inference',origin,future,np.repeat(np.nan,args.points),path).to_csv(output,index=False,encoding='utf-8-sig')
    print(json.dumps({'model':args.model,'model_path':str(model_path),'output':str(output),
                      'forecast_origin':str(origin),'forecast_end':str(future[-1]),'points':args.points,
                      'labels_read':False},indent=2))

if __name__=='__main__':
    main()
