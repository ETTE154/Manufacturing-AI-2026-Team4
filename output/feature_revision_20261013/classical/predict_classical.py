"""Reload new feature-revision models; fixed stored pre-origin context only."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,json
import joblib
import numpy as np
from retrain_classical import OUT,predict,idx_future,prediction_frame

def forecast_saved(model='selected',scenario='dayahead24'):
    base=OUT if scenario=='dayahead24' else OUT/'long745'
    lock=json.loads((base/'selection_lock.json').read_text(encoding='utf-8'))
    if model=='selected':model=lock['selected_model']
    bundle=joblib.load(base/model/'saved_model/model.joblib')
    h=bundle['forecast_context'];points=bundle['forecast_points']
    p=predict(bundle,h,points);idx=idx_future(h,points)
    return prediction_frame(model,scenario,'reloaded_future',0,idx[0],idx,np.full(points,np.nan),p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='selected',choices=['selected','xgboost','ridge','elasticnet','arimax','baseline'])
    p.add_argument('--scenario',default='dayahead24',choices=['dayahead24','long745'])
    p.add_argument('--output',required=True);args=p.parse_args()
    frame=forecast_saved(args.model,args.scenario);dest=Path(args.output)
    if OUT not in dest.resolve().parents:raise ValueError('Write output only within the new classical directory')
    dest.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(dest,index=False,encoding='utf-8-sig',float_format='%.17g')
    print(json.dumps({'output':str(dest),'rows':len(frame),'origin':str(frame.origin.iloc[0]),'model':str(frame.model.iloc[0]),'May_labels_read':False}))
if __name__=='__main__':main()
