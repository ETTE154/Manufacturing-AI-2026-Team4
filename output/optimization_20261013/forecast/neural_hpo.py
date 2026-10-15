"""Budgeted official-core neural HPO on aggregate univariate hourly history.

Old data/models/vendors are read-only. New artifacts live only in this directory.
This is local compact architecture training, not paper benchmark reproduction.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from types import SimpleNamespace
import argparse,copy,hashlib,json,random,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from opt_common import (ROOT,OUT,SEED,THREADS,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS,
                        LONG_VALID_ORIGIN,LONG_TEST_ORIGIN,load_target,history,
                        future,actual,metrics,prediction_frame,day_summary,
                        require_source_approval,write_json,protocol_manifest)

OLD=ROOT/"output/modeling/forecast"
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/"patchtst"))
from patchtst_vendor.models.PatchTST import Model as PatchCore
from itransformer.vendor.model.iTransformer import Model as InvertedCore

MAX_EPOCHS=24
PATIENCE=4
BATCH_SIZE=32
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
DAY_CONFIGS=[
 {"context":24,"d_model":32,"layers":1,"dropout":0.0},
 {"context":72,"d_model":32,"layers":1,"dropout":0.1},
 {"context":168,"d_model":32,"layers":1,"dropout":0.1},
 {"context":336,"d_model":64,"layers":1,"dropout":0.1},
 {"context":720,"d_model":32,"layers":1,"dropout":0.3},
 {"context":168,"d_model":64,"layers":2,"dropout":0.1},
]
LONG_CONFIGS=[
 {"context":24,"d_model":32,"layers":1,"dropout":0.0},
 {"context":72,"d_model":32,"layers":1,"dropout":0.1},
 {"context":168,"d_model":32,"layers":1,"dropout":0.1},
 {"context":336,"d_model":32,"layers":1,"dropout":0.1},
 {"context":512,"d_model":32,"layers":1,"dropout":0.3},
 {"context":168,"d_model":64,"layers":2,"dropout":0.1},
]

def seed_everything():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)

def architecture(family,config,points):
 if family=="patchtst":
  return dict(enc_in=1,seq_len=config["context"],pred_len=points,
     d_model=config["d_model"],d_ff=config["d_model"]*2,e_layers=config["layers"],
     n_heads=4,dropout=config["dropout"],fc_dropout=config["dropout"],
     head_dropout=config["dropout"],individual=False,patch_len=16,stride=8,
     padding_patch="end",revin=True,affine=True,subtract_last=False,
     decomposition=False,kernel_size=25,training_loss=config.get("loss","MSE"))
 return dict(seq_len=config["context"],pred_len=points,d_model=config["d_model"],
     d_ff=config["d_model"]*2,e_layers=config["layers"],n_heads=4,
     dropout=config["dropout"],output_attention=False,use_norm=True,
     embed="timeF",freq="h",class_strategy="projection",factor=5,activation="gelu",training_loss=config.get("loss","MSE"))

def build(family,cfg):
 return (PatchCore if family=="patchtst" else InvertedCore)(SimpleNamespace(**cfg)).cpu()

class Windows(Dataset):
 def __init__(self,values,context,points):
  self.values=torch.tensor(np.asarray(values,dtype=np.float32)).view(-1,1)
  self.context=context;self.points=points
  self.origins=np.arange(context,len(values)-points+1)
  if len(self.origins)==0:raise ValueError("No fully observed supervised window")
  assert self.origins[-1]+points-1==len(values)-1
 def __len__(self):return len(self.origins)
 def __getitem__(self,index):
  origin=int(self.origins[index])
  return self.values[origin-self.context:origin],self.values[origin:origin+self.points]

def forward(family,model,x):
 return model(x) if family=="patchtst" else model(x,None,None,None)

def setup_fit(family,hist,cfg):
 seed_everything()
 values=hist.to_numpy(dtype=float);mean=float(values.mean());scale=float(values.std())
 if scale<=0 or not np.isfinite(scale):raise ValueError("Invalid training scale")
 ds=Windows((values-mean)/scale,cfg["seq_len"],cfg["pred_len"])
 loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,
         generator=torch.Generator().manual_seed(SEED))
 model=build(family,cfg)
 opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
 return model,opt,loader,mean,scale,ds

def epoch(family,model,opt,loader,loss_name="MSE"):
 model.train();total=0.;rows=0
 for x,y in loader:
  opt.zero_grad(set_to_none=True);pred=forward(family,model,x)
  if pred.shape!=y.shape:raise ValueError("Training label alignment error")
  loss=(pred-y).abs().mean() if loss_name=="L1" else (pred-y).pow(2).mean()
  if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  total+=float(loss.detach())*len(x);rows+=len(x)
 return total/rows

def infer(family,model,hist,cfg,mean,scale):
 model.eval()
 values=((hist.to_numpy(dtype=float)[-cfg["seq_len"]:]-mean)/scale).astype(np.float32)
 x=torch.from_numpy(values).view(1,cfg["seq_len"],1)
 with torch.no_grad():out=forward(family,model,x)
 if tuple(out.shape)!=(1,cfg["pred_len"],1):raise ValueError("Unexpected output shape")
 pred=out[0,:,0].numpy().astype(np.float64)*scale+mean
 if not np.isfinite(pred).all():raise ValueError("Nonfinite prediction")
 return pred

def choose_epochs(family,hist,cfg,scenario):
 holdout=168 if scenario=="dayahead24" else 745
 fit_hist=hist.iloc[:-holdout]
 tail=hist.iloc[-holdout:]
 model,opt,loader,mean,scale,ds=setup_fit(family,fit_hist,cfg)
 assert fit_hist.index[-1]<tail.index[0]
 assert ds.origins[-1]+cfg["pred_len"]-1<len(fit_hist)
 eval_origins=pd.date_range(tail.index[0],periods=7,freq="D") if scenario=="dayahead24" else [tail.index[0]]
 best=float("inf");best_epoch=1;stale=0;logs=[];started=time.perf_counter()
 for step in range(1,MAX_EPOCHS+1):
  loss=epoch(family,model,opt,loader,cfg.get("training_loss","MSE"));errors=[]
  for origin in eval_origins:
   observed=history(hist,origin)
   idx=future(origin,cfg["pred_len"])
   pred=infer(family,model,observed,cfg,mean,scale)
   errors.append(metrics(actual(hist,idx),pred)["mae_kwh"])
  value=float(np.mean(errors))
  logs.append({"epoch":step,"training_loss_standardized":loss,"training_loss_name":cfg.get("training_loss","MSE"),"inner_chronological_mae_kwh":value})
  if value<best-1e-7:best=value;best_epoch=step;stale=0
  else:stale+=1
  if stale>=PATIENCE:break
 return best_epoch,{"inner_fitting_hours":len(fit_hist),"inner_fitting_end":str(fit_hist.index[-1]),
    "inner_tail_origin":str(tail.index[0]),"inner_tail_end":str(tail.index[-1]),
    "inner_holdout_hours":holdout,"inner_origins":len(eval_origins),"training_windows":len(ds),
    "training_tail_target_overlap":False,"best_epoch":best_epoch,"best_mae_kwh":best,
    "epochs_run":len(logs),"logs":logs,"seconds":time.perf_counter()-started}

def save_model(path,family,model,cfg,mean,scale,hist,epochs):
 payload={"family":family,"state_dict":model.state_dict(),"architecture":cfg,
     "normalization_mean":mean,"normalization_scale":scale,"seed":SEED,"epochs":int(epochs),
     "input":"Aggregate univariate provisional13-module hourly energy",
     "history_start":str(hist.index[0]),"history_end":str(hist.index[-1]),"history_hours":len(hist),
     "target_sha256":require_source_approval()["target_sha256"],
     "forecast_context":hist.to_numpy(dtype=float)[-cfg["seq_len"]:].tolist(),
     "forecast_context_time":[str(t) for t in hist.index[-cfg["seq_len"]:]]}
 path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
 return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
         "parameter_count":sum(x.numel() for x in model.parameters()),"epochs":epochs,
         "family":family,"architecture":cfg,"normalization_mean":mean,"normalization_scale":scale,
         "forecast_context_in_checkpoint":True,
         "history_hours":len(hist),"history_end":str(hist.index[-1])}

def reload(path):
 torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)
 saved=torch.load(path,map_location="cpu",weights_only=True)
 model=build(saved["family"],saved["architecture"]);model.load_state_dict(saved["state_dict"]);model.eval()
 return model,saved

def refit(family,hist,cfg,epochs,path):
 model,opt,loader,mean,scale,ds=setup_fit(family,hist,cfg);started=time.perf_counter();logs=[]
 for step in range(1,epochs+1):logs.append(epoch(family,model,opt,loader,cfg.get("training_loss","MSE")))
 record=save_model(path,family,model,cfg,mean,scale,hist,epochs)
 record.update({"fit_seconds":time.perf_counter()-started,"training_windows":len(ds),
                "maximum_supervised_label":str(hist.index[-1]),"training_loss_logs":logs,"training_loss_name":cfg.get("training_loss","MSE")})
 return model,mean,scale,record

def path_frame(family,scenario,split,fold,y,origin,model,cfg,mean,scale):
 h=history(y,origin);idx=future(origin,cfg["pred_len"])
 p=infer(family,model,h,cfg,mean,scale)
 return prediction_frame(family,scenario,split,fold,origin,idx,actual(y,idx),p)

def score_paths(frame,scenario):
 if scenario=="dayahead24":return day_summary(frame)
 score=metrics(frame.actual_kwh,frame.predicted_kwh)
 tail=frame[frame.horizon_offset_hours>=240]
 return {"P745":score,"LateP505":metrics(tail.actual_kwh,tail.predicted_kwh)}

def selection_mae(summary,scenario):
 return summary["mean_origin_path_mae_kwh"] if scenario=="dayahead24" else summary["P745"]["mae_kwh"]

def vendor_metadata(family):
 path=OLD/("patchtst/provenance.json" if family=="patchtst" else "itransformer/vendor/provenance.json")
 obj=json.loads(path.read_text(encoding="utf-8"))
 return {"source":str(path),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"record":obj,
   "scope":"Pinned official core, compact local retraining. Namespace-only import changes; unused iTransformer Reformer import is lazy. No published benchmark reproduction.",
   "iTransformer_univariate_limit":"One variate token: no cross-channel attention advantage is demonstrated." if family=="itransformer" else None}

def run(family,scenario,resume=False):
 run_started=time.perf_counter();seed_everything();approval=require_source_approval();y=load_target()
 dest=OUT/scenario/family;dest.mkdir(parents=True,exist_ok=True)
 points=24 if scenario=="dayahead24" else 745
 configs=list(DAY_CONFIGS if scenario=="dayahead24" else LONG_CONFIGS)
 configs.append({"context":168,"d_model":32,"layers":1,"dropout":0.1,"loss":"L1"})
 budget={"configs":len(configs),"MSE_candidates":6,"extra_L1_candidate":1,"seed":SEED,"threads":THREADS,"maximum_epochs":MAX_EPOCHS,"early_stop_patience":PATIENCE,
     "batch_size":BATCH_SIZE,"optimizer":"AdamW","learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,
     "loss":"Six MSE and one L1 candidate; same train-only global normalization and validation selection","context_candidates":[c["context"] for c in configs],
     "versions":{name:importlib.metadata.version(name) for name in ["torch","numpy","pandas"]},
     "claim":"Budgeted local validation search; not a guarantee of global optimality. No internal-test or May tuning."}
 write_json(dest/"search_plan.json",{"family":family,"scenario":scenario,"configs":configs,"budget":budget,"source_approval":approval,
     "protocol":protocol_manifest(y),"vendor":vendor_metadata(family)})
 candidate_results=[];candidate_frames=[]
 for ci,config in enumerate(configs,1):
  cfg=architecture(family,config,points);frames=[];fitlogs=[]
  starts=DAY_FOLD_STARTS if scenario=="dayahead24" else [LONG_VALID_ORIGIN]
  previous=dest/f"candidate{ci}.json"
  reusable=json.loads(previous.read_text(encoding="utf-8")) if resume and previous.exists() else None
  if reusable is not None and reusable["config"]==config:
   for fi,start in enumerate(starts,1):
    fitlog=reusable["fold_fits"][fi-1]
    model,saved=reload(Path(fitlog["refit"]["path"]))
    if saved["target_sha256"]!=approval["target_sha256"] or saved["history_end"]!=str(history(y,start).index[-1]):
     raise ValueError("Saved CV fit source/history differs from current declared fold")
    origins=pd.date_range(start,periods=14,freq="D") if scenario=="dayahead24" else [start]
    for origin in origins:frames.append(path_frame(family,scenario,"validation",fi,y,origin,model,saved["architecture"],saved["normalization_mean"],saved["normalization_scale"]))
    fitlogs.append({**fitlog,"reused_existing_saved_CV_fit":True})
  else:
   for fi,start in enumerate(starts,1):
    hist=history(y,start);epochs,inner=choose_epochs(family,hist,cfg,scenario)
    path=dest/"checkpoints"/f"candidate{ci}_fold{fi}.pt"
    model,mean,scale,fit=refit(family,hist,cfg,epochs,path)
    origins=pd.date_range(start,periods=14,freq="D") if scenario=="dayahead24" else [start]
    for origin in origins:frames.append(path_frame(family,scenario,"validation",fi,y,origin,model,cfg,mean,scale))
    fitlogs.append({"fold":fi,"fit_origin":str(start),"epochs":epochs,"inner":inner,"refit":fit})
  frame=pd.concat(frames,ignore_index=True);summary=score_paths(frame,scenario)
  record={"candidate":ci,"config":config,"architecture":cfg,"validation":summary,"fold_fits":fitlogs}
  candidate_results.append(record);candidate_frames.append(frame)
  write_json(dest/f"candidate{ci}.json",record)
  print("CANDIDATE",family,scenario,ci,"validation_MAE",selection_mae(summary,scenario),flush=True)
 winner=min(range(len(configs)),key=lambda i:selection_mae(candidate_results[i]["validation"],scenario))
 chosen=candidate_results[winner];cfg=chosen["architecture"]
 validation=candidate_frames[winner]
 validation.to_csv(dest/"validation_predictions.csv",index=False,float_format="%.17g")
 # Independently reload selected final validation fold and reproduce its stored path.
 validation_reload=[]
 for fitlog in chosen["fold_fits"]:
  vm,vs=reload(Path(fitlog["refit"]["path"]))
  subset=validation[validation.fold==fitlog["fold"]]
  for origin,rows in subset.groupby("origin",sort=False):
   check=infer(family,vm,history(y,pd.Timestamp(origin)),vs["architecture"],vs["normalization_mean"],vs["normalization_scale"])
   wanted=rows.sort_values("horizon_offset_hours").predicted_kwh.to_numpy()
   difference=float(np.abs(check-wanted).max());np.testing.assert_allclose(check,wanted,rtol=0,atol=1e-8)
   validation_reload.append({"origin":str(origin),"fold":fitlog["fold"],"max_absdiff_kwh":difference})
 reload_diff=max(r["max_absdiff_kwh"] for r in validation_reload)
 lock={"family":family,"scenario":scenario,"selected_candidate":winner+1,"frozen_config":chosen["config"],
    "architecture":cfg,"selection":"Lowest70-origin mean path MAE" if scenario=="dayahead24" else "LowestFeb25 P745 validation MAE",
    "selection_uses_internal_test_labels":False,"selection_uses_may_labels":False,
    "validation":chosen["validation"],"validation_reload_max_absdiff":reload_diff,
    "validation_reload_origins":len(validation_reload),"budget":budget,
    "provenance":vendor_metadata(family),"target_sha256":approval["target_sha256"]}
 write_json(dest/"selection_lock.json",lock)
 if scenario=="dayahead24":
  test_origin=pd.Timestamp("2025-04-01");test_hist=history(y,test_origin)
  frozen_epochs,test_inner=choose_epochs(family,test_hist,cfg,scenario)
  lock["frozen_epochs"]=frozen_epochs;lock["pretest_epoch_choice"]=test_inner
  lock["final_epoch_policy"]="March training-only inner tail selects test epochs; the same epoch count is frozen for full3600 refit. April test scores never select an epoch."
  write_json(dest/"selection_lock.json",lock)
  model,mean,scale,test_fit=refit(family,test_hist,cfg,frozen_epochs,dest/"checkpoints/internal_test.pt")
  tests=[path_frame(family,scenario,"test",0,y,origin,model,cfg,mean,scale) for origin in APRIL_TEST_ORIGINS]
 else:
  frozen_epochs=chosen["fold_fits"][0]["epochs"];lock["frozen_epochs"]=frozen_epochs
  write_json(dest/"selection_lock.json",lock)
  test_hist=history(y,LONG_TEST_ORIGIN)
  model,mean,scale,test_fit=refit(family,test_hist,cfg,frozen_epochs,dest/"checkpoints/internal_test.pt")
  tests=[path_frame(family,scenario,"test",0,y,LONG_TEST_ORIGIN,model,cfg,mean,scale)]
 test_frame=pd.concat(tests,ignore_index=True)
 test_frame.to_csv(dest/"test_predictions.csv",index=False,float_format="%.17g")
 test_summary=score_paths(test_frame,scenario)
 tm,ts=reload(dest/"checkpoints/internal_test.pt");test_reload=[]
 for checked_origin,rows in test_frame.groupby("origin",sort=False):
  checked=infer(family,tm,history(y,pd.Timestamp(checked_origin)),ts["architecture"],ts["normalization_mean"],ts["normalization_scale"])
  wanted=rows.sort_values("horizon_offset_hours").predicted_kwh.to_numpy()
  difference=float(np.abs(checked-wanted).max());np.testing.assert_allclose(checked,wanted,rtol=0,atol=1e-8)
  test_reload.append({"origin":str(checked_origin),"max_absdiff_kwh":difference})
 fm,mean,scale,final_fit=refit(family,y,cfg,frozen_epochs,dest/"saved_model.pt")
 origin=y.index[-1]+pd.Timedelta(hours=1);idx=future(origin,points)
 pred=infer(family,fm,y,cfg,mean,scale)
 reloaded,saved=reload(dest/"saved_model.pt")
 recheck=infer(family,reloaded,y,cfg,saved["normalization_mean"],saved["normalization_scale"])
 np.testing.assert_allclose(pred,recheck,rtol=0,atol=1e-8)
 final_fit["reload_max_absdiff_kwh"]=float(np.abs(pred-recheck).max())
 final_fit["repeat_inference_max_absdiff_kwh"]=float(np.abs(recheck-infer(family,reloaded,y,cfg,saved["normalization_mean"],saved["normalization_scale"])).max())
 prediction_frame(family,scenario,"future",0,origin,idx,np.full(points,np.nan),pred).to_csv(dest/"future_path.csv",index=False,float_format="%.17g")
 results={"status":"complete","model":family,"scenario":scenario,"frozen_config":chosen["config"],"frozen_epochs":frozen_epochs,
     "validation":chosen["validation"],"test":test_summary,"selected_candidate":winner+1,"attempted_configs":len(configs),
     "candidate_results":candidate_results,"test_fit":test_fit,"saved_model":final_fit,
     "runtime":{"seconds_this_invocation":time.perf_counter()-run_started,
       "unique_candidate_CV_training_seconds":sum(f["inner"]["seconds"]+f["refit"]["fit_seconds"] for c in candidate_results for f in c["fold_fits"]),
       "candidate_count":len(configs),"validation_fit_count":sum(len(c["fold_fits"]) for c in candidate_results),
       "validation_origin_count_per_candidate":70 if scenario=="dayahead24" else 1,
       "test_origin_count":29 if scenario=="dayahead24" else 1,
       "inner_epochs_run_total":sum(f["inner"]["epochs_run"] for c in candidate_results for f in c["fold_fits"]),
       "refit_epochs_total":sum(f["refit"]["epochs"] for c in candidate_results for f in c["fold_fits"]),
       "reused_candidate_fits":sum(int(f.get("reused_existing_saved_CV_fit",False)) for c in candidate_results for f in c["fold_fits"]),
       "threads":THREADS},
     "validation_reload_max_absdiff_kwh":reload_diff,"validation_reload_records":validation_reload,
     "test_reload_max_absdiff_kwh":max(r["max_absdiff_kwh"] for r in test_reload),"test_reload_records":test_reload,
     "future_origin":str(origin),"future_end":str(idx[-1]),
     "future_path_rows":points,"may_labels_read":False,"input":"Aggregate univariate only",
     "iTransformer_one_token_limit":family=="itransformer","selection_lock":str(dest/"selection_lock.json"),
     "limitations":["Budgeted local search cannot guarantee global optimum","April and May were viewed in prior project EDA; this is a declared new post-EDA lock","Unit and meter hierarchy are provisional source specifications"]}
 write_json(dest/"results.json",results)
 print("NEURAL_COMPLETE",family,scenario,"valid",selection_mae(results["validation"],scenario),
       "test",selection_mae(results["test"],scenario),"epochs",frozen_epochs,flush=True)

def predict_saved_checkpoint(path):
 model,saved=reload(Path(path))
 observed=pd.Series(saved["forecast_context"],index=pd.to_datetime(saved["forecast_context_time"]),dtype=float)
 predicted=infer(saved["family"],model,observed,saved["architecture"],saved["normalization_mean"],saved["normalization_scale"])
 idx=future(pd.Timestamp(saved["history_end"])+pd.Timedelta(hours=1),saved["architecture"]["pred_len"])
 return idx,predicted,saved

def benchmark(family):
 require_source_approval();y=load_target();hist=history(y,DAY_FOLD_STARTS[-1]).iloc[:-168]
 cfg=architecture(family,DAY_CONFIGS[2],24)
 model,opt,loader,mean,scale,ds=setup_fit(family,hist,cfg)
 started=time.perf_counter();loss=epoch(family,model,opt,loader);elapsed=time.perf_counter()-started
 record={"family":family,"context":168,"points":24,"windows":len(ds),"parameters":sum(p.numel() for p in model.parameters()),
    "epoch_seconds":elapsed,"mse_standardized":loss,"threads":THREADS,"maximum_epochs":MAX_EPOCHS,"patience":PATIENCE}
 write_json(OUT/f"benchmark_{family}.json",record)
 print(json.dumps(record),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",choices=["patchtst","itransformer"],required=True)
 p.add_argument("--scenario",choices=["dayahead24","long745"]);p.add_argument("--benchmark",action="store_true");p.add_argument("--resume",action="store_true")
 args=p.parse_args()
 if args.benchmark:benchmark(args.model)
 elif args.scenario:run(args.model,args.scenario,args.resume)
 else:p.error("--scenario or --benchmark is required")

