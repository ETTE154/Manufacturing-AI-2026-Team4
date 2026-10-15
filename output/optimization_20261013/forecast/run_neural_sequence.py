from pathlib import Path
import os,sys,subprocess,time,json
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(r"E:\AI work\AWS_프로젝트")
OUT=ROOT/"output/optimization_20261013/forecast"
PYTHON=ROOT/"output/modeling/.venv/Scripts/python.exe"
RUNNER=OUT/"neural_hpo.py"
jobs=[("patchtst","dayahead24",True),("itransformer","dayahead24",False),("patchtst","long745",False),("itransformer","long745",False)]
env=os.environ.copy();env["PYTHONIOENCODING"]="utf-8";env["PYTHONDONTWRITEBYTECODE"]="1"
summary=[]
for model,scenario,resume in jobs:
    started=time.perf_counter()
    args=[str(PYTHON),str(RUNNER),"--model",model,"--scenario",scenario]
    if resume:args.append("--resume")
    print("NEURAL_JOB_START",model,scenario,"resume",resume,flush=True)
    log=OUT/f"hpo_{model}_{scenario}.log"
    with log.open("w",encoding="utf-8") as handle:
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",bufsize=1)
        for line in process.stdout:
            handle.write(line);handle.flush();print(line,end="",flush=True)
        result=process.wait()
    item={"model":model,"scenario":scenario,"returncode":result,"seconds":time.perf_counter()-started,"log":str(log)}
    summary.append(item)
    (OUT/"neural_run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if result:raise RuntimeError(f"Neural optimization job failed: {model}/{scenario}; inspect log")
    print("NEURAL_JOB_COMPLETE",json.dumps(item),flush=True)
print("ALL_NEURAL_OPTIMIZATIONS_COMPLETE",flush=True)

