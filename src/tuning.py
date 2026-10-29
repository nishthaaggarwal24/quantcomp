"""Validation-only search over circuit structure (19 configurations)."""
from dataclasses import asdict
import itertools, json
from pathlib import Path
import numpy as np
from .data import make_split, nested_train_validation
from .qnn import QNNConfig, VariationalClassifier

RESULTS=Path(__file__).resolve().parents[1]/"results"

def phase_a(seeds=(0,), epochs=5, output=RESULTS/"tuning_A.json", path=None):
    output=Path(output)
    records=json.loads(output.read_text()) if output.exists() else []
    completed={(r["seed"],json.dumps(r["config"],sort_keys=True)) for r in records}
    configs=[]
    for depth,reup,bw in itertools.product((1,3),(False,True),(1.0,.5)):
        configs.append(QNNConfig("angle",depth,bw,reup))
        configs.append(QNNConfig("zz",depth,bw,reup))
    configs.extend(QNNConfig("amplitude",d,1.0,False) for d in (1,2,3))
    for seed in seeds:
        d=make_split(seed,path)
        for cfg in configs:
            if (seed,json.dumps(asdict(cfg),sort_keys=True)) in completed:
                continue
            xa,xb,ya,yb,_,_=nested_train_validation(d,seed,view="pca" if cfg.encoding=="amplitude" else "angle")
            model=VariationalClassifier(cfg,seed)
            hist=model.fit(xa,ya,xb,yb,epochs=epochs,seed=seed)
            record={"seed":seed,"config":asdict(cfg),"validation_accuracy":max(r["validation_accuracy"] for r in hist),"history":hist}
            records.append(record)
            output.parent.mkdir(exist_ok=True,parents=True)
            output.write_text(json.dumps(records,indent=2))
            completed.add((seed,json.dumps(asdict(cfg),sort_keys=True)))
    return records
