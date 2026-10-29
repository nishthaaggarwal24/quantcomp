"""Experiment runner: classical references, phase A, final held-out evaluation, plots."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from .data import LABELS, make_split, validation_indices, nested_train_validation
from .classical import evaluate_classical
from .qnn import QNNConfig, VariationalClassifier

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/"results"

def run_classical(seeds=range(5), path=None):
    rows=[]
    for seed in seeds:
        for row in evaluate_classical(seed, path): rows.append({"seed":seed,**row})
    return rows

def tune_phase_a(*args,**kwargs):
    from .tuning import phase_a
    return phase_a(*args,**kwargs)

def select_best(path=RESULTS/"tuning_A.json"):
    records=json.loads(Path(path).read_text())
    scores={}
    for r in records:
        key=(r["config"]["encoding"],json.dumps(r["config"],sort_keys=True))
        scores.setdefault(key,[]).append(r["validation_accuracy"])
    by_encoding={}
    for (enc,cfg),vals in scores.items():
        if enc not in by_encoding or np.mean(vals)>by_encoding[enc][0]: by_encoding[enc]=(float(np.mean(vals)),json.loads(cfg))
    return {k:v[1] for k,v in by_encoding.items()}

def final_evaluation(best=None, seeds=range(5), epochs=5, path=None):
    best=best or select_best()
    RESULTS.mkdir(exist_ok=True)
    output=RESULTS/"final_evaluation.json"
    rows=json.loads(output.read_text()) if output.exists() else []
    completed={(r["seed"],r["encoding"]) for r in rows}
    all_cm={}
    for row in rows:
        all_cm.setdefault(row["encoding"],[]).append(confusion_matrix(row["y_true"],row["y_pred"],labels=range(8)))
    for seed in seeds:
        d=make_split(seed,path)
        for enc,cfgdict in best.items():
            if (seed,enc) in completed: continue
            cfg=QNNConfig(**cfgdict); xtr=d.X_train_pca if enc=="amplitude" else d.X_train; xte=d.X_test_pca if enc=="amplitude" else d.X_test
            # Fixed Phase-A epoch budget. Fit on all training rows, then make one test pass.
            final=VariationalClassifier(cfg,seed)
            hist=final.fit(xtr,d.y_train,epochs=epochs,seed=seed,patience=epochs+1)
            proba=final.predict_proba(xte); pred=proba.argmax(axis=1)
            row={"seed":seed,"encoding":enc,"accuracy":float(accuracy_score(d.y_test,pred)),"config":cfgdict,"y_true":d.y_test.tolist(),"y_pred":pred.tolist(),"probabilities":proba.tolist(),"history":hist}
            rows.append(row); all_cm.setdefault(enc,[]).append(confusion_matrix(d.y_test,pred,labels=range(8)))
            output.write_text(json.dumps(rows,indent=2))
            completed.add((seed,enc))
    make_plots(rows,all_cm)
    summary={}
    for enc in best:
        scores=[r["accuracy"] for r in rows if r["encoding"]==enc]
        summary[enc]={"mean_accuracy":float(np.mean(scores)),"std_accuracy":float(np.std(scores,ddof=1)),"n_splits":len(scores),"config":best[enc]}
    (RESULTS/"qnn_summary.json").write_text(json.dumps(summary,indent=2))
    return rows

def make_plots(rows, matrices):
    import matplotlib.pyplot as plt
    import seaborn as sns
    RESULTS.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    # Training curves (recorded for the first final-evaluation split).
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    first={}
    for r in rows: first.setdefault(r["encoding"],r)
    for enc,r in first.items():
        history=r.get("history",[])
        if history:
            ax.plot([h["epoch"] for h in history],[h["loss"] for h in history],marker="o",linewidth=2.2,label=enc.title())
    ax.set(xlabel="Epoch",ylabel="Training cross-entropy",title="Training loss by encoding")
    ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(RESULTS/"training_curves.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # Show each split explicitly: five points are more informative than a sparse boxplot.
    encs=[e for e in ["angle","zz","amplitude"] if any(r["encoding"]==e for r in rows)]
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    colors=sns.color_palette("deep",n_colors=len(encs))
    for x,(enc,color) in enumerate(zip(encs,colors)):
        vals=np.array([r["accuracy"] for r in rows if r["encoding"]==enc])
        jitter=np.linspace(-.09,.09,len(vals))
        ax.scatter(x+jitter,vals*100,s=62,color=color,edgecolor="white",linewidth=.8,zorder=3)
        mean=float(np.mean(vals))*100
        sd=float(np.std(vals,ddof=1))*100 if len(vals)>1 else 0
        ax.errorbar(x,mean,yerr=sd,fmt="D",color="#102a43",capsize=5,markersize=7,zorder=4,
                    label=f"{enc.title()} mean {mean:.1f}% ± {sd:.1f}%")
    ax.axhline(12.5,color="#d97706",linestyle="--",linewidth=1.6,label="8-class random baseline (12.5%)")
    ax.set_xticks(range(len(encs)),[e.title() for e in encs]); ax.set_ylim(bottom=0)
    ax.set(ylabel="Held-out test accuracy (%)",title="QNN test accuracy across five splits")
    ax.legend(frameon=False,loc="upper left",fontsize=9); fig.tight_layout(); fig.savefig(RESULTS/"accuracy_by_encoding.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    depths=sorted({r["config"]["depth"] for r in rows})
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    vals=[[r["accuracy"] for r in rows if r["config"]["depth"]==d] for d in depths]
    ax.boxplot(vals,tick_labels=[str(d) for d in depths],patch_artist=True,
               boxprops={"facecolor":"#d9f0ee","edgecolor":"#167d86"},
               medianprops={"color":"#102a43","linewidth":2},
               whiskerprops={"color":"#167d86"},capprops={"color":"#167d86"})
    for x,group in enumerate(vals,1):
        if group:
            ax.scatter(np.linspace(x-.07,x+.07,len(group)),np.array(group)*100,color="#167d86",s=35,zorder=3)
    ax.set(xlabel="Ansatz depth",ylabel="Held-out test accuracy (%)",title="QNN accuracy grouped by selected depth")
    fig.tight_layout(); fig.savefig(RESULTS/"accuracy_by_depth.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    for enc,ms in matrices.items():
        counts=np.sum(ms,axis=0).astype(float)
        row_totals=counts.sum(axis=1,keepdims=True)
        percent=np.divide(counts,row_totals,out=np.zeros_like(counts),where=row_totals!=0)*100
        fig,ax=plt.subplots(figsize=(7,5.6))
        sns.heatmap(percent,annot=True,fmt=".0f",vmin=0,vmax=100,cmap="Blues",square=True,
                    xticklabels=LABELS,yticklabels=LABELS,cbar_kws={"label":"Share of true class (%)"},ax=ax,
                    annot_kws={"fontsize":7})
        ax.set(xlabel="Predicted label",ylabel="True label",title=f"{enc.title()} encoding · row-normalized confusion")
        ax.tick_params(axis="x",labelrotation=35,labelsize=7); ax.tick_params(axis="y",labelrotation=0,labelsize=7)
        ax.title.set_fontsize(11); ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
        fig.tight_layout(); fig.savefig(RESULTS/f"confusion_{enc}.png",dpi=150,bbox_inches="tight"); plt.close(fig)

def main():
    RESULTS.mkdir(exist_ok=True)
    rows=run_classical()
    (RESULTS/"classical.json").write_text(json.dumps(rows,indent=2))
    summary={}
    for name in sorted({r["model"] for r in rows}):
        scores=[r["accuracy"] for r in rows if r["model"]==name]
        params=[r["parameters"] for r in rows if r["model"]==name]
        known=[p for p in params if p is not None]
        summary[name]={"accuracy_mean":float(np.mean(scores)),"accuracy_std":float(np.std(scores,ddof=1)),"parameters_mean":float(np.mean(known)) if known else None}
    (RESULTS/"classical_summary.json").write_text(json.dumps(summary,indent=2))
    print("Saved classical baselines. Run phase A with: python -m src.runner tune")

def save_demo_model(config=None, seed=0, epochs=5, path=None):
    """Fit selected circuit on all labelled examples and persist inference pipeline."""
    import joblib
    from .data import load_frame, LABEL_TO_ID
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    frame=load_frame(path) if path else load_frame(); y=frame.label.map(LABEL_TO_ID).to_numpy()
    vec=TfidfVectorizer(stop_words="english"); tf=vec.fit_transform(frame.text)
    svd=TruncatedSVD(n_components=8,random_state=seed); z=svd.fit_transform(tf)
    if config is None:
        tuning=json.loads((RESULTS/"tuning_A.json").read_text())
        config=QNNConfig(**max(tuning,key=lambda r:r["validation_accuracy"])["config"])
    scaler=StandardScaler() if config.encoding=="amplitude" else MinMaxScaler((0,np.pi))
    x=scaler.fit_transform(z)
    model=VariationalClassifier(config,seed); model.fit(x,y,epochs=epochs,seed=seed,patience=epochs+1)
    RESULTS.mkdir(exist_ok=True)
    joblib.dump({"vectorizer":vec,"svd":svd,"scaler":scaler,"config":config.__dict__,"weights":model.weights},RESULTS/"demo_model.joblib")
    return RESULTS/"demo_model.joblib"

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="tune": print(json.dumps(tune_phase_a(),indent=2))
    elif len(sys.argv)>1 and sys.argv[1]=="final": print(json.dumps(final_evaluation(),indent=2))
    elif len(sys.argv)>1 and sys.argv[1]=="demo-model": print(save_demo_model())
    else: main()
