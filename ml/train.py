from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"models"; OUT.mkdir(exist_ok=True)
data=load_breast_cancer()
Xtr,Xte,ytr,yte=train_test_split(data.data,data.target,test_size=.2,random_state=42,stratify=data.target)
model=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
model.fit(Xtr,ytr)
pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"roc_auc":roc_auc_score(yte,prob)}
bundle={"model":model,"version":"rf-1.0","metrics":metrics,"feature_names":data.feature_names.tolist()}
joblib.dump(bundle,OUT/"model.joblib")
print(metrics)
