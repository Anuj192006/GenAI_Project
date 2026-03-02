import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report


df=pd.read_csv("7 churn.csv")

df=df.drop(columns=["customerID"])

df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(),inplace=True)

df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

categorical_cols=df.select_dtypes(include="object").columns.tolist()

encoders={}

for col in categorical_cols:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    encoders[col]=le


X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)

dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)


print("\nLogistic Regression Results")
y_pred_log=log_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred_log))
print("Precision:",precision_score(y_test,y_pred_log))
print("Recall:",recall_score(y_test,y_pred_log))
print("F1 Score:",f1_score(y_test,y_pred_log))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_log))
print("Classification Report:\n",classification_report(y_test,y_pred_log))


print("\nDecision Tree Results")
y_pred_dt=dt_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred_dt))
print("Precision:",precision_score(y_test,y_pred_dt))
print("Recall:",recall_score(y_test,y_pred_dt))
print("F1 Score:",f1_score(y_test,y_pred_dt))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_dt))
print("Classification Report:\n",classification_report(y_test,y_pred_dt))


model_data={
    "model":log_model,
    "scaler":scaler,
    "encoders":encoders,
    "feature_names":X.columns.tolist()
}

with open("telco_churn_model.pkl","wb") as f:
    pickle.dump(model_data,f)



def predict_churn(input_dict):

    with open("telco_churn_model.pkl","rb") as f:
        model_data=pickle.load(f)

    model=model_data["model"]
    scaler=model_data["scaler"]
    encoders=model_data["encoders"]
    feature_names=model_data["feature_names"]

    input_df=pd.DataFrame([input_dict])

    for col in encoders:
        input_df[col]=encoders[col].transform(input_df[col])

    input_df=input_df[feature_names]
    input_scaled=scaler.transform(input_df)

    pred=model.predict(input_scaled)
    prob=model.predict_proba(input_scaled)

    if pred[0]==1:
        print("Churn")
    else:
        print("Not Churn")

    print("Probability:",prob)



sample_customer={
    "gender":"Female",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":1,
    "PhoneService":"No",
    "MultipleLines":"No phone service",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":29.85,
    "TotalCharges":29.85
}

predict_churn(sample_customer)