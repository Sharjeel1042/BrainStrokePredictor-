import joblib
import pandas as pd


def predictFromuserInput(inputDic):
        model=joblib.load('model.joblib')

        user_df=pd.DataFrame([inputDic])

        predection=model.predict(user_df)
        predProb=model.predict_proba(user_df)

        
        return predection,predProb