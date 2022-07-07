from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import mlflow.pyfunc
import fasttext
import pandas as pd
import uuid 


class FastTextMLFlowModel(mlflow.pyfunc.PythonModel):

    def __init__(self, params, run_id):
        self.params = params
        self.model_file = "{}/{}.bin".format(params["model_location"], run_id)
        
    def load_context(self, context):
        self.model = fasttext.load_model(self.model_file)
        
    def clear_context(self):
        # model cannot be pickled, so disposing it
        self.model = None
        
    def __predict_label(self, desc):
        import re
        prediction = self.model.predict(desc)[0][0]
        prediction = re.sub('__label__', '', prediction)
        prediction = re.sub('-', ' ', prediction)
        return prediction
        
    def train(self):
      
        self.model = fasttext.train_supervised(
            input=self.params["input"],
            lr=self.params.get("lr", 0.1),
            dim=self.params.get("dim", 100),
            ws=self.params.get("ws", 5),
            epoch=self.params.get("epoch", 5),
            minCount=self.params.get("minCount", 1),
            minCountLabel=self.params.get("minCountLabel", 1),
            minn=self.params.get("minn", 0),
            maxn=self.params.get("maxn", 0),
            neg=self.params.get("neg", 5),
            wordNgrams=self.params.get("wordNgrams", 5),
            loss=self.params.get("loss", "softmax"),
            bucket=self.params.get("bucket", 2000000),
            thread=self.params.get("thread", 4),
            lrUpdateRate=self.params.get("lrUpdateRate", 100),
            t=self.params.get("t", 0.0001),
            label=self.params.get("label", "__label__"),
            verbose=self.params.get("verbose", 2)
        )
        
        # we're saving our model manually instead of relying on cloudpickle
        self.model.save_model(self.model_file)
  
    def evaluate(self, input_features, input_targets):
        import re
        result = input_targets.to_frame()
        result.columns = ["target"]
        result["prediction"] = input_features.apply(lambda x: self.__predict_label(x))
        result["accuracy"] = result["prediction"] == result["target"]
        result["accuracy"] = result["accuracy"].apply(lambda x: float(x))
        accuracies = result.groupby(["target"])["accuracy"].mean()
        return {
            "avg__acc": accuracies.mean(),
            "q_05_acc": accuracies.quantile(0.05),
            "q_25_acc": accuracies.quantile(0.25),
            "q_50_acc": accuracies.quantile(0.50),
            "q_75_acc": accuracies.quantile(0.75),
            "q_95_acc": accuracies.quantile(0.95)
        }      

    def predict(self, context, input_data):
        tmp = input_data
        if not "input" in input_data.columns:
            tmp.columns = ["input"]
        result = tmp['input'].apply(lambda x: self.__predict_label(x))
        return result