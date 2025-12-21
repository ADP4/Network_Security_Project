from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array

import os
import sys
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import mlflow

import dagshub
dagshub.init(repo_owner='ADP4', repo_name='Network_Security_Project', mlflow=True)




class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            logging.info("Model trainer initialisation: model trainer class")

        except Exception as e:
            logging.info("Model trainer initialisation Failed: model trainer class")
            raise Network_Security_Exception(e,sys)
        
    
    @staticmethod
    def evaluate_models(X_train, y_train, X_test, y_test,
                        models: Dict[str, Any],
                        params: Dict[str, dict]) -> Dict[str, dict]:
        try:
            report = {}

            for model_name, model in models.items():
                param_grid = params.get(model_name, {})

                # 1) Tune (if grid exists) else fit directly
                if param_grid:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        scoring="f1",
                        n_jobs=-1,
                        verbose=1,
                    )
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                # 2) Predict
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # 3) Metrics
                train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
                test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

                # 4) Store results for THIS model
                report[model_name] = {
                    "model": best_model,
                    "train_metric": train_metric,
                    "test_metric": test_metric,
                }

            return report

        except Exception as e:
            raise Network_Security_Exception(e, sys) from e


    

# Main training logic
    def train_model(self, X_train:np.ndarray, y_train:np.ndarray, 
                    X_test:np.ndarray, y_test:np.ndarray)-> ModelTrainerArtifact:
        
        try:
            
            
            #Defining candidate models:
            models = {
                "LogisticRegression": LogisticRegression( max_iter=1000, class_weight="balanced", n_jobs= -1), 

                "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),

                "GradientBoosting": GradientBoostingClassifier(random_state=42),

                "AdaBoost": AdaBoostClassifier(random_state=42),

                "SVM_RBF": SVC( probability=True,  # needed for ROC-AUC / predict_proba
                                class_weight="balanced", random_state=42), 

                "XGBoost" :  XGBClassifier(objective="binary:logistic",
                                            n_estimators=200,
                                            eval_metric="logloss",
                                               objective="binary:logistic"
                                            n_jobs=-1) 
                    }

            #Hyperparameter grids (GridSearchCV)

            params = {
                "LogisticRegression" : {"C": [0.1, 1, 10], 
                                        "solver": ["lbfgs"],
                                        "penalty": ["l2"]},

                "RandomForest": {"n_estimators": [100, 200], 
                                "max_depth": [None, 10, 20], 
                                "min_samples_split": [2, 5],
                                 "max_features": ["sqrt", "log2"]},

                "GradientBoosting": {"n_estimators": [100, 200],
                                    "learning_rate": [0.1, 0.05],
                                        "max_depth": [3, 5]},

                "AdaBoost": {"n_estimators": [50, 100, 200],
                            "learning_rate": [0.1, 0.5, 1.0]},

                    "SVM_RBF": {"C": [0.1, 1.0, 10.0],
                                "gamma": ["scale", "auto"],
                                "kernel": ["rbf"]},

                    "XGBoost" : {"n_estimators": [100, 200], 
                                "learning_rate": [0.1, 0.05],
                                "max_depth": [3, 5],
                                "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}
                     }
                
    #Evaluate models with GridSearchCV ------------------
            model_report = ModelTrainer.evaluate_models(X_train=X_train, y_train=y_train,
                                        X_test=X_test, y_test=y_test,
                                        models=models, params=params)
            
            
                
    #Select best model by test F1-score ------------------
            best_model_name = None
            best_model = None
            best_train_metric = None
            best_test_metric = None
            best_test_f1 = -np.inf

            for name, result in model_report.items():
                test_f1 = result["test_metric"].f1_score
                logging.info(f"Model: {name}, Test F1: {test_f1:.4f}")

                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    best_model_name = name
                    best_model = result["model"]
                    best_train_metric = result["train_metric"]
                    best_test_metric = result["test_metric"]

            if best_model is None:
                raise Network_Security_Exception("No suitable model found.", sys)

            logging.info(f"Best model: {best_model_name} | Test F1: {best_test_f1:.4f}")

            # Save metrics for ALL models (comparison table) ------------------
            comparison_rows = []
            for name, result in model_report.items():
                tr = result["train_metric"]
                te = result["test_metric"]
                comparison_rows.append({
                    "model": name,
                    "train_f1": float(tr.f1_score),
                    "train_precision": float(tr.precision_score),
                    "train_recall": float(tr.recall_score),
                    "test_f1": float(te.f1_score),
                    "test_precision": float(te.precision_score),
                    "test_recall": float(te.recall_score),
                    "f1_gap": float(abs(tr.f1_score - te.f1_score)),
                })

            # save inside model_trainer artifact folder
            metrics_dir = os.path.join(os.path.dirname(self.model_trainer_config.trained_model_file_path), "reports")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_csv_path = os.path.join(metrics_dir, "model_comparison.csv")

            import pandas as pd
            pd.DataFrame(comparison_rows).sort_values("test_f1", ascending=False).to_csv(metrics_csv_path, index=False)

            logging.info(f"Saved model comparison metrics to: {metrics_csv_path}")


                    #Check expected score & overfitting ------------------
            expected_score = self.model_trainer_config.expected_accuracy
            overfit_threshold = (self.model_trainer_config.underfitting_overfitting_threshold)

            if best_test_metric.f1_score < expected_score:
                msg = (
                        f"Best model test F1 {best_test_metric.f1_score:.4f} "
                        f"is below expected score {expected_score:.4f}."
                    )
                logging.error(msg)
                raise Network_Security_Exception(msg, sys)

            f1_gap = abs(best_train_metric.f1_score - best_test_metric.f1_score)
            if f1_gap > overfit_threshold:
                msg = (
                        f"Possible overfitting/underfitting: |Train F1 - Test F1| = {f1_gap:.4f} "
                        f"exceeds threshold {overfit_threshold:.4f}."
                    )
                logging.error(msg)
                raise Network_Security_Exception(msg, sys)

            logging.info(
                    f"Best model passed overfitting check. "
                    f"Train F1: {best_train_metric.f1_score:.4f}, "
                    f"Test F1: {best_test_metric.f1_score:.4f}, Gap: {f1_gap:.4f}"
                )

                # Track BEST model in MLflow ------------------
            self.track_mlflow(model_name=best_model_name, best_model=best_model,
                    train_metric=best_train_metric, test_metric=best_test_metric,)

                # Save final model (wrapped with preprocessor) ------------------
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, network_model)

                # Also save raw best model & preprocessor (model pusher)
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)
            save_object("final_model/preprocessor.pkl", preprocessor)

                # Prepare artifact ------------------
            model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_artifact=best_train_metric,
                    test_metric_artifact=best_test_metric,
                )

            logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise Network_Security_Exception(e, sys) from e
        
# MLflow tracking for BEST model-------------------------------------------------

    def track_mlflow(
        self,
        model_name: str,
        best_model,
        train_metric: ClassificationMetricArtifact,
        test_metric: ClassificationMetricArtifact,
    ) -> None:
        """
        Logs metrics and model to MLflow for the best model.
        """
        try:
            #mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
            #mlflow.set_registry_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))

            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

           # mlflow.set_tracking_uri("./mlruns") #---- chnaged

            with mlflow.start_run(run_name=model_name):

                # ---------------- OPTIONAL IMPROVEMENTS START ----------------
                mlflow.set_tag("best_model", model_name)
                mlflow.set_tag("problem_type", "binary_classification")
                mlflow.set_tag("domain", "network_security")
                mlflow.set_tag("dataset", "UCI_Phishing_Websites")
                mlflow.set_tag("target_column", "Result")
                # ---------------- OPTIONAL IMPROVEMENTS END ------------------


                # Log metrics
                mlflow.log_metric("train_f1", float(train_metric.f1_score))
                mlflow.log_metric("train_precision", float(train_metric.precision_score))
                mlflow.log_metric("train_recall", float(train_metric.recall_score))

                mlflow.log_metric("test_f1", float(test_metric.f1_score))
                mlflow.log_metric("test_precision", float(test_metric.precision_score))
                mlflow.log_metric("test_recall", float(test_metric.recall_score))

                # Log parameters
                try:
                    mlflow.log_params(best_model.get_params())
                except Exception:
                    logging.warning("Could not log all model parameters to MLflow.")

                # Log model
                #if tracking_url_type_store != "file":
                #    mlflow.sklearn.log_model(
                #        sk_model=best_model,
                #        artifact_path="model",
                #        registered_model_name=model_name )
                #else:
                #    mlflow.sklearn.log_model(
                #        sk_model=best_model,
                #        artifact_path="model",)

                mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model")

        except Exception as e:
            logging.warning(f"MLflow tracking failed: {e}")


    
    def initiate_model_trainer(self):
        #- Loads transformed train/test arrays.
        #- Splits X and y.
        #- Calls train_model() to train, evaluate, select, log, and save best model
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading transformed train array from: {train_file_path}")
            train_arr = load_numpy_array(train_file_path)

            logging.info(f"Loading transformed test array from: {test_file_path}")
            test_arr = load_numpy_array(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(
                f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, "
                f"X_test: {X_test.shape}, y_test: {y_test.shape}"
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact


        except Exception as e:
            raise Network_Security_Exception(e,sys)