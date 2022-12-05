from xgboost import XGBClassifier
from read_data import pre_process
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = pre_process()

xgb = XGBClassifier(n_estimators=200,
                    learning_rate=0.1)

xgb.fit(x_train, y_train)

pred = xgb.predict(x_test)

accuracy_score(pred, np.array(y_test).astype('int32'))
roc_auc_score(pred, np.array(y_test).astype('int32'))
confusion_matrix = confusion_matrix(np.array(y_test).astype('int32'), pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=xgb.classes_)
cm_display.plot()
plt.show()
