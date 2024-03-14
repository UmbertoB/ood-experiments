from models.vgg16 import get_vgg16
from lib import cifar10, cifar10_outlier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIMS = 10

model = get_vgg16(OUTPUT_DIMS)

x_train, y_train, x_test, y_test = cifar10.get()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)

x_evaluate, y_evaluate = cifar10_outlier.get()

y_pred = model.predict(x_evaluate)

cm = confusion_matrix(y_evaluate, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")

plt.savefig("/data/odd-project/confusion_matrix.png")

