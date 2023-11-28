from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils import do_cv_svm, imprimir_estatisticas, selecionar_melhor_k
from sklearn.metrics import accuracy_score

# Carregue os arquivos de texto
# X_train = np.loadtxt('Features/v1/train_features.txt')
# y_train = np.genfromtxt('Features/v1/train_labels.txt', dtype='str')
# X_teste = np.loadtxt('Features/v1/validation_features.txt')
# y_teste = np.genfromtxt('Features/v1/validation_labels.txt', dtype='str')

# Divida os dados em conjuntos de treinamento e teste
X = np.loadtxt('Features/v2/caracteristicas.txt')
y = np.loadtxt('Features/v2/rotulos.txt', dtype='str')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Em seguida, separamos parte do conjunto de treino para a validação.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)
X_val = ss.transform(X_val)


# Execute a validação cruzada SVM
accs_svm, melhor_modelo = do_cv_svm(X_train, y_train, 10, Cs=[1, 10, 100, 1000], gammas=[
    'scale', 'auto', 2e-2, 2e-3, 2e-4])

# Faça previsões no conjunto de validação (teste)
previsoes_validacao = melhor_modelo.predict(X_test)

# Imprima as estatísticas usando F1-Score e matriz de confusão
imprimir_estatisticas(accs_svm, y_test, previsoes_validacao)

knn, melhor_k, melhor_val = selecionar_melhor_k(
    range(1, 30, 2), X_train, X_val, y_train, y_val)
print("Melhor k na validação: %d (acc=%.2f)" % (melhor_k, melhor_val))

pred = knn.predict(X_test)
print("acurácia no teste: %.2f" % (accuracy_score(y_test, pred)))
