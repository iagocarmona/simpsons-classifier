from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils import do_cv_svm, imprimir_estatisticas
from sklearn.metrics import accuracy_score

# Carregue os arquivos de texto
X_train = np.loadtxt('train_features.txt')
y_train = np.genfromtxt('train_labels.txt', dtype='str')
X_teste = np.loadtxt('validation_features.txt')
y_teste = np.genfromtxt('validation_labels.txt', dtype='str')

# Execute a validação cruzada SVM
accs_svm, melhor_modelo = do_cv_svm(X_train, y_train, 10, Cs=[1, 10, 100, 1000], gammas=[
    'scale', 'auto', 2e-2, 2e-3, 2e-4])

# Faça previsões no conjunto de validação (teste)
previsoes_validacao = melhor_modelo.predict(X_teste)

# Imprima as estatísticas usando F1-Score e matriz de confusão
imprimir_estatisticas(accs_svm, y_teste, previsoes_validacao)

acuracias = []
# Vamos testar os ks ímpares de 1 a 29
ks = list(range(1, 30, 2))

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_teste)
    # armazenar a acurácia no conjunto de teste
    acuracias.append(accuracy_score(y_teste, pred))


def plot_knn_k_acc(ks, acuracias, label=''):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, acuracias)
    ax.set_xticks(ks)
    ax.set_xlabel('k')
    ax.set_ylabel('Acurácia')
    ax.set_title('Acurácia no Conjunto de %s' % (label))
    fig.tight_layout()
    plt.show()


melhor_teste = max(acuracias)
melhor_k_teste = ks[np.argmax(acuracias)]
print("Melhor k no teste: %d (acc=%.2f)" % (melhor_k_teste, melhor_teste))

plot_knn_k_acc(ks, acuracias, 'Teste')
