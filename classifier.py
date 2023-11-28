from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from utils import do_cv_svm, imprimir_estatisticas, plotar_matriz_confusao, selecionar_melhor_k
from sklearn.metrics import accuracy_score, f1_score

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

# Normalizar os dados de treinamento, teste e validação
print("Normalizando os dados...")
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
X_val = ss.transform(X_val)

print("=============== SVM ===============")
# Execute a validação cruzada SVM
accs_svm, melhor_modelo = do_cv_svm(X_train, y_train, 10, Cs=[1, 10, 100, 1000], gammas=[
    'scale', 'auto', 2e-2, 2e-3, 2e-4])

# Faça previsões no conjunto de validação (teste)
previsoes_validacao = melhor_modelo.predict(X_test)

# Imprima as estatísticas usando F1-Score e matriz de confusão
imprimir_estatisticas(accs_svm, y_test, previsoes_validacao)


print("\n\n=============== KNN ===============")
knn, melhor_k, melhor_val = selecionar_melhor_k(
    range(1, 30, 2), X_train, X_val, y_train, y_val)
print("Melhor k na validação: %d (acc=%.2f)" % (melhor_k, melhor_val))

pred = knn.predict(X_test)
print("acurácia no teste: %.2f" % (accuracy_score(y_test, pred)))
print("F1-Score: %.4f" % (f1_score(y_test, pred, average='weighted')))
plotar_matriz_confusao(y_test, pred)
plt.show()

print("\n\n=============== MLP ===============")
# Criando o modelo
mlp = MLPClassifier(max_iter=500, learning_rate='adaptive')

# Definindo os parâmetros
parametros = {
    'hidden_layer_sizes': [(100,), (200,), (300,), (400,), (500,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive']
}

# Executando o GridSearchCV
mlp_grid = GridSearchCV(mlp, parametros, cv=5, n_jobs=-1)
mlp_grid.fit(X_train, y_train)

# Imprimindo os melhores parâmetros
print("Melhores parâmetros: ", mlp_grid.best_params_)
print("Melhor score: ", mlp_grid.best_score_)
print("Acurácia no teste: ", mlp_grid.score(X_test, y_test))

# Fazendo previsões no conjunto de teste
pred = mlp_grid.predict(X_test)

# Imprimindo as estatísticas
imprimir_estatisticas(mlp_grid.best_score_, y_test, pred)
plotar_matriz_confusao(y_test, pred)
plt.show()


print("\n\n=============== RANDOM FOREST ===============")

# Criando o modelo
rf = RandomForestClassifier(random_state=42)

# Definindo os parâmetros
parametros = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Executando o GridSearchCV
rf_grid = GridSearchCV(rf, parametros, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Imprimindo os melhores parâmetros
print("Melhores parâmetros: ", rf_grid.best_params_)
print("Melhor score: ", rf_grid.best_score_)
print("Acurácia no teste: ", rf_grid.score(X_test, y_test))

# Fazendo previsões no conjunto de teste
pred = rf_grid.predict(X_test)

# Imprimindo as estatísticas
imprimir_estatisticas(rf_grid.best_score_, y_test, pred)
plotar_matriz_confusao(y_test, pred)
plt.show()
