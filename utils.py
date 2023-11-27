import itertools
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
import seaborn as sns
from tqdm import tqdm


def selecionar_melhor_svm(Cs, gammas, X_treino: np.ndarray, X_val: np.ndarray,
                          y_treino: np.ndarray, y_val: np.ndarray, n_jobs=4):

    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        return accuracy_score(y_val, pred)

    # gera todas as combinações de parametros C e gamma, de acordo com as listas de valores recebidas por parametro.
    # Na prática faz o produto cartesiano entre Cs e gammas.
    combinacoes_parametros = list(itertools.product(Cs, gammas))

    # Treinar modelos com todas as combinações de C e gamma
    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                            (c, g, X_treino, X_val, y_treino, y_val) for c, g in combinacoes_parametros)

    melhor_val = max(acuracias_val)
    # Encontrar a combinação que levou ao melhor resultado no conjunto de validação
    melhor_comb = combinacoes_parametros[np.argmax(acuracias_val)]
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]

    # Treinar uma SVM com todos os dados de treino e validação usando a melhor combinação de C e gamma.
    svm = SVC(C=melhor_c, gamma=melhor_gamma)
    svm.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm, melhor_comb, melhor_val


def do_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []

    pgb = tqdm(total=cv_splits, desc='Folds avaliados')

    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(
            X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        svm, _, _ = selecionar_melhor_svm(
            Cs, gammas, X_treino, X_val, y_treino, y_val)
        pred = svm.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))

        pgb.update(1)

    pgb.close()

    return acuracias, svm


def calcular_estatisticas(resultados, labels_true, labels_pred):
    f1 = f1_score(labels_true, labels_pred, average='weighted')
    media, desvio, mini, maxi = np.mean(resultados), np.std(
        resultados), np.min(resultados), np.max(resultados)
    return f1, media, desvio, mini, maxi


def imprimir_estatisticas(resultados, labels_true, labels_pred):
    f1, media, desvio, mini, maxi = calcular_estatisticas(
        resultados, labels_true, labels_pred)
    print("F1-Score: %.4f, Resultados: %.2f +- %.2f, min: %.2f, max: %.2f" %
          (f1, media, desvio, mini, maxi))

    # Plotar a matriz de confusão
    plotar_matriz_confusao(labels_true, labels_pred)


def plotar_matriz_confusao(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(labels_true),
                yticklabels=np.unique(labels_true))
    plt.title('Matriz de Confusão')
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.show()
