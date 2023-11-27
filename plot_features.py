import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregue os arquivos de texto
train_features = np.loadtxt('Features/train_features.txt')
train_labels = np.genfromtxt('Features/train_labels.txt', dtype='str')
validation_features = np.loadtxt('Features/validation_features.txt')
validation_labels = np.genfromtxt(
    'Features/validation_labels.txt', dtype='str')

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42)

print(f'Número de imagens de treinamento: {X_train.shape[0]}')
print(f'Número de imagens de teste: {X_test.shape[0]}')
print(f'Número de características: {X_train.shape[1]}')

# Visualize os dados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(y_train)
plt.title('Treinamento')
plt.xlabel('Rótulos')
plt.ylabel('Número de imagens')
plt.subplot(1, 2, 2)
plt.hist(y_test)
plt.title('Teste')
plt.xlabel('Rótulos')
plt.ylabel('Número de imagens')
plt.tight_layout()
plt.show()
