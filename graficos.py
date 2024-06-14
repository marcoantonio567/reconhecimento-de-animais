import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Definição das classes
class_names = [
    "cachorro", "cobra", "tubarão", "baiacu", "porco",
    "papagaio", "jacaré", "pinguim", "salamandra", "sapo"
]

# Exemplos fictícios de classes reais e preditas
true_classes = [
    "cachorro", "cobra", "tubarão", "baiacu", "porco",
    "papagaio", "jacaré", "pinguim", "salamandra", "sapo",
    "cachorro", "cobra", "tubarão", "baiacu", "porco"
]
predicted_classes = [
    "cachorro", "cobra", "sapo", "baiacu", "porco",
    "papagaio", "jacaré", "pinguim", "salamandra", "sapo",
    "cachorro", "tubarão", "tubarão", "baiacu", "porco"
]
confidence_scores = [
    0.95, 0.80, 0.60, 0.90, 0.85, 0.80, 0.75, 0.85, 0.70, 0.65,
    0.88, 0.79, 0.92, 0.85, 0.90
]

# Geração da matriz de confusão
labels = class_names  # Usar a lista de classes fornecida
conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=labels)

# Configurar a figura e os subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico de Dispersão das Pontuações de Confiança
axs[0, 0].scatter(range(len(confidence_scores)), confidence_scores, color='blue')
axs[0, 0].set_title('Gráfico de Dispersão das Pontuações de Confiança')
axs[0, 0].set_xlabel('Índice da Imagem')
axs[0, 0].set_ylabel('Pontuação de Confiança')
axs[0, 0].grid(True)

# Distribuição das Classes Preditas
sns.countplot(x=predicted_classes, ax=axs[0, 1], order=class_names)
axs[0, 1].set_title('Distribuição das Classes Preditas')
axs[0, 1].set_xlabel('Classe')
axs[0, 1].set_ylabel('Contagem')

# Plotando a Matriz de Confusão
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axs[1, 0])
axs[1, 0].set_title('Matriz de Confusão')
axs[1, 0].set_xlabel('Classe Predita')
axs[1, 0].set_ylabel('Classe Verdadeira')

# Gráfico vazio (ou pode ser uma legenda, informações adicionais, etc.)
axs[1, 1].axis('off')

# Ajustar layout
plt.tight_layout()
plt.show()
