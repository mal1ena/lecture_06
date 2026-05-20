# lecture_06
Индивидуальное задание
Выберите вариант по номеру в журнале. Нужно обучить классификатор FashionMNIST с указанными параметрами и оформить краткий отчет.

Что сделать:

загрузить FashionMNIST;
создать модель с одним скрытым слоем указанного размера;
использовать указанную функцию активации, оптимизатор, learning rate, batch_size и количество эпох;
вывести итоговую accuracy на тестовой выборке;
построить/вывести матрицу ошибок или таблицу правильных ответов по классам;
написать вывод: какие параметры дали хороший/плохой результат и почему.
Параметры модели: hidden=48, activation=Sigmoid, optimizer=RMSprop, LR=0.03, batch=64, epochs=6

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Укажите свой вариант
variant = 3

# Заполните параметры по таблице
hidden_size = 48
activation_name = "Sigmoid"
optimizer_name = "RMSprop"
learning_rate = 0.03
batch_size = 64
epochs = 6

# 1. Загрузка данных FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Создание модели с одним скрытым слоем
input_size = 28 * 28
output_size = 10
act_func = nn.Sigmoid()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, hidden_size),
    act_func,
    nn.Linear(hidden_size, output_size)
)

# 3. Выбор оптимизатора
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 4. Обучение
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
print("Training finished.")
```
Epoch [1/6], Loss: 0.8002
Epoch [2/6], Loss: 0.6756
Epoch [3/6], Loss: 0.6432
Epoch [4/6], Loss: 0.6219
Epoch [5/6], Loss: 0.6134
Epoch [6/6], Loss: 0.6090
Training finished.

```
# 5. Оценка качества на тестовой выборке
model.eval()
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.numpy())
        for i in range(len(labels)):
            lbl = labels[i].item()
            class_total[lbl] += 1
            if predicted[i].item() == lbl:
                class_correct[lbl] += 1

accuracy = 100 * correct / total
print(f"\nИтоговая Accuracy на тестовой выборке: {accuracy:.2f}%")
```
Итоговая Accuracy на тестовой выборке: 76.45%

```
# 6. Таблица правильных ответов по классам
print("\nПравильные ответы по классам:")
for i in range(10):
    print(f"{classes[i]:<12}: {class_correct[i]:3d}/{class_total[i]:3d} ({100*class_correct[i]/max(class_total[i],1):.1f}%)")
```

Правильные ответы по классам:
T-shirt/top : 609/1000 (60.9%)
Trouser     : 947/1000 (94.7%)
Pullover    : 506/1000 (50.6%)
Dress       : 740/1000 (74.0%)
Coat        : 731/1000 (73.1%)
Sandal      : 843/1000 (84.3%)
Shirt       : 655/1000 (65.5%)
Sneaker     : 880/1000 (88.0%)
Bag         : 841/1000 (84.1%)
Ankle boot  : 893/1000 (89.3%)

```
# 7. Матрица ошибок
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f"Confusion Matrix (Variant {variant}, Acc: {accuracy:.2f}%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
<img width="726" height="589" alt="image" src="https://github.com/user-attachments/assets/ea4f1034-b6e6-41cc-a7ac-1962ad3dfed3" />

Отчет по индивидуальному заданию
Вариант: 3
Параметры модели: hidden=48, activation=Sigmoid, optimizer=RMSprop, LR=0.03, batch=64, epochs=6
Итоговая accuracy: 76,45%
Лучше всего распознаются классы: Trouser, Ankle boot, Sneaker
Хуже всего распознаются классы: Pullover, T-shirt/top, Shirt
Вывод: Активация Sigmoid подвержена проблеме затухания градиентов, что особенно заметно на FashionMNIST с его визуально схожими классами (напр., Shirt/T-shirt, Coat/Pullover). За 6 эпох модель не успевает выучить тонкие грани. RMSprop с LR=0.03 работает стабильнее обычного SGD, но для Sigmoid обычно рекомендуют более низкий LR (~0.005-0.01) или замену на ReLU, чтобы избежать насыщения сигмоиды. Для улучшения результата рекомендуется: увеличить epochs до 15-20, заменить Sigmoid на ReLU или добавить BatchNorm для стабилизации градиентов.
