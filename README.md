# Sparrow 🐦: A Dynamic MLP Architecture

[![PyPI version](https://badge.fury.io/py/sparrow-mlp.svg)](https://badge.fury.io/py/sparrow-mlp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](#english) | [فارسی](#فارسی)**

---

## English

Sparrow is a PyTorch library for building **Dynamic Multi-Layer Perceptrons (MLPs)**. This architecture learns its own optimal structure by combining two powerful routing mechanisms:

1.  **Dynamic Depth (Global Routing)**: A top-level router learns to activate or bypass entire hidden layers, finding the most efficient computational path for each input.
2.  **Mixture of Experts (Local Routing)**: Each hidden layer is a MoE layer containing several smaller "expert" networks. A sophisticated **Two-Tower Router** within each layer selects the most relevant expert, enabling specialization and efficient, sparse computation.

This results in a highly adaptive neural network that prunes its own depth and width during training, controlled by intuitive hyperparameters for sparsity, load balancing, and exploration.

### Key Features
-   **Dynamic Depth**: Automatically learns which layers to skip.
-   **Mixture of Experts Layers**: Activates a single, specialized group of neurons in each layer.
-   **Two-Tower Routers**: An advanced routing mechanism for intelligent expert selection.
-   **Built-in Optimizations**: Optional support for load balancing and entropy bonus to ensure robust training.
-   **Simple API**: Build and configure complex dynamic models easily with the `SparrowConfig` class.

### Installation
```bash
pip install sparrow-mlp
```

### Quickstart
Here is a complete example of training a `DynamicMLP` on the Iris dataset.
```python
# examples/run_iris.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# فرض می‌کنیم کتابخانه sparrow نصب شده و در دسترس است
from sparrow import SparrowConfig, DynamicMLP

# ۱. بارگذاری و آماده‌سازی داده‌ها
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# ۲. تعریف تنظیمات مدل با استفاده از SparrowConfig
config = SparrowConfig(
    hidden_dim=32,
    num_hidden_layers=2,
    num_experts=4,
    expert_hidden_size=128,
    layer_sparsity_lambda=0.01,
    load_balancing_alpha=0.01,
    entropy_lambda=0.0  # پاداش انتروپی را برای این مثال ساده غیرفعال می‌کنیم
)

# ۳. ساخت مدل با پاس دادن کانفیگ
model = DynamicMLP(input_size=4, output_size=3, config=config)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 500

# ۴. حلقه آموزش
print("--- شروع آموزش مدل DynamicMLP روی دیتاست زنبق ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # مدل در حالت آموزش سه خروجی دارد
    outputs, balancing_loss, entropy = model(X_train_tensor)
    
    # محاسبه تمام مؤلفه‌های زیان
    loss_cls = criterion(outputs, y_train_tensor)
    loss_sparsity = config.layer_sparsity_lambda * model.layer_gates_values.sum()
    loss_balance = config.load_balancing_alpha * balancing_loss
    bonus_entropy = config.entropy_lambda * entropy

    total_loss = loss_cls + loss_sparsity + loss_balance - bonus_entropy
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.4f}')

# ۵. تست و ارزیابی مدل
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted_labels.numpy())
    print(f'\nدقت نهایی روی داده‌های تست: {accuracy * 100:.2f}%')
```
---

## فارسی

`Sparrow` 🐦 یک کتابخانه پایتورچ برای ساخت **پرسپترون‌های چندلایه دینامیک (MLP)** است. این معماری ساختار بهینه خود را با ترکیب دو مکانیزم مسیریابی قدرتمند یاد می‌گیرد:

۱. **عمق دینامیک (مسیریابی سراسری)**: یک مسیریاب سطح بالا یاد می‌گیرد که بر اساس ورودی، کل لایه‌های پنهان را فعال یا از آنها عبور کند و بهینه‌ترین مسیر محاسباتی را برای هر ورودی پیدا کند.
۲. **ترکیبی از متخصصان (مسیریابی محلی)**: هر لایه پنهان یک لایه MoE است که شامل چندین شبکه "متخصص" کوچکتر می‌شود. یک **مسیریاب دو برجی (Two-Tower Router)** پیشرفته در هر لایه، مرتبط‌ترین متخصص را انتخاب می‌کند که منجر به تخصصی شدن و محاسبات بهینه و پراکنده می‌شود.

نتیجه این ترکیب، یک شبکه عصبی بسیار تطبیق‌پذیر است که عمق و عرض خود را در طول آموزش هرس می‌کند و توسط هایپرپارامترهای قابل فهم برای پراکندگی، توزیع بار و اکتشاف کنترل می‌شود.

### قابلیت‌های کلیدی
- **عمق دینامیک**: به طور خودکار یاد می‌گیرد کدام لایه‌ها را رد کند.
- **لایه‌های MoE**: تنها یک گروه تخصصی از نورون‌ها را در هر لایه فعال می‌کند.
- **مسیریاب‌های Two-Tower**: یک مکانیزم مسیریابی پیشرفته برای انتخاب هوشمند متخصص.
- **بهینه‌سازی‌های داخلی**: پشتیبانی اختیاری از توزیع بار (load balancing) و پاداش انتروپی برای تضمین آموزش پایدار.
- **API ساده**: ساخت و تنظیم مدل‌های دینامیک پیچیده به راحتی با کلاس `SparrowConfig`.

### نصب
```bash
pip install sparrow-mlp
```
### شروع سریع
در بالا یک مثال کامل از آموزش `DynamicMLP` روی دیتاست گل زنبق با استفاده از `SparrowConfig` آمده است.