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