import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# ==========================================
# 1. Извлечение данных (Симуляция GEE)
# ==========================================

def get_ndvi_from_gee(region_coords, date_start, date_end):
    """
    Здесь я симулирую результат, который мог бы быть получен через:
    import ee
    ee.Initialize()
    collection = ee.ImageCollection('MODIS/006/MOD13Q1').filterDate(...)
    ...
    """
    pass

def generate_synthetic_ndvi_data(num_samples=500, seq_len=46):
    """
    Генерация данных NDVI (временные ряды), сам датасет использовал локально дома.
    Seq_len = 46 декад (по 8 дней, как в MODIS) за сезон.
    Target: Дни до созревания (Days to Maturity - DTM).
    """
    np.random.seed(42)
    data = []
    targets = []
    
    for _ in range(num_samples):
        # Генерация кривой NDVI: рост -> пик -> спад (созревание)
        # Используем гамма-распределение или кривую Гаусса для формы
        
        t = np.linspace(0, 1, seq_len)
        
        # Случайные параметры кривой
        peak_pos = np.random.uniform(0.4, 0.6) # Пик вегетации (середина сезона)
        steepness = np.random.uniform(5, 10)
        
        # Фаза роста
        growth = np.exp(steepness * (t - peak_pos))
        # Фаза спада (созревание)
        decay = np.exp(-steepness * 0.5 * (t - peak_pos))
        
        # Комбинированная кривая + шум
        ndvi_base = (1 / (1 + growth)) * (1 / (1 + decay))
        # Масштабируем к реальным значениям NDVI (0.1 - 0.9)
        ndvi = 0.1 + 0.8 * ndvi_base + np.random.normal(0, 0.05, seq_len)
        ndvi = np.clip(ndvi, 0, 1)
        
        # Целевая переменная: Дней до созревания (условно, до полной спелости)
        # Допустим, созревание наступает, когда NDVI падает до 0.3 после пика
        # Для синтетики считаем DTM как момент времени (индекс), когда кривая упала
        maturity_idx = int(peak_pos * seq_len + np.random.randint(5, 15))
        if maturity_idx >= seq_len: maturity_idx = seq_len - 1
        
        # Задача регрессии: предсказать индекс времени (или кол-во дней)
        # Для упрощения: Target = индекс декады созревания
        target_val = maturity_idx
        
        data.append(ndvi)
        targets.append(target_val)
        
    return np.array(data), np.array(targets)

class NVDIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # [Batch, Seq, 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # [Batch, 1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Архитектура модели (Transformer)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        return x + self.pe[:x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        x = self.input_projection(x) # [Batch, Seq_Len, D_Model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Используем среднее значение по всей последовательности для регрессии
        # Либо можно брать последний элемент, но среднее часто робастнее для глобального прогноза
        x = x.mean(dim=1) 
        
        return self.regressor(x)

# ==========================================
# 3. Обучение и Визуализация
# ==========================================

def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 6))
    fig.suptitle("NDVI Dynamics and Maturity Prediction", fontsize=16)
    
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if i >= num_samples: break
            
            X_dev, y_dev = X.to(device), y.to(device)
            pred = model(X_dev)
            
            ndvi_series = X_dev[0].cpu().numpy().flatten()
            actual_day = y_dev[0].cpu().item()
            pred_day = pred[0].cpu().item()
            
            ax = axes[i]
            ax.plot(ndvi_series, label='NDVI Time Series', color='green')
            ax.axvline(x=actual_day, color='blue', linestyle='--', label=f'Actual Maturity (Day {actual_day:.0f})')
            ax.axvline(x=pred_day, color='red', linestyle=':', label=f'Predicted (Day {pred_day:.0f})')
            ax.fill_between(range(len(ndvi_series)), 0, ndvi_series, alpha=0.3, color='green')
            ax.legend(loc='upper left')
            ax.set_ylabel("NDVI")
            
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Основной запуск
# ==========================================

if __name__ == "__main__":
    # 1. Данные
    print("Generating synthetic NDVI data...")
    X, y = generate_synthetic_ndvi_data(num_samples=1000)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = NVDIDataset(X_train, y_train)
    test_dataset = NVDIDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer().to(device)
    
    criterion = nn.L1Loss() # L1 Loss эквивалентен MAE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Обучение
    print("Training Transformer model...")
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            # Валидация
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    pred = model(X_batch)
                    val_preds.extend(pred.cpu().numpy())
                    val_targets.extend(y_batch.numpy())
            
            mae = mean_absolute_error(val_targets, val_preds)
            print(f"Epoch {epoch+1}/{epochs} | MAE (days): {mae:.2f}")
            
    # 4. Итоговая оценка
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    final_mae = mean_absolute_error(all_targets, all_preds)
    print(f"\nFinal Test MAE: {final_mae:.2f} days")
    
    # 5. Визуализация
    visualize_predictions(model, test_loader, device)
