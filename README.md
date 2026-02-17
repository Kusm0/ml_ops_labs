# MLOps Labs — Лабораторна робота 1

Проєкт для відстеження експериментів ML (MLflow) на датасеті Spotify Tracks: регресія популярності треків.

## Середовище

- Python 3.10+
- Віртуальне середовище: `python3 -m venv venv`, потім `source venv/bin/activate` (Linux/macOS) або `.\venv\Scripts\activate` (Windows).
- Встановлення залежностей: `pip install -r requirements.txt`.

## Структура

- `data/raw/` — сирі дані (dataset.csv).
- `data/processed/` — оброблений датасет (нормалізація, без outliers, track_genre закодовано); **його використовують експерименти**.
- `notebooks/01_eda.ipynb` — первинний аналіз даних (EDA).
- `src/preprocess.py` — обробка даних: нормалізація (StandardScaler), видалення outliers (IQR), числове кодування track_genre.
- `src/train.py` — навчання регресії з логуванням у MLflow; підтримує моделі: **rf** (RandomForest), **gbm** (GradientBoosting), **hist_gbm** (HistGradientBoosting), **ridge** (Ridge).

## Запуск

1. **EDA:** відкрити та виконати `notebooks/01_eda.ipynb` (Jupyter: `jupyter notebook` або `jupyter lab`).
2. **Обробка даних (один раз перед експериментами):**  
   `python src/preprocess.py` — читає `data/raw/dataset.csv`, зберігає `data/processed/dataset.csv` та `data/processed/genre_mapping.json`.
3. **Навчання (за замовчуванням використовує data/processed):** `python src/train.py`
4. **Навчання з параметрами:**  
   `python src/train.py --max_depth 10 --n_estimators 100`
5. **MLflow UI:** `mlflow ui --backend-store-uri mlruns/`, потім http://127.0.0.1:5000.

## Експерименти

**Іменування:** експеримент `spotify_popularityReg_v1` (схема domain_objective_stage). Теги: `run_type=baseline` / `run_type=tuned`, логуються seed, data_version, code_version для відтворюваності.

**Перед експериментами** один раз виконати обробку даних: `python src/preprocess.py`.

**5 запусків з різними гіперпараметрами** (один baseline + чотири tuned):

```bash
bash scripts/run_experiments.sh
```

Або вручну:

```bash
python src/train.py --run_type baseline
python src/train.py --run_type tuned --max_depth 5 --n_estimators 50
python src/train.py --run_type tuned --max_depth 15 --n_estimators 150
python src/train.py --run_type tuned --max_depth 20 --min_samples_split 5
python src/train.py --run_type tuned --max_depth 8 --n_estimators 200 --min_samples_split 4
```

У MLflow UI використати "Compare" для порівняння метрик (train vs test) та оцінки overfitting.

**Порівняння моделей (RF, GBM, HistGBM, Ridge):**

```bash
python scripts/run_experiments_models.py
# або: bash scripts/run_experiments_models.sh
```

Запускає по одному run на модель з розумними за замовчуванням гіперпараметрами; в UI можна порівняти test_r2, test_rmse тощо по тегу `model_type`.

### Чому метрики скромні (R² ≈ 0.16–0.21, MAE ≈ 16)?

- **Ціль (popularity)** — шкала 0–100; MAE ≈ 16 означає середню похибку ~16 пунктів.
- **Ми передбачуємо популярність лише за аудіо-ознаками** (danceability, energy, loudness тощо). На реальну популярність у Spotify сильніше впливають плейлисти, маркетинг, популярність виконавця, дата релізу — цього в даних немає.
- **Звідси обмежений “стеля” якості**: на одних аудіо-ознаках R² на рівні 0.2–0.3 часто вже непоганий результат; наші runs узгоджені з цим.
- **Що може покращити**: додати ознаки (наприклад, жанр `track_genre`, рік релізу), інші моделі (градієнтний бустинг), або прийняти, що для цієї задачі скромні метрики очікувані.

## Docker Compose (усе в контейнері)

Навчання і MLflow UI працюють у контейнері; логи пишуться в `./mlruns` на хості (bind mount). UI: **http://localhost:5001**.

```bash
# 1. Зібрати образи
docker compose build

# 2. Обробити дані (створить data/processed/dataset.csv)
docker compose run --rm train python src/preprocess.py

# 3. Запустити 5 експериментів (RandomForest) з середини Docker
docker compose run --rm train python scripts/run_experiments.py
# або через bash:
docker compose run --rm train bash scripts/run_experiments.sh

# Запустити експерименти з іншими моделями (RF, GBM, HistGBM, Ridge):
docker compose run --rm train python scripts/run_experiments_models.py

# Один run з своїми параметрами:
docker compose run --rm train python src/train.py --run_type tuned --max_depth 5 --n_estimators 50 --run_name "my_run"

# 4. Запустити MLflow UI
docker compose up mlflow-ui
# або у фоні: docker compose up -d mlflow-ui
```

У браузері відкрийте **http://localhost:5001** — експерименти та runs з `./mlruns`; кнопка "Compare" для порівняння метрик.

Зупинити UI: `Ctrl+C` (або `docker compose stop mlflow-ui` якщо запускали з `-d`).

---

## Docker (без Compose)

Збірка та запуск вручну:

```bash
docker build -t mlops-lab1 .
docker run --rm mlops-lab1
docker run --rm mlops-lab1 python src/train.py --max_depth 20 --n_estimators 50
```

Щоб зберегти логи на хості й дивитися UI локально: монтуйте `./mlruns` і використовуйте `mlflow ui` на хості або в окремому контейнері (див. варіанти вище в історії README).

---

## Чи лабораторну виконано?

Так, якщо виконано всі пункти:

1. **Структура проєкту** — є (notebooks, src, data/raw, .gitignore, requirements.txt).
2. **EDA** — ноутбук `01_eda.ipynb` з завантаженням, перевіркою пропусків, розподілом цільової, кореляцією.
3. **Скрипт навчання** — `src/train.py` з MLflow (параметри, метрики train/test, модель, артефакт feature importance), CLI.
4. **Мінімум 5 експериментів** — запустити навчання 5 разів з різним `max_depth` (наприклад 2, 5, 10, 20, 40). При використанні Docker — обов’язково з монтуванням `-v $(pwd)/mlruns:/app/mlruns -e MLFLOW_TRACKING_URI=file:///app/mlruns`, щоб runs збереглися.
5. **MLflow UI** — виконати `mlflow ui --backend-store-uri mlruns/` і переконатися, що всі runs видно та можна їх порівняти (Compare).

Повідомлення **"Run finished. View with: mlflow ui"** означає, що один run завершився успішно. Щоб лабораторну вважати виконаною, потрібні 5+ таких runs і перегляд їх у MLflow UI.
