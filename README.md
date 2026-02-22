# MLOps Labs — Лабораторні роботи 1, 2 та 3

Проєкт для відстеження експериментів ML (MLflow) на датасеті Spotify Tracks: регресія популярності треків. Лабораторна 2 додає DVC для версіонування даних та пайплайн prepare → train. Лабораторна 3 додає HPO з Optuna, Hydra та MLflow nested runs (класифікація: popularity ≥ threshold).

## Середовище

- Python 3.10+
- Віртуальне середовище: `python3 -m venv venv`, потім `source venv/bin/activate` (Linux/macOS) або `.\venv\Scripts\activate` (Windows).
- Встановлення залежностей: `pip install -r requirements.txt`.

## Структура

- `data/raw/` — сирі дані (dataset.csv); версіонуються через DVC (файл `dataset.csv.dvc`).
- `data/processed/` — оброблений датасет (нормалізація, без outliers, track_genre закодовано); **його використовують експерименти** при запуску без DVC.
- `data/prepared/` — вихід етапу **prepare** DVC-пайплайну: `train.csv`, `test.csv`, `genre_mapping.json`.
- `data/models/` — вихід етапу **train**: збережена модель `model.joblib`.
- `notebooks/01_eda.ipynb` — первинний аналіз даних (EDA).
- `src/preprocess.py` — обробка даних (один файл у data/processed); використовується також у `prepare.py`.
- `src/prepare.py` — етап DVC-пайплайну: raw → train.csv + test.csv у data/prepared.
- `src/train.py` — навчання регресії з логуванням у MLflow; при виклику з DVC читає data/prepared і зберігає модель у data/models. Підтримує моделі: **rf**, **gbm**, **hist_gbm**, **ridge**.
- `src/optimize.py` — HPO (Lab 3): Optuna + Hydra, бінарна класифікація (target = popularity ≥ 50), RF/Logistic Regression, nested MLflow runs.
- `config/` — Hydra-конфігурація для HPO: `config.yaml`, `model/random_forest.yaml`, `model/logistic_regression.yaml`, `hpo/optuna.yaml`, `hpo/random.yaml`, `hpo/grid.yaml`.
- `dvc.yaml` — опис пайплайну DVC (стадії prepare та train).

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

---

## Лабораторна робота 2 — DVC (версіонування даних та пайплайн)

Пайплайн: **prepare** (сирі дані → data/prepared) → **train** (data/prepared → модель у data/models + MLflow).

**Повна документація лабораторної 2:** [LAB2_DVC.md](LAB2_DVC.md) — що зроблено, структура пайплайну, приклади перевірки (кеш, зміна даних/коду, Docker).

### Передумови

- У проєкті вже є `.dvc/config` з remote `mylocal` (шлях `../dvc_storage`). Якщо DVC ще не ініціалізовано: `pip install dvc`, потім `dvc init` і при потребі `dvc remote add -d mylocal ../dvc_storage`.
- Сирі дані: `data/raw/dataset.csv` (має бути присутній).

### Кроки для виконання лабораторної 2

1. **Додати сирі дані під DVC** (один раз):
   ```bash
   dvc add data/raw/dataset.csv
   git add data/raw/dataset.csv.dvc data/raw/.gitignore
   git commit -m "Track raw dataset with DVC"
   mkdir -p ../dvc_storage   # якщо ще немає
   dvc push
   ```

2. **Запустити пайплайн:**
   ```bash
   dvc repro
   ```
   Після успіху з’явиться `dvc.lock`. Закомітити: `git add dvc.yaml dvc.lock`, `git commit -m "Create DVC pipeline"`.

3. **Перевірка кешу DVC:**
   - Повторно запустити `dvc repro` — обидві стадії мають бути пропущені (skipping).
   - Змінити щось лише в `src/train.py` (наприклад, коментар або гіперпараметр) і знову запустити `dvc repro`: виконається лише стадія **train**, **prepare** — з кешу.

### Запуск етапів вручну (без DVC)

```bash
python src/prepare.py data/raw/dataset.csv data/prepared
python src/train.py data/prepared data/models
```

Модель з’явиться в `data/models/model.joblib`, метрики — у MLflow (`mlflow ui`).

### Виконання DVC з середини Docker

У контейнері вже встановлено DVC і git; проєкт і DVC-сховище монтуються з хоста, тому всі команди можна виконувати з контейнера.

1. **Сховище DVC на хості** (один раз): каталог має бути на один рівень вище проєкту, щоб у контейнері шлях `../dvc_storage` вказував на змонтований том:
   ```bash
   mkdir -p ../dvc_storage
   ```

2. **Збірка та запуск команд у контейнері** (з кореня проєкту на хості):
   ```bash
   docker compose build
   ```

   Додати сирі дані в DVC і відправити в remote:
   ```bash
   docker compose run --rm train dvc add data/raw/dataset.csv
   docker compose run --rm train git add data/raw/dataset.csv.dvc .dvc/.gitignore
   ```
   Перед першим `git commit` з контейнера налаштуйте ім’я та email (без `--global`, щоб збереглося в репо на хості):
   ```bash
   docker compose run --rm train git config user.email "your@email.com"
   docker compose run --rm train git config user.name "Your Name"
   ```
   Потім:
   ```bash
   docker compose run --rm train git commit -m "Track raw dataset with DVC"
   docker compose run --rm train dvc push
   ```

   Запустити пайплайн і зафіксувати результат:
   ```bash
   docker compose run --rm train dvc repro
   docker compose run --rm train git add dvc.yaml dvc.lock
   docker compose run --rm train git commit -m "Create DVC pipeline"
   ```

   Перевірка кешу:
   ```bash
   docker compose run --rm train dvc repro
   # Має показати, що стадії пропущені. Потім змініть src/train.py на хості і знову:
   docker compose run --rm train dvc repro
   # Має перезапуститися лише стадія train.
   ```

   Усі зміни (`.dvc`-файли, `dvc.lock`, `data/prepared/`, `data/models/`) з’являються на хості, бо проєкт змонтовано як `.:/app`. Коміти робляться у вашому локальному репозиторії.

---

## Лабораторна робота 3 — HPO (Optuna, Hydra, MLflow nested runs)

Гіперпараметрична оптимізація: бінарна класифікація (target = popularity ≥ поріг), моделі Random Forest та Logistic Regression, метрики F1/ROC-AUC. Кожен trial логується як дочірній (nested) run у MLflow.

### Передумови

- Виконати `dvc repro` або `python src/prepare.py data/raw/dataset.csv data/prepared`, щоб були `data/prepared/train.csv` та `test.csv`.
- Встановити залежності: `pip install -r requirements.txt` (optuna, hydra-core, omegaconf).

### Запуск HPO

З кореня проєкту:

```bash
# За замовчуванням: model=random_forest, sampler=TPE, n_trials=20
python src/optimize.py

# Інша модель, інший sampler, більше trials
python src/optimize.py model=logistic_regression hpo.sampler=random hpo.n_trials=30

# Порівняння двох sampler-ів (TPE та Random) по 20 trials
bash scripts/run_hpo_samplers.sh
# або змінна: N_TRIALS=25 bash scripts/run_hpo_samplers.sh
```

У MLflow UI (експеримент **HPO_Lab3**) буде один parent run на запуск, у ньому — дочірні runs по trial; артефакти: `config_resolved.json`, `best_params.json`, `best_model.pkl`, модель у artifact_path `model`.

### Реєстрація моделі в Registry (опційно)

Якщо використовується MLflow tracking server з backend store, у `config/config.yaml` встановити `mlflow.register_model: true`. Найкраща модель буде зареєстрована і переведена в стадію Staging.

### Запуск ЛР3 з Docker

Виконати по черзі (з кореня проєкту):

```bash
# 1. Зібрати образ (optuna, hydra-core, config/ вже в requirements та Dockerfile)
docker compose build

# 2. Підготувати дані (train.csv, test.csv у data/prepared)
docker compose run --rm train dvc repro
# або без DVC: docker compose run --rm train python src/prepare.py data/raw/dataset.csv data/prepared

# 3. Запустити HPO: 20 trials, TPE (за замовчуванням)
docker compose run --rm train python src/optimize.py hpo.n_trials=20

# 4. Запустити HPO з Random sampler (для порівняння, Крок 8 методички)
docker compose run --rm train python src/optimize.py hpo=random hpo.n_trials=20
```

Один скрипт замість кроків 3–4 (TPE + Random по 20 trials):

```bash
bash scripts/run_hpo_samplers_docker.sh
```

Переглянути результати: запустити MLflow UI та відкрити експеримент **HPO_Lab3**:

```bash
docker compose up mlflow-ui
# У браузері: http://localhost:5001
```

---

## Лабораторна робота 4 — CI/CD (GitHub Actions + CML)

CI-пайплайн: лінтинг (flake8, black), підготовка даних, тренування класифікатора (`scripts/train_ci.py`), тести (pre/post-train, Quality Gate за F1), CML-звіт у Pull Request. Повний датасет для CI зберігається в репо: `data/raw/dataset.csv`.

**Щоб CI проходив у GitHub Actions**, один раз додайте повний датасет у репо (якщо його ще немає в Git):

```bash
# Якщо файл зараз тільки в DVC, він вже є локально в data/raw/
git add data/raw/dataset.csv
git commit -m "Add full dataset for CI (Lab 4)"
git push
```

Увага: GitHub не приймає файли > 100 MB. Якщо `dataset.csv` більший, використовуйте [Git LFS](https://git-lfs.github.com/) або DVC з хмарним remote для CI.

Після push або відкриття PR workflow **Model CI (Train, Test, Report)** запускається автоматично; у PR з’явиться коментар із метриками та confusion matrix (CML).

---

## Лабораторна робота 5 — Оркестрація ML-пайплайнів (Airflow, Docker multi-stage, CI)

**Передумови:** виконані ЛР1–ЛР4 (MLflow, DVC, HPO, CI/CD).

### Що реалізовано

- **Multi-stage Dockerfile:** збірка залежностей у builder-образі, фінальний образ на `python:3.11-slim` з мінімальним набором для запуску скриптів (DVC, MLflow, requirements).
- **Apache Airflow:** окремий compose `docker-compose.airflow.yml` (Postgres + Scheduler + Webserver, LocalExecutor). Образ з ML-залежностями: `Dockerfile.airflow`.
- **DAG `ml_training_pipeline`:** перевірка даних (FileSensor) → DVC repro prepare → тренування (`scripts/train_ci.py`) → оцінка (XCom) → гілка за accuracy (BranchPythonOperator): реєстрація в MLflow Model Registry (Staging) або stop.
- **CI:** у workflow додано перевірку DAG (DagBag, тест `tests/test_dag_integrity.py`) та крок `docker build -t mlops-lab5 .`.

### Запуск Airflow та DAG

```bash
# Перший раз: ініціалізація БД та користувач admin/admin
docker compose -f docker-compose.airflow.yml up -d
# UI: http://localhost:8080

# У UI увімкніть DAG "ml_training_pipeline" і запустіть (Trigger DAG).
# Проєкт змонтовано в /opt/airflow/ml_project, MLFLOW_TRACKING_URI вказує на mlruns там.
```

Опційно: створити `../dvc_storage` і додати volume `../dvc_storage:/opt/airflow/dvc_storage` у `docker-compose.airflow.yml`, якщо потрібен DVC remote.

- **Звіт ЛР5 (архітектура + контрольні питання):** [docs/LAB5_REPORT.md](docs/LAB5_REPORT.md)
- **Інструкції з перевірки (збірка, Airflow, DAG):** [docs/LAB5_VERIFICATION.md](docs/LAB5_VERIFICATION.md)
