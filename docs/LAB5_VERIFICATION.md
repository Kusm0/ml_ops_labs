# Перевірка виконання Лабораторної роботи 5

Покрокові інструкції, щоб переконатися, що збірка, Airflow і DAG працюють коректно.

## 1. Перевірка збірки Docker-образу

З кореня проєкту:

```bash
docker build -t mlops-lab5 .
```

Очікуваний результат: збірка завершується без помилок, образ `mlops-lab5` з’являється в `docker images`.

Опційно — запуск тренування в контейнері (переконатися, що образ працює):

```bash
docker run --rm -v "$(pwd)/mlruns:/app/mlruns" mlops-lab5 python src/train.py --help
```

## 2. Запуск Apache Airflow

Переконайтеся, що в корені проєкту є:
- `data/raw/dataset.csv` (сирі дані для DAG)
- при потребі каталог `../dvc_storage` для DVC remote (або використовуйте лише локальні дані)

Запустіть Airflow:

```bash
docker compose -f docker-compose.airflow.yml up -d
```

Перший раз може знадобитися ініціалізація БД. Якщо сервіси вже описані з `airflow-init`, він створить користувача. Якщо ні — виконайте один раз:

```bash
docker compose -f docker-compose.airflow.yml run --rm airflow-init
```

Дочекайтеся, поки webserver і scheduler будуть готові (30–60 с). Відкрийте в браузері:

**http://localhost:8080**

Логін: `admin`, пароль: `admin` (або значення з вашого compose).

## 3. Перевірка DAG у Airflow

1. У списку DAG-ів має з’явитися **ml_training_pipeline** (без червоних позначок помилок імпорту).
2. Увімкніть DAG (перемикач «Off» → «On»).
3. Натисніть **Trigger DAG** (кнопка ▶), щоб запустити один раз.

## 4. Перевірка виконання пайплайну

1. Відкрийте запущений DAG run і перегляньте граф задач.
2. Задачі мають виконуватися по черзі: **check_data** → **data_prepare** → **model_train** → **evaluate_model** → **branch_on_accuracy** → **register_model** або **stop_pipeline**.
3. Перегляньте логи кожної задачі (клік по задачі → Log), якщо щось пішло не так.
4. Якщо accuracy > 0.85: має виконатися **register_model**. Перевірте в MLflow (локально: `mlflow ui --backend-store-uri mlruns/` або відповідний URI) — модель має з’явитися в Model Registry зі стадією Staging.
5. Якщо accuracy ≤ 0.85: виконається **stop_pipeline**, реєстрації не буде.

## 5. Зупинка Airflow

```bash
docker compose -f docker-compose.airflow.yml down
```

Дані Postgres зберігаються в Docker volume `postgres-db-volume`; при `down -v` том буде видалено.

## Типові проблеми

- **DAG не з’являється або помилка імпорту:** перевірте, що папка `dags/` змонтована і містить `ml_training_pipeline.py`; перегляньте логи scheduler.
- **check_data не проходить:** переконайтеся, що `data/raw/dataset.csv` існує у проєкті на хості (том `.:/opt/airflow/ml_project`).
- **data_prepare падає:** переконайтеся, що в контейнері є DVC і що `dvc repro prepare` виконується з кореня проєкту (`/opt/airflow/ml_project`); при використанні DVC remote налаштуйте том для `dvc_storage`.
- **Помилки прав доступу:** на Linux можливо знадобиться `AIRFLOW_UID=$(id -u)` при запуску compose (див. коментар у `docker-compose.airflow.yml`).
