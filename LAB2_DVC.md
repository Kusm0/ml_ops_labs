# Лабораторна робота 2 — DVC: версіонування даних та пайплайн

Документація виконаної роботи: що зроблено, як це працює, і приклади перевірки.

---

## 1. Що було зроблено

### 1.1. DVC у проєкті

- **requirements.txt** — додано залежність `dvc>=3.0.0`.
- **Каталог `.dvc/`** — ініціалізація DVC:
  - `.dvc/config` — default remote `mylocal` з URL `../dvc_storage` (локальне сховище на рівень вище проєкту).
  - `.dvc/.gitignore` — ігноруються `cache`, `tmp`, `runs`.

### 1.2. Версіонування сирих даних

- Сирі дані **data/raw/dataset.csv** додано під контроль DVC командою `dvc add data/raw/dataset.csv`.
- У Git комітяться лише метафайли:
  - **data/raw/dataset.csv.dvc** — опис файлу (хеш, розмір); сам CSV у Git не потрапляє.
  - **data/raw/.gitignore** (або **.dvc/.gitignore** залежно від версії DVC) — щоб великий файл не потрапляв у репозиторій.
- Кореневий **.gitignore** змінено: замість ігнорування всього `data/` ігноруються лише конкретні шляхи (`data/raw/dataset.csv`, `data/processed/`, `data/prepared/`, `data/models/`), щоб у репо можна було зберігати `data/raw/*.dvc` та `data/raw/.gitignore`.

### 1.3. Рефакторинг: два етапи пайплайну

**Новий скрипт `src/prepare.py` (етап prepare):**

- **Вхід:** сирі дані — `data/raw/dataset.csv` (шлях передається аргументом).
- **Логіка:** використовує функції з `src/preprocess.py`: завантаження CSV, додавання `explicit_numeric`, кодування `track_genre` (LabelEncoder), видалення outliers (IQR), нормалізація (StandardScaler). Потім розділення на train/test (`train_test_split`, test_size=0.2, random_state=42).
- **Вихід:** каталог `data/prepared/` з файлами `train.csv`, `test.csv`, `genre_mapping.json`.
- **Виклик:** `python src/prepare.py data/raw/dataset.csv data/prepared`.

**Оновлений `src/train.py` (етап train):**

- **Вхід:** каталог з підготовленими даними (наприклад `data/prepared`) — читає `train.csv` і `test.csv`. Збережена сумісність: якщо передано один файл або каталог без train/test — використовується старий режим (один CSV + split у пам’яті).
- **Вихід:**
  - Модель на диск: **data/models/model.joblib** (joblib), щоб DVC міг трактувати `data/models` як вихід пайплайну.
  - Метрики та артефакти — як і раніше через MLflow.
- **Виклик з пайплайну:** `python src/train.py data/prepared data/models` (два позиційні аргументи: каталог даних, каталог для моделі).

### 1.4. Опис пайплайну (dvc.yaml)

У корені проєкту створено **dvc.yaml** з двома стадіями:

| Стадія   | Команда                                              | Залежності (deps)              | Виходи (outs)   |
|----------|------------------------------------------------------|-------------------------------|-----------------|
| prepare  | `python src/prepare.py data/raw/dataset.csv data/prepared` | data/raw/dataset.csv, src/prepare.py | data/prepared   |
| train    | `python src/train.py data/prepared data/models`     | data/prepared, src/train.py   | data/models     |

Після першого успішного `dvc repro` з’являється **dvc.lock** — фіксує хеші всіх залежностей і виходів для відтворюваності.

### 1.5. Docker

- **Dockerfile:** встановлено `git` (для `git add`/`commit` з контейнера), у образ копіюються `dvc.yaml` та `.dvc/`.
- **docker-compose.yml:** для сервісу `train` додано томи:
  - `.:/app` — весь проєкт з хоста (код, data, .git, .dvc), щоб результати DVC і git були на хості.
  - `../dvc_storage:/dvc_storage` — сховище DVC (у контейнері `../dvc_storage` з `/app` = `/dvc_storage`).
- Перший `git commit` з контейнера вимагає налаштування імені: `git config user.email` та `git config user.name` (без `--global`, щоб збереглося в репо).

---

## 2. Як показати, що все працює

Нижче — приклади перевірки (локально або через Docker). Для Docker усі команди виконуються з кореня проєкту на хості; передумова: `mkdir -p ../dvc_storage`, `docker compose build`.

### 2.1. Повний цикл: DVC add → push → repro → commit

Показати, що сирі дані версіонуються, пайплайн запускається і результат фіксується в Git.

**Локально (з venv):**
```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .dvc/.gitignore   # або data/raw/.gitignore — дивись вивід dvc add
git config user.email "you@example.com"            # якщо ще не налаштовано
git config user.name "Your Name"
git commit -m "Track raw dataset with DVC"
mkdir -p ../dvc_storage
dvc push

dvc repro
git add dvc.yaml dvc.lock
git commit -m "Create DVC pipeline"
```

**У Docker:**
```bash
docker compose run --rm train dvc add data/raw/dataset.csv
docker compose run --rm train git add data/raw/dataset.csv.dvc .dvc/.gitignore
docker compose run --rm train git config user.email "you@example.com"
docker compose run --rm train git config user.name "Your Name"
docker compose run --rm train git commit -m "Track raw dataset with DVC"
docker compose run --rm train dvc push

docker compose run --rm train dvc repro
docker compose run --rm train git add dvc.yaml dvc.lock
docker compose run --rm train git commit -m "Create DVC pipeline"
```

Очікування: стадії `prepare` і `train` виконуються, з’являються `data/prepared/` (train.csv, test.csv, genre_mapping.json), `data/models/model.joblib`, файл `dvc.lock`.

---

### 2.2. Кеш DVC — нічого не змінилося

Показати, що при повторному запуску без змін DVC нічого не перераховує.

**Команда:**
```bash
dvc repro
# або в Docker:
docker compose run --rm train dvc repro
```

**Очікуваний вивід:** щось на кшталт  
`'data/raw/dataset.csv.dvc' didn't change, skipping`  
`Stage 'prepare' didn't change, skipping`  
`Stage 'train' didn't change, skipping`  

Тобто обидві стадії пропускаються, використовується кеш.

---

### 2.3. Кеш DVC — змінили лише код навчання

Показати, що при зміні лише `src/train.py` перезапускається лише стадія **train**, стадія **prepare** береться з кешу.

1. Змінити щось у **src/train.py**, наприклад:
   - додати коментар у кінці файлу, або
   - змінити гіперпараметр за замовчуванням: `default=10` → `default=12` для `max_depth`.

2. Запустити:
   ```bash
   dvc repro
   # або: docker compose run --rm train dvc repro
   ```

**Очікування:**  
- `Stage 'prepare' didn't change, skipping`  
- `Running stage 'train':` — виконується лише train, prepare не перезапускається.

Це демонструє економію часу: дані й код prepare не змінювалися, DVC перезапускає тільки залежні від змін кроки.

---

### 2.4. Зміна сирих даних — перезапуск усього пайплайну

Показати, що при зміні сирих даних DVC перезапускає обидві стадії.

1. Трохи змінити **data/raw/dataset.csv** (наприклад, додати рядок або змінити одне значення) і зберегти файл.
2. Оновити DVC-метафайл і перезапустити пайплайн:
   ```bash
   dvc add data/raw/dataset.csv
   dvc repro
   # або в Docker:
   docker compose run --rm train dvc add data/raw/dataset.csv
   docker compose run --rm train dvc repro
   ```

**Очікування:** виконуються обидві стадії — спочатку `prepare`, потім `train`. У консолі видно `Running stage 'prepare':` і далі `Running stage 'train':`.

---

### 2.5. Запуск етапів без DVC (вручну)

Показати, що скрипти працюють окремо, без `dvc repro`.

**Локально:**
```bash
python src/prepare.py data/raw/dataset.csv data/prepared
python src/train.py data/prepared data/models
```

**У Docker:**
```bash
docker compose run --rm train python src/prepare.py data/raw/dataset.csv data/prepared
docker compose run --rm train python src/train.py data/prepared data/models
```

Перевірити: з’явилися `data/prepared/train.csv`, `data/prepared/test.csv`, `data/models/model.joblib`; у MLflow — новий run (`mlflow ui` з хоста або з контейнера).

---

### 2.6. Відправка виходів пайплайну в remote (dvc push)

Показати, що виходи пайплайну можна зберегти в DVC-сховищі.

Після `dvc repro`:
```bash
dvc push
# або: docker compose run --rm train dvc push
```

У каталозі `../dvc_storage` (або змонтованому томі) з’являються об’єкти, що відповідають `data/prepared` та `data/models`. Це корисно для бекапу або спільної роботи з даними/моделями.

---

## 3. Короткий чеклист для здачі лабораторної 2

- [ ] DVC ініціалізовано, у репо є `.dvc/config` з local remote.
- [ ] Сирі дані під DVC: є `data/raw/dataset.csv.dvc`, великий CSV не в Git.
- [ ] Є два окремі етапи: `src/prepare.py` (raw → prepared) та `src/train.py` (prepared → model + MLflow).
- [ ] Пайплайн описано в `dvc.yaml` (prepare, train) з deps і outs.
- [ ] `dvc repro` успішно виконується, з’являється `dvc.lock`.
- [ ] Продемонстровано кеш: повторний `dvc repro` — пропуск стадій; зміна лише `train.py` — перезапуск лише train.

Якщо всі пункти виконані і приклади з розділу 2 працюють — лабораторну можна вважати виконаною.
