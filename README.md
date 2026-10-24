# Генерация Python-кода по текстовой задаче

Encoder–decoder Transformer, обученный с нуля на TinySMS. Модель генерирует
Python-функцию `solve()` по условию задачи. Лучший public pass@1 —
**61,77% (593/960)**.

Модель: **51,5 млн параметров**, 5 слоёв encoder и 5 decoder, `d_model=512`,
8 голов, Pre-LayerNorm, RoPE, SwiGLU с `d_ff=2048`, dropout=0.1.

## Установка

```bash
git clone https://github.com/Mishaq23/hw2_nlp.git
cd hw2_nlp
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

На Kaggle/Colab venv не требуется. Устанавливайте зависимости без `--upgrade`,
сохраняя предустановленный PyTorch с CUDA. Обучение и инференс используют одну GPU
и FP32, без AMP.

## Обучение

Используется выборка **2,4 млн исходных строк из всех 60 шардов**
`SlavaYus/tinysms_qa_12m`, sampling seed=42. Версия датасета закреплена в
[конфиге данных](src/configs/data_preparation/tinysms.yaml).
`train.py` автоматически готовит данные и обучает BPE-токенайзер со словарём 8192.

Обучение состоит из двух стадий:

| Параметр | С нуля | Дообучение |
| --- | --- | --- |
| Конфиг | [best_train](src/configs/best_train.yaml) | [best_finetune](src/configs/best_finetune.yaml) |
| Бюджет шагов | 57 500 | 30 000 дополнительных |
| Batch size | 32 | 32 |
| Начальный LR | 3e-4 | 3.666241190635664e-5 |
| Cosine T_max / eta_min | 85 000 / 3e-6 | 30 000 / 3e-6 |
| Train seed | 1 | 2 |

Оптимизатор — AdamW. `epoch` в конфигах означает окно из 500 шагов.
Бюджет первой стадии восстановлен по LR исходного checkpoint; исходный горизонт
cosine в 85 000 шагов сохранён.

Запускайте команды из корня репозитория:

```bash
export CUDA_VISIBLE_DEVICES=0
export TINYSMS_DATA_DIR="$PWD/data/processed/tinysms"
python train.py -cn best_train trainer.device=cuda
python train.py -cn best_finetune trainer.device=cuda
```

Первая стадия сохраняется в `saved/best_stage1/`, вторая — в `saved/best_stage2/`.
Дообучение использует те же готовые данные и токенайзер, восстанавливает веса и
моменты AdamW из полного `saved/best_stage1/model_best.pth`.
При переносе между сессиями сохраните готовые данные и каталог модели.

Для дообучения другого собственного checkpoint:

```bash
python train.py -cn best_finetune trainer.device=cuda \
  trainer.finetune_from=/path/to/model_best.pth writer.run_name=my_finetune
```

## Инференс

Откройте [inference.ipynb](inference.ipynb) в Kaggle или Google Colab, выберите
**одну NVIDIA T4**, включите Internet и задайте `MODEL_SOURCE` — путь к каталогу
модели или ZIP. Веса предоставляются отдельно. Нужны три файла:

- `config.yaml`;
- `tokenizer.json`;
- `model_best_greedy.pth` или `model_best.pth`.

Запустите все ячейки по порядку; поддерживается Kaggle **Save & Run All**.
Ноутбук скачивает только тестовые вопросы и сохраняет `submission.jsonl`:
1919 строк вида `{"code": "..."}` в исходном порядке. Время записывается в
`inference_metrics.json`. Прежние прогоны итоговой модели на T4 укладывались
в лимит **180 секунд**.

Настройки [best_inference](src/configs/best_inference.yaml): greedy с KV cache,
batch=16, максимум 1024 новых токена с остановкой по EOS.
Для запуска из терминала с уже подготовленным `encoded/test`:

```bash
export TINYSMS_DATA_DIR=/path/to/prepared
python inference.py -cn best_inference \
  inferencer.checkpoint_dir=/path/to/model \
  inferencer.from_pretrained=/path/to/model/model_best_greedy.pth
```

CLI сохраняет `data/saved/code_generation/test/predictions.jsonl` с полями
`sample_id` и `code`. Ноутбук экспортирует для бота только `code`.

---

Основано на [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
Лицензия — [MIT](LICENSE).
