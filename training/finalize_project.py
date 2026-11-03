"""Create dataset gallery, model cards and experiment/defense documents from real metrics."""
import csv
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from common.checkpoint import atomic_json
from train_baselines import ROOT


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    pump = ROOT / 'artifacts/anomaly_pump'
    project = ROOT / 'artifacts/ast_project'
    if not (pump / 'deployment.json').exists() or not (project / 'complete.json').exists():
        raise ValueError('Both trained tasks C and B must finish before finalization')
    manifest = read_json(ROOT / 'data/project_manifest.json')
    pump_manifest = read_json(ROOT / 'data/pump_manifest.json')
    deployment = read_json(pump / 'deployment.json')
    gallery = ROOT / 'data/gallery'
    gallery.mkdir(exist_ok=True)
    sample_path = ROOT / 'samples/manifest.json'
    original = [row for row in read_json(sample_path) if not row['id'].startswith('real_')]
    examples, attributions = [], []
    def add(key, title, row, dataset, license, author, source, machine=None):
        wave, sr = sf.read(ROOT / row['file'], dtype='float32', always_2d=True)
        wave = wave.mean(1)[:10 * sr]
        if sr != 48000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=48000)
        target = gallery / f'{key}.wav'
        sf.write(target, wave, 48000, subtype='PCM_16')
        example = dict(id='real_' + key, title=title + ' — датасет', file=f'../data/gallery/{key}.wav',
            description=f'{dataset}; {row.get("split", row.get("original_split", ""))}; автор: {author}; {license}. Источник: {source}',
            synthetic=False, dataset=dataset, source_url=source, license=license, author=author)
        if machine:
            example['machine_id'] = machine
        examples.append(example)
        attributions.append(dict(file=target.relative_to(ROOT).as_posix(), source_file=row['file'],
            source_url=source, author=author, license=license,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            transformation='First up to 10 seconds, mono, resampled to 48kHz, PCM16; demonstration only'))
    for key, label, title in [('voice', 'speech', 'Голос'), ('whistle', 'whistling', 'Свист'),
                              ('clap', 'clapping', 'Хлопки'), ('siren', 'siren', 'Сирена'), ('music', 'music', 'Музыка')]:
        row = sorted([row for row in manifest['splits']['train'] if row['label'] == label], key=lambda row: int(row['freesound_id']))[0]
        add(key, title, row, 'FSD50K-project', row['license'], row['uploader'], row['source_url'])
    for key, condition, part, title in [('pump_normal', 0, 'train', 'Насос: норма'),
                                      ('pump_anomaly', 1, 'test', 'Насос: аномалия')]:
        row = sorted([row for row in pump_manifest['splits'][part] if row['label'] == condition and row['machine_id'] == 'id_00'],
                     key=lambda row: row['file'])[0]
        add(key, title, row, 'DCASE2020-pump', 'CC-BY-NC-SA-4.0', 'NTT Corporation and Hitachi, Ltd.',
            'https://zenodo.org/records/3678171', 'id_00')
    atomic_json(sample_path, examples + original)
    atomic_json(gallery / 'attributions.json', attributions)
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    with (ROOT / 'artifacts/report_esc50/summary.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    if {row['model'] for row in summary} != {'wave_cnn', 'mel_cnn', 'ast'}:
        raise ValueError('ESC-50 comparison must include all three models')
    metrics_b = read_json(project / 'metrics.json')
    metrics_c = read_json(pump / 'metrics.json')
    metrics_lof = read_json(pump / 'lof/metrics.json')
    lines = ['# Как ИИ слышит мир: экспериментальный отчёт', '',
        '## Постановка задачи', '',
        'Цель — показать связь между математическим представлением звука и его машинным распознаванием. '
        'Задачи: реализовать интерактивные преобразования, проверить их аналитически, сравнить волну и спектрограмму '
        'на одинаковом разбиении, обучить классификатор звуков и детектор насоса, сохранить воспроизводимые результаты. '
        'Новые пользовательские аудиозаписи не входят в эксперимент: все обучающие и тестовые данные публичные.', '',
        '## Математическая основа', '',
        'После дискретизации x[n]=f(n/f_s). Для сигнала с ограниченным спектром восстановление требует f_s>2f_max; '
        'перед уменьшением частоты дискретизации нужен антиалиасинговый фильтр. В приложении можно сравнить варианты с фильтром и без него.', '',
        r'Ряд Фурье: $S_N(t)=a_0/2+\sum_{k=1}^N[a_k\cos(k\omega_0t)+b_k\sin(k\omega_0t)]$. '
        r'Коэффициенты $a_k=(2/T)\int_0^T f(t)\cos(k\omega_0t)dt$ и аналогично $b_k$ '
        'выделяют гармоники благодаря ортогональности. Численные интегралы сверены с FFT и известными коэффициентами. '
        'Ряд строится на периодическом продолжении выбранного участка; скачок на его границе создаёт эффект Гиббса.', '',
        r'ДПФ: $X[k]=\sum_{n=0}^{M-1}x[n]e^{-2\pi i kn/M}$. '
        r'При принятой нормировке Парсеваль даёт $\sum_n|x[n]|^2=M^{-1}\sum_k|X[k]|^2$. '
        'Зануление коэффициентов и обратная ДПФ позволяют прослушать вклад частоты или полосы. '
        'Окно Ханна уменьшает спектральную утечку, но расширяет главный лепесток.', '',
        r'STFT: $X(m,k)=\sum_n x[n+mH]w[n]e^{-2\pi i kn/L}$. '
        'Длина окна L определяет компромисс между локализацией во времени и разделением частот. '
        r'Лог-мел для насоса: $10\log_{10}(MS+10^{-10})$, где S — спектр мощности, M — мел-банк. '
        'AST использует свой закреплённый ASTFeatureExtractor, а не эту реализацию насоса.', '',
        r'Классификатор проекта: $p_i=e^{z_i}/\sum_j e^{z_j}$. '
        'Исходная AST AudioSet многометочная: независимые sigmoid не обязаны суммироваться в единицу. '
        'Для насоса score — среднее отрицательных логарифмов вероятности истинного ID по вырезкам. '
        'LOF сравнивает плотность окрестности клипа с плотностью его соседей. '
        r'$TPR=TP/(TP+FN)$, $FPR=FP/(FP+TN)$; ROC меняет порог, AUC оценивает ранжирование. '
        'Порог классификации и AUC — разные характеристики.', '',
        '## Цель и реализация', '',
        'Локальное приложение связывает аудиозапись, ряд Фурье, ДПФ, STFT, лог-мел-признаки, классификацию звуков и обнаружение аномалий насоса. '
        'Математические сервисы отделены от Django и проверены аналитическими тестами. Интерфейс работает локально на русском языке, с числовыми подстановками в формулах.', '',
        '## A. Сравнение представлений на ESC-50', '',
        'Моно 16 кГц, 5 секунд, seed 2026; одинаковые исходные test-фолды. Validation — следующий фолд, train — остальные три. '
        'Пересечения исходников удалены из train/validation (4 Freesound ID, 9 клипов); test сохранён. Это модифицированный протокол. '
        'Один seed; разброс ниже — выборочное SD по пяти фолдам. AST имеет предобучение AudioSet, поэтому преимущество нельзя приписывать одной спектрограмме.', '',
        '| Модель | Accuracy, mean ± SD | Macro-F1, mean ± SD |', '|---|---:|---:|']
    for row in summary:
        lines.append(f"| {row['model']} | {float(row['accuracy_mean']):.4f} ± {float(row['accuracy_std']):.4f} | {float(row['macro_f1_mean']):.4f} ± {float(row['macro_f1_std']):.4f} |")
    lines += ['', '### Параметры и вычислительная стоимость', '',
        '| Модель | Параметров | Среднее время обучения фолда, с | Средний инференс клипа, мс |',
        '|---|---:|---:|---:|']
    with (ROOT / 'artifacts/report_esc50/fold_metrics.csv').open() as handle:
        fold_results = list(csv.DictReader(handle))
    for row in summary:
        parameters = next(fold['parameters'] for fold in fold_results if fold['model'] == row['model'])
        lines.append(f"| {row['model']} | {parameters} | {float(row['training_seconds_mean']):.2f} | {1000 * float(row['inference_seconds_per_clip_mean']):.3f} |")
    lines += ['', 'Время зависит от устройства, кэширования и реализации; это измерения текущей машины, не универсальный benchmark. '
        'CNN обучаются до 30 эпох, AST — до 5 эпох головы и 25 полного дообучения; early stopping по validation macro-F1. '
        'Бюджет эпох близок, но представления, предобучение и стоимость моделей различаются. '
        'AdamW, cosine с warmup 10%, mixup 0.2, временной сдвиг и шум; SpecAugment только у спектральных моделей. '
        'Точные настройки — training/configs/baselines.yaml и ast_esc50.yaml.', '',
        'Полные результаты шума SNR 20/10/5/0 дБ и пофолдовые показатели — artifacts/report_esc50/snr.csv и fold_metrics.csv.', '',
        '| Модель | SNR, дБ | Mean accuracy | Mean macro-F1 |', '|---|---:|---:|---:|']
    with (ROOT / 'artifacts/report_esc50/snr.csv').open() as handle:
        noise_results = list(csv.DictReader(handle))
    for row in summary:
        for level in [20, 10, 5, 0]:
            values = [value for value in noise_results if value['model'] == row['model'] and float(value['snr_db']) == level]
            lines.append(f"| {row['model']} | {level} | {np.mean([float(value['accuracy']) for value in values]):.4f} | {np.mean([float(value['macro_f1']) for value in values]):.4f} |")
    lines += ['',
        '## C. Аномалии насоса', '',
        'DCASE 2020 Task 2 development pump: train только норма, train/validation 90/10 по ID. '
        'Тестовые аномалии не участвуют в обучении и выборе порога. Порог — 90-й перцентиль normal validation по ID. '
        'ID-CNN использует средний −log p истинного ID по 64-кадровым вырезкам. LOF: 20 соседей, novelty=True, '
        'отдельно по ID на 768-мерных эмбеддингах исходной AST. pAUC стандартизована при FPR≤0,1.', '',
        '| Подход | Mean AUC | Mean pAUC |', '|---|---:|---:|',
        f"| ID-CNN | {metrics_c['mean_auc']:.4f} | {metrics_c['mean_pauc']:.4f} |",
        f"| AST + LOF | {metrics_lof['mean_auc']:.4f} | {metrics_lof['mean_pauc']:.4f} |", '',
        '| ID | ID-CNN AUC | ID-CNN pAUC | LOF AUC | LOF pAUC |', '|---|---:|---:|---:|---:|']
    for machine in ['id_00', 'id_02', 'id_04', 'id_06']:
        lines.append(f"| {machine} | {metrics_c[machine]['auc']:.4f} | {metrics_c[machine]['pauc']:.4f} | {metrics_lof[machine]['auc']:.4f} | {metrics_lof[machine]['pauc']:.4f} |")
    lines += ['',
        f"В приложении по заранее установленному правилу выбран **{deployment['approach']}** по среднему test AUC. "
        'Это сравнение двух подходов по test; результаты обоих сохранены. После выбора параметры не подгонялись. '
        'Оценка не доказывает применимость к новому насосу, другому микрофону или реальному оборудованию.', '',
        'Для справки: официальный autoencoder baseline в репозитории DCASE2020 показывает pump mean AUC 0.72591 и pAUC 0.600019. '
        'Это опубликованный результат другой системы, не наш запуск: https://github.com/y-kawagu/dcase2020_task2_baseline .', '',
        '## B. Классификатор проекта без собственных записей', '',
        manifest['protocol'], '',
        'Использованы только клипы с лицензиями CC0/CC BY и исходной разметкой FSD50K. '
        'Whistling взят из расширенной разметки collection; примеры с несколькими целевыми классами исключены. '
        'Исходники группируются по uploader, части не пересекаются по авторам и Freesound ID. '
        'Признак — первые до 10 секунд; это фиксированное правило без выбора участка по качеству test. '
        'Это собственное проектное разбиение, а не стандартная оценка FSD50K. Пользовательские записи не требовались. '
        'Класс свиста малочисленен; размеры test по каждому классу приведены ниже.', '',
        'Обучение использует веса 1/частота класса и выборку с возвращением: в среднем классы предъявляются поровну. '
        'Это не создаёт новые уникальные примеры свиста. Validation и test проходят один раз без балансировки. '
        'Две фазы AST, seed 2026, max_length 1024, batch 4 × accumulation 4, mixed precision, '
        'early stopping по validation macro-F1; полный конфиг сохранён вместе с моделью.', '',
        'Подготовка waveform общая с приложением: training/common/project_audio.py. '
        'После приведения входа к максимуму 48 кГц берутся первые до 10 секунд, затем '
        'librosa soxr_hq приводит звук к 16 кГц и волна дополняется нулями до 160000 отсчётов. '
        'Только после этого применяется ASTFeatureExtractor. Короткая волна и padding признаков '
        'не взаимозаменяемы; этот контракт зафиксирован в audio_preprocessing.json.', '',
        '| Класс | Train | Validation | Test |', '|---|---:|---:|---:|']
    for label, count in metrics_b['class_test_counts'].items():
        train_count = sum(row['label'] == label for row in manifest['splits']['train'])
        val_count = sum(row['label'] == label for row in manifest['splits']['validation'])
        lines.append(f'| {label} | {train_count} | {val_count} | {count} |')
    lines += ['', '| Условие | Accuracy | Macro-F1 |', '|---|---:|---:|']
    for condition, values in metrics_b['conditions'].items():
        lines.append(f"| {condition} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} |")
    lines += ['', 'Эта таблица показывает raw argmax. Ниже — фактическое решение приложения: низкая уверенность переводится в other по порогу, выбранному на validation.', '',
              '| Условие с порогом | Accuracy | Macro-F1 |', '|---|---:|---:|']
    for condition, values in metrics_b['deployed_conditions'].items():
        lines.append(f"| {condition} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} |")
    best_a = max(summary, key=lambda row: float(row['accuracy_mean']))
    mel_a = next(row for row in summary if row['model'] == 'mel_cnn')
    wave_a = next(row for row in summary if row['model'] == 'wave_cnn')
    clean_b = metrics_b['conditions']['clean']
    lines += ['', '## Выводы', '',
        f"В текущем пятифолдовом эксперименте наибольшая средняя accuracy у {best_a['model']} "
        f"({100 * float(best_a['accuracy_mean']):.2f}%). Разность мел-CNN и волна-CNN составляет "
        f"{100 * (float(mel_a['accuracy_mean']) - float(wave_a['accuracy_mean'])):+.2f} процентного пункта. "
        'Это результат конкретного протокола и одного seed, а не доказательство общего превосходства представления.', '',
        f"Для насоса по заранее принятому правилу выбран {deployment['approach']}; mean AUC обоих подходов и различия по ID приведены выше. "
        'Система оценивает отклонение от известных нормальных примеров, не причину неисправности.', '',
        f"Проектная AST получила raw accuracy {clean_b['accuracy']:.4f} и macro-F1 {clean_b['macro_f1']:.4f} "
        'на 516 клипах собственного разбиения. Результаты после validation-порога приведены отдельно. '
        'Свист имеет только один тестовый пример, поэтому его качество требует нового независимого публичного набора.', '',
        '## Ограничения и воспроизводимость', '',
        'Нет оценки на пользовательском микрофоне или собственной аудитории: по решению пользователя используется только публичный датасет. '
        'Галерея содержит фиксированные реальные примеры и явно помеченный синтез. Галерея не заменяет тестовые метрики. '
        'Все данные, веса и результаты локальные; публикация не выполнялась. Веса safetensors; seed, YAML, окружение, манифесты и TensorBoard сохранены.', '',
        '## Источники', '',
        '- ESC-50: https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0).',
        '- DCASE 2020 dev: https://zenodo.org/records/3678171 (CC BY-NC-SA 4.0).',
        '- FSD50K: https://zenodo.org/records/4060432 (лицензия набора и отдельные лицензии клипов; attributions.csv).',
        '- AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 (BSD-3-Clause).',
        '- CLAP: https://huggingface.co/laion/clap-htsat-unfused (Apache-2.0).',
        '- FSD50K audio mirror pinned revision: ' + manifest['mirror_revision'] + '.', '']
    (reports / 'experiment_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (reports / 'defense.md').write_text('''# Сценарий защиты

1. Загрузить реальный пример голоса или свиста из галереи. Показать волну и частоту дискретизации.
2. На коротком участке увеличить число гармоник. Показать уменьшение RMSE, границу N и периодическое продолжение.
3. Перейти к коэффициенту: произведение с синусом/косинусом, положительные и отрицательные площади, ортогональность.
4. Показать спектр и Парсеваля, вырезать частоту, прослушать восстановление. Объяснить утечку и окно Ханна.
5. Сравнить короткое/длинное окно STFT; показать мел-шкалу и точный вход AST.
6. Сравнить проектную AST, исходную многометочную AST и zero-shot CLAP. Добавить шум, показать реальные изменения вероятностей.
7. Выбрать реальный пример насоса с известным ID, показать score, validation-порог, ROC и матрицу ошибок. Объяснить ограничение применимости.
8. Открыть experiment_report.md: показать пофолдовые метрики, два подхода к аномалиям, малочисленный класс свиста и отсутствие собственных записей.

## Вопросы и ответы

- Почему не все вероятности AST суммируются в 1? AudioSet — multi-label, независимые sigmoid. Проектная AST — single-label, softmax.
- Почему спектрограмма может помогать? Она явно представляет частотно-временную структуру. Честная проверка — два CNN сопоставимого размера; AST ещё использует предобучение.
- Почему нельзя настраивать порог на test? Это сделало бы финальную оценку частью обучения и исказило метрики. Порог выбран на validation.
- Почему нельзя диагностировать любой насос? Обучение ограничено конкретным типом и ID; нет эталона нормы новой машины.
- Что означает pAUC? Стандартизованная частичная AUC для низкого уровня ложных тревог FPR≤0,1, по определению scikit-learn.
- Почему нет собственных записей? Это сознательное ограничение текущего эксперимента, согласованное пользователем; перенос на реальный микрофон не подтверждён.
''', encoding='utf-8')
    threshold_b = read_json(project / 'threshold.json')
    project_card = [
        '---', 'library_name: transformers', 'pipeline_tag: audio-classification',
        'tags:', '- audio', '- ast', '- fsd50k', '---', '',
        '# Project sound classifier — local model card', '',
        'Status: trained and evaluated locally; not uploaded. Publication/license decision is pending.', '',
        '## Model and intended use', '',
        '10-class single-label AST for the educational Fourier sound laboratory. '
        'The UI shows softmax probabilities and redirects low-confidence predictions to other. '
        'This is not a general-purpose sound safety or medical classifier.', '',
        'Base: MIT/ast-finetuned-audioset-10-10-0.4593, revision f826b80d28226b62986cc218e5cec390b1096902. '
        'Architecture, preprocessor and English labels are in config.json/preprocessor_config.json; '
        'Russian labels are in labels_ru.json.', '',
        '## Dataset and protocol', '', manifest['protocol'], '',
        '2468 public FSD50K clips; first up to 10 seconds; CC0/CC BY clips only. '
        'No personal recordings and no ESC-50. All ESC-50 source IDs were excluded. '
        'Splits are disjoint by Freesound ID, uploader and identical source SHA256. '
        'This project-specific pooling of dev/eval and collection-level Whistling is not the official FSD50K benchmark.', '',
        '| Label | Train | Validation | Test |', '|---|---:|---:|---:|']
    for label in manifest['labels']:
        counts = [sum(row['label'] == label for row in manifest['splits'][part]) for part in ['train', 'validation', 'test']]
        project_card.append(f'| {label} | {counts[0]} | {counts[1]} | {counts[2]} |')
    project_card += ['', '## Training', '',
        'Seed 2026; AdamW; head 5 epochs then up to 25 full fine-tuning epochs, early stopping on validation macro-F1. '
        'Waveform mixup, shift and noise; SpecAugment. Training sampling uses inverse class frequency with replacement. '
        'It balances expected draws, not unique examples. Full YAML, source snapshot, environment, data SHA256 and '
        'TensorBoard provenance are retained. See metrics.json for completed epochs and wall time.', '',
        '## Evaluation and confidence threshold', '',
        f"Clean raw accuracy: {clean_b['accuracy']:.4f}; macro-F1: {clean_b['macro_f1']:.4f}. "
        f"Validation-only maximum-probability threshold: {threshold_b['max_prob_threshold']:.6f}. "
        'All clean/noisy and raw/thresholded metrics are in metrics.json. No test-based hyperparameter tuning.', '',
        'Whistling has only 6 training, 2 validation and 1 test clip. Its measured test accuracy is statistically '
        'uninformative. Open-world sounds, new microphones and personal-room recordings were not independently evaluated. '
        'Softmax confidence is not a calibrated probability of correctness.', '',
        '## Local usage', '', '```python',
        'from transformers import ASTFeatureExtractor, ASTForAudioClassification',
        'from training.common.project_audio import prepare',
        'processor = ASTFeatureExtractor.from_pretrained("artifacts/ast_project", local_files_only=True)',
        'model = ASTForAudioClassification.from_pretrained("artifacts/ast_project", local_files_only=True).eval()',
        '# prepare(waveform, sample_rate) applies the shared mono/resample/10-second waveform padding.',
        '# Pass its output to processor(..., sampling_rate=16000, return_tensors="pt").',
        '# Apply softmax over model logits; threshold.json defines the UI confidence policy.', '```', '',
        '## Sources and licensing', '',
        'FSD50K: https://zenodo.org/records/4060432 . Full per-clip author, source URL and license: attributions.csv. '
        'Audio mirror revision: ' + manifest['mirror_revision'] + '. '
        'Base model: BSD-3-Clause. Review upstream notices and data attributions before selecting a distribution license. '
        'LICENSE_NOTES.md is part of the deliverable; no publication was performed.', '']
    (project / 'README.md').write_text('\n'.join(project_card), encoding='utf-8')
    pump_card = [
        '# Educational DCASE2020 pump anomaly detector — local model card', '',
        'Status: trained and evaluated locally; not uploaded. Data conditions: CC BY-NC-SA 4.0.', '',
        '## Intended use and data', '',
        'Educational comparison on DCASE 2020 Task 2 development pump IDs 00/02/04/06. '
        'No applicability claim for a new machine, microphone, other sound or real equipment diagnosis. '
        'Training uses normal audio only: 3014 train and 335 validation files; 856 final test files. '
        'Anomalous test audio is never used for fitting or threshold estimation.', '',
        '## Models and scores', '',
        'ID-CNN: four Conv-BN-ReLU-pool blocks, global average pooling, four machine-ID logits. '
        '16 kHz, FFT/window 1024, hop 512, 128 Slaney mel bands, power 2, 10*log10(S+1e-10). '
        'Score is mean -log p(true ID) over 64-frame crops with hop 32 and final tail included. '
        'Weights: model.safetensors; architecture: model_config.json; exact features: feature_config.json.', '',
        'AST+LOF: original pinned AST, float32 mean last_hidden_state including special/padded tokens, 768 dimensions. '
        'One novelty=True LOF with 20 neighbors for each ID, fit on normal train only; score=-score_samples. '
        'All train/validation/test embeddings, row manifest, trained joblib files and package versions are in lof/. '
        'Only load these locally generated joblib artifacts; do not load arbitrary remote pickle files.', '',
        'Thresholds for both approaches are the 90th percentile of normal validation scores per ID. '
        'Early stopping uses normal validation ID accuracy. Full seed/config/environment/source/data provenance retained.', '',
        '## Final evaluation', '', '| Approach | Mean AUC | Standardized pAUC, max FPR 0.1 |', '|---|---:|---:|',
        f"| ID-CNN | {metrics_c['mean_auc']:.4f} | {metrics_c['mean_pauc']:.4f} |",
        f"| AST+LOF | {metrics_lof['mean_auc']:.4f} | {metrics_lof['mean_pauc']:.4f} |", '',
        f"Application default: {deployment['approach']}, selected by the predeclared final mean test AUC comparison. "
        'Both results remain reported. No tuning follows this selection. Per-ID AUC/pAUC, confusion matrices, ROC '
        'and final scores are in metrics.json, roc.json and test_scores.csv for each approach.', '',
        '## Application and limitations', '',
        'analyzer.services.anomaly.infer requires a known machine ID and explicit applicability. '
        'The UI threshold slider explores saved test results without refitting models or overwriting validation thresholds. '
        'A higher score indicates a difference from the available normal examples, not a diagnosed fault.', '',
        '## Sources and licensing', '',
        'Official source: https://zenodo.org/records/3678171 . CC BY-NC-SA 4.0; source archive checksum is in data_manifest.json. '
        'AST upstream BSD-3-Clause. LICENSE_NOTES.md retains the noncommercial data restriction; '
        'publication is a separate reviewed step.', '']
    (pump / 'README.md').write_text('\n'.join(pump_card), encoding='utf-8')
    atomic_json(ROOT / 'artifacts/all_tasks/finalized.json', dict(complete=True, gallery_examples=len(examples),
        report='reports/experiment_report.md', defense='reports/defense.md', publication=False))
    # Update canonical tables only after all real results exist. Tests/visual QA remain distinct.
    replacements = {
        '| A: AST |': '| A: AST | Все пять фолдов обучены и оценены; общая сводка трёх моделей готова |',
        '| C: pump |': f"| C: pump | ID-CNN и AST+LOF обучены, оценены и подключены; выбран {deployment['approach']} |",
        '| B: классы проекта |': '| B: классы проекта | AST обучена на публичном датасете, оценена и подключена; собственные записи не использовались |',
        '| Продукт |': '| Продукт | Обратная связь и презентационный режим реализованы; добавлено 7 реальных примеров с атрибуциями |',
        '| A: AST, полное обучение |': '| A: AST, полное обучение | Завершены все 5 фолдов |',
        '| A: AST, финальная оценка |': '| A: AST, финальная оценка | Завершена; сводка содержит три модели |',
        '| C: насос, классификатор ID и AST + LOF |': '| C: насос, классификатор ID и AST + LOF | Обучены и оценены; модели/эмбеддинги сохранены, приложение подключено |',
        '| B: классификатор классов проекта |': '| B: классификатор классов проекта | Обучена и оценена на публичном подмножестве FSD50K, без собственных записей |',
    }
    for filename in ['AGENTS.md', 'CODEX.md']:
        path = ROOT / filename
        doc = path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(doc):
            for prefix, replacement in replacements.items():
                if line.startswith(prefix):
                    doc[index] = replacement
                    break
        path.write_text('\n'.join(doc) + '\n', encoding='utf-8')
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = readme.replace('| 5. Шаг 7 (ID-CNN/AST+LOF) | код и интерфейс готовы; обучение в очереди |',
                            '| 5. Шаг 7 (ID-CNN/AST+LOF) | модели обучены, оценены и подключены |')
    readme = readme.replace('| 7. Сравнение моделей | CNN готовы; AST обучается |',
                            '| 7. Сравнение моделей | три модели оценены по пяти фолдам |')
    readme = readme.replace('| 8. Проектная AST и детектор | данные и код готовы; обучение после сравнения |',
                            '| 8. Проектная AST и детектор | обучены, оценены и подключены |')
    readme_path.write_text(readme, encoding='utf-8')
    print('Gallery, report, defense and model cards prepared', flush=True)


if __name__ == '__main__':
    main()
