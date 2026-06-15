# Photo Standards DB: Полное техническое объяснение приложения

## 1. Что это за приложение

Это система для подготовки эталонных фотографий, разметки сегментов, обучения YOLO-моделей и последующей проверки деталей на новых изображениях или видеопотоке.

По сути приложение строится вокруг одного доменного цикла:

1. Создать группу изделий.
2. Создать эталоны внутри группы.
3. Загрузить фотографии эталонов.
4. Разметить на эталонных фотографиях ожидаемые сегменты.
5. Назначить одно или несколько reference-фото для проверки.
6. Обучить или импортировать модель.
7. Запустить проверку по фото, снимку с камеры или в realtime.
8. Сохранить результат проверки в историю.

Главная идея приложения: YOLO отвечает за нахождение объектов на текущем кадре, а reference-фото и alignment отвечают за понимание, где именно эти объекты должны находиться относительно эталона.

## 2. Из каких частей состоит система

### Backend

Backend расположен в `server/src/` и построен на `FastAPI`.

Ключевые точки:

- `server/src/main.py` поднимает приложение, storage, task runner, live bus, camera stream manager и realtime inspection streamer.
- `server/src/app/router.py` собирает HTTP API.
- `server/src/app/live/router.py` отдаёт WebSocket-каналы для задач, системной статистики и live-статусов камер.

Основные backend-модули:

- `modules/core/groups` — группы изделий.
- `modules/core/standards` — эталоны и их фотографии.
- `modules/core/segments` — классы сегментов, категории и полигоны.
- `modules/yolo/training` — обучение, импорт, экспорт и активация моделей.
- `modules/yolo/inspection` — пайплайн проверки.
- `modules/cameras` — камеры, preview, снимки, потоковые источники.
- `modules/sam` — полуавтоматическая сегментация через SAM2.
- `modules/tasks` — асинхронные задачи и их статусы.
- `modules/system` — системные метрики.
- `modules/users` — CRUD пользователей.

### Frontend

Frontend расположен в `client/src/` и построен как SPA с маршрутами по разделам приложения:

- `Группы`
- `Обучение`
- `Контроль`
- `История`
- `Камеры`
- `Настройки`

Главный navigation описан в `client/src/app/navigation.ts`.

### Хранилище

Приложение использует сразу два типа хранения:

1. PostgreSQL для метаданных, сущностей, истории проверок, задач и связей.
2. Файловое хранилище `storage/` для изображений, результатов проверок, весов моделей, логов и feature-файлов.

Папки создаются на старте в `server/src/main.py`:

- `storage/standards`
- `storage/inspections`
- `storage/models`
- `storage/logs`

## 3. Главные сущности домена

### Group

Модель: `server/src/modules/core/groups/models.py`

Группа — это верхний контейнер для всего, что относится к одному типу изделия.

Группа объединяет:

- эталоны;
- классы сегментов;
- категории классов сегментов;
- ML-модели;
- историю проверок косвенно через эталоны и модели.

### Standard

Модель: `server/src/modules/core/standards/models.py`

Эталон — это конкретный образец внутри группы. У него есть:

- `name`
- `angle`
- `is_active`
- набор фотографий `StandardImage`

`angle` — не декоративное поле, а часть бизнес-контракта: фронтенд валидирует его через `/api/meta/constants`.

### StandardImage

Модель: `server/src/modules/core/standards/models.py`

Это отдельная фотография эталона. Для неё хранятся:

- `image_path`
- `is_reference`
- `features_path`
- `features_keypoint_count`
- `features_computed_at`

Если фото используется как reference-фото для проверки, для него дополнительно вычисляются и сохраняются feature-данные для alignment.

### SegmentClassGroup

Модель: `server/src/modules/core/segments/models.py`

Это категория классов сегментов. Нужна в первую очередь для организации UI и логического группирования компонентов.

### SegmentClass

Модель: `server/src/modules/core/segments/models.py`

Это контролируемый класс объекта, который должен быть найден на эталоне и при проверке. У класса есть:

- `name`
- `hue`
- связь с группой
- опциональная связь с `SegmentClassGroup`

`hue` используется для визуализации масок, полигонов и результатов.

### SegmentAnnotation

Модель: `server/src/modules/core/segments/models.py`

Это полигоны конкретного класса на конкретном эталонном изображении. Поле `points` хранится как JSON.

Важная деталь: в схеме стоит уникальность `(image_id, segment_class_id)`, то есть на уровне записи аннотации один класс на одном фото представлен одной записью, но внутри `points` может быть набор контуров.

### MlModel

Модель: `server/src/modules/yolo/training/models.py`

Это обученная или импортированная YOLO-модель. В ней хранятся:

- архитектура;
- путь к весам;
- версия;
- параметры обучения;
- классы модели;
- метрики;
- train/val/test ratios;
- размеры датасета;
- флаг `is_active`.

У группы может быть только одна активная модель одновременно. Это закреплено частичным уникальным индексом.

### InspectionResult и InspectionSegmentResult

Модели: `server/src/modules/yolo/inspection/models.py`

`InspectionResult` — сохранённый итог проверки.

В нём хранится:

- эталон;
- модель;
- камера;
- пользователь;
- режим проверки;
- статус passed/failed;
- путь к исходному изображению;
- путь к результирующему изображению;
- количество ожидаемых и совпавших сегментов;
- статус alignment;
- число inlier matches;
- число raw matches;
- матрица homography;
- debug payload;
- заметки.

`InspectionSegmentResult` — детализация по каждому ожидаемому или лишнему сегменту.

Статусы сегмента:

- `ok`
- `missing`
- `extra`
- `unmatched`

### Camera

Модель: `server/src/modules/cameras/models.py`

Камера может быть одного из типов:

- `rtsp`
- `http`
- `usb`

Сохраняются адрес, порт, пути, логин, пароль, device path, timeout и live-статус.

### Task

Модель: `server/src/modules/tasks/models.py`

Все долгие операции идут через задачу:

- обучение модели;
- инспекция;
- импорт модели.

Задача хранит:

- тип;
- статус;
- очередь;
- priority;
- progress;
- stage/message/error;
- payload;
- result;
- связанный entity;
- group_id;
- flags остановки/возобновления;
- checkpoint/run_dir;
- heartbeat.

## 4. Как устроен storage

Файловое хранилище не вспомогательное, а полноценная часть архитектуры.

### Standards storage

Обычно здесь лежат:

- эталонные изображения;
- feature-файлы reference-изображений.

Feature-файлы создаются по пути вида:

`standards/<standard_id>/features/<image_id>.npz`

Это задаётся в `server/src/modules/core/standards/reference_storage.py`.

### Inspections storage

Здесь лежат:

- загруженные фото для проверки;
- snapshot-файлы с камер;
- результирующие изображения с overlay.

### Models storage

Здесь лежат:

- базовые и итоговые веса моделей;
- временные артефакты обучения;
- dataset cache;
- checkpoints/resume-артефакты.

### Logs storage

Отдельно сохраняются server/worker logs.

## 5. Полный путь подготовки данных

### 5.1. Создание группы

Пользователь создаёт группу. После этого в UI группа становится общей осью сразу для нескольких процессов:

- standards_count используется в разделе групп;
- models_count в разделе обучения;
- inspections_count в истории.

Это видно в frontend-контрактах `client/src/types/contracts/groups.ts`.

### 5.2. Создание классов сегментов

Для группы задаются классы сегментов и при необходимости их категории.

Это важно не только для UI, но и для всей дальнейшей логики:

- эти классы используются в разметке эталонов;
- из них собираются class keys для обучения;
- потом они же участвуют в matching и inspection history.

### 5.3. Создание эталона

Пользователь создаёт эталон с именем и углом (`angle`).

Эталон не является одной картинкой. Это набор фото одного образца, среди которых позже выбираются reference views.

### 5.4. Загрузка эталонных фото

Пользователь может загрузить несколько фото для одного эталона. Каждое фото становится `StandardImage`.

Для каждого фото UI показывает:

- является ли оно reference;
- сколько на нём аннотаций;
- размечено ли оно.

### 5.5. Разметка фото

Разметка делается в canvas-редакторе.

Ключевой компонент: `client/src/page-components/segments/components/canvas-surface/canvas-surface.tsx`

Редактор поддерживает:

- рисование полигона;
- редактирование вершин;
- перемещение контура;
- SAM-assisted draft;
- отмену и подтверждение черновика.

Почему это важно для backend:

- именно эти полигоны становятся `expected slots` для будущего matching;
- именно по ним строится датасет для YOLO segmentation/detection разметки;
- именно они же могут маскироваться при alignment.

### 5.6. Пометка reference-фото

Одно или несколько фото могут быть отмечены как `is_reference=True`.

Это означает:

- фото может использоваться как эталонная геометрическая опора при inspection;
- для него будут нужны feature-данные;
- если включён multi-reference, оно попадёт в пул candidate reference views.

## 6. SAM: как помогает разметке

Модуль: `server/src/modules/sam/service.py`

SAM не хранит финальные объекты сам по себе. Он служит вспомогательным инструментом для быстрой разметки.

Поток такой:

1. Пользователь делает клики по изображению.
2. Backend загружает нужное эталонное изображение.
3. SAM2 строит несколько масок и scores.
4. Выбирается лучшая маска.
5. Из маски извлекается крупнейший contour.
6. Contour аппроксимируется до полигона.
7. Polygon возвращается на фронтенд.

Важная деталь: SAM embeddings не пишутся в базу и не сохраняются на диск как отдельный долгоживущий артефакт. Это in-memory cache в модуле `modules/sam/cache.py`.

То есть SAM ускоряет разметку, но не является источником правды. Источник правды — сохранённые полигоны аннотаций.

## 7. Feature pipeline для reference-фото

Это одна из самых важных частей приложения.

### 7.1. Зачем нужны features

Когда приложение сравнивает новое фото с эталоном, ему нужно понять геометрическое соответствие между:

- reference-изображением эталона;
- текущим фото или видеокадром.

Для этого backend заранее хранит признаки reference-фото, чтобы не пересчитывать их на каждой проверке.

### 7.2. Где это реализовано

- `server/src/modules/core/standards/reference_features.py`
- `server/src/modules/core/standards/reference_storage.py`
- `server/src/modules/core/standards/reference_service.py`
- `server/src/modules/core/standards/reference_runtime.py`

### 7.3. Что именно сохраняется

Feature-файл — это `.npz`.

В нём сохраняются:

- `keypoints`
- `descriptors`
- `image_width`
- `image_height`
- `profile_version`
- `feature_profile`
- `max_side`
- `max_keypoints`
- `selection_grid_rows`
- `selection_grid_cols`

Это видно в `save_features()` в `reference_storage.py`.

### 7.4. Что означают keypoints

`keypoints` — это массив формы `(N, 2)`.

Каждая строка — координата заметной точки на изображении reference-фото. Это точки, которые SuperPoint считает хорошими для устойчивого сопоставления между изображениями.

Нужны они для того, чтобы потом по совпавшим точкам построить преобразование от reference к текущему кадру.

### 7.5. Что означают descriptors

`descriptors` — это массив формы `(N, 256)` типа `float32`.

Каждый descriptor — это вектор признаков для соответствующего keypoint.

Именно descriptors позволяют понять, какая точка на reference скорее всего соответствует какой точке на текущем фото.

Упрощённо:

- `keypoint` говорит где находится точка;
- `descriptor` говорит как выглядит локальное окружение точки.

### 7.6. Что ещё важно в feature-файле

- `image_width`, `image_height` нужны, чтобы правильно интерпретировать координаты.
- `profile_version` нужен для принудительной инвалидизации старого формата.
- `max_side`, `max_keypoints`, `selection_grid_*` нужны, чтобы понимать, что файл был построен тем же профилем, который ожидает runtime.

Если профиль изменился, файл считается несовместимым и будет пересчитан.

### 7.7. Как features вычисляются

`compute_and_save_features()` в `reference_service.py`:

1. Загружает reference-фото.
2. Вызывает `compute_features()`.
3. Использует профиль `reference`:
   - большой `max_side`;
   - высокий лимит keypoints;
   - grid-balanced selection.
4. Сохраняет `.npz`.
5. Записывает в БД путь к файлу, количество keypoints и время расчёта.

### 7.8. Как выбираются keypoints

Функция `compute_features()` использует `compute_superpoint_features()` и затем дополнительно ограничивает набор keypoints.

Здесь важен принцип grid-balanced selection:

- точки не должны скучиться в одном текстурном месте;
- бюджет keypoints распределяется по сетке изображения;
- при нехватке точек в ячейках добор идёт по score.

Это делает matching стабильнее на реальных производственных сценах.

### 7.9. Профили feature extraction

В `reference_constants.py` бюджеты разделены по режимам:

- `reference` — самый плотный профиль, считается заранее и кэшируется;
- `photo` — для разовой offline-проверки;
- `video` — для realtime, где важна скорость и предсказуемость.

Это принципиальная инженерная деталь: приложение сознательно не использует одинаково тяжёлый alignment и для одиночной фотографии, и для realtime.

## 8. Как reference-фото попадает в inspection context

Ключевой модуль: `server/src/modules/yolo/inspection/adapters/context.py`

Когда запускается проверка:

1. Загружается эталон.
2. Проверяются выбранные классы сегментов.
3. Выбираются candidate reference images.
4. Для каждого reference-фото гарантируется наличие feature-файла.
5. Загружается активная модель группы.
6. Возвращается `InspectionContext`.

### Важная тонкость

Reference-фото подходит для проверки только если на нём размечены все выбранные пользователем классы.

Это очень важное правило. Если пользователь выбрал 5 классов, а на каком-то reference-фото есть разметка только для 3 из них, это фото не будет использоваться для inspection.

Именно поэтому multi-reference здесь не просто “несколько картинок”, а пул полноценных кандидатов для конкретного набора выбранных классов.

## 9. Alignment pipeline

Это самая нетривиальная часть backend.

Главные файлы:

- `server/src/modules/yolo/inspection/adapters/features.py`
- `server/src/modules/yolo/inspection/domain/alignment.py`
- `server/src/modules/yolo/inspection/domain/reference_masking.py`
- `server/src/modules/yolo/inspection/use_cases/inspect_frame.py`
- `server/src/modules/core/standards/reference_runtime.py`

### 9.1. Цель alignment

Нужно получить преобразование `homography`, которое переносит полигоны с эталона на текущий кадр.

После этого можно понять:

- где на текущем кадре должен быть каждый ожидаемый сегмент;
- какие YOLO detections попали в свои места;
- какие слоты пустые;
- что оказалось лишним.

### 9.2. Статусы alignment

Определены в `AlignmentStatus`:

- `success`
- `insufficient_matches`
- `insufficient_inliers`
- `homography_failed`

### 9.3. Identity shortcut

Если включён `ALIGNMENT_IDENTITY_SHORTCUT`, система сначала может попытаться использовать быстрый shortcut, если ситуация позволяет не делать полноценное сопоставление.

Это optimization path, уменьшающий цену проверки там, где кадр уже по сути совпадает с эталоном.

### 9.4. Основной backend: SuperPoint + LightGlue

Torch runtime собирается в `reference_runtime.py`.

Он:

1. Загружает SuperPoint extractor.
2. Загружает LightGlue matcher.
3. Выбирает `cuda` или `cpu` по конфигу.
4. Прогревает runtime.

Дальше pipeline такой:

1. У reference уже есть сохранённые `keypoints/descriptors`.
2. Для текущего кадра features считаются на лету.
3. `match_feature_arrays()` находит пары matching points.
4. По matching points строится `homography` через RANSAC.
5. Проверяются raw matches, inliers, median error и другие guard-условия.

### 9.5. ORB fallback

Если Torch alignment неудачен или недоступен, при включённом `ALIGNMENT_ORB_FALLBACK` используется запасной путь на OpenCV ORB.

Это менее продвинутый, но практичный fallback для отказоустойчивости.

### 9.6. Masked reference alignment

В `reference_masking.py` есть важная техника: полигоны сегментов можно закрасить на эталоне перед alignment.

Для чего это сделано:

- чтобы matching опирался не только на сам объект, но и на общую сцену;
- чтобы избежать ситуации, когда смена детали внутри слота ломает геометрическое выравнивание;
- чтобы не привязывать pose estimation к тем областям, которые потом и проверяются на наличие/отсутствие.

Алгоритм маскировки:

1. Берутся валидные полигоны.
2. Для каждого полигона вычисляется padded polygon.
3. Берётся медианный цвет фона вне полигонов.
4. Полигон заливается этим цветом.

То есть reference не просто “закрашивается чёрным”, а маскируется более аккуратно, чтобы не вносить грубые артефакты.

### 9.7. Multi-reference selection

Если у эталона несколько reference-фото, система делает alignment для каждого кандидата и выбирает лучший.

Это реализовано в `_select_reference_context_and_align()` в `inspect_frame.py`.

Что оценивается:

- число inliers;
- число raw matches;
- inlier ratio;
- median error;
- coverage projected slots;
- небольшой bonus за primary reference.

Слабые reference view тоже ранжируются и попадают в debug summary. Это полезно для диагностики, почему был выбран именно этот reference, а не другой.

### 9.8. Почему хранится debug payload

`FrameAlignment.to_debug_payload()` собирает отладочный набор полей:

- method;
- status;
- stage;
- reason;
- raw_match_count;
- inlier_count;
- median_error;
- число reference/frame features;
- использовалась ли маскировка;
- размеры frame/reference;
- extra debug.

Это нужно не только разработчику. Этот payload потом используется и для итогового inspection debug, и для realtime status.

### 9.9. Failsafe по неподтверждённой позе

Если `INSPECTION_FAILSAFE_REQUIRE_CONFIRMED_POSE=True`, то при отсутствии надёжного alignment система не рисует missing-полигоны как будто они точно известны.

Вместо этого строится специальный сценарий `scene_pose_unconfirmed`.

Это очень важное бизнес-решение: лучше явно сказать “поза сцены не подтверждена”, чем нарисовать ложные missing-зоны в неверных местах.

### 9.10. Альтернативный режим `yolo_count`

Если `INSPECTION_VERIFICATION_MODE=yolo_count`, alignment вообще отключается.

Тогда проверка идёт только по количеству найденных объектов каждого класса.

Это упрощённый режим, полезный как fallback или для отдельных производственных случаев, где точная геометрия не нужна.

## 10. Как строятся expected slots

Модуль: `server/src/modules/yolo/inspection/domain/matcher.py`

`build_expected_segments()` превращает эталонные полигоны в список ожидаемых сегментов.

Для каждого такого сегмента фиксируются:

- annotation id;
- segment class id;
- class key;
- имя;
- hue;
- reference polygon.

Это список “того, что должно быть на сцене”.

Важно: expected segments строятся именно с выбранного reference-фото, а не с любого изображения эталона.

## 11. YOLO inference

YOLO используется для фактического нахождения объектов на текущем фото или кадре.

Backend потом соединяет два источника знания:

1. `expected slots` из reference-разметки;
2. `detections` из YOLO.

Отдельно хранятся raw counts по классам. Они используются и для count-mode, и для debug payload.

## 12. Matching: как решается, что объект на месте

Это вторая по сложности часть после alignment.

### 12.1. Общая идея

После alignment каждый expected polygon переносится на текущий кадр. Получается projected slot.

Дальше matcher пытается сопоставить projected slot и YOLO detection того же класса.

### 12.2. Что учитывается

Matcher смотрит не только на IoU. Он учитывает:

- IoU;
- расстояние между центрами;
- находится ли центр detection внутри projected slot;
- confidence detection;
- неоднозначность нескольких одинаковых detections в одном slot;
- detections неправильного класса внутри slot;
- дубликаты и вложенные detections;
- relaxed same-class fallback;
- случаи, когда поза сцены не подтверждена.

### 12.3. Статусы результата

#### `ok`

Ожидаемый сегмент найден и разумно совпал со своим projected slot.

#### `missing`

Ожидаемый сегмент не найден в подтверждённой зоне или обнаружен wrong-class scenario.

#### `unmatched`

Что-то найдено рядом со slot, но недостаточно уверенно, чтобы объявить это точным попаданием.

#### `extra`

Найдена лишняя деталь, которая не соответствует ни одному ожидаемому сегменту.

### 12.4. Почему `unmatched` и `missing` разделены

Это важная доменная тонкость.

- `missing` означает: ожидаемый слот пуст или некорректен по сути.
- `unmatched` означает: рядом есть детекция, но она не прошла правила соответствия.

То есть `unmatched` полезен для диагностики сложных случаев, когда модель “что-то увидела”, но геометрия/класс/уверенность не совпали достаточно хорошо.

### 12.5. Wrong-class-in-slot

Если в projected slot находится объект другого класса, это не просто `extra` где-то в стороне.

Matcher может специально пометить такой случай как проблема ожидаемого слота, сохранив информацию о том, какой класс был найден в зоне.

### 12.6. Dedupe и collapse extra detections

Чтобы история и UI не были зашумлены, matcher умеет схлопывать лишние detections и отсеивать дубликаты.

Иначе одно и то же физическое наблюдение могло бы попасть в итог сразу несколькими overlapping extra-объектами.

## 13. Как формируется итог inspection result

Главный use case: `server/src/modules/yolo/inspection/use_cases/inspect_frame.py`

### 13.1. Для обычного alignment-mode

Поток такой:

1. Выбрать reference context.
2. Сделать alignment.
3. Построить expected segments.
4. Запустить YOLO inference.
5. Если homography есть — сделать projected matching.
6. Если homography нет — либо применить scene unconfirmed failsafe, либо построить missing matches.
7. При необходимости отрисовать overlay.
8. Вернуть `InspectionFrameResult`.

### 13.2. Для `yolo_count`

Поток проще:

1. Alignment пропускается.
2. Ожидаемые сегменты считаются как список эталонных объектов.
3. YOLO строит detections и raw counts.
4. `match_segments_by_count()` сравнивает counts по классам.
5. Result собирается без геометрической проекции.

### 13.3. Почему alignment failure не всегда останавливает весь pipeline

Даже если alignment неудачен, система может всё равно выполнить detection и отдать диагностическую картину.

Это полезно потому, что:

- можно увидеть, что YOLO в целом живой;
- можно понять, что провалился именно pose estimation;
- realtime режим может адаптивно редуцировать частоту полного pipeline на пустой сцене.

## 14. Разница между запуском проверки и сохранением проверки

Это очень важный нюанс приложения.

### 14.1. Batch inspection не сохраняется автоматически

Для режимов `photo` и `snapshot` результат сначала живёт как `task.result`.

То есть после завершения проверки пользователь ещё не создал запись в истории. Он только получил временный результат.

Чтобы сохранить проверку в историю, нужен отдельный вызов `/api/yolo/inspection/save`.

Это позволяет:

- просмотреть результат перед сохранением;
- отбросить неудачный прогон;
- добавить notes;
- не засорять историю всеми промежуточными проверками.

### 14.2. Realtime snapshot сохраняется сразу

В realtime-сценарии snapshot через `/snapshot` сразу становится `InspectionResult`.

То есть realtime и batch отличаются по семантике жизненного цикла результата.

### 14.3. Что происходит с временными файлами batch inspection

Для batch inspection есть два слоя хранения:

1. временные файлы задачи;
2. постоянные файлы уже сохранённой истории.

Когда запускается `photo` или `snapshot` inspection:

- исходное изображение попадает во временный inspection storage задачи;
- результат overlay тоже сохраняется как временный файл;
- `task.result` хранит пути к этим временным файлам.

Когда вызывается `/api/yolo/inspection/save`:

1. backend копирует временные файлы в постоянное место для `inspection_id`;
2. создаёт `InspectionResult` и `InspectionSegmentResult`;
3. обновляет `task.result`, добавляя `inspection_id` и уже постоянные пути;
4. удаляет старые временные файлы.

Это сделано в `save_inspection.py`.

### 14.4. Почему сохранение идемпотентно

Если задача уже содержит `inspection_id`, повторный `save` не должен создавать вторую историю поверх первой.

Поэтому backend сначала проверяет, не была ли эта задача уже сохранена. Если была, он возвращает уже существующую запись истории.

Это защищает от дублей при повторном клике в UI или повторной отправке запроса.

### 14.5. Что делает discard

`DELETE /api/yolo/inspection/task/{task_id}`:

- очищает временные файлы inspection task, которые не были привязаны к истории;
- удаляет сам task.

Это важно, чтобы `storage/inspections` не забивался временными результатами, которые пользователь не сохранил.

### 14.6. Как frontend помогает cleanup

Frontend не просто ждёт, что backend когда-нибудь сам уберёт мусор.

В `use-inspection-layout.ts` он делает cleanup при:

- смене режима;
- смене группы;
- смене эталона;
- закрытии вкладки;
- `beforeunload`.

Для этого используются keepalive-запросы на discard task и stop realtime session.

## 15. Inspection modes

Контракты режимов видны и в backend, и во frontend.

### 15.1. `photo`

Пользователь загружает файл.

Backend:

- принимает изображение;
- создаёт inspection task;
- выполняет полный pipeline;
- возвращает task-based результат.

### 15.2. `snapshot`

Есть два варианта источника:

1. Серверная камера из списка камер.
2. Локальная камера устройства браузера.

Если выбрана серверная камера, backend сам делает snapshot через camera stream manager.

Если выбрана локальная камера устройства, фронтенд сначала делает захват кадра из `getUserMedia`, превращает его в `File` и отправляет как обычное изображение.

### 15.3. `realtime`

В этом режиме backend создаёт inspection session, а не разовую задачу.

Источник кадров может быть:

- серверная камера;
- браузерная камера устройства.

Результаты поступают как live status плюс видеопоток с overlay.

## 16. Realtime inspection pipeline

Ключевые файлы:

- `server/src/modules/yolo/inspection/realtime/session.py`
- `server/src/modules/yolo/inspection/realtime/frame_processor.py`

### 16.1. Что делает session

`InspectionSession`:

- хранит контекст проверки;
- знает источник кадров;
- прогревает модель и reference matching runtime;
- принимает browser frames при device-camera сценарии;
- следит за idle timeout;
- публикует последние rendered/raw/result данные.

### 16.2. Почему есть warmup

Перед реальной обработкой session вызывает:

- `warmup_model()`
- `warmup_reference_matching()`

Это уменьшает задержку первого реального кадра и переводит состояние UI в `warming_up`, а не в псевдо-пустую ошибку.

### 16.3. Почему не каждый кадр проходит полный pipeline

`RealtimeFrameProcessor` не делает полный alignment + YOLO на каждом кадре.

Причина простая: это слишком дорого.

Поэтому используется гибридная схема:

1. Раз в `N` кадров запускается полный pipeline.
2. Между полными проходами overlay двигается через motion tracking.

Это даёт компромисс между:

- точностью;
- latency;
- нагрузкой на CPU/GPU.

### 16.4. Адаптация к пустой сцене

Если кадр пустой или alignment не подтверждён, realtime может увеличить интервал между полными проходами.

Идея: не тратить одинаково много ресурсов на сцену, где всё равно нет полезной информации.

### 16.5. Что live-статус возвращает на фронтенд

`InspectionRealtimeStatus` содержит:

- `state` (`warming_up`, `online`, `failed`)
- `matched`
- `total`
- `missing`
- `status`
- `passed`
- alignment-поля
- `details`
- `debug_payload`
- `pose_pipeline`

То есть realtime status уже содержит не только сводку, но и подробную диагностику.

### 16.6. Control plane и media plane в realtime

Realtime в приложении разделён на два независимых канала.

#### Control plane

Это управление состоянием inspection-сессии:

- запуск `/api/yolo/inspection/run`;
- polling статуса `GET /api/yolo/inspection/realtime/sessions/{id}/status`;
- остановка `DELETE /api/yolo/inspection/realtime/sessions/{id}`;
- сохранение snapshot `POST /snapshot`.

Важно: статус realtime сейчас идёт не через WebSocket, а через частый polling. Во frontend он обновляется каждые `500ms`, пока сессия не перешла в `failed`.

#### Media plane

Это передача самих кадров/видео:

- WebRTC `POST /webrtc/offer` — основной путь live-видео;
- MJPEG `GET /stream` — запасной streaming path на backend;
- `POST /frames` — путь для browser camera, если backend принимает jpeg-кадры отдельно.

То есть статус и видео живут в разных транспортных каналах. Это важная архитектурная деталь для отладки: “статус есть, а видео нет” и “видео есть, а статус stale” — это разные классы проблем.

### 16.7. Ограничения realtime-сессий

Realtime ограничен не только ресурсами, но и явными правилами:

- `MAX_REALTIME_INSPECTIONS` ограничивает число одновременных live-проверок;
- если browser source долго не присылает первый кадр, сессия закрывается;
- если browser source перестал присылать кадры, сессия считается idle;
- если клиент долго не проявляет активность, сессия закрывается;
- при `failed` состоянии frontend перестаёт polling status.

Эти правила нужны, чтобы не держать бесконечно “зависшие” live-сессии и не расходовать GPU/CPU без пользы.

## 17. Камеры

### 17.1. Что хранится по камере

Камера может быть RTSP, HTTP или USB. Модель включает:

- адрес и порт;
- path и stream_path;
- device_path;
- auth;
- location;
- active flag;
- timeout;
- last status;
- last error.

### 17.2. Проверка доступности камеры

`test_connection()` и `check_camera_health()` пытаются получить кадр.

У камеры есть статусы:

- `online`
- `offline`
- `unknown`

Фоновая служба live status не переключает камеру в offline после одного случайного сбоя. Используется failure streak и разные интервалы probing для разных scope.

### 17.3. Почему есть scope `cameras` и `inspection`

В `CameraLiveStatusService` есть два режима probing:

- `cameras` — более спокойный для экрана камер;
- `inspection` — более частый для экрана контроля.

То есть в момент, когда пользователь выбирает камеру для inspection, приложение обновляет её доступность агрессивнее.

### 17.4. Snapshot с серверной камеры

Для режима `snapshot` backend может сам захватить один кадр и положить его в `inspections/<task_id>/source.jpg`.

Это реализовано в `take_snapshot()`.

### 17.5. Preview и WebRTC

Preview камер строится через WebRTC, а не через медленные polling-механизмы.

`create_camera_preview_answer()` создаёт peer connection и отдаёт видео-track из общего camera stream manager.

### 17.6. Shared camera streams

`CameraStreamManager` мультиплексирует доступ к потоку: несколько подписчиков могут использовать один backend stream.

Это важно, чтобы:

- preview на экране камер;
- snapshot в inspection;
- realtime inspection

не открывали независимо три физических подключения к одной и той же камере без необходимости.

## 18. Браузерная камера устройства

Frontend-часть:

- `client/src/page-components/inspections/lib/device-camera.ts`
- `client/src/page-components/inspections/hooks/use-device-camera.ts`
- `client/src/page-components/inspections/hooks/use-inspection-workspace.ts`

### 18.1. Как она различается от серверной камеры

Во frontend есть специальный sentinel id:

`__device_camera__`

Это означает, что источник изображения не backend-camera из БД, а локальная камера устройства через браузер.

### 18.2. Ограничения браузерной камеры

`getUserMedia` разрешён только в secure context:

- HTTPS;
- либо localhost.

Поэтому UI явно показывает ошибки вида:

- нет HTTPS;
- доступ запрещён;
- камера не найдена;
- камера занята другим приложением.

### 18.3. Snapshot с device camera

Frontend берёт текущий кадр из `<video>`, рисует его на canvas, сериализует в JPEG и создаёт `File`.

После этого backend воспринимает его как обычное загруженное изображение.

### 18.4. Realtime с device camera

Для realtime фронтенд может передать локальный MediaStream в backend через WebRTC.

Это означает, что backend inspection pipeline может обрабатывать не только камеры, зарегистрированные на сервере, но и камеру текущего пользовательского устройства.

## 19. Обучение модели

Ключевые файлы:

- `server/src/modules/yolo/training/domain/dataset_split.py`
- `server/src/modules/yolo/training/use_cases/run_persisted_training.py`
- `server/src/modules/yolo/training/adapters/repository.py`
- `server/src/modules/yolo/training/adapters/storage.py`
- `server/src/modules/yolo/training/adapters/yolo.py`

### 19.1. Что нужно для старта обучения

Группа должна иметь:

- хотя бы один эталон;
- изображения;
- размеченные изображения;
- классы сегментов.

Это отображается и в UI, и проверяется backend-логикой.

### 19.2. Сбор training data

Backend собирает:

- список классов группы;
- все размеченные фото эталонов;
- polygon annotations;
- class meta.

### 19.3. Split по датасету

`plan_dataset_split()` делает deterministic split.

Ключевые свойства:

- split выполняется по каждому эталону отдельно;
- shuffle детерминированный по `group_id` и ratio;
- для каждого эталона нужно минимум `3` размеченных фото;
- должен существовать хотя бы один элемент train;
- если val ratio > 0, должен существовать val split.

Это защищает от псевдо-обучения на слишком бедном датасете.

### 19.4. Как полигоны превращаются в YOLO label lines

`polygons_to_yolo_lines()` нормализует координаты полигона в диапазон `[0, 1]` относительно ширины и высоты изображения.

### 19.5. Resume обучения

Если training уже начинался и есть checkpoint/dataset cache, backend не всегда пересобирает всё заново.

Но есть важная защита: если состав классов группы изменился, resume запрещается, потому что продолжать обучение на изменившейся схеме классов небезопасно.

### 19.6. Что сохраняется после успешного обучения

В модели обновляются:

- `version`
- `weights_path`
- `metrics`
- `trained_at`
- class_keys/class_meta
- размеры split.

В `task.result` сохраняются ссылки на финальные веса и checkpoint.

### 19.7. Главный контракт классов модели

Одна из самых важных идей системы: YOLO-модель не живёт сама по себе. Её классы должны быть встроены во внутреннее пространство классов приложения.

Внутри приложения основным идентификатором класса является `SegmentClass.id`.

Именно поэтому при обучении в `training/adapters/repository.py`:

- `class_key` для training annotation берётся как строковый UUID внутреннего `SegmentClass`;
- `class_meta` тоже строится вокруг внутренних segment classes.

Это означает, что модель, обученная внутри системы, уже “говорит” на языке внутренних классов приложения.

### 19.8. Почему import модели сложнее, чем просто загрузить `.pt`

У внешней модели её native classes могут называться иначе:

- по имени;
- по произвольному ключу;
- по UUID, если модель ранее была экспортирована из этой же системы.

Поэтому импорт идёт в два шага:

1. preview native-классов;
2. явный mapping на внутренние `SegmentClass`.

Модуль `modules/yolo/interop/domain/mapping.py`:

- пытается автоматически предложить соответствие по UUID или по имени;
- валидирует mapping;
- строит `class_keys` и `class_meta` уже во внутреннем формате.

### 19.9. Что это даёт inspection pipeline

Когда inspection получает YOLO detections, дальше matcher работает не с произвольными “классами модели”, а с классами домена приложения.

Это критично, потому что matching, history и UI должны говорить в терминах:

- конкретного `SegmentClass`;
- его имени;
- его hue;
- его отношения к эталону.

Если бы не было этого mapping-контракта, imported model нельзя было бы надёжно использовать совместно с эталонами и polygon slots.

## 20. Импорт и экспорт моделей

Раздел training поддерживает не только обучение, но и импорт/экспорт.

При импорте frontend сначала делает preview native-классов модели, а потом пользователь маппит их:

- либо на существующие классы сегментов;
- либо создаёт новые.

Это важно, потому что модель не может быть использована осмысленно без согласования её class space с внутренними `SegmentClass` группы.

## 21. Асинхронные задачи

### 21.1. Какие задачи бывают

По `tasks.constants`:

- `training_run`
- `inspection_run`
- `model_import`

### 21.2. Что умеет task lifecycle

`modules/tasks/service.py` поддерживает:

- create;
- list/get;
- progress updates;
- heartbeat;
- pause/resume training;
- cancel.

### 21.3. Почему задачи важны для UX

Фронтенд может:

- показывать stage и progress;
- понимать активный task;
- блокировать конфликтующие действия;
- безопасно освобождать временные inspection results.

### 21.4. Live updates задач

Для задач есть WebSocket `/ws/tasks/{task_id}`.

Это даёт почти realtime-обновление статусов без грубого polling.

### 21.5. Почему task runner вынесен отдельно

Backend запускает отдельный встроенный runner и исполняет задачи в отдельных subprocess.

Это нужно для изоляции тяжёлых операций:

- обучение YOLO;
- импорт моделей;
- inspection jobs.

### 21.6. Очереди и конкуренция за GPU

Одна из центральных частей архитектуры — orchestration тяжёлых задач.

В приложении training, inspection и model import конкурируют за одни и те же ресурсы, в первую очередь за GPU.

Поэтому у задач есть:

- queue;
- priority;
- ограничения concurrency по типу задач;
- правила совместимости запуска.

Базовая идея такая:

- training идёт через GPU queue;
- inspection идёт через GPU queue;
- model import тоже идёт через GPU queue;
- runner не даёт тяжёлым типам бесконтрольно стартовать параллельно, если они конфликтуют по ресурсам.

### 21.7. Почему inspection может приостанавливать обучение

В `start_inspection.py` перед постановкой inspection task вызывается `request_training_pause_for_inspection()`.

Это означает:

- если сейчас идёт обучение;
- а пользователь запускает inspection;
- обучение может быть поставлено на паузу автоматически;
- после завершения inspection runner попытается вернуть paused training в очередь resume.

С точки зрения продукта это логично: inspection — более оперативная пользовательская операция, чем длительное обучение.

### 21.8. Что делает TaskQueueRunner

`infra/queue/runner.py`:

- регулярно опрашивает БД на runnable tasks;
- захватывает задачи;
- запускает каждую задачу в отдельном subprocess;
- отслеживает heartbeat, память, CPU и завершение;
- умеет восстанавливать или фейлить активные задачи после рестарта сервера;
- после завершения inspection может авто-возобновить paused training.

То есть runner — это не просто “цикл, который вызывает функцию”, а полноценный диспетчер фоновых работ.

### 21.9. Почему subprocess на задачу — это важно

Отдельный subprocess на задачу даёт:

- изоляцию тяжёлых библиотек и GPU-контекста;
- более безопасную остановку зависших задач;
- отдельные stdout/stderr logs;
- меньший риск, что training или import сломают основной web-process.

Для объяснения системы это важно: web API и тяжёлые ML-job не выполняются в одном потоке запросов.

## 22. Runtime-конфиг и ограничения системы

У приложения есть два типа ограничений:

1. жёсткие продуктовые/технические контракты;
2. runtime-настройки, которые реально меняют поведение pipeline.

### 22.1. Где лежат настройки

Основные настройки:

- `server/src/app/config.py`
- `server/src/app/inspection_config.py`
- `server/src/constants.py`

### 22.2. Что меняет поведение inspection

Ключевые настройки inspection:

- `INSPECTION_VERIFICATION_MODE`
  - `alignment` или `yolo_count`;
- `INSPECTION_MULTI_REFERENCE_ENABLED`;
- лимит числа reference candidates;
- guard thresholds для weak reference selection;
- `INSPECTION_FAILSAFE_REQUIRE_CONFIRMED_POSE`;
- `ALIGNMENT_BACKEND`;
- `ALIGNMENT_DEVICE`;
- `ALIGNMENT_ORB_FALLBACK`;
- `ALIGNMENT_IDENTITY_SHORTCUT`.

То есть поведение контроля достаточно сильно конфигурируемо даже без переписывания кода.

### 22.3. Что меняет поведение YOLO runtime

Ключевые настройки inference:

- `YOLO_DEVICE`;
- `YOLO_CONF_THRESHOLD`;
- `YOLO_REALTIME_CONF_THRESHOLD`;
- `YOLO_NMS_IOU`;
- `YOLO_EXTRA_CONF_THRESHOLD`;
- `YOLO_HALF`.

Также важно: inspection runtime принимает только `.pt` PyTorch-модели. В `inspection/adapters/yolo.py` ONNX explicitly не поддерживается.

### 22.4. Upload-ограничения

Правила загрузки файлов задаются через `uploads` constants и `infra/storage/file_storage.py`:

- проверяется допустимый `content_type`;
- проверяется `max_size_bytes`;
- тип и размер валидируются до сохранения файла.

### 22.5. `/api/meta/constants` как общий контракт UI

`/api/meta/constants` нужен не только для `angle`.

Через него frontend получает:

- inspection modes и statuses;
- training architectures;
- image size;
- epochs limits;
- batch size limits;
- train/val ratio limits;
- `ratio_sum_max`;
- `min_images_to_train`;
- hue limits для segment classes;
- allowed upload types и max upload size;
- realtime constants.

То есть это единая точка правды для UI-валидации и отображения допустимых параметров системы.

### 22.6. Ограничения browser camera

Для device camera есть отдельные ограничения браузера:

- нужен secure context: HTTPS или localhost;
- нужен доступ к `getUserMedia`;
- камера может быть занята другим приложением;
- браузер может отклонить permission.

Это не backend-ограничение, а важная часть реального эксплуатационного поведения приложения.

## 23. История проверок

Раздел истории не просто список passed/failed.

Backend и frontend сохраняют/отображают:

- эталон;
- model;
- camera;
- mode;
- source image;
- result image;
- notes;
- segment-level breakdown;
- alignment status;
- debug payload.

Во frontend детализация приоритизирует:

1. `missing`
2. `extra`
3. `ok`

Это разумно, потому что в реальной работе оператору в первую очередь нужны отклонения.

## 24. Системные метрики и observability

### 24.1. HTTP и WebSocket

Система отдаёт метрики через:

- `GET /api/system/stats`
- `WS /ws/system/stats`

### 24.2. Что именно показывается

`modules/system/service.py` собирает:

- hostname;
- uptime;
- CPU;
- RAM;
- disk;
- GPU через `nvidia-smi`, если доступно;
- размер storage и разбивку на `standards / inspections / models / logs / other`.

Это полезно и как operational dashboard, и как способ быстро понять, куда уходит место в проекте.

## 25. Frontend-маршруты и как они отражают backend-логику

### `/groups`

Раздел для базовой подготовки данных.

Здесь пользователь:

- создаёт группу;
- редактирует описание;
- видит readiness-показатели;
- открывает эталоны;
- управляет классами сегментов.

### `/groups/:groupId/standards/:standardId`

Здесь пользователь:

- загружает фото;
- назначает reference-фото;
- удаляет лишние изображения;
- переходит в редактор разметки.

### `/groups/:groupId/standards/:standardId/images/:imageId`

Это полноценный annotation workspace.

Отсюда в backend уходят полигоны, которые потом используются и в training, и в inspection.

### `/training`

Отображает модели группы, готовность к обучению, импорт/экспорт, активную модель и текущие training tasks.

### `/inspection`

Это главный runtime-раздел приложения.

Он связывает:

- выбранный эталон;
- выбранные классы;
- выбранную камеру или файл;
- текущую модель группы;
- временный или realtime inspection result.

### `/inspection-history`

Здесь уже видны сохранённые результаты, а не временные task-based outputs.

### `/cameras`

Здесь видно, какие физические/сетевые источники вообще доступны системе.

### `/settings/system`

Экран для live-состояния backend-сервера и storage.

## 26. Ключевые тонкости и неочевидные правила

1. У группы может быть только одна активная модель одновременно.
2. Не каждое reference-фото подходит для каждого inspection: нужны все выбранные классы.
3. Feature-файлы reference-картинок валидируются по профилю и могут инвалидироваться.
4. Batch inspection не создаёт history автоматически.
5. Realtime snapshot, наоборот, сохраняется сразу.
6. Alignment может использовать маскированный эталон, а не исходный reference без изменений.
7. При плохой позе система предпочитает скрыть projected missing polygons, а не рисовать ложные.
8. Realtime не прогоняет полный heavy pipeline на каждом кадре.
9. Device camera и server camera — это два разных класса источников, хотя в UI они выглядят как единый выбор камеры.
10. Resume обучения возможен не всегда: изменение состава классов его ломает.
11. Inspection может приоритетно приостанавливать обучение ради быстрой пользовательской проверки.
12. Временный inspection result живёт отдельно от сохранённой history-записи и требует cleanup.
13. Realtime status и live video идут разными транспортами и отлаживаются отдельно.
14. Импортированная модель обязана быть встроена во внутреннее пространство `SegmentClass`, иначе inspection не сможет корректно работать.
15. SAM ускоряет разметку, но не заменяет хранение аннотаций.
16. Users в проекте есть как CRUD-сущность, но полноценная auth-схема в исследованном backend-слое не прослеживается.

## 27. Практическое чтение приложения по слоям

Если нужно быстро разобраться в проекте по коду, лучше идти в таком порядке:

1. `server/src/main.py`
2. `server/src/app/router.py`
3. `server/src/modules/core/*/models.py`
4. `server/src/modules/core/standards/reference_*`
5. `server/src/modules/yolo/training/*`
6. `server/src/modules/yolo/inspection/use_cases/inspect_frame.py`
7. `server/src/modules/yolo/inspection/domain/matcher.py`
8. `server/src/modules/yolo/inspection/realtime/*`
9. `server/src/modules/cameras/*`
10. `client/src/app/routes/*`
11. `client/src/page-components/inspections/*`
12. `client/src/page-components/segments/*`

## 28. Краткое резюме архитектуры

Приложение не является просто CRUD над фотографиями.

Его центральная инженерная идея такая:

- эталонные полигоны задают ожидаемую структуру сцены;
- reference features и alignment находят геометрическое соответствие между эталоном и новым кадром;
- YOLO находит фактические объекты;
- matcher соединяет ожидание и наблюдение;
- realtime слой делает это достаточно быстро для live-режима;
- history сохраняет уже интерпретированный, аудируемый результат.

Если совсем коротко, то приложение решает задачу не просто “обнаружить объекты на фото”, а “проверить, что правильные объекты находятся в правильных местах относительно эталона, и сохранить это как проверяемый производственный результат”.
