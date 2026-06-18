# Архитектурная база проекта

## 1. Главная идея

Базовая архитектура проекта:

```txt
Modular Monolith
+ Self-contained Modules
+ Public API / Ports
+ Events / Outbox
+ Projections
+ Shared Kernel
```

То есть приложение остаётся одним монолитом, но внутри делится на независимые модули.

```txt
app shell
  ├── module registry
  ├── shared kernel
  ├── event bus / outbox
  └── modules
        ├── tasks
        ├── habits
        ├── focus
        ├── rewards
        ├── punishments
        ├── stats
        └── auth
```

Главное правило:

```txt
Модуль владеет своей моделью.
Другие модули не трогают его внутренности.
```

---

## 2. Цель архитектуры

Эта архитектура нужна, чтобы:

```txt
1. Каждый модуль был максимально самодостаточным.
2. Модуль можно было отключить без разрушения всего приложения.
3. Модули не импортировали внутренности друг друга.
4. Бизнес-логика не размазывалась по проекту.
5. Между частями системы не появлялась паутина связей.
6. Frontend и backend имели похожую логику организации.
7. Проект можно было развивать без постоянного переписывания архитектуры.
```

---

## 3. Общая структура backend

```txt
src/
  app/
    app.ts
    module-registry.ts
    server.ts

  shared/
    db/
    config/
    errors/
    logger/
    event-bus/
    outbox/
    types/

  modules/
    tasks/
    habits/
    focus/
    rewards/
    punishments/
    stats/
    auth/
```

---

## 4. Базовая структура backend-модуля

```txt
modules/tasks/
  ├── domain/
  │   ├── Task.ts
  │   └── task.types.ts
  │
  ├── application/
  │   ├── completeTask.usecase.ts
  │   └── createTask.usecase.ts
  │
  ├── infrastructure/
  │   ├── task.repo.ts
  │   └── task.table.ts
  │
  ├── public/
  │   ├── api.ts
  │   ├── dto.ts
  │   └── events.ts
  │
  ├── routes.ts
  └── index.ts
```

Назначение папок:

```txt
domain/          внутренняя доменная модель модуля
application/     сценарии, use-case'ы, бизнес-действия
infrastructure/  БД, репозитории, внешние адаптеры
public/          публичный интерфейс модуля
routes.ts        HTTP-слой модуля
index.ts         регистрация модуля
```

---

## 5. Что нельзя делать между модулями

Плохо:

```ts
import { Task } from "../tasks/domain/Task";
import { taskRepo } from "../tasks/infrastructure/taskRepo";
import { taskService } from "../tasks/application/taskService";
```

Почему плохо:

```txt
rewards начал зависеть от внутренностей tasks
tasks изменил Task
rewards сломался
```

Так модульность становится фейковой: папки красивые, но связи превращаются в паутину.

---

## 6. Что можно делать между модулями

Модуль может общаться с другим модулем только через:

```txt
1. Public API / Port
2. Events
3. Outbox
4. Projection / Read Model
5. Shared Kernel
```

Нельзя импортировать:

```txt
modules/tasks/domain/*
modules/tasks/application/*
modules/tasks/infrastructure/*
```

Можно импортировать:

```txt
modules/tasks/public/api
modules/tasks/public/events
modules/tasks/public/dto
```

---

# Backend

## 7. Public API

Public API используется, когда нужен ответ прямо сейчас.

Например:

> Модуль rewards хочет узнать, существует ли задача и какая у неё сложность.

Плохо:

```ts
import { Task } from "../tasks/domain/Task";
```

Хорошо:

```ts
const taskInfo = await tasksPublicApi.getRewardInfo(taskId);
```

Пример `tasks/public/api.ts`:

```ts
export type TaskRewardInfo = {
  taskId: string;
  userId: string;
  difficulty: "easy" | "medium" | "hard";
};

export interface TasksPublicApi {
  getRewardInfo(taskId: string): Promise<TaskRewardInfo | null>;
}
```

Модуль `rewards` видит только публичную модель:

```ts
const info = await tasksApi.getRewardInfo(taskId);

if (!info) {
  throw new Error("Task not found");
}

const points = calculatePoints(info.difficulty);
```

Он не знает, что внутри `Task`.

### Когда использовать Public API

Использовать, если:

```txt
нужно получить данные прямо сейчас
нужно проверить существование объекта
нужно выполнить действие и сразу получить результат
нужен синхронный ответ
```

Примеры:

```txt
rewards спрашивает у tasks сложность задачи
punishments спрашивает у habits, кому принадлежит привычка
focus спрашивает у auth текущего пользователя
notes спрашивает у tasks, существует ли связанная задача
```

---

## 8. Port / Interface

Port используется, когда модуль должен зависеть не от конкретного другого модуля, а от возможности.

Модуль не говорит:

```txt
мне нужен tasks
```

Он говорит:

```txt
мне нужен источник информации для награды
```

Пример:

```ts
export interface RewardSourceInfoProvider {
  getRewardSourceInfo(sourceId: string): Promise<{
    sourceId: string;
    userId: string;
    difficulty: "easy" | "medium" | "hard";
  } | null>;
}
```

`rewards` зависит от интерфейса:

```ts
export function createRewardsService(deps: {
  rewardSourceInfoProvider: RewardSourceInfoProvider;
}) {
  return {
    async grantForSource(sourceId: string) {
      const source = await deps.rewardSourceInfoProvider.getRewardSourceInfo(sourceId);

      if (!source) {
        throw new Error("Reward source not found");
      }

      return calculateReward(source.difficulty);
    },
  };
}
```

А при сборке приложения подключается конкретная реализация:

```ts
const rewardsService = createRewardsService({
  rewardSourceInfoProvider: tasksPublicApi,
});
```

### Когда использовать Port

Использовать, если:

```txt
модуль должен быть максимально независимым
источник данных может быть разным
модуль не должен знать конкретно про tasks/habits/focus
один модуль должен одинаково работать с разными источниками
```

Пример:

```txt
tasks      ┐
habits     ├── RewardSourceInfoProvider ──> rewards
focus      ┘
```

---

## 9. Event Bus

Event Bus используется, когда модуль сообщает факт:

```txt
у меня что-то произошло
```

Пример:

```ts
eventBus.publish("task.completed", {
  taskId: task.id,
  userId: task.userId,
  difficulty: task.difficulty,
  completedAt: new Date().toISOString(),
});
```

Другие модули слушают событие:

```ts
eventBus.subscribe("task.completed", async (event) => {
  await rewardsService.grantForCompletedTask(event);
});
```

Модуль `tasks` не знает, что существует `rewards`.

Он просто сообщает:

```txt
задача выполнена
```

### Когда использовать Event Bus

Использовать, если:

```txt
произошёл факт
на этот факт могут реагировать несколько модулей
источник события не должен знать получателей
действие не обязано возвращать результат сразу
```

Примеры:

```txt
task.completed       → rewards начисляет очки
task.completed       → stats обновляет статистику
task.completed       → notifications создаёт уведомление

habit.missed         → punishments усиливает режим
habit.completed      → rewards начисляет очки
focus.sessionEnded   → stats обновляет фокус-время
```

### Где Event Bus использовать не надо

Плохо:

```txt
rewards публикует "task.info.requested"
tasks отвечает "task.info.provided"
rewards ждёт ответ
```

Если нужен ответ, использовать Public API:

```ts
const task = await tasksApi.getRewardInfo(taskId);
```

Event — это не вопрос. Event — это факт.

```txt
Плохо:
  дай мне задачу

Хорошо:
  задача выполнена
```

---

## 10. Transactional Outbox

Transactional Outbox — это надёжная версия событий.

Проблема обычного Event Bus:

```ts
await taskRepo.save(task);

eventBus.publish("task.completed", payload);
```

Если приложение упало между `save` и `publish`, задача будет выполнена, но награда не начислится.

Надёжнее:

```txt
в одной транзакции:
  1. обновили task
  2. записали событие в outbox_events
```

Пример:

```ts
await db.transaction(async (tx) => {
  await taskRepo.save(task, tx);

  await outboxRepo.add(
    {
      type: "task.completed",
      payload: {
        taskId: task.id,
        userId: task.userId,
        difficulty: task.difficulty,
      },
    },
    tx,
  );
});
```

Потом отдельный worker читает события:

```ts
const events = await outboxRepo.getUnprocessed();

for (const event of events) {
  await eventBus.dispatch(event);
  await outboxRepo.markProcessed(event.id);
}
```

### Когда использовать Outbox

Использовать, если событие важно и его нельзя потерять.

Примеры:

```txt
начисление награды
наказание
изменение streak
история действий
статистика
важные уведомления
```

Для старта можно использовать обычный in-memory Event Bus.

Для надёжной архитектуры:

```txt
Domain Events + Transactional Outbox
```

---

## 11. Projection / Read Model

Projection используется, когда модуль строит свою копию данных для чтения.

Например, `stats` не должен лазить в таблицы `tasks`, `habits`, `focus`.

Он слушает события:

```txt
task.completed
habit.missed
focus.session.finished
reward.granted
```

И строит свои таблицы:

```txt
stats_events
daily_stats
user_progress_snapshot
```

Пример:

```ts
eventBus.subscribe("task.completed", async (event) => {
  await statsRepo.addEvent({
    userId: event.userId,
    type: "task.completed",
    sourceId: event.taskId,
    occurredAt: event.completedAt,
  });

  await statsRepo.incrementDailyCompletedTasks(event.userId);
});
```

### Когда использовать Projection

Использовать для:

```txt
дашбордов
статистики
истории
аналитики
лент активности
быстрого чтения агрегированных данных
```

Пример:

```txt
tasks хранит задачи
habits хранит привычки
focus хранит сессии
stats хранит статистику

stats не владеет задачами
stats владеет статистикой
```

---

## 12. Shared Kernel

Shared Kernel — это маленькое общее ядро проекта.

Туда можно класть:

```txt
UserId
EntityId
DateTime
Result
AppError
Pagination
Logger
Config
EventBus
```

Нельзя класть:

```txt
Task
Habit
Reward
Punishment
FocusSession
```

Плохой `shared`:

```txt
shared/
  taskUtils.ts
  habitCalculator.ts
  rewardLogic.ts
  everything.ts
```

Хороший `shared`:

```txt
shared/
  types/
  errors/
  logger/
  event-bus/
  db/
  config/
  validation/
```

---

## 13. Backend: таблица выбора способа связи

| Ситуация                                            | Что использовать        |
| --------------------------------------------------- | ----------------------- |
| Нужно получить данные сейчас                        | Public API              |
| Нужно выполнить действие и получить результат       | Public API              |
| Нужно отвязать модуль от конкретного другого модуля | Port / Interface        |
| Нужно сообщить, что что-то произошло                | Event Bus               |
| Событие нельзя потерять                             | Transactional Outbox    |
| Нужно строить статистику/историю/дашборд            | Projection / Read Model |
| Нужно общее для всех модулей                        | Shared Kernel           |
| Внутри одного модуля                                | Обычные прямые вызовы   |

---

## 14. Полный backend-сценарий

Пользователь выполнил задачу.

### 1. HTTP route

```ts
app.post("/tasks/:taskId/complete", async (req, reply) => {
  const result = await completeTaskUseCase.execute({
    taskId: req.params.taskId,
    userId: req.user.id,
  });

  return reply.send(result);
});
```

### 2. Use-case внутри tasks

```ts
export function createCompleteTaskUseCase(deps: {
  taskRepo: TaskRepo;
  outbox: OutboxRepo;
}) {
  return {
    async execute(input: { taskId: string; userId: string }) {
      const task = await deps.taskRepo.getById(input.taskId);

      if (!task) {
        throw new Error("Task not found");
      }

      task.complete(input.userId);

      await db.transaction(async (tx) => {
        await deps.taskRepo.save(task, tx);

        await deps.outbox.add(
          {
            type: "task.completed",
            payload: {
              taskId: task.id,
              userId: task.userId,
              difficulty: task.difficulty,
              completedAt: new Date().toISOString(),
            },
          },
          tx,
        );
      });

      return {
        taskId: task.id,
        status: "completed",
      };
    },
  };
}
```

### 3. Outbox worker

```ts
for (const event of await outboxRepo.getUnprocessed()) {
  await eventBus.dispatch(event);
  await outboxRepo.markProcessed(event.id);
}
```

### 4. Rewards слушает событие

```ts
eventBus.subscribe("task.completed", async (event) => {
  await rewardsService.grant({
    userId: event.payload.userId,
    sourceType: "task",
    sourceId: event.payload.taskId,
    difficulty: event.payload.difficulty,
  });
});
```

### 5. Stats слушает то же событие

```ts
eventBus.subscribe("task.completed", async (event) => {
  await statsService.recordTaskCompleted({
    userId: event.payload.userId,
    taskId: event.payload.taskId,
    completedAt: event.payload.completedAt,
  });
});
```

Итог:

```txt
tasks не знает про rewards
tasks не знает про stats
rewards не знает внутреннюю модель Task
stats не лезет в таблицы tasks
событие не потеряется из-за outbox
```

---

# Frontend

## 15. Общая структура frontend

```txt
src/
  app/
    App.tsx
    router.tsx
    providers.tsx
    module-registry.ts

  shared/
    ui/
    api/
    lib/
    config/
    stores/

  modules/
    tasks/
    habits/
    focus/
    rewards/
    punishments/
    stats/
    auth/
```

Пример frontend-модуля:

```txt
modules/tasks/
  ├── public.ts
  ├── routes.tsx
  ├── nav.ts
  ├── api/
  │   └── tasksApi.ts
  ├── queries/
  │   └── taskQueries.ts
  ├── model/
  │   ├── taskTypes.ts
  │   └── taskMappers.ts
  ├── pages/
  │   └── TasksPage.tsx
  └── components/
      ├── TaskCard.tsx
      └── TaskPicker.tsx
```

---

## 16. Если на backend events, то что на frontend?

На frontend не нужно полностью копировать backend Event Bus.

На backend events нужны для бизнес-последствий:

```txt
task.completed → reward.granted
habit.missed → punishment.applied
focus.finished → stats.updated
```

На frontend чаще нужно:

```txt
обновить данные
перерисовать экран
показать toast
открыть модалку
перейти на страницу
```

Поэтому главный механизм frontend:

```txt
TanStack Query + invalidateQueries
```

---

## 17. TanStack Query

Используется для данных с сервера.

Например, пользователь нажал “Выполнить задачу”:

```ts
const completeTask = useMutation({
  mutationFn: tasksApi.completeTask,

  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["tasks"] });
    queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    queryClient.invalidateQueries({ queryKey: ["rewards"] });
    queryClient.invalidateQueries({ queryKey: ["stats"] });
  },
});
```

Фронт не начисляет награды сам.

Он просто говорит:

```txt
задача выполнена на бэке
теперь обновим данные
```

### Когда использовать TanStack Query

Использовать для:

```txt
загрузки задач
загрузки привычек
загрузки статистики
мутаций
обновления после действий
кеша данных с сервера
```

Главное правило:

```txt
server-state хранится в query cache
не в Zustand
не в Redux
не в useState
```

---

## 18. Router

Router — лучший способ слабой связи между frontend-модулями.

Например, модуль `rewards` хочет открыть задачу:

```ts
navigate({
  to: "/tasks/$taskId",
  params: { taskId },
});
```

Он не импортирует страницу задачи напрямую.

Он просто ведёт пользователя по маршруту.

### Когда использовать Router

Использовать, если:

```txt
нужно перейти в другой модуль
нужно открыть страницу сущности
нужно сохранить состояние в URL
нужно сделать ссылку, которую можно открыть заново
```

Примеры:

```txt
dashboard → /tasks/123
rewards → /rewards/history
stats → /stats/day/2026-06-13
focus → /focus/session/current
```

---

## 19. Public Module API на frontend

Frontend-модуль тоже должен иметь публичный интерфейс.

Пример:

```txt
modules/tasks/
  public.ts
  routes.tsx
  api.ts
  queries.ts
  components/
```

`tasks/public.ts`:

```ts
export { taskRoutes } from "./routes";
export { tasksApi } from "./api/tasksApi";
export { taskQueryKeys } from "./queries/taskQueries";
export { TaskPicker } from "./components/TaskPicker";
```

Другие модули могут использовать только `public.ts`.

Например, `rewards` хочет дать выбрать задачу:

```tsx
import { TaskPicker } from "@/modules/tasks/public";

<TaskPicker onSelect={handleTaskSelect} />
```

Но нельзя:

```tsx
import { InternalTaskCard } from "@/modules/tasks/components/InternalTaskCard";
```

### Когда использовать Public Module API на frontend

Использовать, если:

```txt
один модуль должен встроить маленькую публичную часть другого
нужен общий picker/select/widget
нужны query keys другого модуля
нужны routes/navItems/providers модуля
```

Примеры:

```txt
rewards использует TaskPicker
dashboard использует public DashboardWidget из tasks
app shell собирает navItems из модулей
router собирает routes из модулей
```

---

## 20. Client Store

Например:

```txt
Zustand
Jotai
Redux
```

Используется только для UI-состояния.

Хорошо:

```txt
открыта ли боковая панель
текущая тема
активная вкладка
локальный черновик
состояние модалки
выбранный layout
```

Плохо:

```txt
все tasks лежат в Zustand
все rewards лежат в Zustand
вся stats лежит в Zustand
```

Для server-state используется TanStack Query.

### Когда использовать Client Store

Использовать, если состояние:

```txt
локальное для клиента
не является главным источником правды
не обязано храниться на сервере
нужно нескольким UI-компонентам
```

Пример:

```ts
type UiState = {
  sidebarOpen: boolean;
  commandPaletteOpen: boolean;
  setSidebarOpen(value: boolean): void;
};
```

---

## 21. UI Event Bus

На фронте можно иметь маленький Event Bus, но только для интерфейсных эффектов.

Пример:

```ts
uiEvents.emit("toast.show", {
  title: "Задача выполнена",
});
```

Или:

```ts
uiEvents.emit("modal.open", {
  name: "create-task",
});
```

### Когда использовать UI Event Bus

Использовать для:

```txt
toast
modal
command palette
shortcuts
sidebar toggle
глобальные UI-события
```

Не использовать для бизнес-логики.

Плохо:

```txt
frontend event task.completed → frontend сам начисляет reward
```

Правильно:

```txt
backend начисляет reward
frontend обновляет rewards query
```

---

## 22. SSE / WebSocket

Используется, если backend сам сообщает frontend о событиях.

Например:

```txt
reward.granted
punishment.applied
stats.updated
focus.session.expired
```

Фронт получает событие и инвалидирует кеш:

```ts
socket.on("reward.granted", () => {
  queryClient.invalidateQueries({ queryKey: ["rewards"] });
  queryClient.invalidateQueries({ queryKey: ["dashboard"] });
});
```

### Когда использовать SSE/WebSocket

Использовать, если нужны:

```txt
живые уведомления
таймеры
синхронизация между вкладками
фоновая обработка
мгновенные обновления
real-time dashboard
```

Для старта можно без этого.

Стартовый вариант:

```txt
mutation → response → invalidateQueries
```

Продвинутый вариант:

```txt
backend event → SSE/WebSocket → invalidateQueries
```

---

## 23. Frontend: таблица выбора

| Ситуация                                          | Что использовать     |
| ------------------------------------------------- | -------------------- |
| Данные с сервера                                  | TanStack Query       |
| После мутации нужно обновить экран                | invalidateQueries    |
| Нужно перейти в другой модуль                     | Router               |
| Нужно встроить публичный компонент другого модуля | Public Module API    |
| Нужно хранить UI-состояние                        | Zustand/Jotai        |
| Нужно показать toast/modal/shortcut               | UI Event Bus         |
| Нужны live-события с backend                      | SSE/WebSocket        |
| Нужно сохранить состояние в ссылке                | Router search params |

---

## 24. Полный пример: задача выполнена

### Backend

```txt
POST /tasks/:id/complete
  ↓
tasks.completeTask()
  ↓
Task меняется внутри tasks
  ↓
task.completed записывается в outbox
  ↓
outbox worker публикует событие
  ↓
rewards начисляет награду
stats обновляет статистику
notifications создаёт уведомление
```

### Frontend

```txt
TasksPage
  ↓
completeTask mutation
  ↓
успешный ответ от backend
  ↓
invalidateQueries:
  - tasks
  - dashboard
  - rewards
  - stats
  ↓
UI сам обновился
```

Если есть live-события:

```txt
backend отправил reward.granted
  ↓
frontend получил событие
  ↓
invalidate rewards/dashboard
  ↓
показал toast
```

---

## 25. Главный алгоритм выбора способа связи

```txt
Вопрос 1:
Мне нужен ответ прямо сейчас?

Да:
  Public API / Port

Нет:
  следующий вопрос

Вопрос 2:
Я хочу сообщить, что что-то произошло?

Да:
  Event

Нет:
  следующий вопрос

Вопрос 3:
Событие нельзя потерять?

Да:
  Transactional Outbox

Нет:
  обычный Event Bus

Вопрос 4:
Мне нужна статистика, история, дашборд?

Да:
  Projection / Read Model

Вопрос 5:
Это UI-данные с сервера?

Да:
  TanStack Query

Вопрос 6:
Это локальное состояние интерфейса?

Да:
  Zustand/Jotai

Вопрос 7:
Это переход между экранами?

Да:
  Router

Вопрос 8:
Это toast/modal/shortcut?

Да:
  UI Event Bus
```

---

## 26. Короткий стандарт связи

### Backend

```txt
Внутри модуля:
  прямые вызовы

Между модулями:
  Public API / Port

Факты:
  Events

Надёжные факты:
  Transactional Outbox

Статистика/история:
  Projections

Общие базовые вещи:
  Shared Kernel
```

### Frontend

```txt
Данные:
  TanStack Query

Обновление данных:
  invalidateQueries

Переходы:
  Router

Публичные части модулей:
  Public Module API

Локальное UI-состояние:
  Zustand/Jotai

UI-эффекты:
  UI Event Bus

Live:
  SSE/WebSocket + invalidateQueries
```

---

## 27. Самая важная формула связи

```txt
Backend:
  Public API — спросить
  Event — сообщить
  Outbox — не потерять
  Projection — быстро читать

Frontend:
  Query — получить данные
  Invalidate — обновить данные
  Router — перейти
  Store — держать UI-состояние
  UI Event — показать эффект
```

Совсем коротко:

```txt
Нужен ответ → Public API
Произошёл факт → Event
Факт нельзя потерять → Outbox
Нужно читать агрегаты → Projection
На фронте изменились серверные данные → invalidateQueries
На фронте нужен переход → Router
На фронте нужен UI-эффект → UI Event Bus
```

---

# Принцип написания кода

## 28. Базовый стиль кода

Базовый стиль написания кода для этой архитектуры:

```txt
Module-first Use-case Style
```

Или подробнее:

```txt
Use-case first
+ Module owns its model
+ Public boundary only
+ Events for facts
+ Outbox for reliability
```

Это значит:

```txt
Код пишется не вокруг классов, функций или слоёв.
Код пишется вокруг модуля и его сценариев.
```

Главный вопрос при написании кода:

```txt
Не "мне писать класс или функцию?"
А:
  1. В каком модуле это действие?
  2. Какой сценарий выполняется?
  3. Какая модель принадлежит этому модулю?
  4. Что модуль отдаёт наружу?
  5. Какое событие публикует?
```

---

## 29. Главное правило написания кода

```txt
Внутри модуля — гибрид.
Между модулями — только контракты.
```

То есть внутри модуля можно использовать:

```txt
функции
классы
структуры
методы
чистые функции
объекты с функциями
```

Но между модулями разрешены только:

```txt
Public API
Ports
DTO
Events
Outbox events
Read Models / Projections
```

---

## 30. Use-case

Use-case — это пользовательский или системный сценарий.

Примеры:

```txt
создать задачу
выполнить задачу
провалить привычку
начать фокус-сессию
начислить награду
применить наказание
```

Use-case координирует действие:

```txt
1. получает входные данные
2. загружает нужную модель
3. вызывает доменную логику
4. сохраняет результат
5. создаёт событие
6. возвращает ответ
```

Псевдокод:

```txt
completeTask(userId, taskId):
  task = taskRepo.findById(taskId)

  if task not found:
    return error

  task.complete(userId)

  taskRepo.save(task)

  outbox.add("task.completed", {
    taskId,
    userId,
    difficulty
  })

  return success
```

Правило:

```txt
Use-case не должен быть огромным сервисом.
Use-case должен быть отдельным понятным сценарием.
```

---

## 31. Domain Model

Domain Model — это внутренняя модель модуля.

Она нужна, если у сущности есть правила и состояния.

Например, `Task`:

```txt
Task.complete(userId):
  if task.owner != userId:
    return error

  if task.status != active:
    return error

  task.status = completed
```

Правильно:

```txt
task.complete(userId)
```

Плохо:

```txt
task.status = completed
```

Почему плохо:

```txt
если статус можно менять из любого места,
модель больше не защищает свои правила
```

Правило:

```txt
Если сущность имеет важные правила — она должна защищать себя сама.
```

Примеры хороших доменных моделей:

```txt
Task
Habit
FocusSession
RewardPolicy
PunishmentRule
Schedule
TimeWindow
Rental
Order
Payment
```

---

## 32. DTO

DTO — это простые данные для передачи наружу.

DTO не должен содержать бизнес-логику.

Пример:

```txt
TaskDto:
  id
  title
  status
  difficulty
```

DTO используется для:

```txt
HTTP-ответов
public API
frontend
events payload
межмодульной передачи данных
```

Правило:

```txt
DTO — это не доменная модель.
DTO — это безопасный снимок данных.
```

Плохо:

```txt
rewards получает Task
```

Хорошо:

```txt
rewards получает TaskRewardInfo
```

---

## 33. Pure Function

Pure function используется для чистой логики.

Например:

```txt
calculateReward(difficulty):
  easy -> 5
  medium -> 10
  hard -> 20
```

Чистая функция:

```txt
не ходит в БД
не вызывает API
не читает файлы
не зависит от текущего времени напрямую
не меняет внешний мир
```

Использовать для:

```txt
расчётов
валидации
маппинга
проверки правил
форматирования
подсчёта наград
подсчёта штрафов
подсчёта streak
```

Правило:

```txt
Если логика может быть чистой функцией — делай её чистой функцией.
```

---

## 34. Repository

Repository отвечает только за хранение и загрузку данных.

Он не должен содержать бизнес-логику.

Хорошо:

```txt
taskRepo.findById(taskId)
taskRepo.save(task)
taskRepo.delete(taskId)
```

Плохо:

```txt
taskRepo.completeTask(taskId)
taskRepo.grantRewardForTask(taskId)
taskRepo.applyPunishment(userId)
```

Почему плохо:

```txt
repository начинает знать бизнес-сценарии
и превращается в скрытый service
```

Правило:

```txt
Repository работает с хранением.
Use-case работает со сценарием.
Domain model работает с правилами.
```

---

## 35. Adapter

Adapter — это работа с внешним миром.

Примеры:

```txt
EmailSender
NotificationClient
PaymentClient
FileStorage
ExternalCalendarClient
SystemBlocker
```

Adapter не должен решать бизнес-логику.

Он только выполняет техническое действие:

```txt
отправить
загрузить
получить
записать
заблокировать
разблокировать
```

Правило:

```txt
Внешний мир прячется за adapter.
Use-case не должен знать детали внешней системы.
```

---

## 36. Event

Event — это факт, который уже произошёл.

Хорошие события:

```txt
task.completed
task.failed
habit.missed
habit.completed
focus.sessionStarted
focus.sessionFinished
reward.granted
punishment.applied
```

Плохие события:

```txt
task.completePlease
task.getInfoRequested
reward.calculateNow
```

Почему плохо:

```txt
это команды или вопросы, а не факты
```

Правило:

```txt
Event — это факт.
Command — это просьба выполнить действие.
Query — это просьба вернуть данные.
```

---

## 37. Outbox Event

Outbox используется, когда событие нельзя потерять.

Например:

```txt
task.completed
habit.missed
reward.granted
punishment.applied
```

Схема:

```txt
в одной транзакции:
  1. меняем состояние модели
  2. записываем событие в outbox
```

Потом worker обрабатывает событие.

Правило:

```txt
Если потеря события ломает смысл приложения — используй outbox.
```

---

## 38. Projection / Read Model

Projection используется для удобного чтения данных.

Например, модуль `stats` не читает напрямую таблицы `tasks`, `habits`, `focus`.

Он слушает события:

```txt
task.completed
habit.missed
focus.sessionFinished
```

И строит свои таблицы:

```txt
daily_stats
activity_log
discipline_score
focus_summary
```

Правило:

```txt
Модуль статистики не владеет задачами.
Модуль статистики владеет статистикой.
```

---

## 39. State Machine

Если у модели есть сложные статусы, их нужно описывать явно.

Например, задача:

```txt
active
completed
failed
cancelled
```

Разрешённые переходы:

```txt
active -> completed
active -> failed
active -> cancelled

completed -> запрещено
failed -> запрещено
cancelled -> запрещено
```

Плохо:

```txt
task.status = "completed"
```

Хорошо:

```txt
task.complete(userId)
```

Почему:

```txt
метод complete проверяет,
можно ли вообще сделать такой переход
```

Правило:

```txt
Если у объекта есть статусы — переходы должны быть явными.
```

---

## 40. Actor / Process

Actor или process нужен не для всего проекта, а только для долгоживущих процессов.

Примеры:

```txt
фокус-сессия
таймер
автоматическое наказание
фоновая проверка дедлайнов
синхронизация с системой
```

Например, `FocusSession`:

```txt
FocusSessionProcess получает команды:
  start
  pause
  resume
  interrupt
  finish
```

Использовать actor/process, если:

```txt
объект живёт во времени
у него есть внутреннее состояние
он реагирует на команды
он может получать события
```

Не использовать actor/process для обычного CRUD.

---

## 41. Таблица: что каким стилем писать

| Что пишем                   | Как писать              | Почему                          |
| --------------------------- | ----------------------- | ------------------------------- |
| Пользовательское действие   | Use-case                | Это отдельный сценарий          |
| Важная сущность с правилами | Domain model            | Модель защищает свои инварианты |
| Простые данные              | DTO / struct / record   | Это просто перенос данных       |
| Чистый расчёт               | Pure function           | Легко тестировать               |
| Работа с БД                 | Repository              | Хранение отдельно от логики     |
| Работа с внешним API        | Adapter / Client        | Внешний мир изолирован          |
| Межмодульный запрос         | Public API / Port       | Нужен ответ прямо сейчас        |
| Межмодульный факт           | Event                   | Что-то уже произошло            |
| Важный факт                 | Outbox Event            | Событие нельзя потерять         |
| Статистика/дашборд          | Projection / Read Model | Быстрое чтение и независимость  |
| Сложные статусы             | State Machine           | Явные разрешённые переходы      |
| Долгий процесс              | Actor / Process         | Есть своё состояние во времени  |

---

## 42. Как выбирать способ написания

### Вопрос 1. Это действие пользователя или системы?

Примеры:

```txt
выполнить задачу
создать привычку
завершить фокус-сессию
начислить награду
```

Использовать:

```txt
Use-case
```

---

### Вопрос 2. Это важная модель с правилами?

Примеры:

```txt
Task
Habit
FocusSession
PunishmentRule
RewardPolicy
```

Использовать:

```txt
Domain model
```

Модель должна иметь методы, которые защищают состояние.

---

### Вопрос 3. Это просто данные для передачи?

Примеры:

```txt
TaskDto
CreateTaskInput
TaskRewardInfo
DashboardView
```

Использовать:

```txt
DTO / plain data
```

---

### Вопрос 4. Это расчёт или проверка?

Примеры:

```txt
calculateReward
calculatePenalty
canCompleteTask
formatDuration
```

Использовать:

```txt
Pure function
```

---

### Вопрос 5. Это работа с БД?

Использовать:

```txt
Repository
```

---

### Вопрос 6. Это работа с внешней системой?

Использовать:

```txt
Adapter / Client
```

---

### Вопрос 7. Другому модулю нужен ответ прямо сейчас?

Использовать:

```txt
Public API / Port
```

---

### Вопрос 8. Нужно сообщить, что что-то произошло?

Использовать:

```txt
Event
```

---

### Вопрос 9. Событие нельзя потерять?

Использовать:

```txt
Outbox
```

---

### Вопрос 10. Нужно быстро читать агрегированные данные?

Использовать:

```txt
Projection / Read Model
```

---

## 43. Пример: задача выполнена

### Сценарий

```txt
Пользователь нажал "Выполнить задачу".
```

### Use-case

```txt
completeTask(userId, taskId):
  task = taskRepo.findById(taskId)

  if task not found:
    return error

  task.complete(userId)

  taskRepo.save(task)

  outbox.add("task.completed", {
    taskId,
    userId,
    difficulty,
    completedAt
  })

  return success
```

### Domain Model

```txt
Task.complete(userId):
  if task.userId != userId:
    return error "access_denied"

  if task.status != active:
    return error "task_not_active"

  task.status = completed

  return success
```

### Event

```txt
task.completed:
  taskId
  userId
  difficulty
  completedAt
```

### Event Handler в rewards

```txt
on task.completed:
  points = calculateReward(difficulty)

  rewardRepo.save({
    userId,
    sourceType: "task",
    sourceId: taskId,
    points
  })

  outbox.add("reward.granted", {
    userId,
    sourceType,
    sourceId,
    points
  })
```

### Event Handler в stats

```txt
on task.completed:
  statsRepo.addActivity({
    userId,
    type: "task.completed",
    sourceId: taskId,
    occurredAt: completedAt
  })

  statsRepo.incrementCompletedTasks(userId)
```

Итог:

```txt
tasks меняет только Task
rewards меняет только Reward
stats меняет только Stats
между модулями ходят только events и DTO
```

---

## 44. Frontend в таком же стиле

Frontend тоже должен быть модульным.

```txt
modules/tasks/
  public.ts
  pages/
  components/
  api/
  queries/
  model/
```

Правила:

```txt
1. Страница собирает сценарий экрана.
2. Компонент отображает данные.
3. Hook содержит frontend-логику.
4. API-клиент вызывает backend.
5. Query хранит server-state.
6. Store хранит только UI-state.
7. Public module API отдаёт наружу только безопасные части.
```

---

## 45. Frontend use-case

На frontend use-case обычно выглядит как hook.

Например:

```txt
useCompleteTask:
  вызвать tasksApi.completeTask
  после успеха invalidate tasks/dashboard/rewards/stats
  показать toast
```

То есть frontend не выполняет бизнес-логику.

Frontend только:

```txt
1. отправляет команду на backend
2. обновляет server-state
3. показывает UI-эффект
```

---

## 46. Frontend domain model

На frontend обычно не нужна полноценная domain model.

Лучше использовать:

```txt
DTO
ViewModel
Mapper
Hook
Component
```

Например:

```txt
TaskDto       данные с backend
TaskCardVm    данные для карточки
toTaskCardVm  mapper
TaskCard      компонент
```

Правило:

```txt
Бизнес-правила живут на backend.
Frontend показывает состояние и отправляет действия.
```

---

## 47. Главная формула стиля кода

```txt
Внутри модуля:
  use-case управляет сценарием
  domain model защищает правила
  repository работает с БД
  adapter работает с внешним миром
  pure functions считают
  state machine защищает переходы

Снаружи модуля:
  public API отвечает на вопросы
  events сообщают факты
  DTO передают данные
  outbox гарантирует доставку важных событий
  projections дают быстрые данные для чтения
```

---

## 48. Короткий стандарт стиля кода

```txt
Если это действие — use-case.
Если это правило сущности — domain model.
Если это расчёт — pure function.
Если это данные наружу — DTO.
Если это БД — repository.
Если это внешний сервис — adapter.
Если это вопрос к модулю — public API.
Если это факт — event.
Если факт нельзя потерять — outbox.
Если это дашборд/статистика — projection.
Если это статус — state machine.
Если это долгий живой процесс — actor/process.
```

---

## 49. Самый важный запрет

Не выбирать стиль как религию:

```txt
только классы
только функции
только FP
только DDD
только event-driven
```

Правильный подход:

```txt
Стиль выбирается по роли кода внутри модульной архитектуры.
```

То есть:

```txt
Use-case — для сценария.
Domain model — для правил.
Function — для расчёта.
Repository — для хранения.
Adapter — для внешнего мира.
Event — для факта.
Projection — для чтения.
```

Это универсальный принцип, который можно переносить в разные языки.

---

# Финальная формула

```txt
Модуль владеет своей моделью.
Use-case управляет сценарием.
Domain model защищает правила.
Repository хранит данные.
Adapter работает с внешним миром.
Public API отвечает на вопросы.
Event сообщает факты.
Outbox не даёт потерять важные факты.
Projection строит удобное чтение.
Frontend не дублирует бизнес-логику, а отправляет действия и обновляет данные.
```
