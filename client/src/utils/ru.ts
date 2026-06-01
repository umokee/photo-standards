export const ru = {
  users: {
    roles: {
      operator: "Оператор",
      inspector: "Инспектор",
      admin: "Администратор",
    },
  },
  standards: {
    angles: {
      front: "Спереди",
      top: "Сверху",
      left: "Слева",
      right: "Справа",
      back: "Сзади",
    },
  },
  inspections: {
    modes: {
      photo: "Загрузка фото",
      snapshot: "Снимок с камеры",
      realtime: "Live-режим",
    },
    statuses: {
      passed: "Пройдено",
      failed: "Не пройдено",
    },
  },
  training: {
    statuses: {
      pending: "В очереди",
      preparing: "Подготовка",
      training: "Обучение",
      saving: "Сохранение",
      paused: "На паузе",
      done: "Готово",
      failed: "Ошибка",
    },
    architectures: {
      "yolo26n-seg": "YOLO v26 Nano",
      "yolo26s-seg": "YOLO v26 Small",
      "yolo26m-seg": "YOLO v26 Medium",
      "yolo26l-seg": "YOLO v26 Large",
      "yolo26x-seg": "YOLO v26 XL",
    },
    metrics: {
      mAP50: "mAP50",
      mAP50_95: "mAP50-95",
      precision: "Precision",
      recall: "Recall",
    },
  },
};
