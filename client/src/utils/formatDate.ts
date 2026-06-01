export const formatDate = (date: string | Date): string => {
  const d = new Date(date);
  const day = d.getDate();
  const year = d.getFullYear();

  const months = [
    "янв.",
    "фев.",
    "мар.",
    "апр.",
    "май",
    "июн.",
    "июл.",
    "авг.",
    "сен.",
    "окт.",
    "ноя.",
    "дек.",
  ];

  const month = months[d.getMonth()];
  return `${day} ${month} ${year}`;
};
