import { useDeferredValue, useMemo, useState } from "react";

type Searchable = {
  id: string;
  name?: string | null;
};

type Options<T> = {
  items: T[];
  getText?: (item: T) => string;
};

export const useSearch = <T extends Searchable>({ items, getText }: Options<T>) => {
  const [search, setSearch] = useState("");

  const normalizedSearch = search.trim().toLowerCase();
  const deferredSearch = useDeferredValue(normalizedSearch);

  const filtered = useMemo(() => {
    if (!deferredSearch) return items;

    return items.filter((item) => {
      const rawText = getText ? getText(item) : (item.name ?? "");
      const text = rawText.toLowerCase();

      return text.includes(deferredSearch);
    });
  }, [items, getText, deferredSearch]);

  const resetSearch = () => setSearch("");

  return {
    search,
    setSearch,
    filtered,
    resetSearch,
  };
};
