import QueryState from "@/components/ui/query-state/query-state";

export function Component() {
  return (
    <QueryState
      isEmpty
      size="page"
      emptyTitle="Выберите камеру"
      emptyDescription="Выберите камеру из списка или добавьте новую"
    />
  );
}
