import QueryState from "@/components/ui/query-state/query-state";

export function Component() {
  return (
    <QueryState
      isEmpty
      size="page"
      emptyTitle="Выберите группу"
      emptyDescription="Выберите группу из списка или создайте новую"
    />
  );
}
