import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useLoaderData } from "react-router-dom";

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);

  const hasStandards = group.standards.length > 0;

  return (
    <QueryState
      isEmpty
      size="page"
      emptyTitle={hasStandards ? "Выберите эталон" : "В этой группе нет эталонов"}
      emptyDescription={
        hasStandards
          ? "Выберите эталон в верхней панели"
          : "Выберите другую группу или создайте эталон"
      }
    />
  );
}
