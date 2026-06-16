import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useLoaderData } from "react-router-dom";

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  return <QueryState isEmpty size="page" emptyTitle={group.standards.length ? "Select reference" : "No references"} emptyDescription={group.standards.length ? "Выбери эталон в верхней панели." : "Сначала добавь эталонные кадры."} />;
}
