import { ClassesWorkspace } from "@/page-components/segments/components/classes-workspace/classes-workspace";
import { useGroupDetailOutletContext } from "./_group-detail";

export function Component() {
  const { group } = useGroupDetailOutletContext();
  return <ClassesWorkspace group={group} />;
}
