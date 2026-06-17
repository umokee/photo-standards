import { useGetGroup } from "@/page-components/groups/api/get-group";
import type { GroupDetail } from "@/types/contracts";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import s from "./_project-assets-strict.module.scss";

type GroupDetailOutletContext = { group: GroupDetail };

export const useGroupDetailOutletContext = () => useOutletContext<GroupDetailOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);

  return (
    <div className={s.assetsRouteShell}>
      <Outlet context={{ group }} />
    </div>
  );
}
