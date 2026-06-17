import { useGetGroup } from "@/page-components/groups/api/get-group";
import type { GroupDetail } from "@/types/contracts";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import p from "../platform-pages.module.scss";

type GroupDetailOutletContext = { group: GroupDetail };

export const useGroupDetailOutletContext = () => useOutletContext<GroupDetailOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);

  return (
    <div className={`${p.page} ${p.assetsWorkspaceV24}`}>
      <Outlet context={{ group }} />
    </div>
  );
}
