import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import { GroupsSidebar } from "@/page-components/groups/components/groups-sidebar";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import { paths } from "../../paths";

export function Component() {
  const { groupId } = useParams();
  const navigate = useNavigate();

  return (
    <SplitLayout>
      <SplitLayout.Sidebar>
        <GroupsSidebar
          activeGroupId={groupId ?? null}
          getSideContent={(group) => group.stats.inspections_count}
          onSelectGroup={(nextGroupId) => navigate(paths.inspectionHistoryGroup(nextGroupId))}
        />
      </SplitLayout.Sidebar>

      <SplitLayout.Content>
        <SplitLayout.Body>
          <Outlet />
        </SplitLayout.Body>
      </SplitLayout.Content>
    </SplitLayout>
  );
}
