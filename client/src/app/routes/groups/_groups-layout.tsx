import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import { CreateGroup } from "@/page-components/groups/components/create-group";
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
          footer={<CreateGroup />}
          getSideContent={(group) => group.stats.standards_count}
          onSelectGroup={(nextGroupId) => navigate(paths.groupDetail(nextGroupId))}
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
