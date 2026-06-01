import { ContentHeader } from "@/components/layouts/content-header/content-header";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { DeleteGroup } from "@/page-components/groups/components/delete-group";
import { UpdateGroup } from "@/page-components/groups/components/update-group";
import { ManageSegmentGroups } from "@/page-components/segments/components/manage-segment-groups/manage-segment-groups";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { GroupDetail } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";

type GroupDetailOutletContext = {
  group: GroupDetail;
};

export const useGroupDetailOutletContext = () => {
  return useOutletContext<GroupDetailOutletContext>();
};

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);

  return (
    <>
      <ContentHeader>
        <ContentHeader.Top
          title={group.name}
          subtitles={[
            ...(group.description ? [group.description] : []),
            `Создана ${formatDate(group.created_at)}`,
          ]}
          meta={[
            `${group.stats.standards_count} эталонов`,
            `${group.stats.images_count} изображений`,
            `${group.stats.annotated_images_count} размечено`,
            `${group.stats.polygons_count} аннотаций`,
            `${group.stats.segment_classes_count} классов`,
          ]}
        >
          <ContentHeader.Actions>
            <DeleteGroup id={group.id} name={group.name} />
            <UpdateGroup group={group} />
            <ManageSegmentGroups group={group} />
            <CreateStandard groupId={group.id} />
          </ContentHeader.Actions>
        </ContentHeader.Top>
      </ContentHeader>

      <Outlet context={{ group }} />
    </>
  );
}
