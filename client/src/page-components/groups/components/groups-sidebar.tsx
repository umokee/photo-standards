import { Sidebar } from "@/components/layouts/sidebar/sidebar";
import Input from "@/components/ui/input/input";
import { QueryBoundary } from "@/components/ui/query-boundary/query-boundary";
import { useSearch } from "@/hooks/use-search";
import useSidebar from "@/hooks/use-sidebar";
import type { GroupListItem } from "@/types/contracts";
import type { ReactNode } from "react";
import { useGetGroups } from "../api/get-groups";

type Props = {
  activeGroupId: string | null;
  footer?: ReactNode;
  getSideContent: (group: GroupListItem) => ReactNode;
  onSelectGroup: (groupId: string) => void;
};

export const GroupsSidebar = ({
  activeGroupId,
  footer,
  getSideContent,
  onSelectGroup,
}: Props) => {
  const { data } = useGetGroups();
  const { close: closeSidebar } = useSidebar();
  const {
    search,
    setSearch,
    filtered: groups,
  } = useSearch({ items: data, getText: (group) => group.name });

  const handleSelectGroup = (groupId: string) => {
    onSelectGroup(groupId);
    closeSidebar();
  };

  return (
    <Sidebar>
      <Sidebar.Header>
        <Sidebar.HeaderTop>
          <Sidebar.Title>Группы</Sidebar.Title>
        </Sidebar.HeaderTop>
        <Input placeholder="Поиск..." noMargin value={search} onChange={setSearch} />
      </Sidebar.Header>

      <QueryBoundary
        size="block"
        loadingText="Загрузка групп..."
        errorTitle="Не удалось загрузить группы"
      >
        <GroupsSidebarList
          activeGroupId={activeGroupId}
          getSideContent={getSideContent}
          groups={groups}
          onSelectGroup={handleSelectGroup}
        />
      </QueryBoundary>

      {footer ? <Sidebar.Footer>{footer}</Sidebar.Footer> : null}
    </Sidebar>
  );
};

type GroupsSidebarListProps = {
  activeGroupId: string | null;
  getSideContent: (group: GroupListItem) => ReactNode;
  groups: GroupListItem[];
  onSelectGroup: (groupId: string) => void;
};

const GroupsSidebarList = ({
  activeGroupId,
  getSideContent,
  groups,
  onSelectGroup,
}: GroupsSidebarListProps) => {
  return (
    <Sidebar.List>
      {groups.map((group) => (
        <Sidebar.Item
          key={group.id}
          active={activeGroupId === group.id}
          onClick={() => onSelectGroup(group.id)}
        >
          <Sidebar.ItemDot />
          <Sidebar.ItemBody>
            <Sidebar.ItemName>{group.name}</Sidebar.ItemName>
          </Sidebar.ItemBody>
          <Sidebar.ItemSide>{getSideContent(group)}</Sidebar.ItemSide>
        </Sidebar.Item>
      ))}
    </Sidebar.List>
  );
};
