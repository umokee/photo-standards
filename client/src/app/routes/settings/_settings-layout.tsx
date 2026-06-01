import { Sidebar } from "@/components/layouts/sidebar/sidebar";
import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import useSidebar from "@/hooks/use-sidebar";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { paths, SettingsSectionPath } from "../../paths";

export function Component() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { close: closeSidebar } = useSidebar();

  const isSystem = pathname.startsWith(paths.settingsSection("system"));

  const handleTo = (section: SettingsSectionPath) => {
    navigate(paths.settingsSection(section));
    closeSidebar();
  };

  return (
    <SplitLayout>
      <SplitLayout.Sidebar>
        <Sidebar>
          <Sidebar.Header>
            <Sidebar.HeaderTop>
              <Sidebar.Title>Настройки</Sidebar.Title>
            </Sidebar.HeaderTop>
          </Sidebar.Header>

          <Sidebar.List>
            <Sidebar.Item active={isSystem} onClick={() => handleTo("system")}>
              <Sidebar.ItemDot />
              <Sidebar.ItemBody>
                <Sidebar.ItemName>Система</Sidebar.ItemName>
                <Sidebar.ItemMeta>Ресурсы и хранилище</Sidebar.ItemMeta>
              </Sidebar.ItemBody>
            </Sidebar.Item>
          </Sidebar.List>
        </Sidebar>
      </SplitLayout.Sidebar>

      <SplitLayout.Content>
        <SplitLayout.Body>
          <Outlet />
        </SplitLayout.Body>
      </SplitLayout.Content>
    </SplitLayout>
  );
}
