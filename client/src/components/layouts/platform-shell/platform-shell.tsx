import { paths } from "@/app/paths";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  Bell,
  Brain,
  Camera,
  ChevronDown,
  CircleDot,
  FolderKanban,
  Grid3X3,
  Home,
  Image,
  ListChecks,
  Menu,
  Moon,
  Plus,
  Search,
  Settings,
  Sparkles,
  Tags,
  X,
  type LucideIcon,
} from "lucide-react";
import * as React from "react";
import { Link, NavLink, Outlet, useLocation, useNavigation } from "react-router-dom";
import s from "./platform-shell.module.scss";

type NavigationItem = {
  to: string;
  icon: LucideIcon;
  label: string;
  section: string;
};

type Props = {
  navigation: readonly NavigationItem[];
};

type CommandItem = {
  to: string;
  icon: LucideIcon;
  label: string;
  hint: string;
  group: string;
};

type SectionId = "assets" | "train" | "inspect";

const pageMeta = [
  { test: (path: string) => path === "/", title: "Home", trail: ["Home"] },
  { test: (path: string) => path.startsWith("/groups"), title: "Projects", trail: ["Home", "Projects"] },
  { test: (path: string) => path.startsWith("/training"), title: "Train", trail: ["Home", "Train"] },
  { test: (path: string) => path.startsWith("/inspection-history"), title: "Runs", trail: ["Home", "Runs"] },
  { test: (path: string) => path.startsWith("/inspection") && !path.startsWith("/inspection-history"), title: "Inspect", trail: ["Home", "Inspect"] },
  { test: (path: string) => path.startsWith("/cameras"), title: "Cameras", trail: ["Home", "Cameras"] },
  { test: (path: string) => path.startsWith("/settings"), title: "System", trail: ["Home", "System"] },
];

function getProjectIdFromPath(pathname: string) {
  return (
    pathname.match(/^\/groups\/([^/]+)/)?.[1] ??
    pathname.match(/^\/training\/([^/]+)/)?.[1] ??
    pathname.match(/^\/inspection\/[^/]+\/groups\/([^/]+)/)?.[1] ??
    pathname.match(/^\/inspection-history\/([^/]+)/)?.[1] ??
    null
  );
}

export const PlatformShell = ({ navigation: _navigation }: Props) => {
  const location = useLocation();
  const routerNavigation = useNavigation();
  const [isMobileOpen, setIsMobileOpen] = React.useState(false);
  const [isCommandOpen, setIsCommandOpen] = React.useState(false);
  const [commandQuery, setCommandQuery] = React.useState("");
  const [isProjectSwitcherOpen, setIsProjectSwitcherOpen] = React.useState(false);
  const [manualProjectId, setManualProjectId] = React.useState<string | null>(null);
  const [expandedSections, setExpandedSections] = React.useState<Record<SectionId, boolean>>({
    assets: true,
    train: true,
    inspect: true,
  });

  const { data: groups = [] } = useQuery(getGroupsQueryOptions());
  const meta = pageMeta.find((item) => item.test(location.pathname)) ?? pageMeta[0];
  const isNavigating = routerNavigation.state !== "idle";
  const isEditorRoute = location.pathname.includes("/standards/") && location.pathname.includes("/images/");
  const projectIdFromPath = getProjectIdFromPath(location.pathname);
  const selectedProject =
    groups.find((group) => group.id === projectIdFromPath) ??
    groups.find((group) => group.id === manualProjectId) ??
    groups[0] ??
    null;

  const selectedProjectId = selectedProject?.id ?? null;
  const assetsActive = location.pathname.startsWith("/groups");
  const trainActive = location.pathname.startsWith("/training");
  const inspectActive = location.pathname.startsWith("/inspection") && !location.pathname.startsWith("/inspection-history");

  const totals = React.useMemo(
    () => ({
      references: groups.reduce((sum, group) => sum + group.stats.standards_count, 0),
      models: groups.reduce((sum, group) => sum + group.stats.models_count, 0),
      runs: groups.reduce((sum, group) => sum + group.stats.inspections_count, 0),
    }),
    [groups]
  );

  React.useEffect(() => {
    if (projectIdFromPath) setManualProjectId(projectIdFromPath);
  }, [projectIdFromPath]);

  const commandItems = React.useMemo<CommandItem[]>(() => {
    const staticItems: CommandItem[] = [
      { to: paths.home(), icon: Home, label: "Home", hint: "Плотный cockpit по всем изделиям", group: "Navigation" },
      { to: paths.groups(), icon: FolderKanban, label: "Projects", hint: "Изделия и их QC-workspace", group: "Workspace" },
      { to: selectedProjectId ? paths.groupDetail(selectedProjectId) : paths.groups(), icon: Image, label: "Assets", hint: "Эталоны, фото, классы и полигоны", group: "Project" },
      { to: selectedProjectId ? paths.trainingOverview(selectedProjectId) : paths.training(), icon: Sparkles, label: "Train", hint: "Модели выбранного изделия", group: "Project" },
      { to: selectedProjectId ? paths.inspectionGroup("photo", selectedProjectId) : paths.inspection(), icon: ListChecks, label: "Inspect", hint: "Проверка изделия по эталону", group: "Project" },
      { to: selectedProjectId ? paths.inspectionHistoryGroup(selectedProjectId) : paths.inspectionHistory(), icon: Activity, label: "Runs", hint: "История проверок", group: "Project" },
      { to: paths.cameras(), icon: Camera, label: "Cameras", hint: "IP/USB источники", group: "System" },
      { to: paths.settingsSection("system"), icon: Settings, label: "System", hint: "Мониторинг и настройки", group: "System" },
    ];

    const projectItems = groups.flatMap<CommandItem>((group) => [
      {
        to: paths.groupDetail(group.id),
        icon: FolderKanban,
        label: group.name,
        hint: `${group.stats.standards_count} refs · ${group.stats.images_count} images · ${group.stats.segment_classes_count} classes`,
        group: "Projects",
      },
      {
        to: paths.trainingGroup(group.id),
        icon: Sparkles,
        label: `${group.name} / Train`,
        hint: `${group.stats.models_count} models · ${group.stats.annotated_images_count} labeled`,
        group: "Train",
      },
      {
        to: paths.inspectionGroup("photo", group.id),
        icon: ListChecks,
        label: `${group.name} / Inspect`,
        hint: `${group.stats.inspections_count} runs`,
        group: "Inspect",
      },
    ]);

    return [...staticItems, ...projectItems];
  }, [groups, selectedProjectId]);

  const normalizedQuery = commandQuery.trim().toLowerCase();
  const filteredCommands = React.useMemo(() => {
    if (!normalizedQuery) return commandItems.slice(0, 14);

    return commandItems
      .filter((item) => `${item.group} ${item.label} ${item.hint}`.toLowerCase().includes(normalizedQuery))
      .slice(0, 20);
  }, [commandItems, normalizedQuery]);

  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandQuery("");
        setIsCommandOpen(true);
        return;
      }

      if (event.key === "Escape") {
        setIsCommandOpen(false);
        setIsProjectSwitcherOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  React.useEffect(() => {
    setIsMobileOpen(false);
    setIsCommandOpen(false);
    setIsProjectSwitcherOpen(false);
  }, [location.pathname]);

  const closeMobile = () => setIsMobileOpen(false);
  const openCommand = () => {
    setCommandQuery("");
    setIsCommandOpen(true);
  };
  const closeCommand = () => setIsCommandOpen(false);
  const toggleSection = (section: SectionId) => {
    setExpandedSections((current) => ({ ...current, [section]: !current[section] }));
  };
  const chooseProject = (projectId: string) => {
    setManualProjectId(projectId);
    setIsProjectSwitcherOpen(false);
  };

  return (
    <div className={clsx(s.root, isEditorRoute && s.editorMode)}>
      <button
        className={clsx(s.mobileBackdrop, isMobileOpen && s.mobileBackdropVisible)}
        type="button"
        aria-label="Закрыть меню"
        tabIndex={isMobileOpen ? 0 : -1}
        onClick={closeMobile}
      />

      <aside className={clsx(s.sidebar, isMobileOpen && s.sidebarOpen)}>
        <Link className={s.logo} to={paths.home()} onClick={closeMobile}>
          <span className={s.logoMark}>
            <span />
            <span />
          </span>
          <span className={s.logoText}>VisionQC</span>
        </Link>

        <button className={s.searchBox} type="button" onClick={openCommand}>
          <Search />
          <span>Search...</span>
          <kbd>Ctrl K</kbd>
        </button>

        <nav className={s.mainNav} aria-label="Основная навигация">
          <NavLink
            to={paths.home()}
            end
            className={({ isActive }) => clsx(s.navItem, isActive && s.navItemActive)}
            onClick={closeMobile}
          >
            <Home />
            <span>Home</span>
          </NavLink>
          <NavLink
            to={paths.groups()}
            className={({ isActive }) => clsx(s.navItem, isActive && s.navItemActive)}
            onClick={closeMobile}
          >
            <FolderKanban />
            <span>Projects</span>
          </NavLink>
        </nav>

        <div className={s.projectContextV19}>
          <div className={s.sidebarSectionTitle}>Current project</div>
          {selectedProject ? (
            <div className={s.projectSwitcherWrapV19}>
              <button className={s.projectSwitcherV19} type="button" onClick={() => setIsProjectSwitcherOpen((value) => !value)}>
                <ProjectAvatar name={selectedProject.name} />
                <span>
                  <strong>{selectedProject.name}</strong>
                  <small>{selectedProject.stats.standards_count} refs · {selectedProject.stats.images_count} images</small>
                </span>
                <ChevronDown />
              </button>
              {isProjectSwitcherOpen ? (
                <div className={s.projectMenuV19}>
                  {groups.map((group) => (
                    <Link
                      className={clsx(group.id === selectedProject.id && s.projectMenuItemActiveV19)}
                      key={group.id}
                      to={paths.groupDetail(group.id)}
                      onClick={() => {
                        chooseProject(group.id);
                        closeMobile();
                      }}
                    >
                      <ProjectAvatar name={group.name} />
                      <span>
                        <strong>{group.name}</strong>
                        <small>{group.stats.standards_count} refs · {group.stats.models_count} models · {group.stats.inspections_count} runs</small>
                      </span>
                    </Link>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <Link className={s.createFirstProjectV19} to={paths.groups()} onClick={closeMobile}>
              <Plus />
              <span>Create first project</span>
            </Link>
          )}
        </div>

        <div className={s.sidebarSectionTitle}>Workspace</div>

        <WorkspaceSection
          id="assets"
          icon={Image}
          title="Assets"
          count={selectedProject?.stats.standards_count ?? 0}
          isOpen={expandedSections.assets}
          active={assetsActive}
          addTo={selectedProjectId ? paths.assetReferences(selectedProjectId) : paths.groups()}
          addTitle="Создать эталон"
          onToggle={toggleSection}
        >
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.assetOverview(selectedProject.id)} icon={FolderKanban} label="Overview" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.assetReferences(selectedProject.id)} icon={Image} label="References" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.assetClasses(selectedProject.id)} icon={Tags} label="Classes" exact onClick={closeMobile} />
            </>
          ) : (
            <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create first project</EmptyTreeLink>
          )}
        </WorkspaceSection>

        <WorkspaceSection
          id="train"
          icon={Sparkles}
          title="Train"
          count={selectedProject?.stats.models_count ?? totals.models}
          isOpen={expandedSections.train}
          active={trainActive}
          addTo={selectedProjectId ? paths.trainingOverview(selectedProjectId) : paths.training()}
          addTitle="Обучить или импортировать модель"
          onToggle={toggleSection}
        >
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.trainingOverview(selectedProject.id)} icon={Sparkles} label="Overview" onClick={closeMobile} />
              <WorkspaceLink to={paths.trainingModels(selectedProject.id)} icon={Brain} label="Models" onClick={closeMobile} />
              <WorkspaceLink to={paths.trainingRuns(selectedProject.id)} icon={Activity} label="Training runs" onClick={closeMobile} />
            </>
          ) : (
            <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create project first</EmptyTreeLink>
          )}
        </WorkspaceSection>

        <WorkspaceSection
          id="inspect"
          icon={ListChecks}
          title="Inspect"
          count={selectedProject?.stats.inspections_count ?? totals.runs}
          isOpen={expandedSections.inspect}
          active={inspectActive || location.pathname.startsWith("/inspection-history")}
          addTo={selectedProjectId ? paths.inspectionGroup("photo", selectedProjectId) : paths.inspection()}
          addTitle="Запустить проверку"
          onToggle={toggleSection}
        >
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.inspectionGroup("photo", selectedProject.id)} icon={ListChecks} label="Station" onClick={closeMobile} />
              <WorkspaceLink to={paths.inspectionHistoryGroup(selectedProject.id)} icon={Activity} label="Runs" onClick={closeMobile} />
            </>
          ) : (
            <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create project first</EmptyTreeLink>
          )}
        </WorkspaceSection>

        <div className={s.sidebarSpacer} />

        <nav className={s.footerNav} aria-label="Служебная навигация">
          <Link to={paths.cameras()} onClick={closeMobile}>
            <Camera />
            <span>Cameras</span>
          </Link>
          <Link to={paths.settingsSection("system")} onClick={closeMobile}>
            <Settings />
            <span>System</span>
          </Link>
        </nav>

        <div className={s.profileCard}>
          <span className={s.profileAvatar}>Q</span>
          <span className={s.profileBody}>
            <strong>local workspace</strong>
            <small>quality-control.local</small>
          </span>
          <span className={s.profileDots}>⋮</span>
        </div>
      </aside>

      <section className={s.workspace}>
        <header className={s.topbar}>
          <div className={s.topbarLeft}>
            <button className={s.mobileMenu} type="button" onClick={() => setIsMobileOpen(true)}>
              <Menu />
            </button>
            <div className={s.breadcrumbs}>
              {meta.trail.map((crumb, index) => (
                <React.Fragment key={`${crumb}-${index}`}>
                  {index > 0 ? <span className={s.crumbSep}>›</span> : null}
                  <span className={index === meta.trail.length - 1 ? s.crumbActive : undefined}>{crumb}</span>
                </React.Fragment>
              ))}
            </div>
          </div>

          <button className={s.topbarCenter} type="button" onClick={openCommand}>
            <Search />
            <span>Search projects, references, cameras...</span>
          </button>

          <div className={s.topbarActions}>
            <Link className={s.pillButton} to={paths.groups()}>
              <Plus />
              Project
            </Link>
            <Link className={s.darkButton} to={selectedProjectId ? paths.inspectionGroup("photo", selectedProjectId) : paths.inspection()}>
              Inspect
            </Link>
            <span className={s.balance}>LOCAL</span>
            <button type="button" aria-label="Приложения"><Grid3X3 /></button>
            <button type="button" aria-label="Уведомления"><Bell /></button>
            <button type="button" aria-label="Тема"><Moon /></button>
          </div>
        </header>

        {isNavigating ? <div className={s.progressLine} /> : null}

        <main className={clsx(s.content, isEditorRoute && s.contentFlush)}>
          <Outlet />
        </main>
      </section>

      {isCommandOpen ? (
        <div className={s.commandOverlay} role="presentation" onMouseDown={closeCommand}>
          <section
            className={s.commandDialog}
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className={s.commandSearchRow}>
              <Search />
              <input
                autoFocus
                placeholder="Search project, reference, model, camera..."
                value={commandQuery}
                onChange={(event) => setCommandQuery(event.target.value)}
              />
              <button type="button" onClick={closeCommand} aria-label="Close search"><X /></button>
            </div>

            <div className={s.commandMetaRow}>
              <span><CircleDot /> {filteredCommands.length} results</span>
              <span>Ctrl K</span>
            </div>

            <div className={s.commandList}>
              {filteredCommands.length ? (
                filteredCommands.map((item) => {
                  const Icon = item.icon;

                  return (
                    <Link className={s.commandItem} key={`${item.group}-${item.to}-${item.label}`} to={item.to} onClick={closeCommand}>
                      <span className={s.commandIcon}><Icon /></span>
                      <span className={s.commandText}>
                        <strong>{item.label}</strong>
                        <small>{item.hint}</small>
                      </span>
                      <b>{item.group}</b>
                    </Link>
                  );
                })
              ) : (
                <div className={s.commandEmpty}>
                  <Search />
                  <strong>Nothing found</strong>
                  <span>Попробуй project, reference, train, inspect или camera.</span>
                </div>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
};

function WorkspaceSection({
  id,
  icon: Icon,
  title,
  children,
  active,
  isOpen,
  count,
  addTo,
  addTitle,
  onToggle,
}: {
  id: SectionId;
  icon: LucideIcon;
  title: string;
  children: React.ReactNode;
  active?: boolean;
  isOpen: boolean;
  count: number;
  addTo: string;
  addTitle: string;
  onToggle: (section: SectionId) => void;
}) {
  return (
    <div className={clsx(s.tree, s.workspaceTreeV19, isOpen && s.treeOpenV19)}>
      <div className={s.treeHeaderWrapV19}>
        <button className={clsx(s.treeHeader, active && s.treeHeaderActive)} type="button" onClick={() => onToggle(id)}>
          <Icon />
          <span>{title}</span>
          <b>{count}</b>
          <ChevronDown className={s.treeChevronV19} />
        </button>
        <Link className={s.treePlusV19} to={addTo} title={addTitle} aria-label={addTitle}>
          <Plus />
        </Link>
      </div>
      {isOpen ? <div className={s.workspaceTreeChildrenV19}>{children}</div> : null}
    </div>
  );
}

function WorkspaceLink({ to, icon: Icon, label, exact, onClick }: { to: string; icon: LucideIcon; label: string; exact?: boolean; onClick?: () => void }) {
  return (
    <NavLink to={to} end={exact} className={({ isActive }) => clsx(s.workspaceLinkV19, isActive && s.workspaceLinkActiveV19)} onClick={onClick}>
      <Icon />
      <span>{label}</span>
    </NavLink>
  );
}

function EmptyTreeLink({ to, onClick, children }: { to: string; onClick?: () => void; children: React.ReactNode }) {
  return (
    <Link className={s.emptyTreeLinkV19} to={to} onClick={onClick}>
      <Plus />
      <span>{children}</span>
    </Link>
  );
}

function ProjectAvatar({ name }: { name: string }) {
  return <span className={s.projectAvatar}>{name.slice(0, 1).toUpperCase()}</span>;
}
