import { paths } from "@/app/paths";
import { getCurrentTheme, setAppTheme, subscribeTheme, type AppTheme } from "@/lib/theme";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  Brain,
  Camera,
  ChevronDown,
  CircleDot,
  FolderKanban,
  Home,
  Image,
  ListChecks,
  Menu,
  Moon,
  Search,
  Settings,
  Sparkles,
  Sun,
  Tags,
  X,
  type LucideIcon,
} from "lucide-react";
import * as React from "react";
import { createPortal } from "react-dom";
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
type Crumb = { label: string; to?: string };
type ProjectLike = {
  id: string;
  name: string;
  stats: {
    standards_count: number;
    images_count: number;
    segment_classes_count: number;
    models_count: number;
    inspections_count: number;
    annotated_images_count?: number;
  };
};

function getProjectIdFromPath(pathname: string) {
  return (
    pathname.match(/^\/groups\/([^/]+)/)?.[1] ??
    pathname.match(/^\/training\/([^/]+)/)?.[1] ??
    pathname.match(/^\/inspection\/[^/]+\/groups\/([^/]+)/)?.[1] ??
    pathname.match(/^\/inspection-history\/([^/]+)/)?.[1] ??
    null
  );
}

function getModeFromPath(pathname: string) {
  return pathname.match(/^\/inspection\/([^/]+)/)?.[1] ?? null;
}

function buildBreadcrumbs(pathname: string, project: ProjectLike | null, projectIdFromPath: string | null): Crumb[] {
  const crumbs: Crumb[] = [{ label: "Home", to: paths.home() }];

  if (pathname === "/") return [{ label: "Home" }];

  if (pathname.startsWith("/groups")) {
    crumbs.push({ label: "Projects", to: paths.groups() });
    if (!projectIdFromPath || !project) return crumbs.map((crumb, index) => index === crumbs.length - 1 ? { label: crumb.label } : crumb);

    crumbs.push({ label: project.name, to: paths.groupDetail(project.id) });
    if (pathname.includes("/references")) crumbs.push({ label: "References" });
    else if (pathname.includes("/classes")) crumbs.push({ label: "Classes" });
    else if (pathname.includes("/standards/") && pathname.includes("/images/")) crumbs.push({ label: "Reference", to: pathname.split("/images/")[0] }, { label: "Editor" });
    else if (pathname.includes("/standards/")) crumbs.push({ label: "Reference" });
    else crumbs.push({ label: "Assets" });
    return crumbs;
  }

  if (pathname.startsWith("/training")) {
    crumbs.push({ label: "Train", to: project ? paths.trainingOverview(project.id) : paths.training() });
    if (project) crumbs.push({ label: project.name, to: paths.trainingOverview(project.id) });
    if (pathname.includes("/models/")) crumbs.push({ label: "Models", to: project ? paths.trainingModels(project.id) : undefined }, { label: "Model" });
    else if (pathname.includes("/models")) crumbs.push({ label: "Models" });
    else if (pathname.includes("/runs")) crumbs.push({ label: "Training runs" });
    else crumbs.push({ label: "Overview" });
    return crumbs;
  }

  if (pathname.startsWith("/inspection-history")) {
    crumbs.push({ label: "Inspect", to: project ? paths.inspectionGroup("photo", project.id) : paths.inspection() });
    crumbs.push({ label: "Runs", to: project ? paths.inspectionHistoryGroup(project.id) : paths.inspectionHistory() });
    if (project) crumbs.push({ label: project.name, to: paths.inspectionHistoryGroup(project.id) });
    if (/^\/inspection-history\/[^/]+\/[^/]+/.test(pathname)) crumbs.push({ label: "Report" });
    return crumbs;
  }

  if (pathname.startsWith("/inspection")) {
    const mode = getModeFromPath(pathname);
    crumbs.push({ label: "Inspect", to: paths.inspection() });
    if (mode) crumbs.push({ label: mode[0]?.toUpperCase() + mode.slice(1), to: paths.inspectionMode(mode as never) });
    if (project) crumbs.push({ label: project.name, to: paths.inspectionGroup(mode ?? "photo", project.id) });
    if (pathname.includes("/standards/")) crumbs.push({ label: "Reference" });
    return crumbs;
  }

  if (pathname.startsWith("/cameras")) return [...crumbs, { label: "Cameras" }];
  if (pathname.startsWith("/settings")) return [...crumbs, { label: "System" }];

  return crumbs;
}

export const PlatformShell = ({ navigation: _navigation }: Props) => {
  const location = useLocation();
  const routerNavigation = useNavigation();
  const [isMobileOpen, setIsMobileOpen] = React.useState(false);
  const [isCommandOpen, setIsCommandOpen] = React.useState(false);
  const [commandQuery, setCommandQuery] = React.useState("");
  const [isProjectSwitcherOpen, setIsProjectSwitcherOpen] = React.useState(false);
  const [manualProjectId, setManualProjectId] = React.useState<string | null>(null);
  const [projectMenuPosition, setProjectMenuPosition] = React.useState({ top: 0, left: 0 });
  const [theme, setThemeState] = React.useState<AppTheme>(() => getCurrentTheme());
  const [expandedSections, setExpandedSections] = React.useState<Record<SectionId, boolean>>({
    assets: true,
    train: true,
    inspect: true,
  });

  const projectSwitcherRef = React.useRef<HTMLButtonElement | null>(null);
  const projectMenuRef = React.useRef<HTMLDivElement | null>(null);

  const { data: groups = [] } = useQuery(getGroupsQueryOptions());
  const isNavigating = routerNavigation.state !== "idle";
  const isEditorRoute = location.pathname.includes("/standards/") && location.pathname.includes("/images/");
  const projectIdFromPath = getProjectIdFromPath(location.pathname);
  const selectedProject =
    groups.find((group) => group.id === projectIdFromPath) ??
    groups.find((group) => group.id === manualProjectId) ??
    groups[0] ??
    null;

  const selectedProjectId = selectedProject?.id ?? null;
  const assetsActive = /^\/groups\/[^/]+/.test(location.pathname);
  const trainActive = location.pathname.startsWith("/training");
  const inspectActive =
    (location.pathname.startsWith("/inspection") && !location.pathname.startsWith("/inspection-history")) ||
    location.pathname.startsWith("/inspection-history");
  const crumbs = React.useMemo(
    () => buildBreadcrumbs(location.pathname, selectedProject, projectIdFromPath),
    [location.pathname, selectedProject, projectIdFromPath]
  );

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

  React.useEffect(() => subscribeTheme(setThemeState), []);

  const toggleTheme = React.useCallback(() => {
    setThemeState((current) => {
      const nextTheme = current === "dark" ? "light" : "dark";
      setAppTheme(nextTheme);
      return nextTheme;
    });
  }, []);

  const updateProjectMenuPosition = React.useCallback(() => {
    const button = projectSwitcherRef.current;
    if (!button) return;
    const rect = button.getBoundingClientRect();
    const width = 286;
    const left = Math.min(rect.right + 10, Math.max(10, window.innerWidth - width - 10));
    setProjectMenuPosition({ top: Math.max(10, rect.top), left });
  }, []);

  React.useEffect(() => {
    if (!isProjectSwitcherOpen) return;
    updateProjectMenuPosition();
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (projectSwitcherRef.current?.contains(target)) return;
      if (projectMenuRef.current?.contains(target)) return;
      setIsProjectSwitcherOpen(false);
    };
    window.addEventListener("resize", updateProjectMenuPosition);
    window.addEventListener("scroll", updateProjectMenuPosition, true);
    window.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener("resize", updateProjectMenuPosition);
      window.removeEventListener("scroll", updateProjectMenuPosition, true);
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [isProjectSwitcherOpen, updateProjectMenuPosition]);

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
        hint: `${group.stats.models_count} models · ${group.stats.annotated_images_count ?? 0} labeled`,
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
  const toggleProjectSwitcher = () => {
    updateProjectMenuPosition();
    setIsProjectSwitcherOpen((value) => !value);
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
          <span className={s.logoMark}><span /><span /></span>
          <span className={s.logoText}>VisionQC</span>
        </Link>

        <button className={s.searchBox} type="button" onClick={openCommand}>
          <Search />
          <span>Search...</span>
          <kbd>Ctrl K</kbd>
        </button>

        <nav className={s.mainNav} aria-label="Основная навигация">
          <NavLink to={paths.home()} end className={({ isActive }) => clsx(s.navItem, isActive && s.navItemActive)} onClick={closeMobile}>
            <Home /><span>Home</span>
          </NavLink>
          <NavLink to={paths.groups()} end className={({ isActive }) => clsx(s.navItem, isActive && s.navItemActive)} onClick={closeMobile}>
            <FolderKanban /><span>Projects</span>
          </NavLink>
        </nav>

        <div className={s.projectContextV19}>
          <div className={s.sidebarSectionTitle}>Current project</div>
          {selectedProject ? (
            <div className={s.projectSwitcherWrapV19}>
              <button ref={projectSwitcherRef} className={s.projectSwitcherV19} type="button" onClick={toggleProjectSwitcher}>
                <ProjectAvatar name={selectedProject.name} />
                <span>
                  <strong>{selectedProject.name}</strong>
                  <small>{selectedProject.stats.standards_count} refs · {selectedProject.stats.images_count} images</small>
                </span>
                <ChevronDown />
              </button>
            </div>
          ) : (
            <Link className={s.createFirstProjectV19} to={paths.groups()} onClick={closeMobile}>
              <span>Create first project</span>
            </Link>
          )}
        </div>

        <div className={s.sidebarSectionTitle}>Workspace</div>

        <WorkspaceSection id="assets" icon={Image} title="Assets" count={selectedProject?.stats.standards_count ?? 0} isOpen={expandedSections.assets} active={assetsActive} onToggle={toggleSection}>
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.assetOverview(selectedProject.id)} icon={FolderKanban} label="Overview" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.assetReferences(selectedProject.id)} icon={Image} label="References" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.assetClasses(selectedProject.id)} icon={Tags} label="Classes" exact onClick={closeMobile} />
            </>
          ) : <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create first project</EmptyTreeLink>}
        </WorkspaceSection>

        <WorkspaceSection id="train" icon={Sparkles} title="Train" count={selectedProject?.stats.models_count ?? totals.models} isOpen={expandedSections.train} active={trainActive} onToggle={toggleSection}>
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.trainingOverview(selectedProject.id)} icon={Sparkles} label="Overview" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.trainingModels(selectedProject.id)} icon={Brain} label="Models" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.trainingRuns(selectedProject.id)} icon={Activity} label="Training runs" exact onClick={closeMobile} />
            </>
          ) : <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create project first</EmptyTreeLink>}
        </WorkspaceSection>

        <WorkspaceSection id="inspect" icon={ListChecks} title="Inspect" count={selectedProject?.stats.inspections_count ?? totals.runs} isOpen={expandedSections.inspect} active={inspectActive} onToggle={toggleSection}>
          {selectedProject ? (
            <>
              <WorkspaceLink to={paths.inspectionGroup("photo", selectedProject.id)} icon={ListChecks} label="Station" exact onClick={closeMobile} />
              <WorkspaceLink to={paths.inspectionHistoryGroup(selectedProject.id)} icon={Activity} label="Runs" exact onClick={closeMobile} />
            </>
          ) : <EmptyTreeLink to={paths.groups()} onClick={closeMobile}>Create project first</EmptyTreeLink>}
        </WorkspaceSection>

        <div className={s.sidebarSpacer} />

        <nav className={s.footerNav} aria-label="Служебная навигация">
          <Link to={paths.cameras()} onClick={closeMobile}><Camera /><span>Cameras</span></Link>
          <Link to={paths.settingsSection("system")} onClick={closeMobile}><Settings /><span>System</span></Link>
        </nav>

        <div className={s.profileCard}>
          <span className={s.profileAvatar}>Q</span>
          <span className={s.profileBody}><strong>local workspace</strong><small>quality-control.local</small></span>
          <span className={s.profileDots}>⋮</span>
        </div>
      </aside>

      <section className={s.workspace}>
        <header className={s.topbar}>
          <div className={s.topbarLeft}>
            <button className={s.mobileMenu} type="button" onClick={() => setIsMobileOpen(true)}><Menu /></button>
            <nav className={s.breadcrumbs} aria-label="Breadcrumbs">
              {crumbs.map((crumb, index) => {
                const isLast = index === crumbs.length - 1;
                return (
                  <React.Fragment key={`${crumb.label}-${index}`}>
                    {index > 0 ? <span className={s.crumbSep}>›</span> : null}
                    {crumb.to && !isLast ? (
                      <Link className={s.crumbLinkV50} to={crumb.to}>{crumb.label}</Link>
                    ) : (
                      <span className={clsx(isLast && s.crumbActive)}>{crumb.label}</span>
                    )}
                  </React.Fragment>
                );
              })}
            </nav>
          </div>

          <button className={s.topbarCenter} type="button" onClick={openCommand}>
            <Search />
            <span>Search projects, references, cameras...</span>
          </button>

          <div className={s.topbarActions}>
            <Link className={s.pillButton} to={paths.groups()}><FolderKanban />Projects</Link>
            <Link className={s.darkButton} to={selectedProjectId ? paths.inspectionGroup("photo", selectedProjectId) : paths.inspection()}>Inspect</Link>
            <button
              type="button"
              aria-label={theme === "dark" ? "Включить светлую тему" : "Включить тёмную тему"}
              aria-pressed={theme === "dark"}
              title={theme === "dark" ? "Включить светлую тему" : "Включить тёмную тему"}
              onClick={toggleTheme}
            >
              {theme === "dark" ? <Sun /> : <Moon />}
            </button>
          </div>
        </header>

        {isNavigating ? <div className={s.progressLine} /> : null}

        <main className={clsx(s.content, isEditorRoute && s.contentFlush)}>
          <Outlet />
        </main>
      </section>

      {isProjectSwitcherOpen && typeof document !== "undefined" ? createPortal(
        <div ref={projectMenuRef} className={s.projectMenuPortalV50} style={{ top: projectMenuPosition.top, left: projectMenuPosition.left }}>
          <div className={s.projectMenuHeaderV50}><span>Projects</span><b>{groups.length} изделий</b></div>
          {groups.map((group) => (
            <Link
              className={clsx(s.projectMenuItemV50, group.id === selectedProject?.id && s.projectMenuItemActiveV19)}
              key={group.id}
              to={paths.groupDetail(group.id)}
              onClick={() => { chooseProject(group.id); closeMobile(); }}
            >
              <ProjectAvatar name={group.name} />
              <span><strong>{group.name}</strong><small>{group.stats.standards_count} refs · {group.stats.models_count} models · {group.stats.inspections_count} runs</small></span>
            </Link>
          ))}
        </div>,
        document.body
      ) : null}

      {isCommandOpen ? (
        <div className={s.commandOverlay} role="presentation" onMouseDown={closeCommand}>
          <section className={s.commandDialog} role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={(event) => event.stopPropagation()}>
            <div className={s.commandSearchRow}>
              <Search />
              <input autoFocus placeholder="Search project, reference, model, camera..." value={commandQuery} onChange={(event) => setCommandQuery(event.target.value)} />
              <button type="button" onClick={closeCommand} aria-label="Close search"><X /></button>
            </div>
            <div className={s.commandMetaRow}><span><CircleDot /> {filteredCommands.length} results</span><span>Ctrl K</span></div>
            <div className={s.commandList}>
              {filteredCommands.length ? filteredCommands.map((item) => {
                const Icon = item.icon;
                return (
                  <Link className={s.commandItem} key={`${item.group}-${item.to}-${item.label}`} to={item.to} onClick={closeCommand}>
                    <span className={s.commandIcon}><Icon /></span>
                    <span className={s.commandText}><strong>{item.label}</strong><small>{item.hint}</small></span>
                    <b>{item.group}</b>
                  </Link>
                );
              }) : (
                <div className={s.commandEmpty}><Search /><strong>Nothing found</strong><span>Попробуй project, reference, train, inspect или camera.</span></div>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
};

function WorkspaceSection({ id, icon: Icon, title, children, active, isOpen, count, onToggle }: {
  id: SectionId;
  icon: LucideIcon;
  title: string;
  children: React.ReactNode;
  active?: boolean;
  isOpen: boolean;
  count: number;
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
      </div>
      {isOpen ? <div className={s.workspaceTreeChildrenV19}>{children}</div> : null}
    </div>
  );
}

function WorkspaceLink({ to, icon: Icon, label, exact, onClick }: { to: string; icon: LucideIcon; label: string; exact?: boolean; onClick?: () => void }) {
  return (
    <NavLink to={to} end={exact} className={({ isActive }) => clsx(s.workspaceLinkV19, isActive && s.workspaceLinkActiveV19)} onClick={onClick}>
      <Icon /><span>{label}</span>
    </NavLink>
  );
}

function EmptyTreeLink({ to, onClick, children }: { to: string; onClick?: () => void; children: React.ReactNode }) {
  return <Link className={s.emptyTreeLinkV19} to={to} onClick={onClick}><span>{children}</span></Link>;
}

function ProjectAvatar({ name }: { name: string }) {
  return <span className={s.projectAvatar}>{name.slice(0, 1).toUpperCase()}</span>;
}
