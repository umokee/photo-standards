import { paths } from "@/app/paths";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  Bell,
  Camera,
  ChevronDown,
  Database,
  FolderOpen,
  Grid3X3,
  HelpCircle,
  Home,
  ListChecks,
  Menu,
  Moon,
  Plus,
  Rocket,
  Search,
  Settings,
  Trash2,
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

const pageMeta = [
  { test: (path: string) => path === "/", title: "Home", trail: ["Home"] },
  { test: (path: string) => path.startsWith("/groups"), title: "Annotate", trail: ["Home", "Annotate"] },
  { test: (path: string) => path.startsWith("/training"), title: "Train", trail: ["Home", "Train"] },
  { test: (path: string) => path.startsWith("/inspection-history"), title: "Runs", trail: ["Home", "Runs"] },
  { test: (path: string) => path.startsWith("/inspection"), title: "Deploy", trail: ["Home", "Deploy"] },
  { test: (path: string) => path.startsWith("/cameras"), title: "Sources", trail: ["Home", "Sources"] },
  { test: (path: string) => path.startsWith("/settings"), title: "Settings", trail: ["Home", "Settings"] },
];

export const PlatformShell = ({ navigation }: Props) => {
  const location = useLocation();
  const routerNavigation = useNavigation();
  const [isMobileOpen, setIsMobileOpen] = React.useState(false);
  const [isCommandOpen, setIsCommandOpen] = React.useState(false);
  const [commandQuery, setCommandQuery] = React.useState("");
  const { data: groups = [] } = useQuery(getGroupsQueryOptions());
  const meta = pageMeta.find((item) => item.test(location.pathname)) ?? pageMeta[0];
  const isNavigating = routerNavigation.state !== "idle";
  const isEditorRoute = location.pathname.includes("/standards/") && location.pathname.includes("/images/");

  const commandItems = React.useMemo<CommandItem[]>(() => {
    const staticItems: CommandItem[] = [
      { to: paths.home(), icon: Home, label: "Home", hint: "Dashboard overview", group: "Navigation" },
      { to: paths.groups(), icon: Database, label: "Annotate", hint: "Datasets, references and images", group: "Navigation" },
      { to: paths.training(), icon: FolderOpen, label: "Train", hint: "Model projects and active weights", group: "Navigation" },
      { to: paths.inspection(), icon: Rocket, label: "Deploy", hint: "Run visual inspection", group: "Navigation" },
      { to: paths.inspectionHistory(), icon: Activity, label: "Runs", hint: "Inspection history and results", group: "Navigation" },
      { to: paths.cameras(), icon: Camera, label: "Sources", hint: "IP cameras and image sources", group: "Navigation" },
      { to: paths.settingsSection("system"), icon: Settings, label: "System", hint: "Runtime, storage and diagnostics", group: "Navigation" },
    ];

    const projectItems = groups.flatMap<CommandItem>((group) => [
      {
        to: paths.groupDetail(group.id),
        icon: Database,
        label: group.name,
        hint: `${group.stats.images_count} images · ${group.stats.standards_count} refs`,
        group: "Datasets",
      },
      {
        to: paths.trainingGroup(group.id),
        icon: FolderOpen,
        label: `${group.name} / train`,
        hint: `${group.stats.models_count} models · ${group.stats.segment_classes_count} classes`,
        group: "Projects",
      },
      {
        to: paths.inspectionGroup("photo", group.id),
        icon: Rocket,
        label: `${group.name} / deploy`,
        hint: `${group.stats.inspections_count} runs`,
        group: "Deploy",
      },
    ]);

    return [...staticItems, ...projectItems];
  }, [groups]);

  const normalizedQuery = commandQuery.trim().toLowerCase();
  const filteredCommands = React.useMemo(() => {
    if (!normalizedQuery) return commandItems.slice(0, 12);

    return commandItems
      .filter((item) => {
        const haystack = `${item.group} ${item.label} ${item.hint}`.toLowerCase();
        return haystack.includes(normalizedQuery);
      })
      .slice(0, 18);
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
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  React.useEffect(() => {
    setIsMobileOpen(false);
    setIsCommandOpen(false);
  }, [location.pathname]);

  const closeMobile = () => setIsMobileOpen(false);
  const openCommand = () => {
    setCommandQuery("");
    setIsCommandOpen(true);
  };
  const closeCommand = () => setIsCommandOpen(false);

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
            <Search />
            <span>Explore</span>
          </NavLink>
        </nav>

        <div className={s.sidebarSectionTitle}>My Projects</div>

        <SidebarTree
          icon={Database}
          title="Annotate"
          to={paths.groups()}
          count={groups.length}
          active={location.pathname.startsWith("/groups")}
          onClick={closeMobile}
        >
          {groups.slice(0, 8).map((group) => (
            <Link key={group.id} to={paths.groupDetail(group.id)} onClick={closeMobile}>
              <DatasetAvatar name={group.name} />
              <span title={group.name}>{group.name}</span>
              <sup>{group.stats.standards_count}</sup>
            </Link>
          ))}
        </SidebarTree>

        <SidebarTree
          icon={FolderOpen}
          title="Train"
          to={paths.training()}
          count={groups.reduce((sum, group) => sum + group.stats.models_count, 0)}
          active={location.pathname.startsWith("/training")}
          onClick={closeMobile}
        >
          {groups.slice(0, 8).map((group) => (
            <Link key={group.id} to={paths.trainingGroup(group.id)} onClick={closeMobile}>
              <ProjectAvatar name={group.name} />
              <span title={group.name}>{group.name}</span>
              <sup>{group.stats.models_count}</sup>
            </Link>
          ))}
        </SidebarTree>

        <SidebarTree
          icon={Rocket}
          title="Deploy"
          to={paths.inspection()}
          count={groups.reduce((sum, group) => sum + group.stats.inspections_count, 0)}
          active={location.pathname.startsWith("/inspection")}
          onClick={closeMobile}
        >
          {groups.slice(0, 6).map((group) => (
            <Link key={group.id} to={paths.inspectionGroup("photo", group.id)} onClick={closeMobile}>
              <span className={s.deployDot} />
              <span title={group.name}>{group.name}</span>
              <sup>{group.stats.inspections_count}</sup>
            </Link>
          ))}
        </SidebarTree>

        <div className={s.sidebarSpacer} />

        <nav className={s.footerNav} aria-label="Служебная навигация">
          <Link to={paths.inspectionHistory()} onClick={closeMobile}>
            <Trash2 />
            <span>Runs</span>
          </Link>
          <Link to={paths.settingsSection("system")} onClick={closeMobile}>
            <Settings />
            <span>Settings</span>
          </Link>
          <Link to={paths.cameras()} onClick={closeMobile}>
            <HelpCircle />
            <span>Sources</span>
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
                  <span className={index === meta.trail.length - 1 ? s.crumbActive : undefined}>
                    {crumb}
                  </span>
                </React.Fragment>
              ))}
            </div>
          </div>

          <button className={s.topbarCenter} type="button" onClick={openCommand}>
            <Search />
            <span>Search projects, standards, cameras...</span>
          </button>

          <div className={s.topbarActions}>
            <Link className={s.pillButton} to={paths.groups()}>
              <Plus />
              Project
            </Link>
            <Link className={s.darkButton} to={paths.inspection()}>
              Deploy
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
                placeholder="Search page, dataset, model, source..."
                value={commandQuery}
                onChange={(event) => setCommandQuery(event.target.value)}
              />
              <button type="button" onClick={closeCommand} aria-label="Close search"><X /></button>
            </div>

            <div className={s.commandMetaRow}>
              <span><ListChecks /> {filteredCommands.length} results</span>
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
                  <span>Попробуй dataset, train, deploy, camera или название изделия.</span>
                </div>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
};

function SidebarTree({
  icon: Icon,
  title,
  to,
  children,
  active,
  count,
  onClick,
}: {
  icon: LucideIcon;
  title: string;
  to: string;
  count: number;
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <div className={s.tree}>
      <Link className={clsx(s.treeHeader, active && s.treeHeaderActive)} to={to} onClick={onClick}>
        <Icon />
        <span>{title}</span>
        <b>{count}</b>
        <ChevronDown />
      </Link>
      <div className={s.treeChildren}>{children}</div>
    </div>
  );
}

function DatasetAvatar({ name }: { name: string }) {
  return <span className={s.datasetAvatar}>{name.slice(0, 1).toUpperCase()}</span>;
}

function ProjectAvatar({ name }: { name: string }) {
  return <span className={s.projectAvatar}>{name.slice(0, 1).toUpperCase()}</span>;
}
