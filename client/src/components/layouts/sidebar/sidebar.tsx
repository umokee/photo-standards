import clsx from "clsx";
import { ReactNode } from "react";
import s from "./sidebar.module.scss";

type ItemProps = {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
};

type DotStatus = "ok" | "warning" | "error";

const dotStatusClassMap = {
  ok: s.dotOk,
  warning: s.dotWarning,
  error: s.dotError,
};

const Root = ({ children }: { children: ReactNode }) => <>{children}</>;

const Header = ({ children }: { children: ReactNode }) => (
  <div className={s.header}>{children}</div>
);

const HeaderTop = ({ children }: { children: ReactNode }) => (
  <div className={s.headerTop}>{children}</div>
);

const Title = ({ children }: { children: ReactNode }) => (
  <span className={s.title}>{children}</span>
);

const List = ({ children }: { children: ReactNode }) => <div className={s.list}>{children}</div>;

const Item = ({ children, active, onClick }: ItemProps) => (
  <button
    type="button"
    className={clsx(s.item, active && s.itemActive)}
    onClick={onClick}
    aria-current={active ? "page" : undefined}
  >
    {children}
  </button>
);

const ItemDot = ({ status }: { status?: DotStatus }) => (
  <span className={clsx(s.itemDot, status && dotStatusClassMap[status])} />
);

const ItemBody = ({ children }: { children: ReactNode }) => (
  <div className={s.itemBody}>{children}</div>
);

const ItemName = ({ children }: { children: ReactNode }) => (
  <span className={s.itemName}>{children}</span>
);

const ItemMeta = ({ children }: { children: ReactNode }) => (
  <span className={s.itemMeta}>{children}</span>
);

const ItemSide = ({ children }: { children: ReactNode }) => (
  <span className={s.itemSide}>{children}</span>
);

const Footer = ({ children }: { children: ReactNode }) => (
  <div className={s.footer}>{children}</div>
);

export const Sidebar = Object.assign(Root, {
  Header,
  HeaderTop,
  Title,
  List,
  Item,
  ItemDot,
  ItemBody,
  ItemName,
  ItemMeta,
  ItemSide,
  Footer,
});
