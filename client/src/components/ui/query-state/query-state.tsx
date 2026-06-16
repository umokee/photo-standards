import clsx from "clsx";
import { AlertTriangle, Box, Loader2, SearchX } from "lucide-react";
import type { ReactNode } from "react";
import s from "./query-state.module.scss";

type Size = "inline" | "block" | "page";

interface Props {
  isLoading?: boolean;
  isError?: boolean;
  isEmpty?: boolean;
  size?: Size;
  loadingText?: string;
  errorTitle?: string;
  errorDescription?: string;
  emptyTitle?: string;
  emptyDescription?: string;
  action?: ReactNode;
  children?: ReactNode;
}

const StateContainer = ({ size, tone, children }: { size: Size; tone?: "loading" | "empty" | "error"; children: ReactNode }) => {
  return <div className={clsx(s.state, s[`state--${size}`], tone && s[`state--${tone}`])}>{children}</div>;
};

const LoadingState = ({ size, text }: { size: Size; text: string }) => {
  return (
    <StateContainer size={size} tone="loading">
      <div className={clsx(s.icon, s.iconLoading)}>
        <Loader2 />
      </div>
      <span className={s.title}>{text}</span>
      {size !== "inline" ? (
        <div className={s.skeletonGrid} aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      ) : null}
    </StateContainer>
  );
};

const ErrorState = ({
  size,
  title,
  description,
  action,
}: {
  size: Size;
  title: string;
  description?: string;
  action?: ReactNode;
}) => {
  return (
    <StateContainer size={size} tone="error">
      <div className={clsx(s.icon, s.iconError)}>
        <AlertTriangle />
      </div>
      <span className={clsx(s.title, s.titleError)}>{title}</span>
      {description && <span className={s.sub}>{description}</span>}
      {action && <div className={s.action}>{action}</div>}
    </StateContainer>
  );
};

const EmptyState = ({
  size,
  title,
  description,
  action,
}: {
  size: Size;
  title: string;
  description?: string;
  action?: ReactNode;
}) => {
  return (
    <StateContainer size={size} tone="empty">
      <div className={s.icon}>
        {description ? <SearchX /> : <Box />}
      </div>
      <span className={s.title}>{title}</span>
      {description && <span className={s.sub}>{description}</span>}
      {action && <div className={s.action}>{action}</div>}
    </StateContainer>
  );
};

export default function QueryState({
  isLoading,
  isError,
  isEmpty,
  size = "block",
  loadingText = "Загрузка...",
  errorTitle = "Не удалось загрузить данные",
  errorDescription = "Попробуйте обновить страницу или повторить действие позже",
  emptyTitle = "Нет данных",
  emptyDescription,
  action,
  children,
}: Props) {
  if (isLoading) {
    return <LoadingState size={size} text={loadingText} />;
  }

  if (isError) {
    return <ErrorState size={size} title={errorTitle} description={errorDescription} action={action} />;
  }

  if (isEmpty) {
    return <EmptyState size={size} title={emptyTitle} description={emptyDescription} action={action} />;
  }

  return children ?? null;
}
