import clsx from "clsx";
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
  action?: React.ReactNode;
  children?: React.ReactNode;
}

const StateContainer = ({ size, children }: { size: Size; children: React.ReactNode }) => {
  return <div className={clsx(s.state, s[`state--${size}`])}>{children}</div>;
};

const LoadingState = ({ size, text }: { size: Size; text: string }) => {
  return (
    <StateContainer size={size}>
      <div className={clsx(s.icon, s.iconLoading)}>
        <span className={s.ring} />
      </div>
      <span className={s.title}>{text}</span>
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
  action?: React.ReactNode;
}) => {
  return (
    <StateContainer size={size}>
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
  action?: React.ReactNode;
}) => {
  return (
    <StateContainer size={size}>
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
    return (
      <ErrorState size={size} title={errorTitle} description={errorDescription} action={action} />
    );
  }

  if (isEmpty) {
    return (
      <EmptyState size={size} title={emptyTitle} description={emptyDescription} action={action} />
    );
  }

  return children ?? null;
}
