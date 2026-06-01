import { parseError } from "@/lib/errors";
import { Component, PropsWithChildren, Suspense } from "react";
import QueryState from "../query-state/query-state";

type Size = "inline" | "block" | "page";

interface ErrorBoundaryProps extends PropsWithChildren {
  size?: Size;
  errorTitle?: string;
  errorDescription?: string;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class QueryErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  reset = () => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.error) {
      const parsed = parseError(this.state.error);

      return (
        <QueryState
          isError
          size={this.props.size ?? "block"}
          errorTitle={this.props.errorTitle ?? parsed.title}
          errorDescription={this.props.errorDescription ?? parsed.description}
        />
      );
    }
    return this.props.children;
  }
}

interface QueryBoundaryProps extends PropsWithChildren {
  size?: Size;
  loadingText?: string;
  errorTitle?: string;
  errorDescription?: string;
  onReset?: () => void;
}

export const QueryBoundary = ({
  children,
  size = "block",
  loadingText = "Загрузка...",
  errorTitle,
  errorDescription,
  onReset,
}: QueryBoundaryProps) => {
  return (
    <QueryErrorBoundary
      size={size}
      errorTitle={errorTitle}
      errorDescription={errorDescription}
      onReset={onReset}
    >
      <Suspense fallback={<QueryState isLoading size={size} loadingText={loadingText} />}>
        {children}
      </Suspense>
    </QueryErrorBoundary>
  );
};
