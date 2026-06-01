import Button from "@/components/ui/button/button";
import QueryState from "@/components/ui/query-state/query-state";
import { parseError } from "@/lib/errors";
import { isRouteErrorResponse, useNavigate, useRouteError } from "react-router-dom";

export default function RouteError() {
  const error = useRouteError();
  const navigate = useNavigate();

  const parsed = isRouteErrorResponse(error)
    ? {
        title: `Ошибка ${error.status}`,
        description:
          typeof error.data === "string" && error.data.trim()
            ? error.data
            : error.statusText || "Не удалось открыть страницу",
      }
    : parseError(error);

  return (
    <QueryState
      isError
      size="page"
      errorTitle={parsed.title}
      errorDescription={parsed.description}
      action={
        <div style={{ display: "flex", gap: 8 }}>
          <Button variant="ghost" size="sm" onClick={() => navigate(0)}>
            Обновить
          </Button>
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
            Назад
          </Button>
        </div>
      }
    >
      {null}
    </QueryState>
  );
}
