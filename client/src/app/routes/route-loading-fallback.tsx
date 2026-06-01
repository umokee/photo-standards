import QueryState from "@/components/ui/query-state/query-state";

export default function RouteLoadingFallback() {
  return <QueryState isLoading size="page" loadingText="Загрузка..." />;
}
