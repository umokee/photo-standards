import QueryState from "@/components/ui/query-state/query-state";
import { useLoaderData } from "react-router-dom";
import { useTrainingModelOutletContext } from "./_training-detail";
import p from "../platform-pages.module.scss";

export function Component() {
  const { modelId } = useLoaderData() as { modelId: string | null };
  const { models } = useTrainingModelOutletContext();
  const model = models.find((item) => item.id === modelId) ?? null;

  if (!modelId) return null;

  return (
    <QueryState isEmpty={!model} size="block" emptyTitle="Model not found">
      {model ? (
        <section className={p.panelCard}>
          <h3>{model.architecture} {model.version ? `v${model.version}` : ""}</h3>
          <p>{model.num_classes ?? "—"} classes · image size {model.imgsz} · {model.is_active ? "active" : "ready"}</p>
        </section>
      ) : null}
    </QueryState>
  );
}
