import { Section } from "@/components/layouts/section/section";
import { Badge } from "@/components/ui/badge/badge";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetInspection } from "@/page-components/inspections/api/get-inspection";
import { InspectionHistoryDetail } from "@/page-components/inspections/components/inspection-history-detail/inspection-history-detail";
import { InspectionHistoryList } from "@/page-components/inspections/components/inspection-history-list/inspection-history-list";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useInspectionHistoryOutletContext } from "./_inspection-history-group";

export function Component() {
  const { inspectionId } = useLoaderData() as { inspectionId: string | null };
  const navigate = useNavigate();

  const { history, selectedGroup, buildInspectionPath } = useInspectionHistoryOutletContext();

  const handleSelectInspection = (nextInspectionId: string) => {
    if (nextInspectionId === inspectionId) {
      navigate(buildInspectionPath(null));
      return;
    }

    navigate(buildInspectionPath(nextInspectionId));
  };

  return (
    <QueryState
      isEmpty={!history.length}
      size="page"
      emptyTitle="Проверок по группе не найдено"
      emptyDescription={`Для группы «${selectedGroup.name}» пока нет сохранённых результатов контроля.`}
    >
      <Section title="Проверки" side={<Badge>{history.length}</Badge>}>
        <InspectionHistoryList
          items={history}
          selectedInspectionId={inspectionId}
          onSelect={handleSelectInspection}
          renderDetail={(item) =>
            item.id === inspectionId && inspectionId ? (
              <ExpandedInspectionHistoryDetail inspectionId={inspectionId} />
            ) : null
          }
        />
      </Section>
    </QueryState>
  );
}

function ExpandedInspectionHistoryDetail({ inspectionId }: { inspectionId: string }) {
  const { data } = useGetInspection(inspectionId);

  return <InspectionHistoryDetail inspection={data} />;
}
