import { paths } from "@/app/paths";
import { Section } from "@/components/layouts/section/section";
import QueryState from "@/components/ui/query-state/query-state";
import { StandardCard } from "@/page-components/standards/components/standard-card/standard-card";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";

export function Component() {
  const navigate = useNavigate();
  const { standardId } = useLoaderData() as { standardId: string | null };
  const { group } = useGroupDetailOutletContext();

  const toggleStandard = (targetStandardId: string) => {
    if (standardId === targetStandardId) {
      navigate(paths.groupDetail(group.id));
      return;
    }

    navigate(paths.standardDetail(group.id, targetStandardId));
  };

  const handleToImageEditor = (imageId: string) => {
    if (!standardId) {
      return;
    }

    navigate(paths.standardImage(group.id, standardId, imageId));
  };

  return (
    <QueryState
      isEmpty={!group.standards.length}
      emptyTitle="Нет эталонов"
      emptyDescription="Создайте эталоны для этой группы"
    >
      <Section title={`Эталоны · ${group.stats.standards_count}`}>
        {group.standards.map((standard) => {
          const isExpanded = standardId === standard.id;

          return (
            <StandardCard
              key={standard.id}
              standard={standard}
              expanded={isExpanded}
              onToggle={() => toggleStandard(standard.id)}
              onToImageEditor={(imageId) => handleToImageEditor(imageId)}
            />
          );
        })}
      </Section>
    </QueryState>
  );
}
