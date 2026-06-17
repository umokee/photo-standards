import { ImagePlus, MousePointer2 } from "lucide-react";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useLoaderData } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  const hasReferences = group.standards.length > 0;

  return (
    <div className={p.inspectEmptyStateV17}>
      <ImagePlus />
      <h2>{hasReferences ? "Выбери эталон" : "Нет эталонных видов"}</h2>
      <p>{hasReferences ? "Эталон определяет, какие детали и полигоны будут перенесены на проверяемое изображение." : "Сначала создай reference view в Annotate и разметь обязательные детали."}</p>
      <span><MousePointer2 /> {hasReferences ? "Выбери reference в верхней панели." : "Вернись в Annotate → Dataset."}</span>
    </div>
  );
}
