import { Badge } from "@/components/ui/badge/badge";
import Button from "@/components/ui/button/button";
import { getFieldError } from "@/lib/errors";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import type { MlModel } from "@/types/contracts";
import { architectureLabel } from "@/utils/labels";
import { useEffect, useMemo, useState } from "react";
import { buildExportModelPayload, useExportModel } from "../../api/export-model";
import {
  buildExportModelFileName,
  formatExportModelOptionLabel,
  isExportableModel,
  sortModelsForExport,
} from "../../lib/export-model-helpers";
import { getModelClassLabels, getModelVersionLabel } from "../../lib/model-helpers";
import shell from "../model-transfer-modal.module.scss";
import s from "./export-model.module.scss";

interface Props {
  models: MlModel[];
  triggerClassName?: string;
}

type ExportModelModalProps = {
  models: MlModel[];
};

export const ExportModel = ({ models, triggerClassName }: Props) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="ghost">Экспорт</Button>
    </Modal.Trigger>
    <Modal.Content>
      <ExportModelModal models={models} />
    </Modal.Content>
  </Modal>
);

const ExportModelModal = ({ models }: ExportModelModalProps) => {
  const close = useModalClose();
  const mutation = useExportModel();

  const exportableModels = useMemo(
    () => models.filter(isExportableModel).sort(sortModelsForExport),
    [models]
  );

  const [selectedModelId, setSelectedModelId] = useState(exportableModels[0]?.id ?? "");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (exportableModels.length === 0) {
      setSelectedModelId("");
      return;
    }

    if (!exportableModels.some((item) => item.id === selectedModelId)) {
      setSelectedModelId(exportableModels[0].id);
    }
  }, [exportableModels, selectedModelId]);

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  const selectedModel = exportableModels.find((item) => item.id === selectedModelId) ?? null;

  const modelOptions = exportableModels.map((model) => ({
    value: model.id,
    label: formatExportModelOptionLabel(model),
  }));

  const classLabels = selectedModel ? getModelClassLabels(selectedModel) : [];
  const previewClassLabels = classLabels.slice(0, 8);

  const handleSubmit = () => {
    const payload = buildExportModelPayload({
      modelId: selectedModelId,
      fileName: selectedModel ? buildExportModelFileName(selectedModel) : "",
    });

    if (!payload.ok) {
      setFormErrors(payload.errors);
      return;
    }

    setFormErrors({});
    mutation.mutate(payload.data);
  };

  return (
    <>
      <Modal.Header>Экспорт модели</Modal.Header>

      <Modal.Body>
        <div className={shell.root}>
          {exportableModels.length === 0 ? (
            <div className={shell.messageBox}>
              Пока нет готовых моделей для экспорта. Сначала завершите обучение или импорт
              модели, чтобы появились `.pt` и `.classes.json`.
            </div>
          ) : (
            <>
              <SurfaceSection title="Источник" hint="Выберите модель для экспорта">
                <Select
                  label="Модель"
                  options={modelOptions}
                  value={selectedModelId}
                  error={formErrors.modelId ?? getFieldError(mutation.error, "modelId")}
                  onChange={(value) => {
                    setSelectedModelId(value);
                    setFormErrors((current) => {
                      const next = { ...current };
                      delete next.modelId;
                      delete next.form;
                      return next;
                    });
                  }}
                />
              </SurfaceSection>

              {selectedModel ? (
                <SurfaceSection
                  title="Сводка"
                  hint="Архив содержит веса модели и файл с описанием классов"
                >
                  <div className={s.summaryCard}>
                    <div className={shell.summaryGrid}>
                      <div className={shell.summaryItem}>
                        <span className={shell.summaryLabel}>Архитектура</span>
                        <strong className={shell.summaryValue}>
                          {architectureLabel(selectedModel.architecture)}
                        </strong>
                      </div>

                      <div className={shell.summaryItem}>
                        <span className={shell.summaryLabel}>Версия</span>
                        <strong className={shell.summaryValue}>
                          {getModelVersionLabel(selectedModel)}
                        </strong>
                      </div>

                      <div className={shell.summaryItem}>
                        <span className={shell.summaryLabel}>Классов</span>
                        <strong className={shell.summaryValue}>
                          {selectedModel.num_classes ?? classLabels.length}
                        </strong>
                      </div>

                      <div className={shell.summaryItem}>
                        <span className={shell.summaryLabel}>Файл архива</span>
                        <strong className={shell.summaryValue}>
                          {buildExportModelFileName(selectedModel)}
                        </strong>
                      </div>
                    </div>

                    {!!previewClassLabels.length ? (
                      <div className={s.sectionBlock}>
                        <div className={s.sectionTitle}>Классы</div>
                        <div className={s.classList}>
                          {previewClassLabels.map((label) => (
                            <Badge key={label}>{label}</Badge>
                          ))}
                          {classLabels.length > previewClassLabels.length ? (
                            <Badge>+{classLabels.length - previewClassLabels.length}</Badge>
                          ) : null}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </SurfaceSection>
              ) : null}
            </>
          )}

          {mutation.isError && (
            <div className={shell.errorBox}>
              {mutation.error?.message ?? "Не удалось подготовить экспорт модели"}
            </div>
          )}
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Закрыть
        </Button>

        <Button disabled={!selectedModel || mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Подготавливаем архив..." : "Скачать архив"}
        </Button>
      </Modal.Footer>
    </>
  );
};
