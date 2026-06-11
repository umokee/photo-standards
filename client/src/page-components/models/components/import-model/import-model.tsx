import { paths } from "@/app/paths";
import { Badge } from "@/components/ui/badge/badge";
import Button from "@/components/ui/button/button";
import { ColorPicker } from "@/components/ui/color-picker/color-picker";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import ToggleCard from "@/components/ui/toggle-card/toggle-card";
import { useAppConstants, useArchitectureOptions, useImageSizeOptions } from "@/constants";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import type { Architecture, ImportedClassMappingDraft } from "@/types/contracts";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { buildImportModelPayload, useImportModel } from "../../api/import-model";
import {
  buildPreviewImportModelPayload,
  usePreviewImportModel,
} from "../../api/preview-import-model";
import {
  buildImportModelCategoryOptions,
  buildImportModelExistingClassOptions,
  buildImportModelPreviewRows,
  isImportModelMappingRowValid,
  type ImportModelMappingMode,
  type ImportModelMappingRow,
  type ImportModelSelectOption,
} from "../../lib/import-model-helpers";
import shell from "../model-transfer-modal.module.scss";
import s from "./import-model.module.scss";

interface Props {
  groupId: string;
}

export const ImportModel = ({ groupId }: Props) => (
  <Modal>
    <Modal.Trigger>
      <Button variant="ghost">Импорт</Button>
    </Modal.Trigger>
    <Modal.Content wide>
      <ImportModelModal groupId={groupId} />
    </Modal.Content>
  </Modal>
);

const ImportModelModal = ({ groupId }: Props) => {
  const navigate = useNavigate();
  const close = useModalClose();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previewRequestIdRef = useRef(0);

  const constants = useAppConstants();
  const architectureOptions = useArchitectureOptions();
  const imageSizeOptions = useImageSizeOptions();
  const { data: group } = useGetGroup(groupId);

  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [architecture, setArchitecture] = useState<Architecture>(
    () => constants.training.architectures.default as Architecture
  );
  const [imageSize, setImageSize] = useState(() => String(constants.training.image_size.default));
  const [activate, setActivate] = useState(true);
  const [rows, setRows] = useState<ImportModelMappingRow[]>([]);

  const previewMutation = usePreviewImportModel();
  const importMutation = useImportModel({
    groupId,
    mutationConfig: {
      onSuccess: (data) => {
        navigate(paths.trainingModel(groupId, data.model_id));
      },
    },
  });

  useEffect(() => {
    if (importMutation.isSuccess) {
      close();
    }
  }, [importMutation.isSuccess, close]);

  const existingClassOptions = useMemo(() => buildImportModelExistingClassOptions(group), [group]);
  const categoryOptions = useMemo(() => buildImportModelCategoryOptions(group), [group]);

  const mappedCount = useMemo(
    () => rows.filter((row) => row.mode === "existing" || row.mode === "new").length,
    [rows]
  );

  const canSubmit =
    !!weightsFile &&
    rows.length > 0 &&
    mappedCount > 0 &&
    rows.every(isImportModelMappingRowValid) &&
    !importMutation.isPending;

  const fileStatus = !weightsFile
    ? "Поддерживается импорт весов YOLO в формате .pt"
    : previewMutation.isPending
      ? "Читаем классы и готовим сопоставления..."
      : previewMutation.isSuccess
        ? "Классы прочитаны. Проверьте сопоставления ниже"
        : previewMutation.isError &&
          "Не удалось прочитать классы. Выберите другой .pt или попробуйте снова";

  const openFilePicker = () => {
    if (!inputRef.current) return;
    inputRef.current.value = "";
    inputRef.current.click();
  };

  const handleFileChange = (file: File | null) => {
    if (!file) return;

    setRows([]);
    previewMutation.reset();
    importMutation.reset();
    setWeightsFile(file);
    runPreview(file);
  };

  const runPreview = (file: File) => {
    const previewPayload = buildPreviewImportModelPayload({
      group_id: groupId,
      weights: file,
    });
    if (!previewPayload.ok) {
      return;
    }

    const requestId = previewRequestIdRef.current + 1;
    previewRequestIdRef.current = requestId;

    previewMutation.mutate(previewPayload.data, {
      onSuccess: (data) => {
        if (previewRequestIdRef.current !== requestId) {
          return;
        }

        setRows(
          buildImportModelPreviewRows(data.native_classes, group, constants.segments.hue.default)
        );
      },
    });
  };

  const handleRowModeChange = (nativeKey: string, mode: ImportModelMappingMode) => {
    setRows((current) =>
      current.map((row) =>
        row.nativeKey === nativeKey
          ? {
              ...row,
              mode,
              segmentClassId: mode === "existing" ? row.segmentClassId : "",
            }
          : row
      )
    );
  };

  const handleRowChange = (
    nativeKey: string,
    patch: Partial<Omit<ImportModelMappingRow, "nativeKey" | "nativeIndex">>
  ) => {
    setRows((current) =>
      current.map((row) => (row.nativeKey === nativeKey ? { ...row, ...patch } : row))
    );
  };

  const handleSubmit = () => {
    if (!weightsFile || !canSubmit) return;

    const mappings: ImportedClassMappingDraft[] = rows
      .filter((row) => row.mode !== "ignore")
      .map((row) => {
        if (row.mode === "existing") {
          return {
            mode: "existing",
            native_key: row.nativeKey,
            segment_class_id: row.segmentClassId,
          };
        }

        return {
          mode: "new",
          native_key: row.nativeKey,
          new_class_name: row.newClassName.trim(),
          new_class_hue: clampHue(
            row.newClassHue,
            constants.segments.hue.min,
            constants.segments.hue.max,
            constants.segments.hue.default
          ),
          new_class_group_id: row.newClassGroupId || null,
        };
      });

    const payload = buildImportModelPayload({
      group_id: groupId,
      architecture,
      imgsz: imageSize,
      activate,
      mappings,
      weights: weightsFile,
    });
    if (!payload.ok) {
      return;
    }

    importMutation.mutate(payload.data);
  };

  return (
    <>
      <Modal.Header>Импорт модели</Modal.Header>

      <Modal.Body>
        <div className={shell.root}>
          <input
            ref={inputRef}
            hidden
            type="file"
            accept=".pt"
            onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
          />

          <SurfaceSection title="Источник" hint="Выберите веса для импорта">
            <div className={s.fileRow}>
              <div className={s.fileMeta}>
                <span className={s.fileName}>{weightsFile?.name ?? "Файл не выбран"}</span>
                <span className={s.fileStatus}>{fileStatus}</span>
              </div>

              <Button variant="ghost" onClick={openFilePicker}>
                {weightsFile ? "Заменить файл" : "Выбрать .pt"}
              </Button>
            </div>
          </SurfaceSection>

          <SurfaceSection
            title="Параметры импорта"
            hint="Эти настройки сохранятся у импортируемой модели"
          >
            <div className={shell.compactGrid}>
              <Select
                label="Архитектура"
                options={architectureOptions}
                value={architecture}
                onChange={(value) => setArchitecture(value as Architecture)}
              />

              <Select
                label="Размер изображения"
                options={imageSizeOptions}
                value={imageSize}
                onChange={setImageSize}
              />
            </div>

            <ToggleCard
              title="Сделать модель активной после импорта"
              checked={activate}
              onChange={setActivate}
            />
          </SurfaceSection>

          <SurfaceSection
            title="Сопоставление классов"
            hint="Для каждого класса можно выбрать существующий класс группы, создать новый или пропустить импорт"
            aside={
              previewMutation.data ? (
                <Badge>{previewMutation.data.native_classes.length} найдено</Badge>
              ) : null
            }
          >
            {previewMutation.data && (
              <div className={s.mappings}>
                {rows.map((row) => (
                  <ImportMappingRow
                    key={row.nativeKey}
                    row={row}
                    existingClassOptions={existingClassOptions}
                    categoryOptions={categoryOptions}
                    minHue={constants.segments.hue.min}
                    maxHue={constants.segments.hue.max}
                    onModeChange={handleRowModeChange}
                    onChange={handleRowChange}
                  />
                ))}
              </div>
            )}
          </SurfaceSection>
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>

        <Button disabled={!canSubmit} onClick={handleSubmit}>
          {importMutation.isPending ? "Ставим в очередь..." : "Импортировать"}
        </Button>
      </Modal.Footer>
    </>
  );
};

type ImportMappingRowProps = {
  row: ImportModelMappingRow;
  existingClassOptions: ImportModelSelectOption[];
  categoryOptions: ImportModelSelectOption[];
  minHue: number;
  maxHue: number;
  onModeChange: (nativeKey: string, mode: ImportModelMappingMode) => void;
  onChange: (
    nativeKey: string,
    patch: Partial<Omit<ImportModelMappingRow, "nativeKey" | "nativeIndex">>
  ) => void;
};

const ImportMappingRow = ({
  row,
  existingClassOptions,
  categoryOptions,
  minHue,
  maxHue,
  onModeChange,
  onChange,
}: ImportMappingRowProps) => {
  const currentHue = clampHue(row.newClassHue, minHue, maxHue, minHue);
  const [isColorOpen, setIsColorOpen] = useState(false);

  return (
    <div className={s.mappingRow}>
      <div className={s.mappingTop}>
        <div className={s.mappingInfo}>
          <span className={s.nativeName}>{row.nativeKey}</span>
          <div className={clsx(s.nativeMeta, row.suggestedMatchReason && s.nativeMetaAuto)}>
            {row.suggestedMatchReason && row.suggestedSegmentClassName ? (
              <>
                <span className={s.autoMatchLabel}>
                  Авто по {row.suggestedMatchReason === "uuid" ? "UUID" : "имени"}:
                </span>{" "}
                <span className={s.autoMatchValue}>{row.suggestedSegmentClassName}</span>
              </>
            ) : (
              <>Индекс в модели: {row.nativeIndex}</>
            )}
          </div>
        </div>
      </div>

      <div className={s.mappingControls}>
        <div className={s.modeTabs}>
          <button
            type="button"
            className={clsx(s.modeBtn, row.mode === "ignore" && s.modeBtnActive)}
            onClick={() => onModeChange(row.nativeKey, "ignore")}
          >
            Игнорировать
          </button>

          <button
            type="button"
            className={clsx(s.modeBtn, row.mode === "existing" && s.modeBtnActive)}
            onClick={() => onModeChange(row.nativeKey, "existing")}
          >
            Сопоставить
          </button>

          <button
            type="button"
            className={clsx(s.modeBtn, row.mode === "new" && s.modeBtnActive)}
            onClick={() => onModeChange(row.nativeKey, "new")}
          >
            Новый класс
          </button>
        </div>

        {row.mode === "ignore" ? (
          <span className={s.inlineMeta}>Класс не будет участвовать в импорте</span>
        ) : null}

        {row.mode === "existing" ? (
          <div className={s.existingRow}>
            <Select
              noMargin
              placeholder="Выберите существующий класс"
              options={existingClassOptions}
              value={row.segmentClassId || null}
              onChange={(value) =>
                onChange(row.nativeKey, {
                  segmentClassId: value,
                })
              }
            />
          </div>
        ) : null}

        {row.mode === "new" ? (
          <>
            <div className={s.newGrid}>
              <div className={s.colorField}>
                <button
                  type="button"
                  className={clsx(s.colorTrigger, isColorOpen && s.colorOpen)}
                  aria-label="Изменить цвет нового класса"
                  onClick={() => setIsColorOpen((prev) => !prev)}
                >
                  <span
                    className={s.colorSwatch}
                    style={{ background: `hsl(${currentHue}, 65%, 55%)` }}
                  />
                </button>
              </div>

              <Input
                noMargin
                value={row.newClassName}
                placeholder="Название нового класса"
                onChange={(value) =>
                  onChange(row.nativeKey, {
                    newClassName: value,
                  })
                }
              />

              <Select
                noMargin
                options={categoryOptions}
                value={row.newClassGroupId || null}
                onChange={(value) =>
                  onChange(row.nativeKey, {
                    newClassGroupId: value,
                  })
                }
              />
            </div>

            {isColorOpen ? (
              <div className={s.colorPanel}>
                <ColorPicker
                  hue={currentHue}
                  onChange={(value) =>
                    onChange(row.nativeKey, {
                      newClassHue: String(value),
                    })
                  }
                />
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
};

function clampHue(value: string | number, min: number, max: number, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(parsed)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, Math.round(parsed)));
}
