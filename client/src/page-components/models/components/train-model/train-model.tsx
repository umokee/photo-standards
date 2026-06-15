import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import {
  useAppConstants,
  useArchitectureOptions,
  useImageSizeOptions,
  useTrainingLimits,
} from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Architecture } from "@/types/contracts";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { buildTrainModelPayload, useTrainModel } from "../../api/train-model";
import shell from "../model-transfer-modal.module.scss";
import s from "./train-model.module.scss";

interface Props {
  groupId: string;
  canTrain: boolean;
  isTrainingLocked: boolean;
}

export const TrainModel = ({ groupId, canTrain, isTrainingLocked }: Props) => (
  <Modal>
    <Modal.Trigger>
      <Button>Запустить обучение</Button>
    </Modal.Trigger>
    <Modal.Content wide>
      <TrainModelModal groupId={groupId} canTrain={canTrain} isTrainingLocked={isTrainingLocked} />
    </Modal.Content>
  </Modal>
);

const TrainModelModal = ({ groupId, canTrain, isTrainingLocked }: Props) => {
  const navigate = useNavigate();
  const close = useModalClose();

  const constants = useAppConstants();
  const architectureOptions = useArchitectureOptions();
  const imageSizeOptions = useImageSizeOptions();
  const trainingLimits = useTrainingLimits();
  const safeRatioSumMax = Math.min(trainingLimits.ratio_sum_max, 100);

  const [architecture, setArchitecture] = useState<Architecture>(
    () => constants.training.architectures.default as Architecture
  );
  const [epochs, setEpochs] = useState(() => String(constants.training.epochs.default));
  const [batchSize, setBatchSize] = useState(() => String(constants.training.batch_size.default));
  const [imageSize, setImageSize] = useState(() => String(constants.training.image_size.default));
  const [trainRatio, setTrainRatio] = useState(() =>
    String(constants.training.train_ratio.default)
  );
  const [valRatio, setValRatio] = useState(() => String(constants.training.val_ratio.default));
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const mutation = useTrainModel({
    groupId,
    mutationConfig: {
      onSuccess: (data) => {
        navigate(paths.trainingModel(groupId, data.model_id));
      },
    },
  });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  const trainPercent = Number(trainRatio || 0);
  const valPercent = Number(valRatio || 0);
  const testPercent = Math.max(0, 100 - trainPercent - valPercent);

  const isRatioInvalid = trainPercent + valPercent > safeRatioSumMax;
  const ratioError = isRatioInvalid
    ? `Сумма долей train и val должна быть не больше ${safeRatioSumMax}%.`
    : undefined;
  const isSubmitDisabled = mutation.isPending || isTrainingLocked || !canTrain || isRatioInvalid;
  const availabilityMessage = isTrainingLocked
    ? "Для этой группы уже запущено обучение. Дождитесь завершения текущей задачи или отмените её"
    : !canTrain
      ? "Для обучения нужны эталоны, фото, классы сегментов и размеченные аннотации"
      : null;

  const handleSubmit = () => {
    if (isSubmitDisabled) return;

    const payload = buildTrainModelPayload({
      group_id: groupId,
      architecture,
      epochs,
      batch_size: batchSize,
      imgsz: imageSize,
      train_ratio: trainRatio,
      val_ratio: valRatio,
    });
    if (payload.ok === false) {
      setFormErrors(payload.errors);
      return;
    }

    setFormErrors({});
    mutation.mutate(payload.data);
  };

  const handleTrainRatioChange = (value: string) => {
    setFormErrors((current) => clearFormErrors(current, ["train_ratio", "val_ratio", "form"]));
    setTrainRatio(
      limitSplitValue({
        value,
        min: trainingLimits.train_ratio.min,
        max: trainingLimits.train_ratio.max,
        otherValue: valRatio,
        ratioSumMax: safeRatioSumMax,
      })
    );
  };

  const handleValRatioChange = (value: string) => {
    setFormErrors((current) => clearFormErrors(current, ["train_ratio", "val_ratio", "form"]));
    setValRatio(
      limitSplitValue({
        value,
        min: trainingLimits.val_ratio.min,
        max: trainingLimits.val_ratio.max,
        otherValue: trainRatio,
        ratioSumMax: safeRatioSumMax,
      })
    );
  };

  return (
    <>
      <Modal.Header>Запустить обучение</Modal.Header>

      <Modal.Body>
        <div className={shell.root}>
          {availabilityMessage ? (
            <div className={shell.messageBox}>{availabilityMessage}</div>
          ) : null}

          <SurfaceSection
            title="Параметры модели"
            hint="Будут использованы для новой обученной версии модели"
          >
            <div className={shell.compactGrid}>
              <Select
                label="Архитектура"
                options={architectureOptions}
                value={architecture}
                error={formErrors.architecture ?? getFieldError(mutation.error, "architecture")}
                onChange={(value) => {
                  setArchitecture(value as Architecture);
                  setFormErrors((current) => clearFormErrors(current, ["architecture", "form"]));
                }}
              />

              <Select
                label="Размер изображения"
                options={imageSizeOptions}
                value={imageSize}
                error={formErrors.imgsz ?? getFieldError(mutation.error, "imgsz")}
                onChange={(value) => {
                  setImageSize(value);
                  setFormErrors((current) => clearFormErrors(current, ["imgsz", "form"]));
                }}
              />
            </div>
          </SurfaceSection>

          <SurfaceSection
            title="Режим обучения"
            hint="Больше эпох и batch size повышают нагрузку на GPU"
          >
            <div className={shell.compactGrid}>
              <Input
                label="Эпохи"
                type="number"
                min={trainingLimits.epochs.min}
                max={trainingLimits.epochs.max}
                step={1}
                value={epochs}
                error={formErrors.epochs ?? getFieldError(mutation.error, "epochs")}
                onChange={(value) => {
                  setEpochs(value);
                  setFormErrors((current) => clearFormErrors(current, ["epochs", "form"]));
                }}
              />

              <Input
                label="Batch size"
                type="number"
                min={trainingLimits.batch_size.min}
                max={trainingLimits.batch_size.max}
                step={1}
                value={batchSize}
                error={formErrors.batch_size ?? getFieldError(mutation.error, "batch_size")}
                onChange={(value) => {
                  setBatchSize(value);
                  setFormErrors((current) => clearFormErrors(current, ["batch_size", "form"]));
                }}
              />
            </div>
          </SurfaceSection>

          <SurfaceSection
            title="Разделение датасета"
            hint="Оставшиеся изображения автоматически попадут в test"
          >
            <div className={shell.compactGrid}>
              <Input
                label="Train %"
                type="number"
                min={trainingLimits.train_ratio.min}
                max={trainingLimits.train_ratio.max}
                step={1}
                value={trainRatio}
                error={formErrors.train_ratio ?? getFieldError(mutation.error, "train_ratio")}
                onChange={handleTrainRatioChange}
              />

              <Input
                label="Val %"
                type="number"
                min={trainingLimits.val_ratio.min}
                max={trainingLimits.val_ratio.max}
                step={1}
                value={valRatio}
                error={formErrors.val_ratio ?? ratioError ?? getFieldError(mutation.error, "val_ratio")}
                onChange={handleValRatioChange}
              />
            </div>

            <div className={clsx(s.splitSummary, isRatioInvalid && s.splitSummaryWarning)}>
              <span className={s.splitSummaryTitle}>Итоговое разбиение</span>

              <div className={s.splitStats}>
                <span>
                  Train <strong>{trainPercent}%</strong>
                </span>
                <span>
                  Val <strong>{valPercent}%</strong>
                </span>
                <span>
                  Test <strong>{testPercent}%</strong>
                </span>
              </div>

              <span className={s.splitHint}>
                {isRatioInvalid && ratioError}
              </span>
            </div>
          </SurfaceSection>

          {formErrors.form ? <div className={shell.errorBox}>{formErrors.form}</div> : null}

          {mutation.isError ? (
            <div className={shell.errorBox}>
              {mutation.error?.message ?? "Не удалось запустить обучение модели"}
            </div>
          ) : null}
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>

        <Button disabled={isSubmitDisabled} onClick={handleSubmit}>
          {mutation.isPending ? "Запускаем..." : "Запустить"}
        </Button>
      </Modal.Footer>
    </>
  );
};

const limitSplitValue = ({
  value,
  min,
  max,
  otherValue,
  ratioSumMax,
}: {
  value: string;
  min: number;
  max: number;
  otherValue: string;
  ratioSumMax: number;
}) => {
  if (value === "") return "";

  const next = Number(value);
  if (Number.isNaN(next)) return "";

  const other = Number(otherValue || 0);
  const safeMax = Math.min(max, 100, ratioSumMax - other);

  return String(Math.max(min, Math.min(next, safeMax)));
};

const clearFormErrors = (errors: Record<string, string>, keys: string[]) => {
  const next = { ...errors };

  for (const key of keys) {
    delete next[key];
  }

  return next;
};
