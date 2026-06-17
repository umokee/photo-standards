
import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { Check, Trash2, X } from "lucide-react";
import type { ReactElement } from "react";
import { useEffect } from "react";
import { useDeleteImage } from "../api/delete-image";
import { useSetReference } from "../api/set-reference";

type SetReferenceImageProps = {
  groupId: string;
  standardId: string;
  imageId: string;
  isReference: boolean;
  buttonClassName?: string;
  showLabel?: boolean;
};

type DeleteStandardImageProps = SetReferenceImageProps;

export const SetReferenceImage = ({
  groupId,
  standardId,
  imageId,
  isReference,
  buttonClassName,
  showLabel = false,
}: SetReferenceImageProps) => {
  const label = isReference ? "Убрать из проверки" : "Использовать в проверке";

  return (
    <ActionTrigger
      trigger={
        <button
          type="button"
          className={buttonClassName}
          aria-label={label}
          title={label}
        >
          {isReference ? <X size={showLabel ? 14 : 12} /> : <Check size={showLabel ? 14 : 12} />}
          {showLabel ? <span>{label}</span> : null}
        </button>
      }
    >
      <SetReferenceImageModal
        groupId={groupId}
        standardId={standardId}
        imageId={imageId}
        isReference={isReference}
      />
    </ActionTrigger>
  );
};

export const DeleteStandardImage = ({
  groupId,
  standardId,
  imageId,
  isReference,
  buttonClassName,
  showLabel = false,
}: DeleteStandardImageProps) => (
  <ActionTrigger
    trigger={
      <button type="button" className={buttonClassName} aria-label="Удалить фото эталона" title="Удалить фото эталона">
        <Trash2 size={showLabel ? 14 : 12} />
        {showLabel ? <span>Удалить</span> : null}
      </button>
    }
  >
    <DeleteStandardImageModal
      groupId={groupId}
      standardId={standardId}
      imageId={imageId}
      isReference={isReference}
    />
  </ActionTrigger>
);

const SetReferenceImageModal = ({
  groupId,
  standardId,
  imageId,
  isReference,
}: SetReferenceImageProps) => {
  const close = useModalClose();
  const mutation = useSetReference({ groupId, standardId });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  return (
    <StandardImageActionModal
      title={isReference ? "Убрать фото из проверки" : "Использовать фото в проверке"}
      description={
        isReference
          ? "Фото будет снято с reference pool и больше не будет участвовать в автоматическом выборе ракурса при проверке. Сам файл и разметка останутся."
          : "Фото будет добавлено в reference pool. Если features ещё не готовы, сервер сначала посчитает их и затем включит фото в проверку."
      }
      confirmLabel={isReference ? "Убрать из проверки" : "Использовать в проверке"}
      pendingLabel={isReference ? "Убираем..." : "Подготавливаем..."}
      isPending={mutation.isPending}
      onConfirm={() => mutation.mutate(imageId)}
    />
  );
};

const DeleteStandardImageModal = ({
  groupId,
  standardId,
  imageId,
  isReference,
}: DeleteStandardImageProps) => {
  const close = useModalClose();
  const mutation = useDeleteImage({ groupId, standardId });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  return (
    <StandardImageActionModal
      title="Удалить фото эталона"
      description={
        isReference
          ? "Это фото участвует в проверке. Если удалить его, оно исчезнет из reference pool вместе с разметкой и features."
          : "Фото, разметка и рассчитанные features будут удалены."
      }
      confirmLabel="Удалить"
      pendingLabel="Удаляем..."
      confirmVariant="danger"
      isPending={mutation.isPending}
      onConfirm={() => mutation.mutate(imageId)}
    />
  );
};

const ActionTrigger = ({
  trigger,
  children,
}: {
  trigger: ReactElement;
  children: ReactElement;
}) => (
  <div onClick={(e) => e.stopPropagation()}>
    <Modal>
      <Modal.Trigger>{trigger}</Modal.Trigger>
      <Modal.Content>{children}</Modal.Content>
    </Modal>
  </div>
);

type StandardImageActionModalProps = {
  title: string;
  description: string;
  confirmLabel: string;
  pendingLabel: string;
  confirmVariant?: "primary" | "danger";
  isPending: boolean;
  onConfirm: () => void;
};

const StandardImageActionModal = ({
  title,
  description,
  confirmLabel,
  pendingLabel,
  confirmVariant = "primary",
  isPending,
  onConfirm,
}: StandardImageActionModalProps) => {
  const close = useModalClose();

  return (
    <>
      <Modal.Header>{title}</Modal.Header>
      <Modal.Body>{description}</Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button variant={confirmVariant} disabled={isPending} onClick={onConfirm}>
          {isPending ? pendingLabel : confirmLabel}
        </Button>
      </Modal.Footer>
    </>
  );
};
