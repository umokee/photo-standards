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
};

type DeleteStandardImageProps = SetReferenceImageProps;

export const SetReferenceImage = ({
  groupId,
  standardId,
  imageId,
  isReference,
}: SetReferenceImageProps) => (
  <ActionTrigger
    trigger={
      <button
        type="button"
        aria-label={
          isReference
            ? "Убрать фото из проверки"
            : "Использовать фото в проверке"
        }
        title={
          isReference
            ? "Убрать фото из проверки"
            : "Использовать фото в проверке"
        }
      >
        {isReference ? <X size={12} /> : <Check size={12} />}
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

export const DeleteStandardImage = ({
  groupId,
  standardId,
  imageId,
  isReference,
}: DeleteStandardImageProps) => (
  <ActionTrigger
    trigger={
      <button type="button" aria-label="Удалить фото эталона">
        <Trash2 size={12} />
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
          ? "Это фото больше не будет участвовать в автоматическом выборе ракурса при проверке."
          : "Это фото будет добавлено в пул ракурсов, из которых система выбирает лучший при проверке."
      }
      confirmLabel={isReference ? "Убрать" : "Использовать"}
      pendingLabel={isReference ? "Убираем..." : "Добавляем..."}
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
          ? "Это фото участвует в проверке. Вы уверены, что хотите удалить его?"
          : "Вы уверены, что хотите удалить это фото?"
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
