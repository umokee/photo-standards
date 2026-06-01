import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { Star, Trash2 } from "lucide-react";
import type { ReactElement } from "react";
import { useEffect } from "react";
import { useDeleteImage } from "../api/delete-image";
import { useSetReference } from "../api/set-reference";

type SetReferenceImageProps = {
  groupId: string;
  standardId: string;
  imageId: string;
};

type DeleteStandardImageProps = SetReferenceImageProps & {
  isReference: boolean;
};

export const SetReferenceImage = ({ groupId, standardId, imageId }: SetReferenceImageProps) => (
  <ActionTrigger
    trigger={
      <button type="button" aria-label="Сделать фото эталонным">
        <Star size={12} />
      </button>
    }
  >
    <SetReferenceImageModal groupId={groupId} standardId={standardId} imageId={imageId} />
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

const SetReferenceImageModal = ({ groupId, standardId, imageId }: SetReferenceImageProps) => {
  const close = useModalClose();
  const mutation = useSetReference({ groupId, standardId });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  return (
    <StandardImageActionModal
      title="Назначить эталонное фото"
      description="Сделать это изображение эталонным для данного изделия?"
      confirmLabel="Назначить"
      pendingLabel="Назначаем..."
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
          ? "Это текущее эталонное фото. Вы уверены, что хотите удалить его?"
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
