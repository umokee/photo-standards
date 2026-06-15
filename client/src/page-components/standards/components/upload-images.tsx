import Button from "@/components/ui/button/button";
import ImageInput from "@/components/ui/image-input/image-input";
import { getFieldError } from "@/lib/errors";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { useEffect, useState } from "react";
import { buildUploadImagesPayload, useUploadImages } from "../api/upload-images";

export const UploadImages = ({ groupId, standardId }: { groupId: string; standardId: string }) => (
  <Modal>
    <Modal.Trigger>
      <Button variant="ghost" size="sm">
        Фото
      </Button>
    </Modal.Trigger>
    <Modal.Content>
      <UploadImagesModal groupId={groupId} standardId={standardId} />
    </Modal.Content>
  </Modal>
);

const UploadImagesModal = ({ groupId, standardId }: { groupId: string; standardId: string }) => {
  const close = useModalClose();
  const [images, setImages] = useState<File[] | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const mutation = useUploadImages({
    groupId,
    mutationConfig: {
      onSuccess: () => setImages(null),
    },
  });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildUploadImagesPayload({ standardId, images });

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    setFormErrors({});
    mutation.mutate(result.data);
  };

  const handleChange = (value: File[] | null) => {
    setFormErrors({});
    setImages(value);
  };

  return (
    <>
      <Modal.Header>Загрузить изображения</Modal.Header>
      <Modal.Body>
        <ImageInput
          multiple
          error={formErrors.images ?? getFieldError(mutation.error, "images")}
          value={images}
          onChange={(value) => handleChange(value ? (value as File[]) : null)}
        />
      </Modal.Body>
      <Modal.Footer>
        <Button disabled={mutation.isPending || !images?.length} onClick={handleSubmit}>
          Загрузить
        </Button>
      </Modal.Footer>
    </>
  );
};
