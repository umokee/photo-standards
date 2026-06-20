import p from "./standard-modal.module.scss";
import Button from "@/components/ui/button/button";
import ImageInput from "@/components/ui/image-input/image-input";
import { getFieldError } from "@/lib/errors";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { CheckCircle2, FileImage, Upload, Wand2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { buildUploadImagesPayload, useUploadImages } from "../api/upload-images";

type UploadImagesProps = {
  groupId: string;
  standardId: string;
  triggerClassName?: string;
};

export const UploadImages = ({ groupId, standardId, triggerClassName }: UploadImagesProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} icon={Upload} variant="ghost" size="sm">
        Загрузить
      </Button>
    </Modal.Trigger>
    <Modal.Content wide>
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
  }, [mutation.isSuccess, close]);

  const totalSize = useMemo(() => {
    if (!images?.length) return "0 MB";
    const mb = images.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024;
    return `${mb.toFixed(mb >= 10 ? 0 : 1)} MB`;
  }, [images]);

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
      <Modal.Header>Загрузить фото</Modal.Header>
      <Modal.Body>
        <ImageInput
          multiple
          error={formErrors.images ?? getFieldError(mutation.error, "images")}
          value={images}
          onChange={(value) => handleChange(value ? (value as File[]) : null)}
        />

        <div className={p.uploadSummaryGrid}>
          <UploadSummary icon={FileImage} label="Выбрано" value={`${images?.length ?? 0} фото`} />
          <UploadSummary icon={Upload} label="Размер" value={totalSize} />
          <UploadSummary icon={Wand2} label="Дальше" value="Разметка" />
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>Отмена</Button>
        <Button icon={CheckCircle2} disabled={mutation.isPending || !images?.length} onClick={handleSubmit}>
          {mutation.isPending ? "Загрузка..." : "Загрузить фото"}
        </Button>
      </Modal.Footer>
    </>
  );
};

function UploadSummary({ icon: Icon, label, value }: { icon: typeof FileImage; label: string; value: string }) {
  return (
    <div className={p.uploadSummaryCard}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
