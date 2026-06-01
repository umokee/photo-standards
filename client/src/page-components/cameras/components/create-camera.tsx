import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import { useEffect, useState } from "react";
import { buildCreateCameraPayload, useCreateCamera } from "../api/create-camera";
import {
  applyCameraFormChange,
  cameraFormFieldKeys,
  initialCameraFormValues,
  type CameraFormValues,
} from "../lib/camera-form";
import { CameraFormFields } from "./camera-form-fields/camera-form-fields";

export const CreateCamera = () => (
  <Modal>
    <Modal.Trigger>
      <Button full variant="ghost" size="sm">
        Новая камера
      </Button>
    </Modal.Trigger>
    <Modal.Content wide>
      <CreateCameraModal />
    </Modal.Content>
  </Modal>
);

const CreateCameraModal = () => {
  const close = useModalClose();
  const mutation = useCreateCamera();
  const [values, setValues] = useState<CameraFormValues>(initialCameraFormValues);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [close, mutation.isSuccess]);

  const handleChange = <K extends keyof CameraFormValues>(key: K, value: CameraFormValues[K]) => {
    setFormErrors({});
    setValues((prev) => applyCameraFormChange(prev, key, value));
  };

  const handleSubmit = () => {
    const result = buildCreateCameraPayload(values);

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    mutation.mutate(result.data);
  };

  const errors = Object.fromEntries(
    cameraFormFieldKeys.map((field) => [
      field,
      formErrors[field] ?? getFieldError(mutation.error, field),
    ])
  ) as Partial<Record<keyof CameraFormValues, string | undefined>>;

  return (
    <>
      <Modal.Header>Добавить камеру</Modal.Header>
      <Modal.Body>
        <CameraFormFields values={values} onChange={handleChange} errors={errors} />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Создаём..." : "Создать"}
        </Button>
      </Modal.Footer>
    </>
  );
};
