import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import type { Camera } from "@/types/contracts";
import { useEffect, useState } from "react";
import { buildUpdateCameraPayload, useUpdateCamera } from "../api/update-camera";
import {
  applyCameraFormChange,
  cameraFormFieldKeys,
  toCameraFormValues,
  type CameraFormValues,
} from "../lib/camera-form";
import { CameraFormFields } from "./camera-form-fields/camera-form-fields";

type UpdateCameraProps = {
  camera: Camera;
  triggerClassName?: string;
};

export const UpdateCamera = ({ camera, triggerClassName }: UpdateCameraProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="ghost">Изменить</Button>
    </Modal.Trigger>
    <Modal.Content wide>
      <UpdateCameraModal camera={camera} />
    </Modal.Content>
  </Modal>
);

const UpdateCameraModal = ({ camera }: { camera: Camera }) => {
  const close = useModalClose();
  const mutation = useUpdateCamera();
  const [values, setValues] = useState<CameraFormValues>(() => toCameraFormValues(camera));
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
    const result = buildUpdateCameraPayload(values, toCameraFormValues(camera));

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    if (!result.data) {
      return;
    }

    mutation.mutate({ id: camera.id, data: result.data });
  };

  const errors = Object.fromEntries(
    cameraFormFieldKeys.map((field) => [
      field,
      formErrors[field] ?? getFieldError(mutation.error, field),
    ])
  ) as Partial<Record<keyof CameraFormValues, string | undefined>>;

  return (
    <>
      <Modal.Header>{`Изменить камеру ${camera.name}`}</Modal.Header>
      <Modal.Body>
        <CameraFormFields values={values} onChange={handleChange} errors={errors} isEditing />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Сохраняем..." : "Сохранить"}
        </Button>
      </Modal.Footer>
    </>
  );
};
