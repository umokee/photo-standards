import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { Trash2 } from "lucide-react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDeleteCamera } from "../api/delete-camera";

interface Props {
  id: string;
  name: string;
}

export const DeleteCamera = ({ id, name }: Props) => (
  <Modal>
    <Modal.Trigger>
      <Button variant="danger">Удалить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <DeleteCameraModal id={id} name={name} />
    </Modal.Content>
  </Modal>
);

const DeleteCameraModal = ({ id, name }: Props) => {
  const close = useModalClose();
  const navigate = useNavigate();
  const mutation = useDeleteCamera({
    mutationConfig: {
      onSuccess: () => navigate(paths.cameras()),
    },
  });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess, close]);

  return (
    <>
      <Modal.Header>Удалить камеру</Modal.Header>
      <Modal.Body>{`Вы уверены, что хотите удалить камеру «${name}»?`}</Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button variant="danger" disabled={mutation.isPending} onClick={() => mutation.mutate(id)}>
          {mutation.isPending ? "Удаляем..." : "Удалить"}
        </Button>
      </Modal.Footer>
    </>
  );
};
