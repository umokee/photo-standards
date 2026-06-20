import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import type { GroupDetail } from "@/types/contracts";
import { useNavigate } from "react-router-dom";
import { useDeleteGroup } from "../api/delete-group";

type GroupActionTarget = Pick<GroupDetail, "id" | "name" | "description">;

type DeleteGroupProps = {
  group: GroupActionTarget;
  triggerClassName?: string;
};

export const DeleteGroup = ({ group, triggerClassName }: DeleteGroupProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="danger" size="sm" aria-label={`Удалить изделие ${group.name}`}>Удалить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <DeleteGroupModal group={group} />
    </Modal.Content>
  </Modal>
);

const DeleteGroupModal = ({ group }: { group: GroupActionTarget }) => {
  const close = useModalClose();
  const navigate = useNavigate();
  const mutation = useDeleteGroup({
    mutationConfig: {
      onSuccess: () => {
        close();
        navigate(paths.groups());
      },
    },
  });

  return (
    <>
      <Modal.Header>Удалить изделие</Modal.Header>
      <Modal.Body>
        <p>
          Изделие «{group.name}» будет удалено вместе с его эталонами, фото, разметкой и историей,
          если это разрешено сервером.
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>Отмена</Button>
        <Button variant="danger" disabled={mutation.isPending} onClick={() => mutation.mutate(group.id)}>
          {mutation.isPending ? "Удаление..." : "Удалить"}
        </Button>
      </Modal.Footer>
    </>
  );
};
