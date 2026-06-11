import { ContentHeader } from "@/components/layouts/content-header/content-header";
import { Section } from "@/components/layouts/section/section";
import { InfoRow } from "@/components/ui/info-row/info-row";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import { useGetCamera } from "@/page-components/cameras/api/get-camera";
import { CameraPreviewPanel } from "@/page-components/cameras/components/camera-preview-panel/camera-preview-panel";
import { DeleteCamera } from "@/page-components/cameras/components/delete-camera";
import { UpdateCamera } from "@/page-components/cameras/components/update-camera";
import {
  buildCameraDisplayUrl,
  formatCameraDateTime,
  formatCameraLastCheckedAgo,
} from "@/page-components/cameras/lib/camera-view";
import { formatDate } from "@/utils/formatDate";
import { useLoaderData } from "react-router-dom";
import s from "./_camera-detail.module.scss";

export function Component() {
  const { cameraId } = useLoaderData() as { cameraId: string };
  const { data: camera } = useGetCamera(cameraId);
  const cameraUrl = buildCameraDisplayUrl(camera);
  const protocolLabel = camera.protocol.toUpperCase();

  return (
    <>
      <ContentHeader>
        <ContentHeader.Top
          title={camera.name}
          subtitles={[`Расположение: ${camera.location || "Расположение не указано"}`]}
          meta={[
            `Создана ${formatDate(camera.created_at)}`,
            `Обновлено: ${formatCameraLastCheckedAgo(camera.last_checked_at)}`,
            `Состояние: ${camera.is_active ? "Активна" : "Отключена"}`,
          ]}
        >
          <ContentHeader.Actions>
            <DeleteCamera id={camera.id} name={camera.name} />
            <UpdateCamera camera={camera} />
          </ContentHeader.Actions>
        </ContentHeader.Top>
      </ContentHeader>

      <div className={s.page}>
        <Section title="Проверка и поток">
          <CameraPreviewPanel camera={camera} />
        </Section>

        <Section title="Сведения">
          <div className={s.detailsGrid}>
            <SurfaceSection title="Основное" transparent>
              <InfoRow label="Протокол" value={protocolLabel} />
              <InfoRow label="Активна в системе" value={camera.is_active ? "Да" : "Нет"} />
              <InfoRow label="Расположение" value={camera.location || "-"} />
              <InfoRow label="Описание" value={camera.description || "-"} />
            </SurfaceSection>

            <SurfaceSection title="Диагностика" transparent>
              <InfoRow
                label="Последняя проверка"
                value={formatCameraDateTime(camera.last_checked_at)}
              />
              <InfoRow
                label="Проверена"
                value={formatCameraLastCheckedAgo(camera.last_checked_at)}
              />
              <InfoRow label="Таймаут" value={`${camera.timeout_sec} сек`} />
              <InfoRow label="Создана" value={formatCameraDateTime(camera.created_at)} />
            </SurfaceSection>

            <SurfaceSection
              className={s.detailsWide}
              title={camera.protocol === "usb" ? "Подключение USB" : "Параметры подключения"}
              transparent
            >
              {camera.protocol === "usb" ? (
                <div className={s.connectionRows}>
                  <InfoRow label="Устройство" value={camera.device_path || "-"} />
                  <InfoRow label="Режим" value="Локальная USB-камера" />
                  <InfoRow label="Авторизация" value="Не используется" />
                </div>
              ) : (
                <div className={s.connectionRows}>
                  <InfoRow label="Хост" value={camera.host || "-"} />
                  <InfoRow label="Порт" value={camera.port != null ? String(camera.port) : "-"} />
                  <InfoRow label="Путь потока" value={camera.stream_path || camera.path || "-"} />
                  <InfoRow label="Username" value={camera.username || "-"} />
                  <InfoRow label="Password" value={camera.username ? "••••••••" : "-"} />
                  <div className={s.connectionWideRow}>
                    <InfoRow label="URL" value={cameraUrl} valueWrap="wrap" />
                  </div>
                </div>
              )}
            </SurfaceSection>
          </div>
        </Section>
      </div>
    </>
  );
}
