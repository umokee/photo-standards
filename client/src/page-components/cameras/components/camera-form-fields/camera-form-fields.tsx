import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import Select from "@/components/ui/select/select";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import ToggleCard from "@/components/ui/toggle-card/toggle-card";
import { RefreshCw } from "lucide-react";
import { useMemo } from "react";
import { useGetUsbCameraDevices } from "../../api/get-usb-camera-devices";
import { CameraFormValues, cameraProtocolOptions } from "../../lib/camera-form";
import s from "./camera-form-fields.module.scss";

interface Props {
  values: CameraFormValues;
  errors?: Partial<Record<keyof CameraFormValues, string | undefined>>;
  onChange: <K extends keyof CameraFormValues>(key: K, value: CameraFormValues[K]) => void;
  isEditing?: boolean;
}

export const CameraFormFields = ({ values, errors, onChange, isEditing = false }: Props) => {
  const isUsb = values.protocol === "usb";
  const usbDevicesQuery = useGetUsbCameraDevices(isUsb);

  const usbOptions = useMemo(() => {
    const options = (usbDevicesQuery.data ?? []).map((device) => ({
      value: device.device_path,
      label: device.name,
    }));

    if (values.device_path && !options.some((option) => option.value === values.device_path)) {
      options.unshift({
        value: values.device_path,
        label: values.device_path,
      });
    }

    return options;
  }, [usbDevicesQuery.data, values.device_path]);

  const selectedUsbDevice = useMemo(() => {
    return (
      usbDevicesQuery.data?.find((device) => device.device_path === values.device_path) ?? null
    );
  }, [usbDevicesQuery.data, values.device_path]);

  const streamPathLabel = values.protocol === "rtsp" ? "Путь потока" : "HTTP endpoint";
  const streamPathPlaceholder = values.protocol === "rtsp" ? "/stream1" : "/video";

  return (
    <div className={s.root}>
      <SurfaceSection
        className={s.formSection}
        title="Основное"
        hint="Базовые данные камеры и режим подключения."
      >
        <Input
          label="Название"
          placeholder="Например, Линия 1"
          value={values.name}
          onChange={(value) => onChange("name", value)}
          error={errors?.name}
        />

        <div className={s.gridTwo}>
          <Select
            label="Протокол"
            options={cameraProtocolOptions}
            value={values.protocol}
            onChange={(value) => onChange("protocol", value as CameraFormValues["protocol"])}
            error={errors?.protocol}
          />

          <Input
            label="Расположение"
            placeholder="Например, Склад / Вход"
            value={values.location}
            onChange={(value) => onChange("location", value)}
            error={errors?.location}
          />
        </div>

        <Input
          label="Описание"
          placeholder="Краткое описание камеры"
          value={values.description}
          onChange={(value) => onChange("description", value)}
          error={errors?.description}
        />
      </SurfaceSection>

      <SurfaceSection
        className={s.formSection}
        title="Подключение"
        hint={
          isUsb
            ? "Выберите найденную USB-камеру. Путь сохранится автоматически."
            : "Для RTSP и HTTP используем один основной endpoint потока."
        }
        aside={
          isUsb ? (
            <Button
              variant="ghost"
              size="sm"
              icon={RefreshCw}
              disabled={usbDevicesQuery.isFetching}
              onClick={() => {
                void usbDevicesQuery.refetch();
              }}
            >
              {usbDevicesQuery.isFetching ? "Обновление..." : "Обновить"}
            </Button>
          ) : null
        }
      >
        {isUsb ? (
          <>
            <QueryState
              isLoading={usbDevicesQuery.isLoading}
              isError={usbDevicesQuery.isError}
              isEmpty={!usbOptions.length}
              size="block"
              loadingText="Ищем USB-камеры..."
              errorTitle="Не удалось получить список USB-камер"
              errorDescription="Проверьте подключение устройства и попробуйте обновить список"
              emptyTitle="USB-камеры не найдены"
              emptyDescription="Подключите устройство и обновите список"
            >
              <Select
                label="USB-камера"
                placeholder="Выберите устройство"
                options={usbOptions}
                value={values.device_path || null}
                onChange={(value) => onChange("device_path", value)}
                error={errors?.device_path}
              />
            </QueryState>

            {selectedUsbDevice ? (
              <div className={s.metaNote}>
                {selectedUsbDevice.device_path}
                {selectedUsbDevice.width && selectedUsbDevice.height
                  ? ` · ${selectedUsbDevice.width}×${selectedUsbDevice.height}`
                  : ""}
                {selectedUsbDevice.backend ? ` · ${selectedUsbDevice.backend}` : ""}
              </div>
            ) : null}
          </>
        ) : (
          <>
            <div className={s.hostPortRow}>
              <Input
                label="Хост"
                placeholder="192.168.1.50"
                value={values.host}
                onChange={(value) => onChange("host", value)}
                error={errors?.host}
                noMargin
              />

              <span className={s.separator}>:</span>

              <Input
                label="Порт"
                placeholder={values.protocol === "rtsp" ? "554" : "80"}
                value={values.port}
                onChange={(value) => onChange("port", value)}
                error={errors?.port}
                type="number"
                min={1}
                max={65535}
                noMargin
              />
            </div>

            <Input
              label={streamPathLabel}
              placeholder={streamPathPlaceholder}
              value={values.stream_path}
              onChange={(value) => onChange("stream_path", value)}
              error={errors?.stream_path}
            />
          </>
        )}
      </SurfaceSection>

      {!isUsb ? (
        <SurfaceSection
          className={s.formSection}
          title="Авторизация"
          hint="Если у камеры нет логина и пароля, поля останутся пустыми и в API уйдут null."
        >
          <ToggleCard
            title="Есть логин и пароль"
            hint="Включайте только для камер, где поток закрыт авторизацией."
            checked={values.has_auth}
            onChange={(checked) => onChange("has_auth", checked)}
          />

          {values.has_auth ? (
            <>
              <div className={s.gridTwo}>
                <Input
                  label="Логин"
                  placeholder="admin"
                  value={values.username}
                  onChange={(value) => onChange("username", value)}
                  error={errors?.username}
                />

                <Input
                  label="Пароль"
                  placeholder={isEditing ? "Оставьте пустым, чтобы не менять" : "••••••••"}
                  value={values.password}
                  onChange={(value) => onChange("password", value)}
                  error={errors?.password}
                  type="password"
                />
              </div>

              {isEditing ? (
                <div className={s.metaNote}>
                  Пустой пароль при редактировании оставит сохранённое значение без изменений.
                </div>
              ) : null}
            </>
          ) : null}
        </SurfaceSection>
      ) : null}

      <SurfaceSection
        className={s.formSection}
        title="Дополнительно"
        hint="Параметры работы камеры в системе."
      >
        <div className={s.gridTwo}>
          <Input
            label="Timeout (сек)"
            placeholder="5"
            value={values.timeout_sec}
            onChange={(value) => onChange("timeout_sec", value)}
            error={errors?.timeout_sec}
            type="number"
            min={1}
            max={60}
          />

          <ToggleCard
            title="Активна в системе"
            hint="Неактивная камера не используется при захвате и проверках."
            checked={values.is_active}
            onChange={(checked) => onChange("is_active", checked)}
          />
        </div>
      </SurfaceSection>
    </div>
  );
};
