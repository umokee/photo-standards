import { ContentHeader } from "@/components/layouts/content-header/content-header";
import { Section } from "@/components/layouts/section/section";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetSystemStats } from "@/page-components/settings/api/get-system-stats";
import { StatCards } from "@/page-components/settings/components/stat-cards/stat-cards";
import { StorageSummaryCard } from "@/page-components/settings/components/storage-summary-card/storage-summary-card";
import { useSystemStatsLive } from "@/page-components/settings/lib/use-system-stats-live";

const numberFormatter = new Intl.NumberFormat("ru-RU");

function formatCount(value: number) {
  return numberFormatter.format(value);
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;

  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = -1;

  do {
    value /= 1024;
    unitIndex += 1;
  } while (value >= 1024 && unitIndex < units.length - 1);

  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatMegabytes(value: number | null) {
  if (value == null) return "-";
  return `${formatCount(value)} MB`;
}

function getPercent(used: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((used / total) * 100)));
}

function formatPercent(value: number) {
  return `${Math.round(value)}%`;
}

function formatUptime(totalSec: number) {
  const days = Math.floor(totalSec / 86400);
  const hours = Math.floor((totalSec % 86400) / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}д`);
  if (hours > 0 || days > 0) parts.push(`${hours}ч`);
  parts.push(`${minutes}м`);

  return parts.join(" ");
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString("ru-RU");
}

export function Component() {
  const { data, isLoading, isError } = useGetSystemStats();

  useSystemStatsLive();

  if (!data) {
    return <QueryState isLoading={isLoading} isError={isError} size="page" />;
  }

  const cpuPercent = data.resources.cpu_percent;
  const ramPercent = getPercent(
    data.resources.memory_used_bytes,
    data.resources.memory_total_bytes
  );
  const diskPercent = getPercent(data.resources.disk_used_bytes, data.resources.disk_total_bytes);
  const gpuPercent = data.gpu.utilization_percent ?? 0;
  const gpuMemoryPercent =
    data.gpu.memory_used_mb != null && data.gpu.memory_total_mb != null
      ? getPercent(data.gpu.memory_used_mb, data.gpu.memory_total_mb)
      : null;

  const gpuHint = data.gpu.available
    ? [
        `Нагрузка ${gpuPercent}%`,
        data.gpu.temperature_c != null ? `${data.gpu.temperature_c}°C` : null,
        data.gpu.memory_used_mb != null && data.gpu.memory_total_mb != null
          ? `${formatMegabytes(data.gpu.memory_used_mb)} / ${formatMegabytes(data.gpu.memory_total_mb)} (${gpuMemoryPercent}%) VRAM`
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "Сервер не видит видеокарту";

  return (
    <QueryState isLoading={isLoading} isError={isError} size="page">
      <ContentHeader>
        <ContentHeader.Top
          title="Система"
          subtitles={["Метрики сервера, хранилища и состояния приложения"]}
          meta={[`Последнее обновление: ${formatTime(data.updated_at)}`]}
        />
      </ContentHeader>

      <Section title="Ресурсы" bordered>
        <StatCards
          items={{
            CPU: {
              value: formatPercent(cpuPercent),
              hint: `${formatCount(data.resources.cpu_count_logical)} логических потоков`,
            },
            RAM: {
              value: `${formatBytes(data.resources.memory_used_bytes)} / ${formatBytes(data.resources.memory_total_bytes)} (${formatPercent(ramPercent)})`,
              hint: "Оперативная память сервера",
            },
            Disk: {
              value: `${formatBytes(data.resources.disk_used_bytes)} / ${formatBytes(
                data.resources.disk_total_bytes
              )} (${formatPercent(diskPercent)})`,
              hint: "Системный диск",
            },
            GPU: {
              value: data.gpu.name ?? "Недоступна",
              hint: gpuHint,
            },
            Host: {
              value: data.system.hostname,
              hint: "Имя машины, на которой запущен сервер",
            },
            Uptime: {
              value: formatUptime(data.system.uptime_sec),
              hint: "Время работы backend-процесса",
            },
          }}
        />
      </Section>

      <Section title="Хранилище">
        <StorageSummaryCard
          value={formatBytes(data.storage.used_bytes)}
          items={[
            {
              label: "Эталоны",
              value: formatBytes(data.storage.categories.standards_bytes),
            },
            {
              label: "Проверки",
              value: formatBytes(data.storage.categories.inspections_bytes),
            },
            {
              label: "Модели",
              value: formatBytes(data.storage.categories.models_bytes),
            },
            {
              label: "Логи",
              value: formatBytes(data.storage.categories.logs_bytes),
            },
            {
              label: "Прочее",
              value: formatBytes(data.storage.categories.other_bytes),
            },
          ]}
        />
      </Section>
    </QueryState>
  );
}
