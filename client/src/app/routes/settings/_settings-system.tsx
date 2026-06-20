
import QueryState from "@/components/ui/query-state/query-state";
import { useGetSystemStats } from "@/page-components/settings/api/get-system-stats";
import { useSystemStatsLive } from "@/page-components/settings/lib/use-system-stats-live";
import type { LucideIcon } from "lucide-react";
import { Activity, Cpu, Gauge, HardDrive, MemoryStick, Server, ShieldCheck, Thermometer, Timer, Zap } from "lucide-react";
import s from "./_settings-strict.module.scss";

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
  if (value == null) return "—";
  return `${formatCount(value)} MB`;
}

function getPercent(used: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((used / total) * 100)));
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
  const ramPercent = getPercent(data.resources.memory_used_bytes, data.resources.memory_total_bytes);
  const diskPercent = getPercent(data.resources.disk_used_bytes, data.resources.disk_total_bytes);
  const gpuPercent = data.gpu.utilization_percent ?? 0;
  const gpuMemoryPercent = data.gpu.memory_used_mb != null && data.gpu.memory_total_mb != null ? getPercent(data.gpu.memory_used_mb, data.gpu.memory_total_mb) : null;
  const healthScore = Math.max(0, Math.min(100, Math.round(100 - Math.max(cpuPercent, ramPercent, diskPercent, gpuPercent) * 0.55)));

  return (
    <QueryState isLoading={isLoading} isError={isError} size="page">
      <section className={s.hero}>
        <div>
          <h1>Система</h1>
          <p>Ресурсы, GPU и хранилище.</p>
        </div>
        <div className={s.healthCard}>
          <span>Состояние</span>
          <strong>{healthScore}</strong>
          <small>обновлено {formatTime(data.updated_at)}</small>
        </div>
      </section>

      <div className={s.grid}>
        <section className={s.telemetryPanel}>
          <div className={s.cardHead}>
            <div><h3>Ресурсы</h3><p>CPU, RAM, диск и GPU.</p></div>
            <span className={s.softBadge}>live</span>
          </div>
          <div className={s.metricGrid}>
            <ResourceCard icon={Cpu} label="CPU" value={`${Math.round(cpuPercent)}%`} hint={`${formatCount(data.resources.cpu_count_logical)} потоков`} percent={cpuPercent} />
            <ResourceCard icon={MemoryStick} label="RAM" value={`${formatBytes(data.resources.memory_used_bytes)} / ${formatBytes(data.resources.memory_total_bytes)}`} hint={`${ramPercent}% занято`} percent={ramPercent} />
            <ResourceCard icon={HardDrive} label="Disk" value={`${formatBytes(data.resources.disk_used_bytes)} / ${formatBytes(data.resources.disk_total_bytes)}`} hint={`${diskPercent}% занято`} percent={diskPercent} />
            <ResourceCard icon={Zap} label="GPU" value={data.gpu.name ?? "нет GPU"} hint={data.gpu.available ? `${gpuPercent}% · ${data.gpu.temperature_c ?? "—"}°C` : "GPU недоступен"} percent={gpuPercent} muted={!data.gpu.available} />
          </div>
        </section>

        <aside className={s.hostPanel}>
          <div className={s.cardHead}>
            <div><h3>Хост</h3></div>
          </div>
          <HostInfo icon={Server} label="Имя хоста" value={data.system.hostname} />
          <HostInfo icon={Timer} label="Время работы" value={formatUptime(data.system.uptime_sec)} />
          <HostInfo icon={Gauge} label="Память GPU" value={data.gpu.memory_used_mb != null && data.gpu.memory_total_mb != null ? `${formatMegabytes(data.gpu.memory_used_mb)} / ${formatMegabytes(data.gpu.memory_total_mb)} (${gpuMemoryPercent}%)` : "—"} />
          <HostInfo icon={Thermometer} label="Температура" value={data.gpu.temperature_c != null ? `${data.gpu.temperature_c}°C` : "—"} />
        </aside>
      </div>

      <section className={s.storagePanel}>
        <div className={s.cardHead}>
          <div>
            <h3>Хранилище</h3>
            <p>Файлы проекта.</p>
          </div>
          <span className={s.softBadge}>{formatBytes(data.storage.used_bytes)}</span>
        </div>
        <div className={s.storageGrid}>
          <StorageTile icon={ShieldCheck} label="Эталоны" value={data.storage.categories.standards_bytes} total={data.storage.used_bytes} />
          <StorageTile icon={Activity} label="Проверки" value={data.storage.categories.inspections_bytes} total={data.storage.used_bytes} />
          <StorageTile icon={Zap} label="Модели" value={data.storage.categories.models_bytes} total={data.storage.used_bytes} />
          <StorageTile icon={Server} label="Логи" value={data.storage.categories.logs_bytes} total={data.storage.used_bytes} />
          <StorageTile icon={HardDrive} label="Другое" value={data.storage.categories.other_bytes} total={data.storage.used_bytes} />
        </div>
      </section>
    </QueryState>
  );
}

function ResourceCard({ icon: Icon, label, value, hint, percent, muted }: { icon: LucideIcon; label: string; value: string; hint: string; percent: number; muted?: boolean }) {
  return (
    <div className={s.resourceCard} data-muted={muted ? "true" : undefined}>
      <div><Icon /><span>{label}</span></div>
      <strong>{value}</strong>
      <small>{hint}</small>
      <div className={s.track}><span style={{ width: `${Math.max(0, Math.min(100, Math.round(percent)))}%` }} /></div>
    </div>
  );
}

function StorageTile({ icon: Icon, label, value, total }: { icon: LucideIcon; label: string; value: number; total: number }) {
  const percent = getPercent(value, total || 1);
  return (
    <div className={s.storageTile}>
      <Icon />
      <span>{label}</span>
      <strong>{formatBytes(value)}</strong>
      <div className={s.track}><span style={{ width: `${percent}%` }} /></div>
      <small>{percent}% хранилища</small>
    </div>
  );
}

function HostInfo({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className={s.hostInfo}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
