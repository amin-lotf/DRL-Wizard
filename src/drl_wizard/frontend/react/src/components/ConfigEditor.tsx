import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { titleFromKey } from "../lib/format";
import type { ConfigRecord, ConfigValue, FieldMeta } from "../types/api";

interface ConfigEditorProps {
  title: string;
  description?: string;
  config: ConfigRecord | null;
  meta?: Record<string, FieldMeta>;
  onChange: (nextConfig: ConfigRecord) => void;
  lockedKeys?: string[];
}

interface ReadonlyConfigPanelProps {
  title: string;
  description?: string;
  config: ConfigRecord | null;
}

function sortEntries(
  config: ConfigRecord,
  meta?: Record<string, FieldMeta>,
): Array<[string, ConfigValue]> {
  return Object.entries(config).sort(([leftKey], [rightKey]) => {
    const leftOrder = meta?.[leftKey]?.order ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = meta?.[rightKey]?.order ?? Number.MAX_SAFE_INTEGER;
    if (leftOrder !== rightOrder) {
      return leftOrder - rightOrder;
    }
    return leftKey.localeCompare(rightKey);
  });
}

function isObjectLike(value: ConfigValue): value is Record<string, ConfigValue> | ConfigValue[] {
  return typeof value === "object" && value !== null;
}

function FieldFrame({
  label,
  description,
  children,
  locked = false,
}: {
  label: string;
  description?: string;
  children: ReactNode;
  locked?: boolean;
}) {
  return (
    <label className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-text">{label}</span>
        {locked ? <span className="label">Locked</span> : null}
      </div>
      {children}
      {description ? <p className="text-xs leading-5 text-muted">{description}</p> : null}
    </label>
  );
}

function NumberField({
  value,
  integer,
  disabled,
  onCommit,
}: {
  value: number;
  integer: boolean;
  disabled: boolean;
  onCommit: (nextValue: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  return (
    <input
      className="input"
      type="number"
      step={integer ? 1 : "any"}
      value={draft}
      disabled={disabled}
      onChange={(event) => {
        const nextDraft = event.target.value;
        setDraft(nextDraft);
        const parsed = integer ? Number.parseInt(nextDraft, 10) : Number.parseFloat(nextDraft);
        if (!Number.isNaN(parsed)) {
          onCommit(parsed);
        }
      }}
      onBlur={() => {
        const parsed = integer ? Number.parseInt(draft, 10) : Number.parseFloat(draft);
        if (Number.isNaN(parsed)) {
          setDraft(String(value));
          return;
        }
        onCommit(parsed);
        setDraft(String(parsed));
      }}
    />
  );
}

function TextField({
  value,
  disabled,
  onCommit,
}: {
  value: string;
  disabled: boolean;
  onCommit: (nextValue: string) => void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  return (
    <input
      className="input"
      value={draft}
      disabled={disabled}
      onChange={(event) => {
        const nextValue = event.target.value;
        setDraft(nextValue);
        onCommit(nextValue);
      }}
    />
  );
}

function JsonField({
  value,
  disabled,
  onCommit,
}: {
  value: Record<string, ConfigValue> | ConfigValue[];
  disabled: boolean;
  onCommit: (nextValue: Record<string, ConfigValue> | ConfigValue[]) => void;
}) {
  const serialized = useMemo(() => JSON.stringify(value, null, 2), [value]);
  const [draft, setDraft] = useState(serialized);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(serialized);
    setError(null);
  }, [serialized]);

  return (
    <div className="space-y-2">
      <textarea
        className="textarea"
        value={draft}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          try {
            const parsed = JSON.parse(draft) as Record<string, ConfigValue> | ConfigValue[];
            onCommit(parsed);
            setError(null);
          } catch {
            setError("Invalid JSON. The previous value is still in use.");
            setDraft(serialized);
          }
        }}
      />
      {error ? <p className="text-xs text-warning">{error}</p> : null}
    </div>
  );
}

function ConfigField({
  fieldKey,
  value,
  meta,
  locked,
  onChange,
}: {
  fieldKey: string;
  value: ConfigValue;
  meta?: FieldMeta;
  locked: boolean;
  onChange: (nextValue: ConfigValue) => void;
}) {
  const label = meta?.label ?? titleFromKey(fieldKey);
  const description = meta?.description;
  const enumChoices = meta?.enum_choices ?? null;

  if (enumChoices?.length) {
    return (
      <FieldFrame label={label} description={description} locked={locked}>
        <select
          className="input"
          value={String(value)}
          disabled={locked}
          onChange={(event) => onChange(event.target.value)}
        >
          {enumChoices.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </FieldFrame>
    );
  }

  if (typeof value === "boolean") {
    return (
      <FieldFrame label={label} description={description} locked={locked}>
        <label className="flex h-11 items-center justify-between rounded-lg border border-border bg-page/70 px-3">
          <span className="text-sm text-text">{value ? "Enabled" : "Disabled"}</span>
          <input
            type="checkbox"
            checked={value}
            disabled={locked}
            onChange={(event) => onChange(event.target.checked)}
          />
        </label>
      </FieldFrame>
    );
  }

  if (typeof value === "number") {
    return (
      <FieldFrame label={label} description={description} locked={locked}>
        <NumberField
          value={value}
          integer={Number.isInteger(value)}
          disabled={locked}
          onCommit={onChange}
        />
      </FieldFrame>
    );
  }

  if (isObjectLike(value)) {
    return (
      <FieldFrame label={label} description={description} locked={locked}>
        <JsonField value={value} disabled={locked} onCommit={onChange} />
      </FieldFrame>
    );
  }

  return (
    <FieldFrame label={label} description={description} locked={locked}>
      <TextField value={value === null ? "" : String(value)} disabled={locked} onCommit={onChange} />
    </FieldFrame>
  );
}

export function ConfigEditor({
  title,
  description,
  config,
  meta,
  onChange,
  lockedKeys = [],
}: ConfigEditorProps) {
  if (!config) {
    return (
      <section className="panel p-5">
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        <p className="mt-2 text-sm text-muted">Loading configuration…</p>
      </section>
    );
  }

  const lockedKeySet = new Set(lockedKeys);
  const entries = sortEntries(config, meta);

  return (
    <section className="panel p-5">
      <div className="mb-5 space-y-1">
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        {description ? <p className="text-sm text-muted">{description}</p> : null}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        {entries.map(([fieldKey, value]) => (
          <ConfigField
            key={fieldKey}
            fieldKey={fieldKey}
            value={value}
            meta={meta?.[fieldKey]}
            locked={lockedKeySet.has(fieldKey)}
            onChange={(nextValue) =>
              onChange({
                ...config,
                [fieldKey]: nextValue,
              })
            }
          />
        ))}
      </div>
    </section>
  );
}

export function ReadonlyConfigPanel({
  title,
  description,
  config,
}: ReadonlyConfigPanelProps) {
  if (!config) {
    return (
      <section className="panel p-5">
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        <p className="mt-2 text-sm text-muted">No configuration loaded.</p>
      </section>
    );
  }

  return (
    <section className="panel p-5">
      <div className="mb-5 space-y-1">
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        {description ? <p className="text-sm text-muted">{description}</p> : null}
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {Object.entries(config).map(([key, value]) => (
          <div key={key} className="rounded-lg border border-border bg-page/60 p-3">
            <div className="label mb-1">{titleFromKey(key)}</div>
            {isObjectLike(value) ? (
              <pre className="overflow-x-auto text-xs leading-6 text-[#d6e7df]">
                {JSON.stringify(value, null, 2)}
              </pre>
            ) : (
              <div className="break-words text-sm text-text">{String(value)}</div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
