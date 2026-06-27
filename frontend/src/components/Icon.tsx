/** Thin wrapper over Google Material Symbols (loaded in index.html). */
interface IconProps {
  name: string;
  className?: string;
  /** Optical size in px; also drives the glyph weight via font-variation-settings. */
  size?: number;
  filled?: boolean;
  title?: string;
}

export function Icon({ name, className = "", size = 24, filled = true, title }: IconProps) {
  return (
    <span
      className={`material-symbols-outlined select-none ${className}`}
      style={{
        fontSize: size,
        fontVariationSettings: `'FILL' ${filled ? 1 : 0}, 'wght' 400, 'GRAD' 0, 'opsz' ${size}`,
      }}
      title={title}
      aria-hidden={title ? undefined : true}
    >
      {name}
    </span>
  );
}
