import Link from "next/link";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link className="brand" href="/" aria-label="BARRACUDA home">
      <span className="brandMark" aria-hidden="true">B</span>
      <span className="brandCopy">
        <strong>BARRACUDA</strong>
        {!compact && <small>Immune cell inference</small>}
      </span>
    </Link>
  );
}
