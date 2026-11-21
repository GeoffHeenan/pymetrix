"""``pymetrix-labbook``: check a project's lab book, list its entries, or serve read-only audit tools over MCP."""

from __future__ import annotations

import argparse
import sys
import zipfile

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pymetrix-labbook", description="Check or list the lab book of a pymetrix project.")
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("verify", help="check the chain, the save log, the steps and a replay")
    check.add_argument("project")
    check.add_argument("--no-replay", action="store_true", help="check the lab book only, without opening the project")
    check.add_argument("--trust", action="store_true", help="run the project's typed code in the replay (read it first)")
    show = sub.add_parser("show", help="list the lab book's entries")
    show.add_argument("project")
    audit = sub.add_parser("audit", help="serve read-only audit tools over MCP (stdio), for an auditor's assistant")
    audit.add_argument("project")
    args = parser.parse_args(argv)

    if args.action == "audit":
        from .audit import build_audit_server

        build_audit_server(args.project).run("stdio")
        return 0

    if args.action == "verify":
        from .verify import verify

        result = verify(args.project, replay=not args.no_replay, trust=True if args.trust else None)
        print(result)
        return 0 if result.ok else 1

    from .chain import Chain
    from .recorder import FILE, KEY
    from .verify import describe_actor

    with zipfile.ZipFile(args.project) as z:
        try:
            blob = z.read(f"attachments/{KEY}/{FILE}")
        except KeyError:
            print(f"{args.project} has no lab book", file=sys.stderr)
            return 1
    for e in Chain.from_bytes(blob).entries:
        data = e.get("data", {})
        what = (data.get("code") or ", ".join(data.get("codes", [])) or data.get("path", "") or data.get("title", "")
                or data.get("text", "") or data.get("reason", "")
                or (f"{len(data['specifications'])} specification(s)" if "specifications" in data else "")
                or data.get("brief", {}).get("title", "") or data.get("hash", ""))
        first = str(what).strip().splitlines()[0] if str(what).strip() else ""
        by = describe_actor(e.get("by")) if "by" in e else ""
        print(f"{e.get('seq', '?'):>4}  {e.get('time', '')}  {e.get('event', '?'):<9} {e.get('during', ''):<13} {by:<24} {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
