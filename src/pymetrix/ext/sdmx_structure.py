"""Reading an SDMX 2.1 structure message: a dataflow's dimensions with the codes each takes.

A dimension's codes are found as SDMX 2.1 allows them to be given, in this order:

1. the dimension's own representation (``LocalRepresentation/Enumeration``), by agency, id and version;
2. its concept's representation (``ConceptIdentity`` → the concept's ``CoreRepresentation/Enumeration``);
3. a code list in the message with the same id under another agency or version (the IMF's flows belong to one
   agency, ``IMF.RES``, while their code lists may be kept by another, ``IMF``);
4. failing those, the code list asked for on its own (``codelist/{agency}/{id}/{version}``);

and then narrowed by the ``ContentConstraint``s attached to the dataflow or its data structure (the codes that
have data). A dimension whose codes cannot be found still gets the codes a constraint lists, without names; if there
are none, its ``note`` says why, and codes can be typed. References may be ``Ref`` elements or URNs.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

from ..connectors import Dimension
from ..errors import PyMetrixError

__all__ = ["Reference", "codelists", "dimensions", "reference"]

_URN = re.compile(r"=(?P<agency>[^:]+):(?P<id>[^(]+)\((?P<version>[^)]*)\)(?:\.(?P<item>.+))?$")
_ANY_VERSION = ("", "latest", "*", "+")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    return None if element is None else next((c for c in element if _local(c.tag) == name), None)


def _text(element: ET.Element, name: str) -> str:
    """The English (else the first) ``Name`` or ``Description`` of an SDMX artefact."""
    found = [c for c in element if _local(c.tag) == name]
    for c in found:
        if c.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "en") == "en":
            return " ".join((c.text or "").split())
    return " ".join((found[0].text or "").split()) if found else ""


@dataclass(frozen=True)
class Reference:
    """A reference to an artefact: its ``agency``, ``id`` and ``version`` (empty: any), and for an item (a concept)
    its scheme's id in ``parent``."""

    agency: str
    id: str
    version: str = ""
    parent: str = ""

    def __str__(self) -> str:
        return f"{self.agency}:{self.id}({self.version or 'latest'})"


def reference(element: ET.Element | None, *path: str) -> Reference | None:
    """The reference under ``element`` along ``path`` (local tag names): a ``Ref`` element or a ``URN``."""
    node = element
    for name in path:
        node = _child(node, name)
    if node is None:
        return None
    ref = _child(node, "Ref")
    if ref is not None and ref.attrib.get("id"):
        a = ref.attrib
        parent = a.get("maintainableParentID", "")
        return Reference(a.get("agencyID", ""), a["id"], a.get("maintainableParentVersion", "") if parent else a.get("version", ""), parent)
    urn = _child(node, "URN")
    found = _URN.search(" ".join((urn.text or "").split())) if urn is not None else None
    if found is None:
        return None
    if found.group("item"):  # a concept: SCHEME_AGENCY:SCHEME(VERSION).CONCEPT
        return Reference(found.group("agency"), found.group("item"), found.group("version"), found.group("id"))
    return Reference(found.group("agency"), found.group("id"), found.group("version"))


CodeList = tuple[tuple[str, str, str], ...]  # (code, name, description)


def codelists(elements: list[ET.Element]) -> dict[tuple[str, str, str], CodeList]:
    """The code lists in a message by (agency, id, version)."""
    return {(e.attrib.get("agencyID", ""), e.attrib["id"], e.attrib.get("version", "")):
            tuple((c.attrib["id"], _text(c, "Name"), _text(c, "Description")) for c in e if _local(c.tag) == "Code" and "id" in c.attrib)
            for e in elements if _local(e.tag) == "Codelist" and "id" in e.attrib}


def _version_key(version: str) -> tuple:
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.]", version or "0"))


def _exact(lists: dict[tuple[str, str, str], CodeList], ref: Reference) -> CodeList | None:
    if ref.version not in _ANY_VERSION:
        return lists.get((ref.agency, ref.id, ref.version))
    versions = sorted((v for (a, i, v) in lists if a == ref.agency and i == ref.id), key=_version_key)
    return lists[(ref.agency, ref.id, versions[-1])] if versions else None


def _same_id(lists: dict[tuple[str, str, str], CodeList], ref: Reference) -> CodeList | None:
    found = sorted(((a, v) for (a, i, v) in lists if i == ref.id), key=lambda av: (av[0] == ref.agency, _version_key(av[1])))
    return lists[(found[-1][0], ref.id, found[-1][1])] if found else None


def _constraints(elements: list[ET.Element], flow: Reference, dsd: Reference | None) -> dict[str, tuple[set[str] | None, set[str]]]:
    """The codes the content constraints attached to ``flow`` (or ``dsd``) include (None: not limited) and exclude, by
    dimension; a dimension no constraint names is not in the result."""
    included: dict[str, set[str]] = {}
    excluded: dict[str, set[str]] = {}
    for c in elements:
        if _local(c.tag) != "ContentConstraint":
            continue
        attachment = _child(c, "ConstraintAttachment")
        targets = [(_local(t.tag), reference(attachment, _local(t.tag))) for t in (attachment if attachment is not None else [])]
        if not any((kind == "Dataflow" and r is not None and r.id == flow.id and r.agency in ("", flow.agency))
                   or (kind == "DataStructure" and r is not None and dsd is not None and r.id == dsd.id)
                   for kind, r in targets):
            continue
        for region in (r for r in c if _local(r.tag) == "CubeRegion"):
            into = included if region.attrib.get("include", "true").lower() != "false" else excluded
            for kv in (k for k in region if _local(k.tag) == "KeyValue" and "id" in k.attrib):
                into.setdefault(kv.attrib["id"], set()).update(" ".join((v.text or "").split()) for v in kv if _local(v.tag) == "Value")
    return {dim: (included.get(dim), excluded.get(dim, set())) for dim in set(included) | set(excluded)}


def dimensions(root: ET.Element, flow: Reference, dsd_element: ET.Element, dsd: Reference | None,
               fetch_codelist: Callable[[Reference], bytes] | None = None) -> tuple[Dimension, ...]:
    """The dimensions of ``dsd_element`` (by position) with their codes, found as the module's docstring says.
    ``fetch_codelist(ref)`` asks for a code list missing from the message (its answer: a structure message)."""
    elements = list(root.iter())
    lists = codelists(elements)
    concepts: dict[tuple[str, str, str], ET.Element] = {}  # (scheme agency, scheme id, concept id)
    for scheme in (e for e in elements if _local(e.tag) == "ConceptScheme"):
        for concept in (c for c in scheme if _local(c.tag) == "Concept" and "id" in c.attrib):
            concepts[(scheme.attrib.get("agencyID", ""), scheme.attrib.get("id", ""), concept.attrib["id"])] = concept
    by_concept_id = {cid: e for (_a, _s, cid), e in concepts.items()}
    allowed = _constraints(elements, flow, dsd)
    dims = [e for e in dsd_element.iter() if _local(e.tag) == "Dimension" and "id" in e.attrib]
    dims.sort(key=lambda e: int(e.attrib.get("position", "0") or 0))
    out = []
    for d in dims:
        dim_id = d.attrib["id"]
        concept_ref = reference(d, "ConceptIdentity")
        concept = None
        if concept_ref is not None:
            concept = concepts.get((concept_ref.agency, concept_ref.parent, concept_ref.id))
            concept = concept if concept is not None else by_concept_id.get(concept_ref.id)
        concept = concept if concept is not None else by_concept_id.get(dim_id)
        local = reference(d, "LocalRepresentation", "Enumeration")
        core = reference(concept, "CoreRepresentation", "Enumeration") if concept is not None else None
        refs = [r for r in (local, core) if r is not None]
        codes = next((c for c in (_exact(lists, r) for r in refs) if c), None)  # 1 and 2
        codes = codes or next((c for c in (_same_id(lists, r) for r in refs) if c), None)  # 3
        note = ""
        if not codes and refs and fetch_codelist is not None:  # 4
            try:
                more = codelists(list(ET.fromstring(fetch_codelist(refs[0])).iter()))
                codes = _exact(more, refs[0]) or _same_id(more, refs[0])
            except (PyMetrixError, ConnectionError, OSError, ET.ParseError) as exc:
                note = f"its code list {refs[0]} could not be loaded ({exc})"
            if not codes and not note:
                note = f"its code list {refs[0]} was not in the provider's answers"
        elif not codes:
            note = f"its code list {refs[0]} was not in the provider's answer" if refs else "the structure gives it no code list"
        codes = codes or ()
        included, excluded = allowed.get(dim_id, (None, set()))
        if codes:
            narrowed = tuple(c for c in codes if (included is None or c[0] in included) and c[0] not in excluded)
            codes = narrowed or codes  # a constraint that matches none of the codes is not believed
            note = ""
        elif included:
            codes = tuple((x, "", "") for x in sorted(included - excluded))
            note = f"{note}: the codes listed are those the dataflow's constraint allows, without their names"
        name = _text(concept, "Name") if concept is not None else ""
        out.append(Dimension(dim_id, name, tuple((c, n) for c, n, _ in codes),
                             descriptions=tuple((c, t) for c, _, t in codes if t), note=note))
    return tuple(out)
