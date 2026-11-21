"""References in the help link to the work: through doi.org when they give a DOI, to JSTOR for a stable ID, to a web
address they give, or to a search for the citation (pymetrix.help.reference_markdown; scripts/check_references.py checks the DOIs in CI)."""

from urllib.parse import parse_qs, urlparse

from pymetrix import help as hp


def test_a_doi_links_through_doi_org():
    line = hp.reference_markdown("Dickey, D. A. and Fuller, W. A. (1979). JASA 74, 427-431. doi:10.2307/2286348.")
    assert line == "- Dickey, D. A. and Fuller, W. A. (1979). JASA 74, 427-431. [doi:10.2307/2286348](https://doi.org/10.2307/2286348)."
    assert "(https://doi.org/10.1016/0304-4076(92)90104-Y)" in hp.reference_markdown(
        "Kwiatkowski et al. (1992). Journal of Econometrics 54. https://doi.org/10.1016/0304-4076(92)90104-Y")
    # an old Wiley DOI (SICI) holds ; < > ( ): the DOI is whole, and the link escapes what Markdown would misread
    sici = "10.1002/(sici)1099-1255(199611)11:6<601::aid-jae417>3.0.co;2-t"
    line = hp.reference_markdown(f"MacKinnon, J. G. (1996). Journal of Applied Econometrics 11(6), 601-618. doi:{sici}")
    assert line.endswith(f"[doi:{sici}](https://doi.org/10.1002/(sici)1099-1255(199611)11:6%3C601::aid-jae417%3E3.0.co%3B2-t)")


def test_a_jstor_stable_id_links_to_jstor_and_a_doi_comes_first():
    line = hp.reference_markdown("MacKinlay, A. C. (1997). Event studies. JEL 35(1), 13-39. jstor:2729691")
    assert line == "- MacKinlay, A. C. (1997). Event studies. JEL 35(1), 13-39. [jstor:2729691](https://www.jstor.org/stable/2729691)"
    chapter = hp.reference_markdown("Wooldridge, J. M. (2010). Econometric Analysis, chapter 5. MIT Press. jstor:j.ctt5hhcfr.9.")
    assert chapter.endswith("[jstor:j.ctt5hhcfr.9](https://www.jstor.org/stable/j.ctt5hhcfr.9).")
    both = hp.reference_markdown("A. (2000). T. J 1, 1-2. doi:10.2307/2286348 jstor:2286348")
    assert "(https://doi.org/10.2307/2286348)" in both and "jstor.org" not in both


def test_a_web_address_links_and_anything_else_searches_for_the_citation():
    assert hp.reference_markdown("See https://www.census.gov/x13 for the manual.") == "- See <https://www.census.gov/x13> for the manual."
    citation = "Zellner, A. (1962). Seemingly unrelated regressions. JASA 57."
    line = hp.reference_markdown(citation)
    url = line[line.index("](") + 2:-2]
    assert line.startswith(f"- {citation} ([search](") and parse_qs(urlparse(url).query)["q"] == [citation]


def test_help_pages_and_topics_show_linked_references():
    hp._load_all()
    page = next(t for t in hp.topics() if t.references)
    assert "scholar.google.com" in hp.topic_markdown(page) or "doi.org" in hp.topic_markdown(page)
