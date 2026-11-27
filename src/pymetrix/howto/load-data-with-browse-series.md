---
summary: find series at FRED, the IMF, the ECB, Eurostat or the World Bank, load them, and get the same data again from the script.
audience: everyone, newcomers first
order: 2
---

# Load data with Browse series (FRED, IMF, ECB, Eurostat, World Bank)

Find series at a data provider by searching or browsing, load them into the workspace as a dataset, and keep the
step so the script reads the same series again. The step recorded is the provider's address (`fred://GDPC1`), not
the search, so the script runs without the browser.

## Before you start

- A connection to the internet. The providers' public web services are used; nothing else is installed.
- What you are after, in the provider's terms. This page loads the ECB's monthly exchange rates of the US dollar and
  the pound sterling against the euro (dataflow `EXR`), US real GDP from FRED (`GDPC1`) and the World Bank's GDP in
  current dollars for Samoa and Tonga (`NY.GDP.MKTP.CD`).
- For FRED, whether you will search. Reading a FRED series by its ID needs nothing; searching FRED needs a free API
  key from the St. Louis Fed (set `FRED_API_KEY`, or type it in the browser, which keeps it for the session only).
  The World Bank and the SDMX providers (the ECB, Eurostat, the OECD, the IMF, the BIS) need no key.

## In the app

1. File → Browse series…. Provider: SDMX (ECB, Eurostat, OECD, IMF, BIS or your own), Source: ECB. Type
   `exchange rates` in the search box and press Search.
2. Choose the row EXR (Exchange Rates). Its dimensions appear under Series key, a box each. Choose FREQ: M, CURRENCY:
   USD and GBP, CURRENCY_DENOM: EUR, EXR_TYPE: SP00 and EXR_SUFFIX: A. The Key line shows the address,
   `sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A`; a dimension with nothing chosen takes any code.
3. Name: fx, then Load. The dataset `fx` has a series per currency, `gbp` and `usd`, labelled with the ECB's titles.
   Add to Load from connector… instead opens that dialog with the codes filled in, to give the first and last
   periods or other options before loading.
4. A series whose code you know needs no browsing: File → Load from connector…, with Data service: fred, Series
   codes: GDPC1 and Save as: gdp.
5. For the World Bank: Data service: worldbank, Series codes: NY.GDP.MKTP.CD, and under Advanced settings, Other
   parameters: country=WSM;TON. Save as: wb. Several countries make a panel, by country and year. In Browse series…
   the World Bank's countries are ticked in the Countries box instead.

Each load is one line of View → Script, `fx = Dataset.read('sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A', vintage='latest')`,
which reads the series afresh when the script runs.

## In code

```python
from pymetrix import Dataset, connectors

flows = connectors.search("sdmx", "exchange rates", source="ECB")   # the ECB's dataflows: ECB/EXR, Exchange Rates
exr = connectors.structure("sdmx", "ECB/EXR")                       # its dimensions and their codes
address = exr.address(FREQ="M", CURRENCY=["USD", "GBP"], CURRENCY_DENOM="EUR", EXR_TYPE="SP00", EXR_SUFFIX="A")
address                                                              # 'sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A'
fx = Dataset.read(address)                                           # monthly: gbp and usd
```

A code you know is read directly, and periods and options go after `?`:

```python
gdp = Dataset.read("fred://GDPC1")                                   # quarterly; no key needed to read by ID
wb = Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON")         # a panel by country and year
found = connectors.search("worldbank", "gdp")                        # NY.GDP.MKTP.CD first
connectors.to_frame(found)[["code", "title"]]
fx.write("fx.parquet")                                               # the values as loaded, to keep them
```

## Check the result

- The dataset's summary line: `fx` is monthly, `gdp` quarterly, `wb` a panel of two countries. A frequency other than
  the provider's means the wrong series or key.
- The Variables tab: each series' label is the provider's title ("US dollar/Euro ECB reference exchange rate"), and
  its source names the provider and the code (`ECB SDMX: EXR/M.USD+GBP.EUR.SP00.A`,
  `FRED, Federal Reserve Bank of St. Louis: GDPC1`). Where the provider says when it last updated the data (the
  World Bank, Eurostat, the IMF), that date is the series' vintage.
- The first and last values against the provider's own web page for the series.
- View → Script: the address, and no search or key in it.

## Pitfalls

- **Searching FRED needs a key; reading it does not.** Without one the browser says so and the search returns
  nothing. Type the key in the browser (FRED API key) or set `FRED_API_KEY`; it is never saved in the project.
- **Series from one address must share a frequency.** `fred://GDPC1,UNRATE` (quarterly and monthly) is refused:
  load them as two datasets and convert one with Data → Transform → Convert frequency….
- **The script reads the latest data, and so does the project.** Run again next month, or reopened, the step reads
  the provider again and gets revised numbers. For FRED, give Vintage (as published on) a date: ALFRED keeps the
  data as published that day. For the others, write the dataset to a file when you load it (`fx.write("fx.parquet")`)
  and load that file with File → Load data…: a project keeps the data of a file it loaded.
- **Searches and lists are kept for a while.** Searches are cached for a day and lists of dataflows and codes for a
  week; tick Ask again (skip the cache) in the browser, or pass `refresh=True`, when a provider has just added
  something.
- **SDMX keys are positional.** Each dimension has its place in the key, in the dataflow's order; build it in the
  browser or with `structure(...).address(...)` rather than by hand.

## See also

- [Data connectors](pymetrix:topic/data-connectors), in the help.
- [The user guide's Data from FRED, the World Bank and SDMX services](../../../docs/guide.md#data-from-fred-the-world-bank-and-sdmx-services)
  and [Finding series](../../../docs/guide.md#finding-series): every option of the addresses, the cache, and an
  institution's own SDMX service.
- [Save, reopen and share a project, and replay its script](save-and-share-a-project.md): keeping the data as
  loaded.
