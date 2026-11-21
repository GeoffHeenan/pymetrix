# Analogue Projects And Positioning

Draft note for the pymetrix repository, October 2026.

## Why This Note Exists

pymetrix is unusual in combining three things:

- an open-source replacement path for workflows often done in commercial software;
- an integrated Python workbench rather than a single estimator library;
- substantial AI-assisted development, with human responsibility for design, validation and releases.

There are few exact precedents for all three together. There are, however, useful analogues.

## Closest Analogues

### SageMath

SageMath is probably the closest intellectual analogue. Its stated mission is to create "a viable free open source alternative to Magma, Maple, Mathematica and Matlab" ([SageMath](https://www.sagemath.org/index)). Like pymetrix, it is not just one algorithm; it is an integration layer over many mathematical systems, exposed through a coherent user-facing environment.

Relevance for pymetrix:

- integrates existing open-source mathematical libraries;
- targets workflows historically dominated by commercial tools;
- offers a unified environment rather than a narrow package;
- makes openness, reproducibility and extensibility part of the value proposition.

### QGIS

QGIS is a free and open-source geographic information system used for daily GIS viewing, data capture, advanced analysis, map production, atlases and reports ([QGIS documentation](https://docs.qgis.org/3.44/pdf/en/QGIS-3.44-AboutQGIS-en.pdf)).

Relevance for pymetrix:

- professional domain workbench;
- GUI plus plugins plus data formats plus analysis;
- credible open-source alternative to commercial workflow software;
- strong example of positioning around workflow, not just features.

### KiCad

KiCad is an open-source electronic design automation suite. Its project goal is to provide a cross-platform electronics design application for professional electronics designers while staying approachable to newer users ([KiCad](https://www.kicad.org/about/kicad/)).

Relevance for pymetrix:

- professional users take priority in direction-setting;
- open-source alternative in a domain with established commercial incumbents;
- integrated suite, not merely a library;
- useful example of governance and contributor expectations for technical software.

### FreeCAD

FreeCAD is an open-source parametric 3D modeler for designing real objects, with modelling history and extensible workbenches ([FreeCAD](https://www.freecad.org/)).

Relevance for pymetrix:

- workbench architecture;
- Python scripting and GUI coexist;
- open-source alternative to commercial design workflows;
- useful caution: extensibility and openness matter, but workflow polish and reliability are critical for professional trust.

### GNU Octave

GNU Octave is a scientific programming language whose syntax is largely compatible with MATLAB and which can run many MATLAB scripts ([GNU Octave](https://octave.org/)).

Relevance for pymetrix:

- compatibility-oriented open-source alternative;
- useful precedent for easing migration from commercial tools;
- less close to pymetrix's strategy, because pymetrix is not aiming to clone EViews or MATLAB syntax.

### Blender

Blender is a free and open-source 3D creation suite covering modelling, rendering, animation, video editing, VFX, compositing, texturing and simulation ([Blender](https://www.blender.org/)).

Relevance for pymetrix:

- demonstrates that an open-source suite can move beyond "free alternative" framing and become its own ecosystem;
- shows the long-term value of integrated workflows and a strong project identity.

### OpenFOAM

OpenFOAM is mature open-source computational fluid dynamics software used in industry and academia, often as an alternative to expensive proprietary CFD packages ([OpenFOAM Foundation](https://openfoam.org/news/funding-2026/)).

Relevance for pymetrix:

- powerful technical engine competing with commercial workflows;
- strong extensibility and scientific depth;
- cautionary lesson: a powerful engine without a smooth integrated workflow can remain difficult for non-specialists.

## Lessons For pymetrix

1. Lead with workflow, not estimator count.

   The strongest analogue projects are workbenches or environments. pymetrix should be presented as "applied macroeconometrics in Python, from data to forecast report", not as a list of estimators.

2. Integrate rather than rewrite.

   SageMath and QGIS show the value of being a coherent layer over specialized components. pymetrix should wrap mature libraries where possible and write native code only where the integrated data, expression, model or reproducibility layer requires it.

3. Treat plugins as central.

   QGIS and FreeCAD are examples of ecosystems where plugins and workbenches matter. For pymetrix, plugins are the natural route for institutional data stores, model collections, extra methods, course material and interoperability.

4. Support migration without becoming a clone.

   Octave is a compatibility model; QGIS and KiCad are workflow alternatives. pymetrix should interoperate with EViews, IRIS and Dynare where useful, but should not define itself as a clone of any of them.

5. Professional credibility depends on validation.

   Technical users will trust pymetrix when results are cross-checked, defaults are documented, project files are reproducible, and reports carry provenance.

6. AI assistance should be visible but not central to the product pitch.

   The fact that pymetrix has been developed with substantial AI assistance is relevant for governance and trust. The product pitch should still be about validated, human-maintained, open-source software.

## Suggested Positioning

pymetrix is an integrated Python workbench for applied macroeconometrics: data, estimation, model solving, forecasting, reports and reproducible projects in one workflow.

It belongs in the same broad family as SageMath, QGIS and KiCad: open-source technical workbenches that bring professional workflows out of proprietary environments and into extensible, inspectable software.

## What Not To Claim

pymetrix should not be described as:

- an EViews clone;
- a MATLAB replacement;
- a GUI for statsmodels;
- a complete replacement for IRIS, Dynare or OGP model suites;
- an AI-generated econometrics package.

Better:

> pymetrix is a Python-native macroeconometrics workbench, built from validated open-source components and native modelling infrastructure, with guided forms that generate reproducible Python.
