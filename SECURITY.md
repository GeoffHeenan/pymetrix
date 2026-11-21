# Security

pymetrix runs on your own computer with your own permissions. This page says where it runs code
that you did not just type, and how it keeps that under your control.

## Projects

A project replays its steps when it is opened.

- **Dialog steps** run only pymetrix's own commands (estimate, test, solve and so on), with the
  settings saved in the project. The expressions those settings hold are checked before they are
  evaluated: row conditions (`where=`, a sample's condition) may use only column names, literals,
  comparisons, `and`/`or`/`not`, arithmetic and a short list of functions, and model files'
  expressions likewise; a step whose condition is refused is kept, not run.
- **Typed code**, from the Code and Immediate windows, is ordinary Python, which can do anything a
  program can. It runs on opening only when the project is trusted:
  - projects you saved yourself, with all their code run, are trusted;
  - for anyone else's project, pymetrix shows the code and asks first. You can open it without
    running the code: the data and dialog steps are restored, and the code is kept, marked as
    not run.
- **How trust is remembered.** By a SHA-256 fingerprint of the code itself, kept in
  `trusted_projects.json` in `~/.pymetrix` (or the folder named by `PYMETRIX_HOME`). Re-saving a
  project keeps its trust; any change to the code asks again.
- **In scripts,** `load_project(path, trust=...)` makes the same choice explicit, and
  `project_code(path)` shows the code before you decide.

Not yet available: a list for revoking trust, and signed projects.

## Plugins

Plugins are Python packages, and a plugin's code runs when it loads. Install plugins only from
sources you trust, as you would any Python package. The app asks before a newly installed plugin
first runs, showing only what its package's metadata says (name, version, author, licence, and
which kinds of extension it adds), since importing it would already run it. Scripts cannot stop to
ask: they load a new plugin with a warning. A plugin disabled in **Help → Plugins** (or with
`pymetrix.plugins.disable`) stays disabled in the app and in scripts; the choices are kept in
`plugins.json` in `~/.pymetrix` (or `$PYMETRIX_HOME`).

## Data connectors

The bundled connectors (FRED, the World Bank, SDMX providers) send requests only when you read
data from them, and only to the provider named in the request. They send no data of yours.

## AI assistance

pymetrix has no AI features yet. The planned design keeps data local by default and shows what
would be sent before anything leaves the machine; see the
[note on AI in the UI](docs/notes/ai-assistance-in-the-ui.md).

## Reporting a problem

Please report security problems privately, not in a public issue. Use GitHub's private
vulnerability reporting for this repository: the **Security** tab, then **Report a vulnerability**.
If you cannot use it, email hello@pymetrix.org instead.
Please include the version, what you did, and what happened.
