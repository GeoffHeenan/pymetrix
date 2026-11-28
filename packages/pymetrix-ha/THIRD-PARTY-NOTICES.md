# Third-party notices: pymetrix-ha

The plugin's own code and documentation are under the BSD-3-Clause licence in `LICENSE`. It bundles no third-party
data. It depends on sequence-jacobian (MIT), which is installed from PyPI and is not in the plugin's sdist or wheel.

The Krusell-Smith example (`src/pymetrix_ha/examples/_krusell_smith.py`) adapts the blocks of sequence-jacobian's
Krusell-Smith notebook (`notebooks/krusell_smith.ipynb`), so that its results can be checked against the numbers the
notebook prints. sequence-jacobian's licence:

> MIT License
>
> Copyright (c) 2022 Sequence-Jacobian Team
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
