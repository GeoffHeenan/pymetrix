# Trying PyMetrix in your browser

This codespace has PyMetrix installed, with its desktop app running on a desktop in your browser. The
first-party plugins are installed too: finance (the **Finance** menu) and the research lab book.

1. **Open the app.** In the **Ports** tab (bottom panel), open **PyMetrix desktop** (port 6080)
   with the globe icon, then click **Connect**. The password is `vscode`. The app opens with the
   Lab 3 demo, a quarterly forecast round: look at **History** and **Script**, open objects from
   **Objects**, and try the menus.
2. **Other demos:** **File → Open demo project…**, or the course material under **Help**.
3. **Python and notebooks:** the notebooks in `examples/course/` run here in VS Code; choose the
   Python kernel.
4. **The lab book** records your work as you go. Save a project, then in the terminal run
   `pymetrix-labbook show <project>.pymx` to list its entries, and `pymetrix-labbook verify <project>.pymx` to check
   them.
5. **Seasonal adjustment:** STL, classical and TRAMO-SEATS work here (TRAMO-SEATS is Eurostat's JDemetra+: Java 21
   and its libraries come with the codespace). X-13 needs the Census Bureau's x13as program, which is not installed.
6. **If the app was closed:** run `bash .devcontainer/start-app.sh` in the terminal, or
   `pymetrix-gui --demo consumption --maximized` for another demo.

The browser desktop is slower than the app on your own computer; the local install is in the
README. Codespaces use your GitHub account's free monthly allowance (about 30 hours on this
4-core machine). Stop the codespace when you finish, and delete it when you no longer need it.
