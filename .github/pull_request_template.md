### Proposed changes
Include a brief description of the changes being proposed, and why they are necessary.

### Closes issues (optional)
- Closes Issue #000
- Closes Issue #000

### Checklist (replace `[ ]` with `[x]` to check off)
- [ ] Remove any unused Python packages from `Load packages`
- [ ] Remove any unused/empty code cells
- [ ] Remove any guidance cells (e.g. `General advice`)
- [ ] Ensure that all code cells follow the [PEP8 standard](https://www.python.org/dev/peps/pep-0008/) for code. The `jupyterlab_code_formatter` tool can be used to format code cells to a consistent style: select each code cell, then click `Edit` and then one of the `Apply X Formatter` options (`YAPF` or `Black` are recommended).
- [ ] Include relevant tags in the final notebook cell and re-use tags if possible
- [ ] Clear all outputs, run notebook from start to finish, and save the notebook in the state where all cells have been sequentially evaluated
